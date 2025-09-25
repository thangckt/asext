from collections import deque

import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import Prism, convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.lammpsdata import _write_masses
from ase.io.lammpsrun import _mass2element, _parse_box_bound, get_max_index
from ase.parallel import paropen


#####ANCHOR: Read LAMMPS DUMP file
def _lammps_data_to_ase_atoms(
    data,
    colnames,
    cell,
    celldisp,
    pbc=False,
    atomsobj=Atoms,
    order=True,
    specorder=None,
    prismobj=None,
    units="metal",
):
    """Extract positions and other per-atom parameters and create Atoms

    :param data: per atom data
    :param colnames: index for data
    :param cell: cell dimensions
    :param celldisp: origin shift
    :param pbc: periodic boundaries
    :param atomsobj: function to create ase-Atoms object
    :param order: sort atoms by id. Might be faster to turn off.
    Disregarded in case `id` column is not given in file.
    :param specorder: list of species to map lammps types to ase-species
    (usually .dump files to not contain type to species mapping)
    :param prismobj: Coordinate transformation between lammps and ase
    :type prismobj: Prism
    :param units: lammps units for unit transformation between lammps and ase
    :returns: Atoms object
    :rtype: Atoms

    Notes:
    - The original function in `ase.io.lammpsrun.lammps_data_to_ase_atoms` can not recover the atom types. This function is modified to save the atom types if `type` column is given in the LAMMPS dump file.
    """
    if len(data.shape) == 1:
        data = data[np.newaxis, :]

    # read IDs if given and order if needed
    if "id" in colnames:
        ids = data[:, colnames.index("id")].astype(int)
        if order:
            sort_order = np.argsort(ids)
            data = data[sort_order, :]

    # determine the elements
    if "element" in colnames:
        # priority to elements written in file
        elements = data[:, colnames.index("element")]
    elif "mass" in colnames:
        # try to determine elements from masses
        elements = [_mass2element(m) for m in data[:, colnames.index("mass")].astype(float)]
    elif "type" in colnames:
        # fall back to `types` otherwise
        elements = data[:, colnames.index("type")].astype(int)

        # reconstruct types from given specorder
        if specorder:
            elements = [specorder[t - 1] for t in elements]
    else:
        # todo: what if specorder give but no types?
        # in principle the masses could work for atoms, but that needs
        # lots of cases and new code I guess
        raise ValueError("Cannot determine atom types form LAMMPS dump file")

    def get_quantity(labels, quantity=None):
        try:
            cols = [colnames.index(label) for label in labels]
            if quantity:
                return convert(data[:, cols].astype(float), quantity, units, "ASE")

            return data[:, cols].astype(float)
        except ValueError:
            return None

    # Positions
    positions = None
    scaled_positions = None
    if "x" in colnames:
        # doc: x, y, z = unscaled atom coordinates
        positions = get_quantity(["x", "y", "z"], "distance")
    elif "xs" in colnames:
        # doc: xs,ys,zs = scaled atom coordinates
        scaled_positions = get_quantity(["xs", "ys", "zs"])
    elif "xu" in colnames:
        # doc: xu,yu,zu = unwrapped atom coordinates
        positions = get_quantity(["xu", "yu", "zu"], "distance")
    elif "xsu" in colnames:
        # xsu,ysu,zsu = scaled unwrapped atom coordinates
        scaled_positions = get_quantity(["xsu", "ysu", "zsu"])
    else:
        raise ValueError("No atomic positions found in LAMMPS output")

    velocities = get_quantity(["vx", "vy", "vz"], "velocity")
    charges = get_quantity(["q"], "charge")
    forces = get_quantity(["fx", "fy", "fz"], "force")
    # !TODO: how need quaternions be converted?
    quaternions = get_quantity(["c_q[1]", "c_q[2]", "c_q[3]", "c_q[4]"])

    # convert cell
    cell = convert(cell, "distance", units, "ASE")
    celldisp = convert(celldisp, "distance", units, "ASE")
    if prismobj:
        celldisp = prismobj.vector_to_ase(celldisp)
        cell = prismobj.update_cell(cell)

    if quaternions is not None:
        out_atoms = atomsobj(
            symbols=elements,
            positions=positions,
            cell=cell,
            celldisp=celldisp,
            pbc=pbc,
        )
        out_atoms.new_array("quaternions", quaternions, dtype=float)
    elif positions is not None:
        # reverse coordinations transform to lammps system
        # (for all vectors = pos, vel, force)
        if prismobj:
            positions = prismobj.vector_to_ase(positions, wrap=True)

        out_atoms = atomsobj(
            symbols=elements,
            positions=positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell,
        )
    elif scaled_positions is not None:
        out_atoms = atomsobj(
            symbols=elements,
            scaled_positions=scaled_positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell,
        )

    if velocities is not None:
        if prismobj:
            velocities = prismobj.vector_to_ase(velocities)
        out_atoms.set_velocities(velocities)
    if charges is not None:
        out_atoms.set_initial_charges([charge[0] for charge in charges])
    if forces is not None:
        if prismobj:
            forces = prismobj.vector_to_ase(forces)
        # !TODO: use another calculator if available (or move forces
        #        to atoms.property) (other problem: synchronizing
        #        parallel runs)
        calculator = SinglePointCalculator(out_atoms, energy=0.0, forces=forces)
        out_atoms.calc = calculator

    # process the extra columns of fixes, variables and computes
    #    that can be dumped, add as additional arrays to atoms object
    for colname in colnames:
        # determine if it is a compute, fix or
        # custom property/atom (but not the quaternian)
        if (
            colname.startswith("f_")
            or colname.startswith("v_")
            or colname.startswith("d_")
            or colname.startswith("d2_")
            or (colname.startswith("c_") and not colname.startswith("c_q["))
        ):
            out_atoms.new_array(colname, get_quantity([colname]), dtype="float")

        elif colname.startswith("i_") or colname.startswith("i2_"):
            out_atoms.new_array(colname, get_quantity([colname]), dtype="int")
        elif colname == "type":
            try:
                out_atoms.new_array(colname, data[:, colnames.index("type")], dtype="int")
            except ValueError:
                pass  # in case type is not integer

    return out_atoms


def read_lammps_dump_text(file: str, index=-1, **kwargs):
    """Process cleartext lammps dumpfiles

    :param fileobj: filestream providing the trajectory data
    :param index: integer or slice object (default: get the last timestep)
    :returns: list of Atoms objects
    :rtype: list

    Notes:
    - This function is a modified version of `ase.io.lammpsrun.read_lammps_dump_text` to allow storing atom types if `type` column is given in the LAMMPS dump file.
    """
    fileobj = paropen(file)
    # Load all dumped timesteps into memory simultaneously
    lines = deque(fileobj.readlines())
    index_end = get_max_index(index)

    n_atoms = 0
    images = []

    # avoid references before assignment in case of incorrect file structure
    cell, celldisp, pbc, info = None, None, False, {}

    while len(lines) > n_atoms:
        line = lines.popleft()

        if "ITEM: TIMESTEP" in line:
            line = lines.popleft()
            # !TODO: pyflakes complains about this line -> do something
            ntimestep = int(line.split()[0])  # NOQA
            info["timestep"] = ntimestep

        if "ITEM: NUMBER OF ATOMS" in line:
            line = lines.popleft()
            n_atoms = int(line.split()[0])

        if "ITEM: BOX BOUNDS" in line:
            cell, celldisp, pbc = _parse_box_bound(line, lines)

        if "ITEM: ATOMS" in line:
            colnames = line.split()[2:]
            datarows = [lines.popleft() for _ in range(n_atoms)]
            data = np.loadtxt(datarows, dtype=str, ndmin=2)
            out_atoms = _lammps_data_to_ase_atoms(
                data=data,
                colnames=colnames,
                cell=cell,
                celldisp=celldisp,
                atomsobj=Atoms,
                pbc=pbc,
                **kwargs,
            )
            out_atoms.info.update(info)
            images.append(out_atoms)

        if len(images) > index_end >= 0:
            break

    return images[index]


#####ANCHOR: Write LAMMPS DATA file
def write_lammps_data(
    file: str,
    atoms: Atoms,
    *,
    specorder: list = None,
    reduce_cell: bool = False,
    force_skew: bool = False,
    prismobj: Prism = None,
    write_image_flags: bool = False,
    masses: bool = False,
    velocities: bool = False,
    units: str = "metal",
    bonds: bool = True,
    atom_style: str = "atomic",
):
    """Write atomic structure data to a LAMMPS data file.

    Parameters
    ----------
    fd : file|str
        File to which the output will be written.
    atoms : Atoms
        Atoms to be written.
    specorder : list[str], optional
        Chemical symbols in the order of LAMMPS atom types, by default None
    force_skew : bool, optional
        Force to write the cell as a
        `triclinic <https://docs.lammps.org/Howto_triclinic.html>`__ box,
        by default False
    reduce_cell : bool, optional
        Whether the cell shape is reduced or not, by default False
    prismobj : Prism|None, optional
        Prism, by default None
    write_image_flags : bool, default False
        If True, the image flags, i.e., in which images of the periodic
        simulation box the atoms are, are written.
    masses : bool, optional
        Whether the atomic masses are written or not, by default False
    velocities : bool, optional
        Whether the atomic velocities are written or not, by default False
    units : str, optional
        `LAMMPS units <https://docs.lammps.org/units.html>`__,
        by default 'metal'
    bonds : bool, optional
        Whether the bonds are written or not. Bonds can only be written
        for atom_style='full', by default True
    atom_style : {'atomic', 'charge', 'full'}, optional
        `LAMMPS atom style <https://docs.lammps.org/atom_style.html>`__,
        by default 'atomic'

    Notes:
    - This function is a modified version of `ase.io.lammpsdata.write_lammps_data` to allow writing atom types based on `atoms.arrays['type']` if it exists. Otherwise, the atom types are assigned based on the order of `specorder` or sorted chemical symbols.
    """

    fd = paropen(file, "w")

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError("Can only write one configuration to a lammps data file!")
        atoms = atoms[0]

    fd.write("(written by ASE)\n\n")

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    fd.write(f"{n_atoms} atoms\n")

    if specorder is not None:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        species = specorder
    elif "type" in atoms.arrays:
        species = _get_symbols_by_types(atoms)
    else:
        # This way it is assured that LAMMPS atom types are always
        # assigned predictably according to the alphabetic order
        species = sorted(set(symbols))

    n_atom_types = len(species)
    fd.write(f"{n_atom_types} atom types\n\n")

    bonds_in = []
    if bonds and (atom_style == "full") and (atoms.arrays.get("bonds") is not None):
        n_bonds = 0
        n_bond_types = 1
        for i, bondsi in enumerate(atoms.arrays["bonds"]):
            if bondsi != "_":
                for bond in bondsi.split(","):
                    dummy1, dummy2 = bond.split("(")
                    bond_type = int(dummy2.split(")")[0])
                    at1 = int(i) + 1
                    at2 = int(dummy1) + 1
                    bonds_in.append((bond_type, at1, at2))
                    n_bonds = n_bonds + 1
                    if bond_type > n_bond_types:
                        n_bond_types = bond_type
        fd.write(f"{n_bonds} bonds\n")
        fd.write(f"{n_bond_types} bond types\n\n")

    if prismobj is None:
        prismobj = Prism(atoms.get_cell(), reduce_cell=reduce_cell)

    # Get cell parameters and convert from ASE units to LAMMPS units
    xhi, yhi, zhi, xy, xz, yz = convert(prismobj.get_lammps_prism(), "distance", "ASE", units)

    fd.write(f"0.0 {xhi:23.17g}  xlo xhi\n")
    fd.write(f"0.0 {yhi:23.17g}  ylo yhi\n")
    fd.write(f"0.0 {zhi:23.17g}  zlo zhi\n")

    if force_skew or prismobj.is_skewed():
        fd.write(f"{xy:23.17g} {xz:23.17g} {yz:23.17g}  xy xz yz\n")
    fd.write("\n")

    if masses:
        _write_masses(fd, atoms, species, units)

    # Write (unwrapped) atomic positions.  If wrapping of atoms back into the
    # cell along periodic directions is desired, this should be done manually
    # on the Atoms object itself beforehand.
    fd.write(f"Atoms # {atom_style}\n\n")

    if write_image_flags:
        scaled_positions = atoms.get_scaled_positions(wrap=False)
        image_flags = np.floor(scaled_positions).astype(int)

    # when `write_image_flags` is True, the positions are wrapped while the
    # unwrapped positions can be recovered from the image flags
    pos = prismobj.vector_to_lammps(
        atoms.get_positions(),
        wrap=write_image_flags,
    )

    types = _get_types(atoms, species)

    if atom_style == "atomic":
        # Convert position from ASE units to LAMMPS units
        pos = convert(pos, "distance", "ASE", units)
        for i, r in enumerate(pos):
            s = types[i]
            line = f"{i + 1:>6} {s:>3} {r[0]:23.17g} {r[1]:23.17g} {r[2]:23.17g}"
            if write_image_flags:
                img = image_flags[i]
                line += f" {img[0]:6d} {img[1]:6d} {img[2]:6d}"
            line += "\n"
            fd.write(line)
    elif atom_style == "charge":
        charges = atoms.get_initial_charges()
        # Convert position and charge from ASE units to LAMMPS units
        pos = convert(pos, "distance", "ASE", units)
        charges = convert(charges, "charge", "ASE", units)
        for i, (q, r) in enumerate(zip(charges, pos)):
            s = types[i]
            line = f"{i + 1:>6} {s:>3} {q:>5} {r[0]:23.17g} {r[1]:23.17g} {r[2]:23.17g}"
            if write_image_flags:
                img = image_flags[i]
                line += f" {img[0]:6d} {img[1]:6d} {img[2]:6d}"
            line += "\n"
            fd.write(line)
    elif atom_style == "full":
        charges = atoms.get_initial_charges()
        # The label 'mol-id' has apparenlty been introduced in read earlier,
        # but so far not implemented here. Wouldn't a 'underscored' label
        # be better, i.e. 'mol_id' or 'molecule_id'?
        if atoms.has("mol-id"):
            molecules = atoms.get_array("mol-id")
            if not np.issubdtype(molecules.dtype, np.integer):
                raise TypeError(
                    f'If "atoms" object has "mol-id" array, then '
                    f"mol-id dtype must be subtype of np.integer, and "
                    f"not {molecules.dtype!s:s}."
                )
            if (len(molecules) != len(atoms)) or (molecules.ndim != 1):
                raise TypeError(
                    'If "atoms" object has "mol-id" array, then '
                    "each atom must have exactly one mol-id."
                )
        else:
            # Assigning each atom to a distinct molecule id would seem
            # preferableabove assigning all atoms to a single molecule
            # id per default, as done within ase <= v 3.19.1. I.e.,
            # molecules = np.arange(start=1, stop=len(atoms)+1,
            # step=1, dtype=int) However, according to LAMMPS default
            # behavior,
            molecules = np.zeros(len(atoms), dtype=int)
            # which is what happens if one creates new atoms within LAMMPS
            # without explicitly taking care of the molecule id.
            # Quote from docs at https://lammps.sandia.gov/doc/read_data.html:
            #    The molecule ID is a 2nd identifier attached to an atom.
            #    Normally, it is a number from 1 to N, identifying which
            #    molecule the atom belongs to. It can be 0 if it is a
            #    non-bonded atom or if you don't care to keep track of molecule
            #    assignments.

        # Convert position and charge from ASE units to LAMMPS units
        pos = convert(pos, "distance", "ASE", units)
        charges = convert(charges, "charge", "ASE", units)
        for i, (m, q, r) in enumerate(zip(molecules, charges, pos)):
            s = types[i]
            line = f"{i + 1:>6} {m:>3} {s:>3} {q:>5} {r[0]:23.17g} {r[1]:23.17g} {r[2]:23.17g}"
            if write_image_flags:
                img = image_flags[i]
                line += f" {img[0]:6d} {img[1]:6d} {img[2]:6d}"
            line += "\n"
            fd.write(line)
        if bonds and (atoms.arrays.get("bonds") is not None):
            fd.write("\nBonds\n\n")
            for i in range(n_bonds):
                bond_type = bonds_in[i][0]
                at1 = bonds_in[i][1]
                at2 = bonds_in[i][2]
                fd.write(f"{i + 1:>3} {bond_type:>3} {at1:>3} {at2:>3}\n")
    else:
        raise ValueError(atom_style)

    if velocities and atoms.get_velocities() is not None:
        fd.write("\n\nVelocities\n\n")
        vel = prismobj.vector_to_lammps(atoms.get_velocities())
        # Convert velocity from ASE units to LAMMPS units
        vel = convert(vel, "velocity", "ASE", units)
        for i, v in enumerate(vel):
            fd.write(f"{i + 1:>6} {v[0]:23.17g} {v[1]:23.17g} {v[2]:23.17g}\n")

    fd.flush()
    fd.close()
    return


def _get_types(atoms: Atoms, species: list):
    if "type" in atoms.arrays:
        types = atoms.arrays["type"]
    else:
        symbols = atoms.get_chemical_symbols()
        types = [species.index(symbols[i]) + 1 for i in range(len(symbols))]
    return types


def _get_symbols_by_types(atoms: Atoms):
    unique_types, first_idx = np.unique(atoms.arrays["type"], return_index=True)
    symbols_by_type = [atoms.symbols[i] for i in first_idx]
    return symbols_by_type
