from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.atoms import Atoms

import random
from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np
import polars as pl
from ase import units
from ase.calculators.lammps import Prism
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data
from thkit.config import validate_config
from thkit.io import read_yaml, write_yaml

from alff.util.ase_cell import rotate_struct_property
from alff.util.key import SCHEMA_ASE_BUILD


#####ANCHOR ASE build structure
def build_struct(argdict: dict) -> Atoms:
    """
    Build atomic configuration, using library [`ase.build`](https://wiki.fysik.dtu.dk/ase/ase/build/build.html#)

    Supported structure types:
    - `bulk`: sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite.
    - `molecule`: molecule
    - `mx2`: MX2
    - `graphene`: graphene

    Args:
        argdict (dict): Parameters dictionary

    Returns:
        struct (Atoms): ASE Atoms object

    Notes:
        - `build.graphene()` does not set the cell c vector along z axis, so we need to modify it manually.
    """
    from ase.build import bulk, graphene, molecule, mx2

    ### validate input
    validate_config(config_dict=argdict, schema_file=SCHEMA_ASE_BUILD)

    ### Build structure with `ase.build`
    structure_type = argdict["structure_type"]
    ase_build_arg = deepcopy(argdict["ase_build_arg"])
    if structure_type == "bulk":
        ase_build_arg["name"] = argdict["chem_formula"]
        struct = bulk(**ase_build_arg)
    elif structure_type == "molecule":
        struct = molecule(**ase_build_arg)
    elif structure_type == "mx2":
        ase_build_arg["formula"] = argdict["chem_formula"]
        ase_build_arg["size"] = argdict.get("size", [1, 1, 1])
        ase_build_arg["vacuum"] = ase_build_arg.get("vacuum", 0.0)
        struct = mx2(**ase_build_arg)
    elif structure_type == "graphene":
        ase_build_arg["formula"] = argdict.get("chem_formula", "C2")
        ase_build_arg["a"] = ase_build_arg.get("a", 2.46)
        ase_build_arg["size"] = ase_build_arg.get("size", [1, 1, 1])
        ase_build_arg["thickness"] = 0.0
        struct = graphene(**ase_build_arg)
        ### Make the cell c vector along z axis
        real_thickness = argdict["ase_build_arg"].get("thickness", 3.35)
        c = struct.cell
        c[2, 2] = real_thickness
        struct.set_cell(c)
        struct.center()

    elif structure_type == "compound":
        print("not implemented yet")  # see `place_elements()` in dpgen
        ### May generate compound structure separately, and save as extxyz, then use option `from_extxyz` in `pdict`

    ### Make some modification on the built structure
    ## repeat cell
    supercell = argdict.get("supercell", [1, 1, 1])
    struct = struct.repeat(supercell)

    ## pbc
    pbc = argdict.get("pbc", [True, True, True])
    struct.set_pbc(pbc)

    ### Add vacuum padding (total vacuum distance both sides)
    vacuum_dists = argdict.get("set_vacuum", None)
    if vacuum_dists is not None:
        struct = set_vacuum(struct, vacuum_dists)

    # TODO: check ase_build_arg based on each function
    # labels: enhancement
    # use function config.argdict_to_schemadict to get the schema dict for each function
    return struct


#####ANCHOR ASE atoms manipulation
def strain_struct(struct_in: Atoms, strains: list = [0, 0, 0]) -> Atoms:
    """
    Apply engineering strain to an ASE Atoms structure along lattice vectors a, b, c.

    Args:
        struct (Atoms): ASE Atoms object.
        strains (list[float]): Engineering strains [ε_a, ε_b, ε_c]. New_length = old_length * (1 + ε).

    Returns:
        atoms (Atoms): New strained structure with scaled cell and atom positions.
    """
    strains = np.asarray(strains, dtype=float)
    assert strains.shape == (3,), "'factors' must be a sequence of 3 floats."

    struct = struct_in.copy()
    cell = struct.get_cell()
    scaled_cell = np.array([cell[i] * (1.0 + strains[i]) for i in range(3)])
    struct.set_cell(scaled_cell, scale_atoms=True)
    return struct


def perturb_struct(struct: Atoms, std_disp: float) -> Atoms:
    """Perturb the atoms by random displacements. This method adds random displacements to the atomic positions. [See more](https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.rattle)"""
    struct = struct.copy()
    seed_number = random.randrange(2**16)
    struct.rattle(stdev=std_disp, seed=seed_number)
    return struct


def slice_struct(struct_in: Atoms, slice_num=(1, 1, 1), tol=1.0e-5) -> Atoms:
    """
    Slice structure into the first subcell by given numbers along a, b, c (cell vector) directions.
    """
    struct = struct_in.copy()
    cell = struct.get_cell()

    ### shrink cell
    new_cell = np.array([cell[i] / slice_num[i] for i in range(3)])
    struct.set_cell(new_cell, strain_atoms=False)
    struct.wrap()  # wrap all atoms into the new cell

    ### Remove duplicate atoms (within tolerance)
    positions = struct.get_positions()
    unique_indices = []
    seen = []
    for i, pos in enumerate(positions):
        if not any(np.allclose(pos, s, atol=tol) for s in seen):
            unique_indices.append(i)
            seen.append(pos)

    ### make new struct
    new_struct = struct[unique_indices]
    new_struct.set_cell(new_cell, strain_atoms=False)
    new_struct.set_pbc(struct.get_pbc())

    print(f"\tInput structure: {len(struct)} atoms. \n\tSliced structure: {len(new_struct)} atoms.")
    return new_struct


def align_struct_min_pos(struct: Atoms) -> Atoms:
    """Align min atoms position to the min cell corner (0,0,0)"""
    min_pos = np.min(struct.positions, axis=0)
    struct.positions -= min_pos
    struct.wrap()
    return struct


def set_vacuum(struct_in: Atoms, distances: list = [0.0, 0.0, 0.0]) -> Atoms:
    """This function *sets* vacuum along cell vectors a, b, c.

    Args:
        struct (Atoms): ASE Atoms object to add vacuum.
        distances (list): Distances to add along cell vectors a, b, c (not x, y, z dims in Cartersian axes). Must be a list of 3 floats.

    Returns:
        struct: A new Atoms object with an expanded cell and centered atoms.

    Notes:
        - `atoms.center()` sets vacuum on both sides of the cell along the specified axis. So the total vacuum is *twice the input value*. This function is different in that, it set total vacuum equal to the input value.
    """
    assert len(distances) == 3, "'distances' must be a list of 3 floats."

    struct = struct_in.copy()
    for i in range(3):
        if distances[i] > 0:
            struct.center(vacuum=distances[i] / 2, axis=i)
    return struct


#####ANCHOR: ASE convert file formats
def poscar2lmpdata(
    poscar_file: str,
    lmpdata_file: str,
    atom_style: str = "atomic",
) -> list[str]:
    ### Note: The order of atom_names in lammpsdata is computed as: sorted(set(atoms.get_chemical_symbols()))
    ### REF: https://gitlab.com/ase/ase/-/blob/master/ase/io/lammpsdata.py
    """Convert POSCAR file to LAMMPS data file."""
    struct = read(poscar_file, format="vasp")
    write_lammps_data(lmpdata_file, struct, atom_style=atom_style, masses=True)
    symbols = struct.get_chemical_symbols()
    atom_names = sorted(set(symbols))
    return atom_names


def extxyz2lmpdata(
    extxyz_file: str,
    lmpdata_file: str,
    atom_style: str = "atomic",
) -> list[str]:
    """Convert extxyz file to LAMMPS data file.
    Note:
        - need to save 'original_cell' to able to revert the original orientation of the crystal.
        - Use `atoms.info['specorder']` to set atom types when convert from `extxyz` to `lammpsdata` file. Remember to convert its string format to list format by `split()`.
    """
    struct = read(extxyz_file, format="extxyz", index="-1")

    if "specorder" in struct.info:
        atom_names = struct.info["specorder"].split()  # convert string to list
        write_lammps_data(
            lmpdata_file, struct, atom_style=atom_style, masses=True, specorder=atom_names
        )
        write_yaml({"specorder": struct.info["specorder"]}, f"{lmpdata_file}.specorder.yml")
    else:
        write_lammps_data(lmpdata_file, struct, atom_style=atom_style, masses=True)
        symbols = struct.get_chemical_symbols()
        atom_names = sorted(set(symbols))

    ### Save some information
    original_cell = struct.cell
    np.savetxt(
        f"{lmpdata_file}.original_cell",
        original_cell,
        header="Revert original orientation by: ori_vec = Prism(original_cell).vector_to_ase()",
    )
    pbc = [1 if p else 0 for p in struct.get_pbc()]
    return atom_names, pbc


def lmpdata2extxyz(lmpdata_file: str, extxyz_file: str, original_cell_file: str = None):
    """Convert LAMMPS data file to extxyz file."""
    from ase.stress import voigt_6_to_full_3x3_stress  # full_3x3_to_voigt_6_stress

    atoms = read(lmpdata_file, format="lammps-data")
    ### recover original orientation
    if original_cell_file is None:
        original_cell_file = (
            f"{lmpdata_file}.original_cell" if glob(f"{lmpdata_file}.original_cell") else None
        )

    if original_cell_file is not None:
        original_cell = np.loadtxt(original_cell_file)
        p = Prism(original_cell)
        # atoms = align_struct_min_pos(atoms)
        atoms.positions = p.vector_to_ase(atoms.positions)
        atoms.cell = p.vector_to_ase(atoms.cell)
        if atoms.calc is not None and hasattr(atoms.calc, "results"):
            if "forces" in atoms.calc.results:
                atoms.calc.results["forces"] = p.vector_to_ase(atoms.calc.results["forces"])
            if "stress" in atoms.calc.results:  # stress in Voigt notation
                stress_3x3 = voigt_6_to_full_3x3_stress(atoms.calc.results["stress"])
                stress_3x3_rotate = p.tensor2_to_ase(stress_3x3)
                atoms.calc.results["stress"] = stress_3x3_rotate

    write(extxyz_file, atoms, format="extxyz")
    return


def lmpdump2extxyz(
    lmpdump_file: str,
    extxyz_file: str,
    original_cell_file: str = None,
    stress_file: str = None,
    lammps_units: str = "metal",
):
    ### Ref: /ase/io/lammpsrun.py; /ase/calculators/lammpslib.py/propagate()
    """Convert LAMMPS dump file to extxyz file.

    Args:
        lmpdump_file (str): Path to the LAMMPS dump file.
        extxyz_file (str): Path to the output extxyz file.
        original_cell_file (str, optional): Path to the text file contains original_cell. It should a simple text file that can write/read with numpy. If not provided, try to find in the same directory as `lmpdump_file` with the extension `.original_cell`. Defaults to None.
        stress_file (str, optional): Path to the text file contains stress tensor. Defaults to None.

    Restriction:
        - Current ver: stress is mapped based on frame_index, it requires that frames in text stress file must be in the same "length and order" as in the LAMMPS dump file.
        - `struct.info.get("timestep")` is a new feature in ASE 3.25 ?
    """

    if lammps_units == "metal":
        # stress_unit = units.bar
        stress_unit = units.bar / (units.eV / units.Angstrom**3)  # bar to eV/A^3

    ### Read stress file once if provided
    if stress_file is not None:
        stress_df = pl.read_csv(stress_file, separator=" ", comment_prefix="#")

    if original_cell_file is not None:
        old_cell = np.loadtxt(original_cell_file)

    ###
    struct_list = read(lmpdump_file, format="lammps-dump-text", index=":")
    new_struct_list = [None] * len(struct_list)
    for i, struct in enumerate(struct_list):
        ### init calc if not exist
        if struct.calc is None:
            struct.calc = SinglePointCalculator(atoms=struct)
        ### map stress
        if stress_file is not None:
            timestep = struct.info.get("timestep", None)
            if timestep:
                df = stress_df.filter(pl.col("time") == timestep)
                if df.height > 0:
                    stress = [df[k][0] for k in ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]]
                    stress = np.array(stress, dtype=float) * stress_unit
                    struct.calc.results["stress"] = stress
                    ### Energy
                    if "pe" in df.columns:
                        struct.calc.results["energy"] = float(df["pe"][0])
        ### Recover original orientation
        if original_cell_file is not None:
            struct = rotate_struct_property(struct, old_cell, wrap=False)

        new_struct_list[i] = struct
    write_extxyz(extxyz_file, new_struct_list)
    return


def write_extxyz(outfile: str, structs: list[Atoms]) -> None:
    """Write a list of Atoms object to an extxyz file. The exited `ase.io.write` function does not support writing file if the parent directory does not exist. This function will overcome this problem.

    Args:
        structs (list): List of Atoms object.
        outfile (str): Path to the output file.
    """
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    write(outfile, structs, format="extxyz")
    return


def read_extxyz(extxyz_file: str, index=":") -> list[Atoms]:
    """Read extxyz file. The exited `ase.io.read` returns a single Atoms object if file contains only one frame. This function will return a list of Atoms object.

    Args:
        extxyz_file (str): Path to the output file.

    Returns:
        list: List of Atoms object.

    Note:
        - `ase.io.read` returns a single Atoms object or a list of Atoms object, depending on the `index` argument.
            - `index=":"` will always return a list.
            - `index=0` or `index=-1` will return a single Atoms object.
        - this function will always return a list of Atoms object, even `index=0` or `index=-1`
    """
    struct_list = read(extxyz_file, format="extxyz", index=index)
    if not isinstance(struct_list, list):  ### Ensure the result is always a list
        struct_list = [struct_list]
    return struct_list


#####ANCHOR: ASE check structure
def check_bad_box_extxyz(
    extxyz_file: str,
    criteria: dict = {"length_ratio": 100, "wrap_ratio": 0.5, "tilt_ratio": 0.5},
) -> list[int]:
    """Check structure in extxyz file whether it has bad box.
    Return:
        a file remarking the bad box frames.
    """
    struct = read(extxyz_file, index="-1", format="extxyz")
    is_bad_box = check_bad_box(struct, criteria)
    if is_bad_box:
        with Path(f"{extxyz_file}.bad_box").open("w") as f:
            f.write("This frame has bad box.")
    return is_bad_box


def check_bad_box(
    struct: Atoms,
    criteria: dict = {"length_ratio": 20, "wrap_ratio": 0.5, "tilt_ratio": 0.5},
) -> bool:
    """
    Check if a simulation box is "bad" based on given criteria.

    Args:
    -----
    struct : ase.Atoms
        Atoms object containing the atomic structure.
    criteria : dict
        A dictionary of criteria to check, which contains pairs of {'criteria_name': threshold_value}.
        Available criteria:
        - `length_ratio`: The ratio of the longest to the shortest cell vector.
          - Formula: max(|a|, |b|, |c|) / min(|a|, |b|, |c|)
          - Prevents highly elongated simulation boxes.
        - `wrap_ratio`: Checks if one cell vector component is excessively wrapped around another.
          - Formula: [b_x / a_x, c_y / b_y, c_x / a_x]
          - Prevents excessive skewing.
        - `tilt_ratio`: Measures tilting of cell vectors relative to their axes.
          - Formula: [b_x / b_y, c_y / c_z, c_x / c_z]
          - Avoids excessive tilting that may disrupt periodic boundaries.

    Returns:
    --------
    is_bad : bool
        True if the simulation box violates any of the given criteria, otherwise False.

    Raises:
    -------
    RuntimeError
        If an unknown criterion key is provided.
    """
    cell = struct.cell.array

    is_bad = False
    for key, value in criteria.items():
        if key == "length_ratio":
            lens = np.linalg.norm(cell, axis=1)
            ratio = np.max(lens) / np.min(lens)
            if ratio > value:
                is_bad = True
        elif key == "wrap_ratio":
            ratio = [
                cell[1, 0] / cell[0, 0],
                cell[2, 1] / cell[1, 1],
                cell[2, 0] / cell[0, 0],
            ]
            if np.max(np.abs(ratio)) > value:
                is_bad = True
        elif key == "tilt_ratio":
            ratio = [
                cell[1, 0] / cell[1, 1],
                cell[2, 1] / cell[2, 2],
                cell[2, 0] / cell[2, 2],
            ]
            if np.max(np.abs(ratio)) > value:
                is_bad = True
        else:
            raise RuntimeError(f"Unknown criteria: {key}")
    return is_bad
