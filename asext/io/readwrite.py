from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.atoms import Atoms

from glob import glob
from pathlib import Path

import numpy as np
import polars as pl
from ase import units
from ase.calculators.lammps import Prism
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from asext.cell import rotate_struct_property
from asext.io.lmpdata import _get_symbols_by_types, read_lammps_dump_text, write_lammps_data


#####ANCHOR: Read/Write extxyz file
def read_extxyz(extxyz_file: str, index=":") -> list[Atoms]:
    """Read extxyz file. The existing `ase.io.read` returns a single Atoms object if file contains only one frame. This function will return a list of Atoms object.

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


def write_extxyz(outfile: str, structs: list[Atoms]) -> None:
    """Write a list of Atoms object to an extxyz file. The existing `ase.io.write` function does not support writing file if the parent directory does not exist. This function will overcome this problem.

    Args:
        structs (list): List of Atoms object.
        outfile (str): Path to the output file.
    """
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    write(outfile, structs, format="extxyz")
    return


#####ANCHOR: Read/Write LAMMPS data file
def read_lmpdump(lmpdump_file: str, index=-1, units="metal", **kwargs) -> list[Atoms]:
    """Shortcut to `ase.io.lammpsrun.read_lammps_dump` function.

    Args:
        lmpdump_file (str): Path to the LAMMPS dump file.
        index (int | list[int]): integer or slice object (default: get the last timestep)
        order (bool): sort atoms by id. Might be faster to turn off. Default: True
        specorder (list[str]): list of species to map lammps types to ase-species. Default: None
        units (str): lammps units for unit transformation between lammps and ase

    Returns:
        list: List of Atoms object.
    """
    struct_list = read_lammps_dump_text(lmpdump_file, index=index, units=units, **kwargs)
    if not isinstance(struct_list, list):  ### Ensure the result is always a list
        struct_list = [struct_list]
    return struct_list


def write_lmpdata(
    file: str,
    atoms: Atoms,
    *,
    specorder: list = None,
    reduce_cell: bool = False,
    force_skew: bool = False,
    prismobj: Prism = None,
    write_image_flags: bool = False,
    masses: bool = True,
    velocities: bool = False,
    units: str = "metal",
    bonds: bool = True,
    atom_style: str = "atomic",
) -> None:
    """Shortcut to `ase.io.lammpsdata.write_lammps_data` function.

    Args:
        file (str): File to which the output will be written.
        atoms (Atoms): Atoms to be written.
        specorder (list[str], optional): Chemical symbols in the order of LAMMPS atom types, by default None
        force_skew (bool, optional): Force to write the cell as a `triclinic <https://docs.lammps.org/Howto_triclinic.html>` box, by default False
        reduce_cell (bool, optional): Whether the cell shape is reduced or not, by default False
        prismobj (Prism|None, optional): Prism, by default None
        write_image_flags (bool): default False. If True, the image flags, i.e., in which images of the periodic simulation box the atoms are, are written.
        masses (bool, optional): Whether the atomic masses are written or not, by default True
        velocities (bool, optional): Whether the atomic velocities are written or not, by default False
        units (str, optional): `LAMMPS units <https://docs.lammps.org/units.html>`, by default 'metal'
        bonds (bool, optional): Whether the bonds are written or not. Bonds can only be written for atom_style='full', by default True
        atom_style : {'atomic', 'charge', 'full'}, optional. `LAMMPS atom style <https://docs.lammps.org/atom_style.html>`, by default 'atomic'

    Returns:
        None
    """
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    write_lammps_data(
        file,
        atoms,
        specorder=specorder,
        reduce_cell=reduce_cell,
        force_skew=force_skew,
        prismobj=prismobj,
        write_image_flags=write_image_flags,
        masses=masses,
        velocities=velocities,
        units=units,
        bonds=bonds,
        atom_style=atom_style,
    )
    return


#####ANCHOR: Convert formats
def extxyz2lmpdata(
    extxyz_file: str,
    lmpdata_file: str,
    masses: bool = True,
    units: str = "metal",
    atom_style: str = "atomic",
    **kwargs,
) -> list[str]:
    """Convert extxyz file to LAMMPS data file.
    Note:
        - need to save 'original_cell' to able to revert the original orientation of the crystal.
        - Use `atoms.arrays['type']` to set atom types when convert from `extxyz` to `lammpsdata` file.
    """
    struct = read(extxyz_file, format="extxyz", index="-1")

    write_lmpdata(lmpdata_file, struct, masses=masses, units=units, atom_style=atom_style, **kwargs)
    if "type" in struct.arrays:
        atom_names = _get_symbols_by_types(struct)
    else:
        atom_names = sorted(set(struct.get_chemical_symbols()))

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
    index: int | slice = -1,
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
    struct_list = read_lmpdump(lmpdump_file, index=index, units=lammps_units)
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


# def poscar2lmpdata(
#     poscar_file: str,
#     lmpdata_file: str,
#     atom_style: str = "atomic",
# ) -> list[str]:
#     ### Note: The order of atom_names in lammpsdata is computed as: sorted(set(atoms.get_chemical_symbols()))
#     ### REF: https://gitlab.com/ase/ase/-/blob/master/ase/io/lammpsdata.py
#     """Convert POSCAR file to LAMMPS data file."""
#     struct = read(poscar_file, format="vasp")
#     write_lmpdata(lmpdata_file, struct, atom_style=atom_style, masses=True)
#     symbols = struct.get_chemical_symbols()
#     atom_names = sorted(set(symbols))
#     return atom_names
