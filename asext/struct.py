from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.atoms import Atoms

import random
from pathlib import Path

import numpy as np

# from ase import units
from ase.io import read


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
