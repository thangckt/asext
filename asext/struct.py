"""Module for ASE Atoms structure manipulation and checking."""

from collections.abc import Sequence
from typing import cast

import numpy as np

from ase.atoms import Atoms
from ase.data import covalent_radii
from ase.io import read
from ase.neighborlist import neighbor_list


#####ANCHOR ASE atoms manipulation
def strain_struct(
    input_struct: Atoms,
    strains: Sequence[float] = (0.0, 0.0, 0.0),
) -> Atoms:
    """Apply engineering strain to an ASE Atoms structure along lattice vectors a, b, c.

    Args:
        input_struct (Atoms): ASE Atoms object.
        strains (list[float]): Engineering strains [ε_a, ε_b, ε_c]. New_length = old_length * (1 + ε).

    Returns:
        atoms (Atoms): New strained structure with scaled cell and atom positions.
    """
    strains_arr = np.asarray(strains, dtype=float)
    if strains_arr.shape != (3,):
        raise ValueError("'strains' must be a sequence of 3 floats.")

    struct = input_struct.copy()
    cell = struct.cell.array
    scaled_cell = np.array([cell[i] * (1.0 + strains_arr[i]) for i in range(3)])
    struct.set_cell(scaled_cell, scale_atoms=True)
    return struct


def random_displace_struct(struct: Atoms, std_disp: float, seed=42) -> Atoms:
    """Apply random displacements to atomic positions, using [Atoms.rattle](https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.rattle) which eventually calls `np.random.RandomState(seed).normal` to generate random samples from a normal (Gaussian) distribution.

    Args:
        struct (Atoms): ASE Atoms object to perturb.
        std_disp (float): Standard deviation of the random displacements in Angstrom.
        seed (int, optional): Seed for the random number generator. Default is 42 for reproducibility.

    Returns:
        struct (Atoms): New structure with random displacements.
    """
    new_struct = struct.copy()
    new_struct.rattle(stdev=std_disp, seed=seed)
    return new_struct


def slice_struct(
    struct_in: Atoms,
    slice_num: Sequence[float] = (1, 1, 1),
    tol: float = 1.0e-5,
) -> Atoms:
    """Slice structure into the first subcell by given numbers along a, b, c (cell vector) directions."""
    struct = struct_in.copy()
    cell = struct.cell.array

    ### shrink cell
    new_cell = np.array([cell[i] / slice_num[i] for i in range(3)])
    struct.set_cell(new_cell, scale_atoms=False)
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
    new_struct.set_cell(new_cell, scale_atoms=False)
    new_struct.set_pbc(struct.get_pbc())

    print(f"\tInput structure: {len(struct)} atoms. \n\tSliced structure: {len(new_struct)} atoms.")
    return new_struct


def align_struct_min_pos(struct: Atoms) -> Atoms:
    """Align min atoms position to the min cell corner (0,0,0)."""
    min_pos = np.min(struct.positions, axis=0)
    struct.positions -= min_pos
    struct.wrap()
    return struct


def set_vacuum(input_struct: Atoms, distances: Sequence[float] = (0.0, 0.0, 0.0)) -> Atoms:
    """This function *sets* vacuum along cell vectors a, b, c.

    Args:
        input_struct (Atoms): ASE Atoms object to add vacuum.
        distances (list): Distances to add along cell vectors a, b, c (not x, y, z dims in Cartersian axes). Must be a list of 3 floats.

    Returns:
        struct: A new Atoms object with an expanded cell and centered atoms.

    Notes:
        - `atoms.center()` sets vacuum on both sides of the cell along the specified axis. So the total vacuum is *twice the input value*. This function is different in that, it set total vacuum equal to the input value.
    """
    if len(distances) != 3:
        raise ValueError("'distances' must be a sequence of 3 floats.")

    struct = input_struct.copy()
    for i in range(3):
        if distances[i] > 0:
            struct.center(vacuum=distances[i] / 2, axis=i)
    return struct


#####ANCHOR ASE check structure
def check_bad_box_extxyz(
    extxyz_file: str,
    criteria: dict | None = None,
) -> bool:
    """Check structure in extxyz file whether it has bad box.

    Args:
        extxyz_file (str): Path to the extxyz file containing the structure to check.
        criteria (dict, optional): A dictionary of criteria to check

    Returns:
        a file remarking the bad box frames.
    """
    struct = read(extxyz_file, index="-1", format="extxyz")
    struct = cast(Atoms, struct)  # for type checking
    return check_bad_box(struct, criteria)


def check_bad_box(
    struct: Atoms,
    criteria: dict | None = None,
) -> bool:
    """Check if a simulation box is "bad" based on given criteria.

    Args:
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
        is_bad : bool
            True if the simulation box violates any of the given criteria, otherwise False.

    Raises:
        RuntimeError: If an unknown criterion key is provided.
    """
    eps = 1.0e-12

    ### Helper functions
    def _safe_ratio(num: float, den: float) -> float:
        if abs(den) < eps:
            return np.inf if abs(num) >= eps else 0.0
        return num / den

    default_criteria = {"length_ratio": 20, "wrap_ratio": 0.5, "tilt_ratio": 0.5}
    if criteria is None:
        criteria = default_criteria
    else:
        default_criteria.update(
            criteria
        )  # Use provided criteria, but fill in any missing ones with defaults
        criteria = default_criteria

    cell = struct.cell.array

    is_bad = False
    for key, value in criteria.items():
        if key == "length_ratio":
            lens = np.linalg.norm(cell, axis=1)
            min_len = np.min(lens)
            ratio = np.inf if min_len < eps else float(np.max(lens) / min_len)
            if ratio > value:
                is_bad = True
        elif key == "wrap_ratio":
            ratio = [
                _safe_ratio(cell[1, 0], cell[0, 0]),
                _safe_ratio(cell[2, 1], cell[1, 1]),
                _safe_ratio(cell[2, 0], cell[0, 0]),
            ]
            if np.max(np.abs(ratio)) > value:
                is_bad = True
        elif key == "tilt_ratio":
            ratio = [
                _safe_ratio(cell[1, 0], cell[1, 1]),
                _safe_ratio(cell[2, 1], cell[2, 2]),
                _safe_ratio(cell[2, 0], cell[2, 2]),
            ]
            if np.max(np.abs(ratio)) > value:
                is_bad = True
        else:
            raise RuntimeError(f"Unknown criteria: {key}")
    return is_bad


def check_atoms_too_close(struct: Atoms) -> None:
    ### https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/utilities/__init__.py
    """Check if any atoms are too close to each other.

    Args:
        struct (Atoms): ASE Atoms object to check.

    Raises:
        ValueError: If any pair of atoms are closer than the sum of their covalent radii (with a small tolerance).

    Notes:
         This function is adapted from [gpaw](https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/utilities/__init__.py)
    """
    radii = covalent_radii[struct.numbers] * 0.01
    dists = neighbor_list("d", struct, radii)
    if len(dists):
        raise ValueError(f"Atoms are too close, e.g. {dists[0]} Å")
    return


def check_atoms_too_close_to_boundary(struct: Atoms, dist: float = 0.2) -> None:
    """Check if any atoms are too close to the boundary of the box.

    Args:
        struct (Atoms): ASE Atoms object to check.
        dist (float): Distance threshold in Å. Atoms closer than this distance to the boundary will raise an error.

    Raises:
        ValueError: If any atom is closer than the specified distance to the boundary of the box.

    Notes: This function is adapted from [gpaw](https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/utilities/__init__.py)
    """
    for axis_v, recip_v, pbc in zip(struct.cell, struct.cell.reciprocal(), struct.pbc):
        if pbc:
            continue
        L = np.linalg.norm(axis_v)
        if L < 1e-12:  # L==0 means no boundary
            continue
        spos_a = struct.positions @ recip_v
        eps = dist / L
        if (spos_a < eps).any() or (spos_a > 1 - eps).any():
            raise ValueError("Atoms too close to boundary")
    return
