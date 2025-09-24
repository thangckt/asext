from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.atoms import Atoms

from copy import deepcopy

import numpy as np

# from asext.struct import align_struct_min_pos
from ase.cell import Cell
from ase.io import read, write


#####ANCHOR ASE new_cell
class AseCell(Cell):
    # ase.cell.Cell.standard_form() return a right-handed lower triangular cell . See https://gitlab.com/ase/ase/-/blob/master/ase/cell.py?ref_type=heads#L333
    # This class make some utility functions to convert the cell to a right-handed upper triangular cell
    def __init__(self, array: np.ndarray):
        super().__init__(array)

    def lower_triangular_form(self) -> tuple[Cell, np.ndarray]:
        """Rename original function `Cell.standard_form()`, see https://gitlab.com/ase/ase/-/blob/master/ase/cell.py?ref_type=heads#L333"""
        return self.standard_form(form="lower")

    def upper_triangular_form(self) -> tuple[Cell, np.ndarray]:
        """Rotate axes such that the unit cell is an upper triangular matrix."""
        return self.standard_form(form="upper")


def make_upper_triangular_cell(atoms: Atoms, zero_tol: float = 1.0e-12) -> Atoms:
    """Atoms with a box is an *upper triangular matrix* is a requirement to run `NPT` class in ASE.
    [[ ax, ay, az ]
     [  0, by, bz ]
     [  0,  0, cz ]]
    """
    new_cell = AseCell(atoms.cell.array).upper_triangular_form()[0]
    atoms = rotate_struct_property(atoms, new_cell.array, wrap=True)
    ### Zero out the small cell-lengths
    # modified_cell = atoms.cell.array.copy()  # Ensure we don't modify the original directly
    # modified_cell[modified_cell < zero_tol] = 0.0
    # atoms.set_cell(modified_cell, strain_atoms=True)
    return atoms


def make_lower_triangular_cell(atoms: Atoms, zero_tol: float = 1.0e-12) -> Atoms:
    """Converts the cell matrix of `atoms` into a *lower triangular*, to be used in LAMMPS:
    [[ ax,  0,  0 ]
     [ bx, by,  0 ]
     [ cx, cy, cz ]]
    """
    new_cell = AseCell(atoms.cell.array).lower_triangular_form()[0]
    atoms = rotate_struct_property(atoms, new_cell.array, wrap=True)
    ### Zero out the small cell-lengths
    # modified_cell = atoms.cell.array.copy()  # Ensure we don't modify the original directly
    # modified_cell[modified_cell < zero_tol] = 0.0
    # atoms.set_cell(modified_cell, strain_atoms=True)
    return atoms


def make_triangular_cell_extxyz(extxyz_file: str, form: str = "lower") -> None:
    """Make the cell of atoms in extxyz file to be triangular.
    Args:
        extxyz_file (str): Path to the extxyz file.
        form (str): 'upper' or 'lower'. Defaults to 'lower'.
    """
    struct_list = read(extxyz_file, format="extxyz", index=":")
    new_struct_list = []
    for atoms in struct_list:
        if form == "upper":
            atoms = make_upper_triangular_cell(atoms)
        elif form == "lower":
            atoms = make_lower_triangular_cell(atoms)
        new_struct_list.append(atoms)
    write(extxyz_file, new_struct_list, format="extxyz")
    return


#####ANCHOR ASE cell rotation
class CellTransform:
    ### Note: There is a [Prism class](https://gitlab.com/ase/ase/-/blob/master/ase/calculators/lammps/coordinatetransform.py?ref_type=heads#L88) which tranforms the cell between LAMMPS and ASE. But It looks complicated, and may used to live update the cell during the simulation.
    """Tranform the cell and atom properties from `old_cell` to `new_cell` orientations.

    The idea is compute a linear transformation that maps the old cell to the new cell. `A = solve(old_cell, new_cell) = old_cell^(-1) new_cell`

    Generally, this linear transformation `A` can include rotation R + shear/reshape U (stretching and shearing), i.e., `A = R * U`.

    Therefore, this transformation can be used in two ways:
    1. Directly apply `A` that includes both rotation and shear/stretch. (should avoid using this, since it is not clear how to transform properties like stress/forces)
    2. Extract only the rotation part `R` from `A` (using polar decomposition), and use it to rotate vectors/tensors, ignoring shear/reshape change.
        - Extract the closest pure rotation `R` from `A` (using polar decomposition)
        - Use that `R` to rotate positions, forces, stress, etc.

    Args:
        old_cell (np.ndarray): 3x3 matrix represent the old cell.
        new_cell (np.ndarray): 3x3 matrix represent the new cell.
        pure_rotation (bool): If True, only use the rotation part of the transformation. Defaults to True.

    Note:
        - `np.linalg.solve(A, B)` solves `AX = B` for `X`. May fail if `A` is singular (square matrix with a determinant of zero, det(A)=0).
        - Rotation matrix is derived from QR decomposition of the cell, following [Prism class](https://gitlab.com/ase/ase/-/blob/master/ase/calculators/lammps/coordinatetransform.py?ref_type=heads#L88)
    """

    def __init__(self, old_cell: np.ndarray, new_cell: np.ndarray, pure_rotation: bool = True):
        self.old_cell = np.asarray(old_cell, dtype=float)
        self.new_cell = np.asarray(new_cell, dtype=float)
        if self.old_cell.shape != (3, 3) or self.new_cell.shape != (3, 3):
            raise ValueError("Cells must be 3x3 matrices.")

        A = np.linalg.solve(self.old_cell, self.new_cell)
        if pure_rotation:
            self.R = _polar_rotation(A)  # ratation matrix
            ## check if R is orthogonal
            if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-9):
                raise ValueError("R is not orthogonal; polar decomposition may have failed.")
        else:
            self.R = A
        return

    def vectors_forward(self, vec: np.ndarray) -> np.ndarray:
        """Rotate vectors from the old_cell's orient to the new_cell's orient.

        Args:
            vec (np.ndarray): Nx3 matrix represent the vector properties. (positions, forces, etc. each row is a vector)

        Returns:
            np.ndarray: Rotated vectors.
        """
        vec = np.asarray(vec, dtype=float)
        return vec @ self.R

    def vectors_backward(self, vec: np.ndarray) -> np.ndarray:
        """Rotate vectors back from the new_cell to the old_cell. Same as [Prism.vector_to_ase](https://gitlab.com/ase/ase/-/blob/master/ase/calculators/lammps/coordinatetransform.py?ref_type=heads#L249)"""
        vec = np.asarray(vec, dtype=float)
        return vec @ self.R.T

    def tensor_forward(self, tensor: np.ndarray) -> np.ndarray:
        """Rotate the tensor from the old_cell's orient to the new_cell's orient.
        (T' = Rᵀ T R) rotates the tensor into the rotated coordinate system

        Args:
            tensor (np.ndarray): 3x3 matrix represent the tensor properties. (e.g., 3x3 stress tensor)

        Returns:
            np.ndarray: Transformed tensor.
        """
        tensor = np.asarray(tensor, dtype=float)
        return self.R.T @ tensor @ self.R

    def tensor_backward(self, tensor: np.ndarray) -> np.ndarray:
        """Rotate the tensor back from the new_cell to the old_cell. Same as [Prism.tensor_to_ase](https://gitlab.com/ase/ase/-/blob/master/ase/calculators/lammps/coordinatetransform.py?ref_type=heads#L278)
        (T = R T' Rᵀ) rotates the tensor back into the original coordinate system
        """
        tensor = np.asarray(tensor, dtype=float)
        return self.R @ tensor @ self.R.T


def _polar_rotation(A: np.ndarray) -> np.ndarray:
    """Return the closest proper rotation to matrix A (polar decomposition).

    The purpose of this function is to get only the orientation difference, ignoring any shear/stretch.

    Remind: Given a linear transformation `A=old_cell^(-1) new_cell` (carry old cell vectors into the new cell vectors), we can decompose it into a rotation R and a symmetric positive semi-definite matrix U (which represents stretch/shear) such that `A = R * U`. The rotation matrix R captures the pure rotational component of the transformation, while U captures the deformation (stretching and shearing) component.
    """
    from scipy.linalg import polar

    R, U = polar(A)
    # Ensure right-handed rotation (det = +1)
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R


def rotate_struct_property(
    struct: Atoms,
    new_cell: np.ndarray,
    wrap: bool = False,
    custom_vector_props: list[str] | None = None,
    custom_tensor_props: list[str] | None = None,
) -> Atoms:
    """
    Rotate atomic structure and its properties to match a new cell orientation.

    Args:
        struct (ase.Atoms): Atoms object.
        new_cell (np.ndarray): 3x3 matrix represent the new cell.
        wrap (bool): If True, wrap atoms into the new cell.
        custom_vector_props (list): List of vector properties to rotate. This allows to set vector properties with custom names.
        custom_tensor_props (list): List of tensor properties to rotate. This allows to set tensor properties with custom names.

    Returns:
        ase.Atoms: Atoms object with rotated properties.

    Note:
        - Important note: `deepcopy(struct)` copies the `struct.calc` object, but `struct.copy()` does not.
    """
    from ase.stress import voigt_6_to_full_3x3_stress  # full_3x3_to_voigt_6_stress

    custom_vector_props = custom_vector_props or []
    custom_tensor_props = custom_tensor_props or []

    struct = deepcopy(struct)
    old_cell = struct.cell.array.copy()
    rot = CellTransform(old_cell, new_cell)
    # struct = align_struct_min_pos(struct)

    ### Rotate positions and cell
    struct.positions = rot.vectors_forward(struct.positions)
    struct.cell = rot.vectors_forward(old_cell)
    # note: use `atoms.cell = rot.vectors_backward(atoms.cell.array)` may return non-zero small value (ex: 1e-16) in the cell matrix
    if wrap:
        struct.wrap()

    ### Rotate built-in properties
    if struct.calc and hasattr(struct.calc, "results"):
        res = struct.calc.results
        if "forces" in res:
            struct.calc.results["forces"] = rot.vectors_forward(res["forces"])
        if "stress" in res:  # stress is in Voigt notation
            stress_3x3 = voigt_6_to_full_3x3_stress(res["stress"])
            stress_3x3_rotate = rot.tensor_forward(stress_3x3)
            struct.calc.results["stress"] = stress_3x3_rotate

    ### Rotate custom properties
    for prop in custom_vector_props:
        if prop in struct.arrays:
            struct.arrays[prop] = rot.vectors_forward(struct.arrays[prop])
    for prop in custom_tensor_props:
        if prop in struct.info:
            tensor = np.asarray(struct.info[prop])
            assert tensor.shape == (3, 3), f"Tensor property {prop} must be a 3x3 matrix."
            struct.info[prop] = rot.tensor_forward(tensor)
    return struct


#####ANCHOR functions from other sources
