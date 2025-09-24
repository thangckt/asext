### REF: https://gitlab.com/ase/ase/-/blob/master/ase/test/cell/test_standard_form.py
#        https://gitlab.com/ase/ase/-/blob/master/ase/test/calculator/lammps/test_prism.py

import numpy as np
import pytest
from ase import Atoms
from ase.test.calculator.lammps.test_prism import make_array
from asext.cell import AseCell, CellTransform
from asext.struct import check_bad_box
from numpy.testing import assert_allclose


#####ANCHOR Test AseCell
def test_lower_triangular_form():
    TOL = 1e-10
    rng = np.random.RandomState(0)

    for _ in range(20):
        cell0 = rng.uniform(-1, 1, (3, 3))
        for sign in [-1, 1]:
            cell = AseCell(sign * cell0)
            rcell, Q = cell.lower_triangular_form()
            assert_allclose(rcell @ Q, cell, atol=TOL)
            assert_allclose(np.linalg.det(rcell), np.linalg.det(cell))
            assert_allclose(rcell.ravel()[[1, 2, 5]], 0, atol=TOL)


def test_upper_triangular_form():
    TOL = 1e-10
    rng = np.random.RandomState(0)

    for _ in range(20):
        cell0 = rng.uniform(-1, 1, (3, 3))
        for sign in [-1, 1]:
            cell = AseCell(sign * cell0)
            rcell, Q = cell.upper_triangular_form()
            assert_allclose(rcell @ Q, cell, atol=TOL)
            assert_allclose(np.linalg.det(rcell), np.linalg.det(cell))
            assert_allclose(rcell.ravel()[[3, 6, 7]], 0, atol=TOL)


#####ANCHOR Test CellTransform
ref_stress = np.array([[1, 5, 7], [2, 1, 4], [3, 5, 1]])


@pytest.mark.parametrize("structure", ("sc", "bcc", "fcc", "hcp"))
@pytest.mark.parametrize("wrap", (False, True))
@pytest.mark.parametrize("pbc", (False, True))
def test_cell_transform(structure: str, pbc: bool, wrap: bool):
    """Test if vector conversion works as expected"""
    rng = np.random.default_rng(42)
    positions = 20.0 * rng.random((10, 3)) - 10.0
    array = make_array(structure)
    atoms = Atoms(positions=positions, cell=array, pbc=pbc)
    vectors_ref = atoms.get_positions(wrap=wrap)

    rot = CellTransform(atoms.cell.array, atoms.cell.array)
    # vector
    vectors = rot.vectors_forward(vectors_ref)
    vectors = rot.vectors_backward(vectors)
    np.testing.assert_allclose(vectors, vectors_ref)
    # tensor
    tensor = rot.tensor_forward(ref_stress)
    tensor = rot.tensor_backward(tensor)
    np.testing.assert_allclose(tensor, ref_stress)


def test_check_bad_box():
    ortho_cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    bad_wrap_cell = np.array([[10, 0, 0], [6, 8, 0], [5, 4, 12]])
    bad_tilt_cell = np.array([[10, 0, 0], [0, 8, 0], [7, 4, 12]])
    atoms1 = Atoms(cell=ortho_cell)
    atoms2 = Atoms(cell=bad_wrap_cell)
    atoms3 = Atoms(cell=bad_tilt_cell)
    assert check_bad_box(atoms1) == False
    assert check_bad_box(atoms2) == True
    assert check_bad_box(atoms3) == True
