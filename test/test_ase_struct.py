import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from asext.struct import set_vacuum

# TODO: Add all tests for ase_struct.py


class TestAddVacuum:
    def test_set_vacuum_orthogonal(self):
        struct = bulk("Na", "sc", a=2.0, cubic=True)
        # Add vacuum along each direction
        distances = [7.0, 5.0, 6.0]
        new_struct = set_vacuum(struct, distances)
        new_cell = new_struct.cell.array  # The new cell should be [7.0, 5.0, 6.0]
        assert np.allclose(np.diag(new_cell), [7.0, 5.0, 6.0], atol=1e-6)
        # Atoms should be centered within the vacuum
        center = new_struct.get_center_of_mass()
        for c in center:
            assert 2.0 < c < 4.0, f"Atom center {c} not within expected vacuum region"

    # def test_set_vacuum_triclinic(self):
    #     struct = bulk("Si", "diamond", a=3.57)
    #     distances = [5.0, 5.0, 5.0]
    #     new_struct = set_vacuum(struct, distances)
    #     cell = struct.cell.array
    #     # Check new cell vectors are longer by adding the vacuum amount
    #     for i in range(3):
    #         orig_len = np.linalg.norm(cell[i])
    #         new_len = np.linalg.norm(new_struct.get_cell()[i])
    #         frac_pos = struct.get_scaled_positions(wrap=False)
    #         span = (np.max(frac_pos[:, i]) - np.min(frac_pos[:, i])) * orig_len
    #         assert np.isclose(new_len, span + distances[i], atol=1e-6), (
    #             f"Cell vector {i} length incorrect"
    #         )

    def test_set_vacuum_invalid_distances(self):
        cell = np.eye(3) * 5.0
        atoms = Atoms("Na", positions=[[0, 0, 0]], cell=cell, pbc=True)
        with pytest.raises(AssertionError):
            set_vacuum(atoms, [1.0, 2.0])

    # def test_set_vacuum_zero_vector(self):
    #     cell = np.zeros((3, 3))
    #     atoms = Atoms("Na", positions=[[0, 0, 0]], cell=cell, pbc=True)
    #     with pytest.raises(ValueError):
    #         set_vacuum(atoms, [1.0, 1.0, 1.0])
