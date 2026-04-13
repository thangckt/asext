"""Module for custom ASE calculators and related utilities."""

import numpy as np
from openbabel import openbabel

from ase.calculators.calculator import Calculator, all_properties
from ase.units import kJ, mol


class OpenBabelFFCalculator(Calculator):
    """A custom ASE calculator that uses Open Babel's force field library.

    Supports calculation of energy, forces, and stress tensors.

    The stress tensor is computed using the virial theorem from forces calculated
    by the Open Babel force field:

        σ_αβ = -(1/V) * Σᵢ rᵢ^α * fᵢ^β

    Args:
        forcefield (str): The name of the [Open Babel force field](https://openbabel.org/docs/Forcefields/Overview.html) to use (e.g. "uff", "mmff94", etc.). Default is "uff".
        kwargs: Additional keyword arguments passed to the ASE's base `Calculator` class.

    Note:
        - `openbabel` is no longer maintained, and not supported in Python 3.13+. Use [`openbabel-wheel`](https://github.com/njzjz/openbabel-wheel) instead.
        - This calculator extends [the original implementation](https://github.com/otayfuroglu/DeepConf/blob/main/ase_ff.py) with stress calculations.
        - Stress computation is based on the virial formula (Irving & Kirkwood, 1950; Ray & Rahman, 1984).
        - Stress is most meaningful for periodic systems; use with caution for isolated molecules.

    References:
        Irving, J. H.; Kirkwood, J. G. (1950). "The Statistical Mechanical Theory of Transport Processes IV."
        Journal of Chemical Physics, 18(6), 817-829. https://doi.org/10.1063/1.1747782
    """

    implemented_properties = ("energy", "forces", "stress")

    def __init__(self, forcefield="uff", **kwargs):
        super().__init__(**kwargs)
        self.forcefield = forcefield

    def calculate(self, atoms=None, properties=all_properties, system_changes=all_properties):
        super().calculate(atoms, properties, system_changes)

        # Convert ASE atoms to Open Babel molecule
        obmol = self._atoms_to_obmol(atoms)

        # Set up the force field
        obff = openbabel.OBForceField.FindForceField(self.forcefield)
        if obff is None:
            raise ValueError(f"Force field {self.forcefield} not found in Open Babel.")

        obff.Setup(obmol)

        ### Compute energy
        energy = obff.Energy() * kJ / mol

        ### Compute forces
        forces = np.zeros((obmol.NumAtoms(), 3))
        #  obff.GetGradient()  # Ensure gradients are updated
        for i in range(obmol.NumAtoms()):
            atom = obmol.GetAtom(i + 1)  # Open Babel uses 1-based indexing
            force = obff.GetGradient(atom)  # Ensure gradients are updated
            forces[i:] = np.array([force.GetX(), force.GetY(), force.GetZ()]) * kJ / mol

        ### Compute stress (use virial stress formula)
        stress = self._compute_virial_stress(atoms, forces)

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress

    #  @staticmethod
    #  def ase_to_obmol(atoms):
    #      """Convert ASE Atoms object to Open Babel molecule."""
    #      obmol = openbabel.OBMol()
    #      obconversion = openbabel.OBConversion()
    #      obconversion.SetInAndOutFormats("xyz", "xyz")
    #
    #      # Write ASE atoms to XYZ format
    #      xyz_data = atoms.get_positions()
    #      symbols = atoms.get_chemical_symbols()
    #      natoms = len(symbols)
    #      xyz_string = f"{natoms}\n\n"
    #      for symbol, coord in zip(symbols, xyz_data):
    #          xyz_string += f"{symbol} {coord[0]} {coord[1]} {coord[2]}\n"
    #
    #      # Read XYZ data into OBMol
    #      obconversion.ReadString(obmol, xyz_string)
    #      return obmol

    @staticmethod
    def _atoms_to_obmol(atoms):
        """Convert an Atoms object to an OBMol object.

        Args:
            atoms: Atoms

        Returns:
            obmol: OBMol
        """
        obmol = openbabel.OBMol()
        for atom in atoms:
            a = obmol.NewAtom()
            a.SetAtomicNum(int(atom.number))
            a.SetVector(atom.position[0], atom.position[1], atom.position[2])
        # Automatically add bonds to molecule
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        return obmol

    def _compute_virial_stress(self, atoms, forces):
        """Compute stress tensor from forces and positions using virial formula.

        The stress tensor is calculated using the virial stress formula:

            σ_αβ = -(1/V) * Σᵢ rᵢ^α * fᵢ^β

        where:

            r_i is the position of atom i

            f_i is the force on atom i

            V is the volume of the system

            α, β ∈ {x, y, z}

        The negative sign follows the convention where compressive stress is negative.

        References:
            - Irving, J. H.; Kirkwood, J. G. (1950). "The Statistical Mechanical Theory
              of Transport Processes IV. The equations of hydrodynamics."
              The Journal of Chemical Physics, 18(6), 817-829.
              https://doi.org/10.1063/1.1747782

            - Ray, J. R.; Rahman, A. (1984). "Statistical mechanics of water: Equation
              of state from computer simulations." The Journal of Chemical Physics, 80(10),
              4423-4428. https://doi.org/10.1063/1.447211

            - LAMMPS Documentation - Compute Stress/Atom:
              https://docs.lammps.org/compute_stress_atom.html

        Args:
            atoms (ase.Atoms): ASE Atoms object with cell definition
            forces (np.ndarray): Atomic forces of shape (natoms, 3) in eV/Å

        Returns:
            np.ndarray: Stress tensor of shape (3, 3) in same units as energy/volume
        """
        V = atoms.get_volume()
        stress_tensor = np.zeros((3, 3))

        for i, atom in enumerate(atoms):
            r = atom.position
            f = forces[i]
            stress_tensor += np.outer(r, f)

        return -stress_tensor / V  # Negative convention for stress
