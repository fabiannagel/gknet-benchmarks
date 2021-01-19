from __future__ import annotations
import warnings

from typing import List
from calculators.calculator import Calculator, Result

from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.calculator import PropertyNotImplementedError

import numpy as np
import itertools
import math


class AseLennardJonesPair(Calculator):
    
    def __init__(self, box_size: float, n: int, R: np.ndarray, sigma: float, epsilon: float, r_cutoff: float):
        super().__init__(box_size, n, R)
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cutoff = r_cutoff

    @property
    def description(self):
        return "ASE Lennard-Jones Calculator"

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float) -> AseLennardJonesPair:
        obj: AseLennardJonesPair = super().from_ase_atoms(atoms, sigma, epsilon, r_cutoff)
        obj._atoms = atoms
        obj._atoms.calc = LennardJones(sigma=obj._sigma, epsilon=obj._epsilon, rc=obj._r_cutoff)
        return obj
    
    @classmethod
    def create_potential(cls, box_size: float, n: int, R: np.ndarray, sigma: float, epsilon: float, r_cutoff: float) -> AseLennardJonesPair:
        obj: AseLennardJonesPair = super().create_potential(box_size, n, R, sigma, epsilon, r_cutoff)
        obj._atoms = bulk('Ar', cubic=True) * obj._compute_supercell_multipliers('Ar', obj._n)
        
        # ASE initializes the system in an energy minimum, so forces = 0.
        # We can introduce strain by upscaling the super cell like this:
        # self._atoms.set_cell(1.05 * self._atoms.get_cell(), scale_atoms=True)    
        obj._atoms.calc = LennardJones(sigma=obj._sigma, epsilon=obj._epsilon, rc=obj._r_cutoff)
        return obj

        
    def _generate_R(self, n: int, scaling_factor: float) -> np.ndarray:
        print("ASE/NumPy PRNG")
        return np.random.uniform(size=(n, 3)) * scaling_factor

    def _compute_supercell_multipliers(self, element: str, n: int) -> List[float]:
        if element != 'Ar': raise NotImplementedError('Unknown cubic unit cell size for element ' + element)
        argon_unit_cell_size = 4
        dimension_mulitplier = math.floor((np.cbrt(n / argon_unit_cell_size)))
        
        actual_n = argon_unit_cell_size * dimension_mulitplier**3 
        if (actual_n != n):
            warnings.warn('{} unit cell size causes deviation from desired n={}. Final atom count n={}'.format(element, str(n), str(actual_n)), RuntimeWarning)
        return list(itertools.repeat(dimension_mulitplier, 3))

    def _compute_properties(self) -> Result:
        energies = self._atoms.get_potential_energies()
        forces = self._atoms.get_forces()
        stresses = self._atoms.get_stresses()
        return Result(self, energies, forces, stresses)
