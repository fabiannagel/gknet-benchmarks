from __future__ import annotations
import warnings

from typing import List, Optional
from calculators.calculator import Calculator, Result

from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.calculator import PropertyNotImplementedError
from ase.constraints import voigt_6_to_full_3x3_stress


import numpy as np
import itertools
import math


class AseLennardJonesPair(Calculator):
    
    def __init__(self, box_size: float, n: int, R: np.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float):
        super().__init__(box_size, n, R, True)
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cutoff = r_cutoff
        self._r_onset = r_onset


    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float, r_onset: float) -> AseLennardJonesPair:
        obj: AseLennardJonesPair = super().from_ase_atoms(atoms, sigma, epsilon, r_cutoff, r_onset)
        obj._atoms = atoms
        obj._calc = LennardJones(sigma=sigma, epsilon=epsilon, rc=r_cutoff, ro=r_onset, smooth=True)
        return obj    


    @classmethod
    def create_potential(cls, n: int, sigma: float, epsilon: float, r_cutoff: Optional[float], r_onset: Optional[float]) -> AseLennardJonesPair:
        '''
        Create a cubic Argon bulk structure using ASE.
        If omitted, r_cutoff is set to half the maximum super cell lattice vector magnitude.
        If omitted, r_onset is set to 0.8 * r_cutoff
        '''        
        atoms = bulk('Ar', cubic=True) * cls._compute_supercell_multipliers('Ar', n)
        
        if r_cutoff is None:
            max_box_length = np.max([np.linalg.norm(uv) for uv in atoms.get_cell().array])
            r_cutoff = 0.5 * max_box_length

        if r_onset is None:
            r_onset = 0.8 * r_cutoff 

        # Optional: Introduce strain by upscaling the super cell
        # self._atoms.set_cell(1.05 * self._atoms.get_cell(), scale_atoms=True)    
        return cls.from_ase_atoms(atoms, sigma, epsilon, r_cutoff, r_onset)


    @property
    def r_onset(self) -> float:
        return self._r_onset


    @property
    def r_cutoff(self) -> float:
        return self._r_cutoff


    @property
    def description(self):
        return "ASE Neighbor List"


    @property
    def pairwise_distances(self):
        return self._atoms.get_all_distances(mic=True) 


    def _generate_R(self, n: int, scaling_factor: float) -> np.ndarray:
        print("ASE/NumPy PRNG")
        return np.random.uniform(size=(n, 3)) * scaling_factor


    def _compute_supercell_multipliers(element: str, n: int) -> List[float]:
        if element != 'Ar': raise NotImplementedError('Unknown cubic unit cell size for element ' + element)
        argon_unit_cell_size = 4
        dimension_mulitplier = math.floor((np.cbrt(n / argon_unit_cell_size)))
        
        actual_n = argon_unit_cell_size * dimension_mulitplier**3 
        if (actual_n != n):
            warnings.warn('{} unit cell size causes deviation from desired n={}. Final atom count n={}'.format(element, str(n), str(actual_n)), RuntimeWarning)
        return list(itertools.repeat(dimension_mulitplier, 3))


    def _compute_properties(self) -> Result:
        self._calc.atoms = None
        # self._calc.nl = None
        self._calc.calculate(atoms=self._atoms)

        energy = self._calc.results['energy']
        energies = self._calc.results['energies']
        forces = self._calc.results['forces']
        stress = voigt_6_to_full_3x3_stress(self._calc.results['stress'])
        stresses = voigt_6_to_full_3x3_stress(self._calc.results['stresses'])

        return Result(self, self._n, energy, energies, forces, stress, stresses)


    def warm_up(self):
        self._calc.nl = None
        self._compute_properties()

        
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_atoms']
        return state
  
    def __setstate__(self, state):
        self.__dict__.update(state)
        error_fn = lambda *args, **kwargs: print("Pickled instance cannot compute new data")
        self._atoms = error_fn