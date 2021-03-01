from __future__ import annotations
from typing import List
import warnings

from calculators.calculator import Calculator, Result
from ase import Atoms
from ase.build import bulk
from asax.lj import LennardJones
import jax.numpy as jnp
from jax import random

import numpy as np
import itertools
import math

class AsaxLennardJonesPair(Calculator):
    _atoms: Atoms

    def __init__(self, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool) -> None:
        super().__init__(box_size, n, R, stress)
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cutoff = r_cutoff
        self._r_onset = r_onset
        self._stress = stress

    @property
    def description(self) -> str:
        return "ASAX Lennard-Jones Calculator (stress={})".format(str(self._stress))

    @property
    def pairwise_distances(self):
        return self._atoms.get_all_distances(mic=True) 

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool) -> AsaxLennardJonesPair:
        obj: AsaxLennardJonesPair = super().from_ase_atoms(atoms, sigma, epsilon, r_cutoff, r_onset, stress)
        obj._atoms = atoms
        obj._atoms.calc = LennardJones(obj._epsilon, obj._sigma, obj._r_cutoff, obj._r_onset, x64=True, stress=stress)
        return obj

    @classmethod
    def create_potential(cls, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool) -> AsaxLennardJonesPair:
        obj: AsaxLennardJonesPair = super().create_potential(box_size, n, R, sigma, epsilon, r_cutoff, r_onset, stress)
        obj._atoms = bulk('Ar', cubic=True) * obj._compute_supercell_multipliers('Ar', obj._n)
        obj._atoms.calc = LennardJones(obj._epsilon, obj._sigma, obj._r_cutoff, obj._r_onset, x64=True, stress=stress)
        return obj

    def _generate_R(self, n: int, scaling_factor: float) -> jnp.ndarray:
         # TODO: Build a global service to manage and demand PRNGKeys for JAX-based simulations
        print("JAX PRNG")
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        return random.uniform(subkey, shape=(n, 3)) * scaling_factor

    # TODO: Duplicate code from AseLennardJonesPair(). Make this a mixin or something? Or inherit Ase -> Asax?
    def _compute_supercell_multipliers(self, element: str, n: int) -> List[float]:
        if element != 'Ar': raise NotImplementedError('Unknown cubic unit cell size for element ' + element)
        argon_unit_cell_size = 4
        dimension_mulitplier = math.floor((np.cbrt(n / argon_unit_cell_size)))

        actual_n = argon_unit_cell_size * dimension_mulitplier**3 
        if (actual_n != n):
            warnings.warn('{} unit cell size causes deviation from desired n={}. Final atom count n={}'.format(element, str(n), str(actual_n)), RuntimeWarning)
        return list(itertools.repeat(dimension_mulitplier, 3))

    def _compute_properties(self) -> Result:
        # TODO: Implement atom-wise energies in ASAX
        # TODO: Add atom-wise stresses to ASAX once implemented for JAX-MD

        energy = self._atoms.get_potential_energy()
        energies = None
        forces = self._atoms.get_forces()
        force = np.sum(forces)
        stress = self._atoms.get_stress()
        stresses = None

        return Result(self, self._n, energy, energies, forces, stress, stresses, None)
