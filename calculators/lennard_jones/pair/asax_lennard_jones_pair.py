from __future__ import annotations

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

    def __init__(self, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float) -> None:
        super().__init__(box_size, n, R)
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cutoff = r_cutoff
        self._r_onset = r_onset

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float, r_onset: float) -> AsaxLennardJonesPair:
        obj: AsaxLennardJonesPair = super().from_ase_atoms(atoms, sigma, float, epsilon, r_cutoff, r_onset)
        obj._atoms = atoms
        obj._atoms.calc = LennardJones(obj._epsilon, obj._sigma, obj._r_cutoff, obj._r_onset, x64=True, stress=True)
        return obj

    @classmethod
    def create_potential(cls, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float) -> AsaxLennardJonesPair:
        obj: AsaxLennardJonesPair = super().create_potential(box_size, n, R, sigma, epsilon, r_cutoff, r_onset)
        obj._atoms = bulk('Ar', cubic=True) * obj._compute_supercell_multipliers('Ar', obj._n)
        obj._atoms.calc = LennardJones(obj._epsilon, obj._sigma, obj._r_cutoff, obj._r_onset, x64=True, stress=True)
        return obj

    def _generate_R(self, n: int, scaling_factor: float) -> jnp.ndarray:
         # TODO: Build a global service to manage and demand PRNGKeys for JAX-based simulations
        print("JAX PRNG")
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        return random.uniform(subkey, shape=(n, 3)) * scaling_factor

    # TODO: Duplicate code from AseLennardJonesPair(). Make this a mixin?
    def _compute_supercell_multipliers(self, element: str, n: int) -> List[float]:
        if element != 'Ar': raise NotImplementedError('Unknown cubic unit cell size for element ' + element)
        dimension_mulitplier = math.floor((np.cbrt(n / 4)))
        return list(itertools.repeat(dimension_mulitplier, 3))

    def calculate(self) -> Result:
        energy = self._atoms.calc.get_potential_energy(self._atoms)

        # atoms = Atoms(positions=[[0, 0, 0], [8, 0, 0]])
        # calc = LennardJones(self._epsilon, self._sigma, self._r_cutoff, self._r_onset, stress=True)
        # energy = calc.get_potential_energy(atoms)
        # print(energy)

        return Result([energy], None, None)
    
        # TODO: asax does not output atom-wise energies. do we actually need those?
        # energies = [self._atoms.get_potential_energy()]
        # forces = self._atoms.get_forces()
        # stresses = [self._atoms.get_stress()]
        # return Result(energies, forces, stresses)