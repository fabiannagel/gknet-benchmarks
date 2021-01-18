from __future__ import annotations
from typing import Callable

from ase.atoms import Atoms
from calculators.calculator import Calculator, Result

import jax.numpy as jnp
from jax import grad, random, jit
from jax_md import space, energy, quantity
from jax.config import config
config.update("jax_enable_x64", True)

class JmdLennardJonesPair(Calculator):

    def __init__(self, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float):
        super().__init__(box_size, n, R)
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cutoff = r_cutoff
        self._r_onset = r_onset

        self._displacement_fn, self._shift_fn = self._create_periodic_space()
        self._atomwise_energy_fn, self._total_energy_fn, self._force_fn = self._create_property_functions()

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float, r_onset: float) -> JmdLennardJonesPair:
        return super().from_ase_atoms(atoms, sigma, epsilon, r_cutoff, r_onset)

    @classmethod
    def create_potential(cls, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float) -> JmdLennardJonesPair:
        return super().create_potential(box_size, n, R, sigma, epsilon, r_cutoff, r_onset)

    def _generate_R(self, n: int, scaling_factor: float) -> jnp.ndarray:
         # TODO: Build a global service to manage and demand PRNGKeys for JAX-based simulations
        print("JAX PRNG")
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        return random.uniform(subkey, shape=(n, 3)) * scaling_factor

    def _create_periodic_space(self):
        displacement_fn, shift_fn = space.periodic(self._box_size)
        return jit(displacement_fn), jit(shift_fn)

    def _create_property_functions(self):
        atomwise_energy_fn = energy.lennard_jones_pair(self._displacement_fn, sigma=self._sigma, epsilon=self._epsilon, r_onset=self._r_onset, r_cutoff=self._r_cutoff, per_particle=True)
        total_energy_fn = lambda R: jnp.sum(atomwise_energy_fn(R))
        force_fn = lambda R: quantity.force(total_energy_fn)(R)
        return atomwise_energy_fn, total_energy_fn, force_fn

    def calculate(self) -> Result:
        def wrapped_computation(R):
            energies = self._atomwise_energy_fn(R)
            forces = self._force_fn(R)
            stresses = None
            return energies, forces, stresses
        
        # energies, forces, stresses = jit(wrapped_computation)(self._R)
        energies, forces, stresses = wrapped_computation(self._R)
        return Result(energies, forces, stresses)    
        # return None
