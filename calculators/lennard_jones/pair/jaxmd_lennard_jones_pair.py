from __future__ import annotations
from typing import Callable, Optional, Tuple

from ase.atoms import Atoms
from calculators.calculator import Calculator, Result

from asax.utils import _transform

import jax.numpy as jnp
from jax import grad, random, jit, value_and_grad
from jax_md import space, energy, quantity
from jax.config import config
config.update("jax_enable_x64", True)

class JmdLennardJonesPair(Calculator):

    def __init__(self, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool):
        super().__init__(box_size, n, R)
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cutoff = r_cutoff
        self._r_onset = r_onset
        self._stress = stress

        if not stress:
            self._displacement_fn, self._shift_fn = self._create_periodic_space()
            self._atomwise_energy_fn, self._total_energy_fn, self._force_fn = self._create_property_functions()
            return

        # (self._displacement_fn, self._shift_fn), self._properties_fn = self._create_stress_functions()        
        (self._displacement_fn, self._shift_fn), self._properties_fn = self._create_stress_functions()

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool) -> JmdLennardJonesPair:
        return super().from_ase_atoms(atoms, sigma, epsilon, r_cutoff, r_onset, stress)

    @classmethod
    def create_potential(cls, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool) -> JmdLennardJonesPair:
        return super().create_potential(box_size, n, R, sigma, epsilon, r_cutoff, r_onset, stress)

    def _generate_R(self, n: int, scaling_factor: float) -> jnp.ndarray:
         # TODO: Build a global service to manage and demand PRNGKeys for JAX-based simulations
        print("JAX PRNG")
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        return random.uniform(subkey, shape=(n, 3)) * scaling_factor

    def _create_periodic_space(self):
        displacement_fn, shift_fn = space.periodic(self._box_size)
        return jit(displacement_fn), jit(shift_fn)

    def _create_stress_functions(self) -> Tuple[space.Space, Callable[[space.Array]]]:
        displacement_fn, shift_fn = space.periodic(self._box_size)

        def energy_under_strain(R: space.Array, strain: space.Array) -> space.Array:
            def displacement_under_strain(Ra: space.Array, Rb: space.Array, **unused_kwargs) -> space.Array:
                ones = jnp.eye(N=3, M=3, dtype=jnp.double)
                transform = ones + strain
                return _transform(transform, displacement_fn(Ra, Rb))

            energy = self._get_energy_fn(displacement_under_strain, per_particle=True)
            return energy(R)

        def compute_properties(R: space.Array) -> space.Array:
            zeros = jnp.zeros((3, 3), dtype=jnp.double)
            atomwise_energies = energy_under_strain(R, zeros)            
            total_energy_fn = lambda R, zeros: jnp.sum(energy_under_strain(R, zeros))
            forces = grad(total_energy_fn, argnums=(0))(R, zeros)
            stress = grad(total_energy_fn, argnums=(1))(R, zeros)
            return atomwise_energies, forces, stress
            # return value_and_grad(energy_under_strain, argnums=(0, 1))(R, zeros)

        return (jit(displacement_fn), jit(shift_fn)), compute_properties
        # TODO: How would this work with computing atomwise energies as a first step?
        # return (jit(displacement_fn), jit(shift_fn)), jit(compute_properties)

    def _create_property_functions(self):
        atomwise_energy_fn = self._get_energy_fn(per_particle=True)
        total_energy_fn = lambda R: jnp.sum(atomwise_energy_fn(R))
        force_fn = lambda R: quantity.force(total_energy_fn)(R)
        return jit(atomwise_energy_fn), jit(total_energy_fn), jit(force_fn)     

    def _get_energy_fn(self, displacement_fn: Optional[energy.DisplacementFn], per_particle=True) -> Callable[[energy.Array], energy.Array]:
        if displacement_fn is None: 
            displacement_fn = self._displacement_fn
        return energy.lennard_jones_pair(displacement_fn, sigma=self._sigma, epsilon=self._epsilon, r_onset=self._r_onset, r_cutoff=self._r_cutoff, per_particle=per_particle)       
        
    def calculate(self) -> Result:
        if not self._stress:
            energies = self._atomwise_energy_fn(self._R)
            forces = self._force_fn(self._R)    
            return Result(energies, forces, None)    

        # old style to unpack
        # total_energy, (R_grad, stress) = self._properties_fn(self._R)
        # forces = -R_grad

        atomwise_energies, forces, stress = self._properties_fn(self._R)
        return Result(atomwise_energies, forces, [stress])
