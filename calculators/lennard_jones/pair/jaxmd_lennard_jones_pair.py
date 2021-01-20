from __future__ import annotations
from typing import Callable, Optional, Tuple

from ase.atoms import Atoms
from jax.api import jacfwd
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
        self._volume = box_size ** 3
        (self._displacement_fn, self._shift_fn), self._properties_fn = self._initialize_jaxmd_primitives()

    def _initialize_jaxmd_primitives(self):
        if not self._stress:
            return self._initialize_equilibirium_potential()
        return self._initialize_strained_potential()

    @property
    def description(self) -> str:
        return "JAX-MD Lennard-Jones Calculator (stress={})".format(str(self._stress))

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool) -> JmdLennardJonesPair:
        return super().from_ase_atoms(atoms, sigma, epsilon, r_cutoff, r_onset, stress)

    @classmethod
    def create_potential(cls, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool) -> JmdLennardJonesPair:
        return super().create_potential(box_size, n, R, sigma, epsilon, r_cutoff, r_onset, stress)

    def _generate_R(self, n: int, scaling_factor: float) -> jnp.ndarray:
         # TODO: Build a global service to manage and demand PRNGKeys for JAX-based simulations
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        return random.uniform(subkey, shape=(n, 3)) * scaling_factor

    def _initialize_strained_potential(self) -> Tuple[space.Space, Callable[[space.Array]]]:
        displacement_fn, shift_fn = space.periodic(self._box_size)

        def energy_under_strain(R: space.Array, strain: space.Array) -> space.Array:
            def displacement_under_strain(Ra: space.Array, Rb: space.Array, **unused_kwargs) -> space.Array:
                ones = jnp.eye(N=3, M=3, dtype=jnp.double)
                transform = ones + strain
                return _transform(transform, displacement_fn(Ra, Rb))

            energy = self._get_energy_fn(displacement_under_strain, per_particle=True)
            return energy(R)

        def compute_properties(R: space.Array) -> Tuple[float, float, float, float, float, float]:
            zeros = jnp.zeros((3, 3), dtype=jnp.double)      

            atomwise_energies_fn = energy_under_strain      
            atomwise_energies = atomwise_energies_fn(R, zeros)            
            total_energy_fn = lambda R, zeros: jnp.sum(atomwise_energies_fn(R, zeros))
            total_energy = total_energy_fn(R, zeros)
        
            forces = grad(total_energy_fn, argnums=(0))(R, zeros) * -1
            force = jnp.sum(forces)
            
            stress = grad(total_energy_fn, argnums=(1))(R, zeros) / self._volume
            stresses = jacfwd(atomwise_energies_fn, argnums=(1))(R, zeros) / self._volume
            
            return total_energy, atomwise_energies, force, forces, stress, stresses
            # return atomwise_energies, forces, stresses
            # return value_and_grad(energy_under_strain, argnums=(0, 1))(R, zeros)

        return (jit(displacement_fn), jit(shift_fn)), jit(compute_properties)

    def _initialize_equilibirium_potential(self) -> Tuple[space.Space, Callable[[space.Array]]]:
        displacement_fn, shift_fn = space.periodic(self._box_size)

        def compute_properties(R: space.Array) -> Tuple[float, float, float, float, float, float]:
            atomwise_energies_fn = self._get_energy_fn(displacement_fn, per_particle=True)
            atomwise_energies = atomwise_energies_fn(R) 
            total_energy_fn = lambda R: jnp.sum(atomwise_energies_fn(R))
            total_energy = total_energy_fn(R)
            forces = grad(total_energy_fn)(R) * -1
            force = jnp.sum(forces)

            return total_energy, atomwise_energies, force, forces, None, None
            # return atomwise_energies, forces

        return (jit(displacement_fn), jit(shift_fn)), jit(compute_properties)

    def _get_energy_fn(self, displacement_fn: Optional[energy.DisplacementFn], per_particle=True) -> Callable[[energy.Array], energy.Array]:
        if displacement_fn is None: 
            displacement_fn = self._displacement_fn
        return energy.lennard_jones_pair(displacement_fn, sigma=self._sigma, epsilon=self._epsilon, r_onset=self._r_onset, r_cutoff=self._r_cutoff, per_particle=per_particle)       
        
    def _compute_properties(self) -> Result:
        

        # if not self._stress:
        #     energies, forces = self._properties_fn(self._R)
        #     return Result(self, energies, forces, None)
            # energies = self._atomwise_energy_fn(self._R)
            # forces = self._force_fn(self._R)    
            # return Result(energies, forces, None)    

        # old style to unpack
        # total_energy, (R_grad, stress) = self._properties_fn(self._R)
        # forces = -R_grad

        energy, atomwise_energies, force, forces, stress, stresses = self._properties_fn(self._R)
        return Result(self, energy, atomwise_energies, force, forces, stress, stresses)
