from __future__ import annotations
from typing import Callable, Optional, Tuple
from functools import partial

from ase.atoms import Atoms
from jax.api import jacfwd, vmap
from calculators.calculator import Calculator, Result

from asax.utils import _transform, get_displacement

import jax.numpy as jnp
from jax import grad, random, jit, value_and_grad
from jax_md import space, energy, quantity
from jax.config import config
config.update("jax_enable_x64", True)

class JmdLennardJonesPair(Calculator):

    # TODO: box_size and displacement_fn are two different ways to initialize
    # Either we can create our own displacement_fn via space.periodic(box_size) ...
    # ... or use the passed displacement_fn that is used when importing an ASE object
    # Unite in new parameter box_size_or_displacement? What is the proper way to do this?

    # TODO: Create lightweight type for LJ parameters?
    def __init__(self, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool, displacement_fn: Optional[Callable]):
        super().__init__(box_size, n, R)
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cutoff = r_cutoff
        self._r_onset = r_onset
        self._stress = stress
        self._volume = box_size ** 3
        self._displacement_fn, self._properties_fn = self._initialize_jaxmd_primitives(displacement_fn)

    def _initialize_jaxmd_primitives(self, displacement_fn):
        if not self._stress:
            return self._initialize_equilibirium_potential(displacement_fn)
        return self._initialize_strained_potential(displacement_fn)

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool) -> JmdLennardJonesPair:
        displacement_fn = get_displacement(atoms)
        return super().from_ase_atoms(atoms, sigma, epsilon, r_cutoff, r_onset, stress, displacement_fn)


    @classmethod
    def create_potential(cls, box_size: float, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool) -> JmdLennardJonesPair:
        return super().create_potential(box_size, n, R, sigma, epsilon, r_cutoff, r_onset, stress)

    @property
    def description(self) -> str:
        return "JAX-MD Lennard-Jones Calculator (stress={})".format(str(self._stress))

    @property
    @partial(jit, static_argnums=0)
    def pairwise_distances(self):
        # displacement_fn takes two vectors Ra and Rb
        # space.map_product() vmaps it twice along rows and columns such that we can input matrices
        dR_dimensionwise_fn = space.map_product(self._displacement_fn)
        dR_dimensionwise = dR_dimensionwise_fn(self._R, self._R)    # ... resulting in 4 dimension-wise distance matrices shaped (n, n, 3)

        # Computing the vector magnitude for every row vector:
        # First, map along the first axis of the initial (n, n, 3) matrix. the "output" will be (n, 3)
        # Secondly, within the mapped (n, 3) matrix, map along the zero-th axis again (one atom).
        # Here, apply the magnitude function for the atom's displacement row vector.
        magnitude_fn = lambda x: jnp.sqrt(jnp.sum(x**2))
        vectorized_fn = vmap(vmap(magnitude_fn, in_axes=0), in_axes=0)
        return vectorized_fn(dR_dimensionwise)

    def _generate_R(self, n: int, scaling_factor: float) -> jnp.ndarray:
         # TODO: Build a global service to manage and demand PRNGKeys for JAX-based simulations
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        return random.uniform(subkey, shape=(n, 3)) * scaling_factor


    def _initialize_strained_potential(self, displacement_fn: Optional[Callable]) -> Tuple[space.Space, Callable[[space.Array]]]:
        print("Tracing _initialize_strained_potential")
        
        if displacement_fn is None:
            displacement_fn, _ = space.periodic(self._box_size)

        def energy_under_strain(R: space.Array, strain: space.Array) -> space.Array:
            print("Tracing energy_under_strain")

            def displacement_under_strain(Ra: space.Array, Rb: space.Array, **unused_kwargs) -> space.Array:
                print("Tracing displacement_under_strain")
                ones = jnp.eye(N=3, M=3, dtype=jnp.double)
                transform = ones + strain
                return _transform(transform, displacement_fn(Ra, Rb))

            energy = self._get_energy_fn(displacement_under_strain, per_particle=True)
            return energy(R)

        def compute_properties(R: space.Array) -> Tuple[float, float, float, float, float, float]:
            print("Tracing compute_properties")

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

        return jit(displacement_fn), jit(compute_properties)


    def _initialize_equilibirium_potential(self, displacement_fn: Optional[Callable]) -> Tuple[space.Space, Callable[[space.Array]]]:
        if displacement_fn is None:
            displacement_fn, _ = space.periodic(self._box_size)

        def compute_properties(R: space.Array) -> Tuple[float, float, float, float, float, float]:
            atomwise_energies_fn = self._get_energy_fn(displacement_fn, per_particle=True)
            atomwise_energies = atomwise_energies_fn(R) 
            total_energy_fn = lambda R: jnp.sum(atomwise_energies_fn(R))
            total_energy = total_energy_fn(R)
            forces = grad(total_energy_fn)(R) * -1
            force = jnp.sum(forces)

            return total_energy, atomwise_energies, force, forces, None, None

        return jit(displacement_fn), jit(compute_properties)


    def _get_energy_fn(self, displacement_fn: energy.DisplacementFn, per_particle=True) -> Callable[[energy.Array], energy.Array]:
        return energy.lennard_jones_pair(displacement_fn, sigma=self._sigma, epsilon=self._epsilon, r_onset=self._r_onset, r_cutoff=self._r_cutoff, per_particle=per_particle)       
        

    def _compute_properties(self) -> Result:
        energy, atomwise_energies, force, forces, stress, stresses = self._properties_fn(self._R)
        return Result(self, energy, atomwise_energies, force, forces, stress, stresses)
