from __future__ import annotations
from typing import Callable, Optional, Tuple
from functools import partial
from utils import new_get_displacement

from calculators.calculator import Calculator
from calculators.result import Result

from ase.atoms import Atoms
from jax_md import space, energy, quantity
from periodic_general import periodic_general, transform
import jax.numpy as jnp
from jax import grad, random, jit, value_and_grad
from jax.api import jacfwd, vmap
from jax.config import config
config.update("jax_enable_x64", True)


class JmdLennardJonesPair(Calculator):

    # TODO: box_size and displacement_fn are two different ways to initialize
    # Either we can create our own displacement_fn via space.periodic(box_size) ...
    # ... or use the passed displacement_fn that is used when importing an ASE object
    # Unite in new parameter box_size_or_displacement? What is the proper way to do this?

    # TODO: Create lightweight type for LJ parameters?
    def __init__(self, box: jnp.ndarray, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool, stresses: bool, jit: bool, displacement_fn: Optional[Callable]):
        super().__init__(box, n, R, stress)
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cutoff = r_cutoff
        self._r_onset = r_onset 
        self._stress = stress
        self._stresses = stresses
        self._jit = jit
        self._displacement_fn, self._properties_fn = self._initialize_potential(displacement_fn, stress)


    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool, stresses: bool, adjust_radii: bool, jit: bool) -> JmdLennardJonesPair:
        displacement_fn = new_get_displacement(atoms)
        
        # JAX-MD's LJ implementation multiplies onset and cutoff by sigma. To be compatible w/ ASE's implementation, we need to perform these adjustments.
        if adjust_radii:
            r_onset /= sigma
            r_cutoff /= sigma

        return super().from_ase_atoms(atoms, sigma, epsilon, r_cutoff, r_onset, stress, stresses, jit, displacement_fn)


    @classmethod
    def create_potential(cls, box_size: float, n: int, R_scaled: Optional[jnp.ndarray], sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool, stresses: bool, jit: bool) -> JmdLennardJonesPair:
        '''Initialize a Lennard-Jones potential from scratch using scaled atomic coordinates. If omitted, random coordinates will be generated.'''
        return super().create_potential(box_size, n, R_scaled, sigma, epsilon, r_cutoff, r_onset, stress, stresses, jit, None)


    @property
    def description(self) -> str:
        return "JAX-MD Pair (stress={}, stresses={}, jit={})".format(self._stress, self._stresses, self._jit)


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


    def _initialize_potential(self, displacement_fn: Optional[Callable], stress: bool) -> Tuple[space.Space, Callable[[space.Array]]]:
        if displacement_fn is None:
            displacement_fn, _ = periodic_general(self._box)
        
        energy_fn = energy.lennard_jones_pair(displacement_fn, sigma=self._sigma, epsilon=self._epsilon, r_onset=self._r_onset, r_cutoff=self._r_cutoff, per_particle=True)       

        def compute_properties_with_stress(deformation: jnp.ndarray, R: space.Array):   
            # 1) Set the box under strain using a symmetrized deformation tensor
            # 2) Override the box in the energy function
            # 3) Derive forces, stress and stresses as gradients of the deformed energy function

            symmetrized_strained_box_fn = lambda deformation: transform(jnp.eye(3) + (deformation + deformation.T) * 0.5, self._box)            
            deformation_energy_fn = lambda deformation, R: energy_fn(R, box=symmetrized_strained_box_fn(deformation))
            atomwise_energies = deformation_energy_fn(deformation, R)

            total_deformation_energy_fn = lambda deformation, R: jnp.sum(deformation_energy_fn(deformation, R))            
            total_energy = total_deformation_energy_fn(deformation, R)

            deformation_force_fn = lambda deformation, R: grad(total_deformation_energy_fn, argnums=1)(deformation, R) * -1
            forces = deformation_force_fn(deformation, R)

            # TODO: Why so ugly?
            stress = None
            if self._stress:
                stress_fn = lambda deformation, R: grad(total_deformation_energy_fn, argnums=0)(deformation, R) / jnp.linalg.det(self._box)
                stress = stress_fn(deformation, R)

            stresses = None
            if self._stresses:            
                stresses_fn = lambda deformation, R: jacfwd(deformation_energy_fn, argnums=0)(deformation, R) / jnp.linalg.det(self._box)
                stresses = stresses_fn(deformation, R)

            return total_energy, atomwise_energies, forces, stress, stresses


        def compute_properties(R: space.Array) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
            total_energy_fn = lambda R: jnp.sum(energy_fn(R))
            forces_fn = quantity.force(total_energy_fn)

            total_energy = total_energy_fn(R)
            atomwise_energies = energy_fn(R)
            forces = forces_fn(R)

            return total_energy, atomwise_energies, forces


        final_compute_properties = None

        if self._stress or self._stresses:
            final_compute_properties = compute_properties_with_stress
        else:
            final_compute_properties = compute_properties
        
        if self._jit:
            return jit(displacement_fn), jit(final_compute_properties)
        return displacement_fn, final_compute_properties


    def _compute_properties(self) -> Result:
        if self._stress or self._stresses:
            deformation = jnp.zeros_like(self._box)
            total_energy, atomwise_energies, forces, stress, stresses = self._properties_fn(deformation, self._R)
            return Result(self, self._n, total_energy, atomwise_energies, forces, stress, stresses)
        
        total_energy, atomwise_energies, forces = self._properties_fn(self._R)

        # TODO: Convert to numpy arrays

        return Result(self, self._n, total_energy, atomwise_energies, forces, None, None)

    def warm_up(self):
        if not self._jit:
            raise RuntimeError("Warm-up only implemented for jit=True")
        self._compute_properties()

    
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_displacement_fn']
        del state['_properties_fn']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.
        error_fn = lambda *args, **kwargs: print("Pickled instance cannot compute new data")
        self._displacement_fn = error_fn
        self._properties_fn = error_fn