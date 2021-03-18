from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Dict
from functools import partial
import warnings

import jax_utils
from jax_utils import XlaMemoryFlag
from calculators.calculator import Calculator
from calculators.result import Result
from ase.atoms import Atoms
from jax_md import space, energy, quantity
from periodic_general import periodic_general, transform
import jax.numpy as jnp
from jax import grad, jit
from jax.api import jacfwd
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_log_compiles", 1)

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
        self._memory_allocation_mode = jax_utils.get_memory_allocation_mode()
        self._displacement_fn, self._properties_fn = self._initialize_potential(displacement_fn, stress)


    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool, stresses: bool, adjust_radii: bool, jit: bool) -> JmdLennardJonesPair:
        displacement_fn = jax_utils.new_get_displacement(atoms)
        
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
        return jax_utils.get_calculator_description(self)


    @property
    def memory_allocation_mode(self) -> XlaMemoryFlag:
        return self._memory_allocation_mode


    @property
    @partial(jit, static_argnums=0)
    def pairwise_distances(self):
        return jax_utils.compute_pairwise_distances(self._displacement_fn, self._R)


    def _generate_R(self, n: int, scaling_factor: float) -> jnp.ndarray:
        return jax_utils.generate_R(n, scaling_factor)


    def _initialize_potential(self, displacement_fn: Optional[Callable], stress: bool) -> Tuple[space.DisplacementFn, Callable[[space.Array]]]:
        # update global reference to avoid side effects
        # TODO: Probably not neessary, fields are private. If modified, unexpected behavior shouldn't be unexpected. Verify results first, and then remove later.
        sigma, epsilon, r_onset, r_cutoff = self._sigma, self._epsilon, self._r_onset, self._r_cutoff
        box = self._box     
        compute_stress = self._stress
        compute_stresses = self._stresses
        use_jit = self._jit

        if displacement_fn is None:
            warnings.warn("Using default periodic_general")
            displacement_fn, _ = periodic_general(box)

        def strained_potential_fn(R: space.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            # 1) Set the box under strain using a symmetrized deformation tensor
            # 2) Override the box in the energy function
            # 3) Derive forces, stress and stresses as gradients of the deformed energy function

            # define a default energy function, an infinitesimal deformation and a function to apply the transformation to the box
            energy_fn = energy.lennard_jones_pair(displacement_fn, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, per_particle=True)                     
            deformation = jnp.zeros_like(box)

            # a function to symmetrize the deformation tensor and apply it to the box
            transform_box_fn = lambda deformation: transform(jnp.eye(3) + (deformation + deformation.T) * 0.5, box) 

            # atomwise and total energy functions that act on the transformed box. same for force, stress and stresses.
            deformation_energy_fn = lambda deformation, R: energy_fn(R, box=transform_box_fn(deformation))
            total_energy_fn = lambda deformation, R: jnp.sum(deformation_energy_fn(deformation, R))            

            force_fn = lambda deformation, R: grad(total_energy_fn, argnums=1)(deformation, R) * -1

            stress = None
            if compute_stress:
                stress_fn = lambda deformation, R: grad(total_energy_fn, argnums=0)(deformation, R) / jnp.linalg.det(box)
                stress = stress_fn(deformation, R)  

            stresses = None
            if compute_stresses:
                stresses_fn = lambda deformation, R: jacfwd(deformation_energy_fn, argnums=0)(deformation, R) / jnp.linalg.det(box)
                stresses = stresses_fn(deformation, R)

            total_energy = total_energy_fn(deformation, R)
            atomwise_energies = deformation_energy_fn(deformation, R)
            forces = force_fn(deformation, R)

            return total_energy, atomwise_energies, forces, stress, stresses


        def unstrained_potential_fn(R: space.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            energy_fn = energy.lennard_jones_pair(displacement_fn, sigma=sigma, epsilon=epsilon, r_onset=r_onset, r_cutoff=r_cutoff, per_particle=True)       
            total_energy_fn = lambda R: jnp.sum(energy_fn(R))
            forces_fn = quantity.force(total_energy_fn)

            total_energy = total_energy_fn(R)
            atomwise_energies = energy_fn(R)
            forces = forces_fn(R)

            return total_energy, atomwise_energies, forces


        if compute_stress or compute_stresses:
            if use_jit:
               return jit(displacement_fn), jit(strained_potential_fn)
            return displacement_fn, strained_potential_fn

        if use_jit:
            return jit(displacement_fn), jit(unstrained_potential_fn)
        return displacement_fn, unstrained_potential_fn


    def _compute_properties(self) -> Result:
        if self._stress or self._stresses:
            # deformation = jnp.zeros_like(self._box)
            # total_energy, atomwise_energies, forces, stress, stresses = self._properties_fn(deformation, self._R)
            total_energy, atomwise_energies, forces, stress, stresses = self._properties_fn(self._R)
            return Result(self, self._n, total_energy, atomwise_energies, forces, stress, stresses)
        
        total_energy, atomwise_energies, forces = self._properties_fn(self._R)

        return Result(self, self._n, total_energy, atomwise_energies, forces, None, None)


    def _perform_warm_up(self):
        if not self._jit:
            raise RuntimeError("Warm-up only implemented for jit=True")
        self._compute_properties()
        print("Warm-up finished")

    
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_displacement_fn']
        del state['_properties_fn']

        # del state['from_ase_atoms']
        # del state['create_potential']
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.
        error_fn = lambda *args, **kwargs: print("Pickled instance cannot compute new data")
        self._displacement_fn = error_fn
        self._properties_fn = error_fn

        # self.from_ase_atoms = error_fn
        # self.create_potential = error_fn