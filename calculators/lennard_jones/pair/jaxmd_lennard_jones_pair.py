from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Dict
from functools import partial
import warnings

import jax_utils
from jax_utils import PotentialFn, XlaMemoryFlag
from calculators.calculator import Calculator
from calculators.result import JaxResult, Result
from ase.atoms import Atoms
from jax_md import space
from jax_md.space import DisplacementFn
from periodic_general import periodic_general
import jax.numpy as jnp
from jax import jit
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
        self._displacement_fn, self._potential_fn = self._initialize_potential(displacement_fn)


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
    def description(self, include_memory_allocation=False) -> str:
        base_description = "JAX-MD Pair (stress={}, stresses={}, jit={}{})"
        if include_memory_allocation:
            memory_allocation = ", memory allocation={}".format(self._memory_allocation_mode)
            return base_description.format(self._stress, self._stresses, self._jit, memory_allocation)
        return base_description.format(self._stress, self._stresses, self._jit, "")


    @property
    def memory_allocation_mode(self) -> XlaMemoryFlag:
        return self._memory_allocation_mode


    @property
    @partial(jit, static_argnums=0)
    def pairwise_distances(self):
        return jax_utils.compute_pairwise_distances(self._displacement_fn, self._R)


    def _generate_R(self, n: int, scaling_factor: float) -> jnp.ndarray:
        return jax_utils.generate_R(n, scaling_factor)


    def _initialize_potential(self, displacement_fn: Optional[DisplacementFn]) -> Tuple[space.DisplacementFn, PotentialFn]:
        if displacement_fn is None:
            warnings.warn("Using default periodic_general")
            displacement_fn, _ = periodic_general(self._box)

        if self._stress or self._stresses:
            strained_potential = jax_utils.get_strained_pair_potential(self._box, displacement_fn, self._sigma, self._epsilon, self._r_cutoff, self._r_onset, self._stress, self._stresses)
            return jax_utils.jit_if_wanted(self._jit, displacement_fn, strained_potential)

        unstrained_potential = jax_utils.get_unstrained_pair_potential(self._box, displacement_fn, self._sigma, self._epsilon, self._r_cutoff, self._r_onset)
        return jax_utils.jit_if_wanted(self._jit, displacement_fn, unstrained_potential)


    def _compute_properties(self) -> Result:
        properties = self._potential_fn(self._R)
        return JaxResult(self, self._n, *properties)


    def _perform_warm_up(self):
        if not self._jit:
            raise RuntimeError("Warm-up only implemented for jit=True")
            
        self._compute_properties()
        print("Warm-up finished")

    
    def __getstate__(self) -> Dict:
        return jax_utils.get_state(self)


    def __setstate__(self, state):
        jax_utils.set_state(self, state)