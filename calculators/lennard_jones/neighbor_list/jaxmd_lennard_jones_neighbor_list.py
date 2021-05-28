from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict
import warnings
from functools import partial
import jax_utils
from jax_utils import PotentialFn, XlaMemoryFlag
from calculators.calculator import Calculator
from calculators.result import JaxResult, Result

from ase.atoms import Atoms
from jax_md import space, energy
from jax_md.space import DisplacementFn
from jax_md.energy import NeighborFn, NeighborList
# from periodic_general import periodic_general
import jax.numpy as jnp
from jax import jit
from jax.config import config
config.update("jax_enable_x64", False)
# config.update("jax_log_compiles", 0)

class JmdLennardJonesNeighborList(Calculator):
    _short_description = "JAX-MD Neighbor List"
    
    _energy_fn: Callable[[space.Array, NeighborList], space.Array] = None
    _neighbor_fn: NeighborFn = None
    _neighbors: NeighborList = None

    def __init__(self, box: jnp.ndarray, n: int, R: jnp.ndarray, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool, stresses: bool, jit: bool, displacement_fn: Optional[Callable], skip_initialization=False):
        super().__init__(box, n, R, stress)
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cutoff = r_cutoff
        self._r_onset = r_onset 
        self._stress = stress
        self._stresses = stresses
        self._jit = jit
        self._memory_allocation_mode = jax_utils.get_memory_allocation_mode()

        # np.array causes strange indexing errors with neighbor lists now and then
        self._box = jnp.array(self._box)
        self._R = jnp.array(self._R)

        # for OOM benchmarks to still obtain a data-only instance when OOM occurs in _initialize_potential()
        if not skip_initialization:
            elapsed_seconds, return_val = self._time_execution(self._initialize_potential, displacement_fn)
            self._initialization_time = elapsed_seconds
            self._displacement_fn, self._potential_fn = return_val
            # self._displacement_fn, self._potential_fn = self._initialize_potential(displacement_fn)


    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool, stresses: bool, adjust_radii: bool, jit: bool, skip_initialization=False) -> JmdLennardJonesNeighborList:
        displacement_fn, shift_fn = jax_utils.get_displacement(atoms)
        
        # JAX-MD's LJ implementation multiplies onset and cutoff by sigma. To be compatible w/ ASE's implementation, we need to perform these adjustments.
        if adjust_radii:
            r_onset /= sigma
            r_cutoff /= sigma

        return super().from_ase_atoms(atoms, sigma, epsilon, r_cutoff, r_onset, stress, stresses, jit, displacement_fn, skip_initialization)


    @classmethod
    def create_potential(cls, box_size: float, n: int, R_scaled: Optional[jnp.ndarray], sigma: float, epsilon: float, r_cutoff: float, r_onset: float, stress: bool, stresses: bool, jit: bool, skip_initialization=False) -> JmdLennardJonesNeighborList:
        '''Initialize a Lennard-Jones potential from scratch using scaled atomic coordinates. If omitted, random coordinates will be generated.'''
        return super().create_potential(box_size, n, R_scaled, sigma, epsilon, r_cutoff, r_onset, stress, stresses, jit, None, skip_initialization)


    @property
    def description(self, include_memory_allocation=False) -> str:
        base_description = self._short_description + " (stress={}, stresses={}, jit={}{})"
        # base_description = "JAX-MD Neighbor List (stress={}, stresses={}, jit={}{})"
        if include_memory_allocation:
            memory_allocation = ", memory allocation={}".format(self._memory_allocation_mode)
            return base_description.format(self._stress, self._stresses, self._jit, memory_allocation)
        return base_description.format(self._stress, self._stresses, self._jit, "")


    @property
    def short_description(self) -> str:
        return self._short_description


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
            displacement_fn, _ = space.periodic_general(self._box)
        
        if self._neighbors is None:
            self._neighbor_fn, self._energy_fn = energy.lennard_jones_neighbor_list(displacement_fn, self._box, sigma=self._sigma, epsilon=self._epsilon, r_onset=self._r_onset, r_cutoff=self._r_cutoff, per_particle=True)
            self._neighbors = self._neighbor_fn(self._R)

        if self._stress or self._stresses:
            strained_potential = jax_utils.get_strained_neighbor_list_potential(self._energy_fn, self._neighbors, self._box, self._stress, self._stresses)
            return jax_utils.jit_if_wanted(self._jit, displacement_fn, strained_potential)
        
        unstrained_potential = jax_utils.get_unstrained_neighbor_list_potential(self._energy_fn, self._neighbors)
        return jax_utils.jit_if_wanted(self._jit, displacement_fn, unstrained_potential)


    def _compute_properties(self) -> Result:
        properties = self._potential_fn(self._R)
        return JaxResult(self, self._n, *properties)


    def _perform_warm_up(self):
        if not self._jit:
            raise NotImplementedError("Warm-up only implemented for jit=True")

        self._compute_properties()
        print("Warm-up finished")

    
    def __getstate__(self):
        return jax_utils.get_state(self)

    def __setstate__(self, state):
        jax_utils.set_state(self, state)