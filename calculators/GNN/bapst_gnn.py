from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Dict
from functools import partial
import warnings

import jax_utils
from jax_utils import PotentialFn, XlaMemoryFlag
from calculators.calculator import Calculator
from calculators.result import JaxResult, Result
from ase.atoms import Atoms
from jax_md import space, energy
from jax_md.energy import NeighborFn, NeighborList
from jax_md.space import DisplacementFn
from jax_md.nn import InitFn
from periodic_general import periodic_general
import time
import jax.numpy as jnp
from jax import jit, random
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_log_compiles", 1)

class BapstGNN(Calculator):
    _short_description = "GNN Neighbor List"
    
    _energy_fn: Callable[[space.Array, NeighborList], space.Array] = None
    _init_fn: InitFn = None
    _neighbor_fn: NeighborFn = None
    _neighbors: NeighborList = None

    # TODO: box_size and displacement_fn are two different ways to initialize
    # Either we can create our own displacement_fn via space.periodic(box_size) ...
    # ... or use the passed displacement_fn that is used when importing an ASE object
    # Unite in new parameter box_size_or_displacement? What is the proper way to do this?

    # TODO: Create lightweight type for LJ parameters?
    def __init__(self, box: jnp.ndarray, n: int, R: jnp.ndarray, r_cutoff: float, stress: bool, stresses: bool, jit: bool, displacement_fn: Optional[Callable], skip_initialization=False):
        super().__init__(box, n, R, stress)
        self._r_cutoff = r_cutoff
        self._stress = stress
        self._stresses = stresses
        self._jit = jit
        self._memory_allocation_mode = jax_utils.get_memory_allocation_mode()

        self._box = jnp.array(self._box)
        self._R = jnp.array(self._R)
        
        if not skip_initialization:
            elapsed_seconds, return_val = self._time_execution(self._initialize_potential, displacement_fn)
            self._initialization_time = elapsed_seconds
            self._displacement_fn, self._potential_fn = return_val
            # self._displacement_fn, self._potential_fn = self._initialize_potential(displacement_fn)

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, r_cutoff: float, stress: bool, stresses: bool, jit: bool, skip_initialization=False) -> BapstGNN:
        displacement_fn = jax_utils.new_get_displacement(atoms)
        return super().from_ase_atoms(atoms, r_cutoff, stress, stresses, jit, displacement_fn, skip_initialization)


    @classmethod
    def create_potential(cls, box_size: float, n: int, R_scaled: Optional[jnp.ndarray], r_cutoff: float, stress: bool, stresses: bool, jit: bool, skip_initialization=False) -> BapstGNN:
        '''Initialize a Lennard-Jones potential from scratch using scaled atomic coordinates. If omitted, random coordinates will be generated.'''
        return super().create_potential(box_size, n, R_scaled, r_cutoff, stress, stresses, jit, None, skip_initialization)


    @property
    def description(self, include_memory_allocation=False) -> str:
        base_description = self._short_description + " (stress={}, stresses={}, jit={}{})"
        # base_description = "Bapst et al. 2020 (GNN) (stress={}, stresses={}, jit={}{})"
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
            displacement_fn, _ = periodic_general(self._box)

        if self._neighbors is None:
            self._neighbor_fn, self._init_fn, self._energy_fn = energy.graph_network_neighbor_list(displacement_fn, self._box, r_cutoff=self._r_cutoff, dr_threshold=0.0)
            self._neighbors = self._neighbor_fn(self._R, extra_capacity=6)

            key = random.PRNGKey(0)
            self._params = self._init_fn(key, self._R, self._neighbors)
        
        # predicted = vmap(train_energy_fn, (None, 0))(params, example_positions)

        potential: PotentialFn
        if self._stress or self._stresses:
            potential = jax_utils.get_strained_gnn_potential(self._energy_fn, self._neighbors, self._params, self._box, self._stress, self._stresses)
            return jax_utils.jit_if_wanted(self._jit, displacement_fn, potential)

        potential = jax_utils.get_unstrained_gnn_potential(self._energy_fn, self._neighbors, self._params)
        return jax_utils.jit_if_wanted(self._jit, displacement_fn, potential)                   


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