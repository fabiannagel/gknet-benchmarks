from ase.calculators.lj import LennardJones

import jax_utils
from md_driver import MdDriver
import time
from typing import List, Tuple, Dict, Callable

import numpy as np
from ase import units, Atoms
from jax import jit, lax, grad
from jax.config import config
import jax.numpy as jnp
from jax_md import simulate, space, energy, quantity, util
from jax_md.energy import NeighborFn, NeighborList
from jax_md.simulate import ApplyFn, ShiftFn, Array

from jax_utils import DisplacementFn, EnergyFn, NVEState

config.update("jax_enable_x64", False)


class JaxmdNeighborListNve(MdDriver):
    displacement_fn: DisplacementFn
    shift_fn: ShiftFn
    energy_fn: EnergyFn
    neighbor_fn: NeighborFn
    apply_fn: ApplyFn

    initial_state: NVEState
    initial_neighbor_list: NeighborList
    final_state: NVEState

    def __init__(self, atoms: Atoms, dt: float, batch_size: int, dr_threshold=1 * units.Angstrom, nl_extra_capacity=100):
        super().__init__(atoms, jnp.float32(dt), batch_size)
        self.dr_threshold = jnp.float32(dr_threshold)
        self.nl_extra_capacity = jnp.int16(nl_extra_capacity)
        self.box = jnp.float32(atoms.get_cell().array)
        self.R = jnp.float32(atoms.get_positions())
        self._initialize()

    @property
    def description(self) -> str:
        return "JAX-MD"

    def _initialize(self):
        self.displacement_fn, self.shift_fn = self._setup_space()
        self.neighbor_fn, self.energy_fn = self._setup_potential(self.displacement_fn)
        self.initial_neighbor_list = self.neighbor_fn(self.R, extra_capacity=self.nl_extra_capacity)
        self.initial_state, self.apply_fn = self._setup_nve(self.energy_fn, self.shift_fn)

    def _setup_space(self) -> Tuple[DisplacementFn, ShiftFn]:
        displacement_fn, shift_fn = space.periodic_general(self.box, fractional_coordinates=False)
        return jit(displacement_fn), jit(shift_fn)

    def _setup_potential(self, displacement_fn: DisplacementFn) -> Tuple[NeighborFn, EnergyFn]:
        # we require a calculator to obtain the initial NVEState
        lj_parameters = jax_utils.get_argon_lennard_jones_parameters()
        self.atoms.calc = LennardJones(sigma=lj_parameters['sigma'], epsilon=lj_parameters['epsilon'], rc=lj_parameters['rc'], ro=lj_parameters['ro'], smooth=True)

        normalized_ro = lj_parameters['ro'] / lj_parameters['sigma']
        normalized_rc = lj_parameters['rc'] / lj_parameters['sigma']

        # TODO: use same unstrained potential as asax
        neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(displacement_fn, self.box,
                                                                    sigma=lj_parameters['sigma'],
                                                                    epsilon=lj_parameters['epsilon'],
                                                                    r_onset=normalized_ro,
                                                                    r_cutoff=normalized_rc,
                                                                    dr_threshold=self.dr_threshold)
                                                                    # per_particle=True)

        return neighbor_fn, energy_fn

    def _setup_nve(self, energy_fn: EnergyFn, shift_fn: ShiftFn) -> Tuple[NVEState, ApplyFn]:
        _, apply_fn = simulate.nve(energy_fn, shift_fn, dt=self.dt)
        return jax_utils.get_initial_nve_state(self.atoms), apply_fn

    def _get_step_fn(self, neighbor_fn: NeighborFn, apply_fn: ApplyFn):
        def step_fn(i, state):
            state, neighbors = state
            neighbors = neighbor_fn(state.position, neighbors)
            state = apply_fn(state, neighbor=neighbors)
            return state, neighbors

        return step_fn

    def _run_md(self, steps: int, verbose: bool):
        step_fn = self._get_step_fn(self.neighbor_fn, self.apply_fn)
        self._run_md_loop(steps, step_fn, self.initial_state, self.initial_neighbor_list, verbose)

    def _run_md_loop(self, steps: int, step_fn: Callable, state: NVEState, neighbors: NeighborList, verbose: bool):
        i = 0

        while i < steps:
            batch_start_time = time.monotonic()
            state, neighbors = lax.fori_loop(0, self.batch_size, step_fn, (state, neighbors))

            if neighbors.did_buffer_overflow:
                neighbors = self.neighbor_fn(state.position)
                print("Steps {}/{}: Neighbor list overflow, recomputing...".format(i, steps))
                continue

            self._batch_times += [round((time.monotonic() - batch_start_time) * 1000, 2)]

            if verbose:
                print("Steps {}/{} took {} ms".format(i, steps, self.batch_times[-1]))
            i += self.batch_size

        state.position.block_until_ready()
        state.mass.block_until_ready()
        state.force.block_until_ready()
