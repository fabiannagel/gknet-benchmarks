from ase import units

import jax_utils
from md_driver import MdDriver
from ase.md import VelocityVerlet
from ase.atoms import Atoms
from asax.lj import LennardJones
import time


class AsaxNeighborListNve(MdDriver):

    def __init__(self, atoms: Atoms, dt: float, batch_size: int, dr_threshold=1 * units.Angstrom, nl_extra_capacity=100):
        super().__init__(atoms, dt, batch_size)
        self.dr_threshold = dr_threshold
        self.nl_extra_capacity = nl_extra_capacity

        lj_parameters = jax_utils.get_argon_lennard_jones_parameters()
        # TODO: Pass nl_extra_capacity to asax LJ calculator (as jnp.int16)
        self.atoms.calc = LennardJones(sigma=lj_parameters['sigma'], epsilon=lj_parameters['epsilon'], rc=lj_parameters['rc'], ro=lj_parameters['ro'], stress=False, dr_threshold=self.dr_threshold)
        self.dyn = VelocityVerlet(atoms, timestep=dt)

    @property
    def description(self) -> str:
        return "ASAX"

    def _run_md(self, steps: int, verbose: bool):
        i = 0
        while i < steps:
            i += self.batch_size
            batch_start_time = time.monotonic()
            self.dyn.run(self.batch_size)

            # elapsed time for simulating the last batch (in milliseconds)
            self._batch_times += [round((time.monotonic() - batch_start_time) * 1000, 2)]

            if verbose:
                print("Steps {}/{} took {} ms".format(i, steps, self.batch_times[-1]))