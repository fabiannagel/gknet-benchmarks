from ase.calculators.lj import LennardJones

import jax_utils
from md_driver import MdDriver
from ase.atoms import Atoms
from ase.md import VelocityVerlet
import time


class AseNeighborListNve(MdDriver):

    def __init__(self, atoms: Atoms, dt: float, batch_size: int):
        super().__init__(atoms, dt, batch_size)
        lj_parameters = jax_utils.get_argon_lennard_jones_parameters()
        self.atoms.calc = LennardJones(sigma=lj_parameters['sigma'], epsilon=lj_parameters['epsilon'], rc=lj_parameters['rc'], ro=lj_parameters['ro'], smooth=True)
        self.dyn = VelocityVerlet(atoms, timestep=dt)

    @property
    def description(self) -> str:
        return "ASE"

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