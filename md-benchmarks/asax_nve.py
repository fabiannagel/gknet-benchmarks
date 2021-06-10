from md_driver import MdDriver
from ase.md import VelocityVerlet
from ase.atoms import Atoms
from asax.lj import LennardJones
from ase.calculators.lj import LennardJones as aseLJ
import time


class AsaxNeighborListNve(MdDriver):

    def __init__(self, atoms: Atoms, dt: float, batch_size: int):
        super().__init__(atoms, dt, batch_size)
        sigma = 2.0
        epsilon = 1.5
        rc = 10.0
        ro = 6.0
        self.atoms.calc = LennardJones(epsilon, sigma, rc, ro, stress=False)
        self.dyn = VelocityVerlet(atoms, timestep=dt)

    @property
    def description(self) -> str:
        return "ASAX"

    def _run_md(self, steps: int, write_stress: bool, verbose: bool):
        i = 0

        while i < steps:
            i += self.batch_size
            batch_start_time = time.monotonic()
            self.dyn.run(self.batch_size)

            # elapsed time for simulating the last batch (in milliseconds)
            self._batch_times += [round((time.monotonic() - batch_start_time) * 1000, 2)]

            if verbose:
                print("Steps {}/{} took {} ms".format(i, steps, self.batch_times[-1]))