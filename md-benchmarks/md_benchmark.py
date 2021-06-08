from typing import Type
from md_driver import MdDriver
from ase_nve import AseNeighborListNve
from jax_nve_nl import JaxmdNeighborListNVE
from asax_nve import AsaxNeighborListNve
from ase import Atoms, units
import jax_utils
import pickle
from jax import config
config.update("jax_enable_x64", True)
# TODO:

def persist_step_times(md: MdDriver):
    with open(md.description, 'wb') as handle:
        pickle.dump(md.step_times, handle)


def run(clazz: Type[MdDriver], atoms: Atoms, dt: float, steps: int, batch_size: int, write_stress: bool, verbose: bool):
    md = clazz(atoms, dt, batch_size)
    md.run(steps, write_stress, verbose)
    print("MD Driver:             \t {}".format(md.description))
    print("Total simulation time: \t {} seconds".format(md.total_simulation_time))
    print("Mean time per batch:   \t {} ms".format(md.mean_batch_time))
    print("Average time per step: \t {} ms\n".format(md.mean_step_time))
    # TODO: persist_step_times()


atoms = jax_utils.initialize_cubic_argon(multiplier=2)
dt = 5 * units.fs
steps = 1000
batch_size = 5
write_stress = False
verbose = False

run(AseNeighborListNve, atoms, dt, steps, batch_size, write_stress, verbose)
run(JaxmdNeighborListNVE, atoms, dt, steps, batch_size, write_stress, verbose)
run(AsaxNeighborListNve, atoms, dt, steps, batch_size, write_stress, verbose)

