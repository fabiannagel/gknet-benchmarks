from typing import Type
from md_driver import MdDriver
from ase_nve import AseNeighborListNve
from jax_nve_nl import JaxmdNeighborListNVE
from asax_nve import AsaxNeighborListNve
from ase import Atoms, units
import jax_utils
from jax import config
config.update("jax_enable_x64", True)


def run(clazz: Type[MdDriver], atoms: Atoms, dt: float, steps: int, batch_size: int, write_stress: bool, verbose: bool) -> MdDriver:
    md = clazz(atoms, dt, batch_size)
    print("MD Driver:             \t {} (n = {})".format(md.description, len(atoms)))
    print("Benchmark in progress...")
    md.run(steps, write_stress, verbose)
    print("Total simulation time: \t {} seconds".format(md.total_simulation_time))
    print("Mean time per batch:   \t {} ms".format(md.mean_batch_time))
    print("Average time per step: \t {} ms\n".format(md.mean_step_time))
    return md


atoms = jax_utils.initialize_cubic_argon(multiplier=14)
dt = 5 * units.fs
steps = 1000
batch_size = 5
write_stress = False
verbose = False

runs = 10
results = {}

for md_driver in [AseNeighborListNve, JaxmdNeighborListNVE, AsaxNeighborListNve]:
# for md_driver in [JaxmdNeighborListNVE, AsaxNeighborListNve]:
    mean_step_times = []

    for i in range(runs):
        md = run(md_driver, atoms, dt, steps, batch_size, write_stress, verbose)
        mean_step_times += [md.mean_step_time]
        print(md.total_simulation_time)

    print("{} \t\t n = {} \t\t ms/step ({} runs): \t\t {}".format(md.description, len(atoms), runs, mean_step_times))



