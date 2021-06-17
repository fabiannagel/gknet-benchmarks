from typing import Type, Dict
from md_driver import MdDriver
from ase_nve import AseNeighborListNve
from jax_nve_nl import JaxmdNeighborListNve
from asax_nve import AsaxNeighborListNve
from ase import Atoms, units
import jax_utils
from jax import config
import utils
config.update("jax_enable_x64", True)


def get_results_dict(steps: int, batch_size: int, runs: int, dt: float):
    return {
        'run_info': {
            'steps': steps,
            'batch_size': batch_size,
            'runs': runs,
            'dt': dt
        }
    }


def run_md_driver(clazz: Type[MdDriver], atoms: Atoms, dt: float, steps: int, batch_size: int, write_stress: bool, verbose: bool) -> MdDriver:
    md = clazz(atoms, dt, batch_size)
    print("MD Driver:             \t {} (n = {})".format(md.description, len(atoms)))
    print("Benchmark in progress...")
    md.run(steps, write_stress, verbose)
    print("Total simulation time: \t {} seconds".format(md.total_simulation_time))
    print("Mean time per batch:   \t {} ms".format(md.mean_batch_time))
    print("Average time per step: \t {} ms\n".format(md.mean_step_time))
    return md


def perform_runs(md_driver: Type[MdDriver], atoms: Atoms, dt: float, steps: int, batch_size: int, runs: int):
    total_simulation_times = []
    mean_step_times = []

    for i in range(runs):
        md = run_md_driver(md_driver, atoms, dt, steps, batch_size, write_stress=False, verbose=False)
        total_simulation_times += [md.total_simulation_time]
        mean_step_times += [md.mean_step_time]

    print("{} \t\t n = {} \t\t ms/step ({} runs): \t\t {}".format(md.description, len(atoms), runs, mean_step_times))
    return total_simulation_times, mean_step_times


super_cells = list(filter(lambda atoms: len(atoms) > 500, utils.load_super_cells("../super_cells")))
steps = 1000
batch_size = 5
runs = 3
dt = 5 * units.fs
results = get_results_dict(steps, batch_size, runs, dt)

for md_driver in [AseNeighborListNve, JaxmdNeighborListNve, AsaxNeighborListNve]:
    results[md_driver] = {}

    for atoms in super_cells:
        total_simulation_times, mean_step_times = perform_runs(md_driver, atoms, dt, steps, batch_size, runs)
        results[md_driver][len(atoms)] = {
            'total_simulation_seconds': total_simulation_times,
            'mean_step_milliseconds': mean_step_times
        }

utils.persist(results, "md_benchmark_results.pickle")






