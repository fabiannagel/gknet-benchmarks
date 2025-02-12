from typing import Type, Dict, List

import jax_utils
from md_driver import MdDriver
from ase_nve import AseNeighborListNve
from jax_nve_nl import JaxmdNeighborListNve
from asax_nve import AsaxNeighborListNve
from ase import Atoms, units
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


def save_oom_event(md_driver_key: str):
    print("{} went OOM at n = {}".format(md_driver_key, len(atoms)))
    oom_md_drivers.append(md_driver_key)


def save_results(md_driver_key: str, atoms: Atoms, total_simulation_seconds: List[float], mean_step_milliseconds: List[float]):
    if not total_simulation_seconds or not mean_step_milliseconds:
        return

    if not md_driver_key in results.keys():
        results[md_driver_key] = {}

    results[md_driver_key][len(atoms)] = {
        "total_simulation_seconds": total_simulation_seconds,
        "mean_step_milliseconds": mean_step_milliseconds
    }


def run_md_driver(md_driver: Type[MdDriver], atoms: Atoms, dt: float, steps: int, batch_size: int, *args, **kwargs) -> MdDriver:
    md = md_driver(atoms, dt, batch_size, *args, **kwargs)
    # print("MD Driver:             \t {} (n = {})".format(md.description, len(atoms)))
    # print("Benchmark in progress...")
    md.run(steps)
    # print("Total simulation time: \t {} seconds".format(md.total_simulation_time))
    # print("Mean time per batch:   \t {} ms".format(md.mean_batch_time))
    # print("Average time per step: \t {} ms\n".format(md.mean_step_time))
    return md


def perform_runs(md_driver: Type[MdDriver], md_driver_key: str, atoms: Atoms, dt: float, steps: int, batch_size: int, runs: int, *args, **kwargs):
    if md_driver_key in oom_md_drivers:
        print("{:<70} went OOM before. Skipping...".format(md_driver_key))
        return

    total_simulation_seconds = []
    mean_step_milliseconds = []

    try:
        for i in range(runs):
            print("{:<70} Performing run {}/{}".format(md_driver_key, i+1, runs))

            md = run_md_driver(md_driver, atoms, dt, steps, batch_size, *args, **kwargs)
            total_simulation_seconds += [md.total_simulation_time]
            mean_step_milliseconds += [md.mean_step_time]

    except RuntimeError:
        save_oom_event(md_driver_key)

    print(mean_step_milliseconds)
    save_results(md_driver_key, atoms, total_simulation_seconds, mean_step_milliseconds)


def run_ase(atoms: Atoms, dt: float, steps: int, batch_size: int, runs: int):
    key = str(AseNeighborListNve)
    perform_runs(AseNeighborListNve, key, atoms, dt, steps, batch_size, runs)


def run_jax_md(atoms: Atoms, dt: float, steps: int, batch_size: int, runs: int):
    key = str(JaxmdNeighborListNve)
    perform_runs(JaxmdNeighborListNve, key, atoms, dt, steps, batch_size, runs)


def run_asax(atoms: Atoms, dt: float, steps: int, batch_size: int, runs: int):
    key = str(AsaxNeighborListNve)
    perform_runs(AsaxNeighborListNve, key, atoms, dt, steps, batch_size, runs)


super_cells = list(filter(lambda atoms: len(atoms) >= 1000, utils.load_super_cells("../super_cells")))
# super_cells = [jax_utils.initialize_cubic_argon(multiplier=14)]
print("n = {}".format([len(atoms) for atoms in super_cells]))

steps = 5000
batch_size = 5
runs = 10
dt = 5 * units.fs

results = get_results_dict(steps, batch_size, runs, dt)
oom_md_drivers: List[str] = []

for atoms in super_cells:
    print("\nn = {}".format(len(atoms)))

    # run_ase(atoms, dt, steps, batch_size, runs)
    run_jax_md(atoms, dt, steps, batch_size, runs)
    run_asax(atoms, dt, steps, batch_size, runs)

# print(results)
utils.persist(results, "asax_jaxmd_5000_steps.pickle")