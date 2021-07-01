import sys
from typing import Dict, Set
import numpy as np
from ase import units

# TODO: Remove from git!
if not '/Users/fabian/git/gknet-benchmarks' in sys.path:
    sys.path.insert(0, '/Users/fabian/git/gknet-benchmarks')

import matplotlib.pyplot as plt
import utils


def get_atom_counts() -> Set:
    atom_counts = []
    for md_driver_name, benchmarked_systems in results.items():
        if md_driver_name == 'run_info':
            continue

        atom_counts += list(results[md_driver_name].keys())
    return sorted(set(atom_counts))


def convert_label(raw_label: str) -> str:
    if "jax" in raw_label:
        return "JAX-MD"

    if "asax" in raw_label:
        return "ASAX"

    if "ase" in raw_label:
        return "ASE"

    return raw_label


def pretty_print_run_info(run_info: Dict) -> str:
    concatenated = ""
    for key, value in run_info.items():
        if key == 'dt':
            value = str(value / units.fs) + " fs"

        concatenated += str(key) + " = " + str(value) + ", "

    return concatenated[:-2]


results: Dict = utils.load("asax_jaxmd_5000_steps.pickle")
atom_counts = get_atom_counts()

for md_driver_name, benchmarked_systems in results.items():
    if md_driver_name == 'run_info':
        continue

    mean_step_milliseconds = []
    step_milliseconds_min = []
    step_milliseconds_max = []

    for n, runtimes in benchmarked_systems.items():
        total_simulation_seconds = runtimes['total_simulation_seconds']
        step_milliseconds = runtimes['mean_step_milliseconds']  # ms/step as the mean of all measured runtimes per step

        mean_step_milliseconds += [np.mean(step_milliseconds)]
        step_milliseconds_min += [np.min(step_milliseconds)]
        step_milliseconds_max += [np.max(step_milliseconds)]

    adapted_atom_counts = atom_counts[:len(mean_step_milliseconds)]
    converted_label = convert_label(md_driver_name)
    plt.plot(adapted_atom_counts, mean_step_milliseconds, label=converted_label)

plt.title("{}\n{}".format("NVE runtime per step for increasing atom count", pretty_print_run_info(results['run_info'])))
plt.xlabel("Number of atoms")
plt.ylabel("Computation time per MD step [ms]")
# plt.yscale("log")
plt.legend()
plt.show()


