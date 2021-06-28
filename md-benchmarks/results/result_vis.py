import sys
from typing import Dict
import numpy as np

if not '/Users/fabian/git/gknet-benchmarks' in sys.path:
    sys.path.insert(0, '/Users/fabian/git/gknet-benchmarks')

import matplotlib.pyplot as plt
import utils


def get_atom_counts():
    atom_counts = []
    for md_driver_name, benchmarked_systems in results.items():
        if md_driver_name == 'run_info':
            continue

        atom_counts += list(results[md_driver_name].keys())
    return sorted(set(atom_counts))


results: Dict = utils.load("jaxmd_asax_benchmark_results.pickle")
run_info = results['run_info']
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
    plt.plot(adapted_atom_counts, mean_step_milliseconds, label=md_driver_name)

plt.xlabel("Number of atoms")
plt.ylabel("Computation time per MD step [ms]")
# plt.yscale("log")
plt.legend()
plt.show()

