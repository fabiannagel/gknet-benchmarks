from typing import Dict, Set
import numpy as np
from ase import units
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import utils


def get_atom_counts(results: Dict) -> Set:
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


def plot_md_results(file_name: str):
    results: Dict = utils.load(file_name)
    atom_counts = get_atom_counts(results)

    for md_driver_name, benchmarked_systems in results.items():
        if md_driver_name == 'run_info':
            continue

        mean_step_milliseconds = []
        standard_deviations = []
        step_milliseconds_min = []
        step_milliseconds_max = []

        for n, runtimes in benchmarked_systems.items():
            total_simulation_seconds = runtimes['total_simulation_seconds']
            step_milliseconds = runtimes['mean_step_milliseconds']  # ms/step as the mean of all measured runtimes per step

            mean_step_milliseconds += [np.mean(step_milliseconds)]
            standard_deviations += [np.std(step_milliseconds)]
            step_milliseconds_min += [np.min(step_milliseconds)]
            step_milliseconds_max += [np.max(step_milliseconds)]

        adapted_atom_counts = atom_counts[:len(mean_step_milliseconds)]
        converted_label = convert_label(md_driver_name)

        # shade by min/max values
        plt.fill_between(adapted_atom_counts, step_milliseconds_min, step_milliseconds_max, alpha=0.2)

        # shade by 2 standard deviations
        # y_start = np.array(mean_step_milliseconds) - 2 * np.array(standard_deviations)
        # y_end = np.array(mean_step_milliseconds) + 2 * np.array(standard_deviations)
        # plt.fill_between(adapted_atom_counts, y_start, y_end, alpha=0.2)

        plt.plot(adapted_atom_counts, mean_step_milliseconds, label=converted_label, linestyle='--', marker='o', markersize=5)

    plt.title("{}\n{}".format("NVE runtime for increasing atom count of Lennard-Jones Argon", pretty_print_run_info(results['run_info'])))
    plt.xlabel("Number of atoms")
    plt.ylabel("Computation time per MD step [ms]")
    # plt.yscale("log")
    plt.legend()
    plt.show()


plot_md_results("asax_jaxmd_5000_steps.pickle")
