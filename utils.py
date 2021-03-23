from vibes.helpers.supercell import make_cubic_supercell
from calculators.calculator import Calculator
import os
from typing import Callable, Iterable, List
from calculators.result import Result
import matplotlib.pyplot as plt
import numpy as np
import pickle
from itertools import groupby
from ase.build import bulk



def generate_unit_cells(z_max: int, unit_cell_size):
    ns = []
    for i in range(z_max):
        n = unit_cell_size * (i+1)**3
        ns.append(n)
    return ns


def generate_system_sizes(start: int, stop: int, step=100) -> List[int]:
    return list(range(start, stop+step, step))


def generate_cubic_system_sizes(start: int, stop: int, step=100) -> List[int]:
    system_sizes = []

    for n in range(start, stop+step, step):
        atoms = bulk("Ar", cubic=True)
        atoms, _ = make_cubic_supercell(atoms, target_size=n)
        system_sizes.append(len(atoms))

    return sorted(set(system_sizes))


def create_output_path(runs: int) -> str:
    results_dir = "results/{}_runs/".format(runs)
    output_path = os.path.join(os.getcwd(), results_dir)

    if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
        raise RuntimeError("Output folder {} is not empty".format(output_path))

    os.makedirs(output_path, exist_ok=True)
    return output_path


def persist_results(results: List[Result], runs: int, descriptor=''):  
    if descriptor:
        file_name = "results_{}_{}_runs.pickle".format(descriptor, runs)
        output_folder_suffix = "{}_{}".format(descriptor, runs)
    else:
        file_name = "results_{}_runs.pickle".format(runs)
        output_folder_suffix = str(runs)

    base_path = create_output_path(output_folder_suffix)
    output_path = os.path.join(base_path, file_name) 

    with open(output_path, 'wb') as handle:
        pickle.dump(results, handle)


def group_by(iterable: Iterable, key: Callable):
    '''Sorts and groups the iterable by the provided key.'''
    iterable = sorted(iterable, key=key)
    return groupby(iterable, key)


def get_system_sizes(results: List[Result]) -> List[int]:
    unique_ns = set([r.n for r in results])
    return sorted(unique_ns)


def load_results_from_pickle(file_path: str) -> List[Result]:
    with open(file_path, 'rb') as handle:
        results = pickle.load(handle)
    return results


def load_calculators_from_pickle(file_path: str) -> List[Calculator]:
    with open(file_path, 'rb') as handle:
        calculators = pickle.load(handle)
    return calculators


def plot_runtimes(results: List[Result], 
                  plot_title: str = None,
                  plot_file_name: str = None, 
                  shade_by: str = None,
                  scatter = False):
    
    runs = []
    system_sizes = get_system_sizes(results)
    fig, ax = plt.subplots(figsize=(20, 10))

    for key, results_per_calculator in group_by(results, lambda r: r.calculator.description):
        results_per_calculator = list(results_per_calculator)    
        
        computation_times = []
        standard_deviations = []
        mins = []
        maxs = []

        for key, mergeable_results in group_by(results_per_calculator, lambda r: r.n):
            mergeable_results = list(mergeable_results)
            runs.append(len(mergeable_results))

            mergeable_computation_times = [r.computation_time for r in mergeable_results]
           
            computation_times.append(np.mean(mergeable_computation_times))
            standard_deviations.append(np.std(mergeable_computation_times))      
            mins.append(np.min(mergeable_computation_times))
            maxs.append(np.max(mergeable_computation_times))     
            
            if scatter:
                current_system_sizes = [r.n for r in mergeable_results]
                ax.scatter(current_system_sizes, mergeable_computation_times)
        
        
        ax.plot(system_sizes, computation_times, label=results_per_calculator[0].calculator.description)
        # plt.errorbar(system_sizes, computation_times, yerr=np.array(computation_times)*10, uplims=True, lolims=True, label='uplims=True, lolims=True')

        if shade_by == 'minmax':
            ax.fill_between(system_sizes, mins, maxs, alpha=0.2)

        elif shade_by == 'std':
            y_start = np.array(computation_times) - np.array(standard_deviations)
            y_end = np.array(computation_times) + np.array(standard_deviations)
            ax.fill_between(system_sizes, y_start, y_end, alpha=0.2)
    
        
    if len(set(runs)) > 1:
        raise RuntimeError("Inconsistent number of runs in results")
        
    title = "{}\nAverages of {} runs. {} shading.".format(plot_title, runs[0], shade_by)
    ax.set_title(title)
    ax.set_xlabel("Number of atoms")
    ax.set_xticks(system_sizes)
    ax.set_ylabel("Computation time [s]")
    ax.set_yscale("log")
    ax.legend()
    
    if plot_file_name:
        fig.savefig(plot_file_name)



def get_calculator_description(results: List[Result]):
    descriptions = set(map(lambda r: r.calculator.description, results))
    if len(descriptions) > 1:
        raise RuntimeError("Expected only results of a single calculator, got multiple.")
    return list(descriptions)[0]


def contains_multiple_memory_allocation_modes(results: List[Result]) -> bool:
    # Breaking results down by XLA memory allocation modes only makes sense for JAX-MD results
    if not all("JAX-MD" in r.calculator.description for r in results):
        return False

    memory_allocation_modes = set(map(lambda r: r.calculator.memory_allocation_mode, results))
    return len(memory_allocation_modes) > 1   
    

def plot_runtime_variances(results: List[Result], ):
    '''For results of a single calculator, visualize the runtimes over indices separately for each system sizes'''

    if not results:
        raise RuntimeError("Provided result list is empty.")

    if contains_multiple_memory_allocation_modes(results):
        raise RuntimeError("Provided result list was computed with multiple memory allocation modes.")

    system_sizes = sorted(set(map(lambda r: r.calculator.n, results)))

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.15)
    n_columns = 2
    n_rows = int(np.ceil(len(system_sizes) / n_columns))    
    
    suptitle = "Computation time of multiple runs with the same system size\nCalculator: {}".format(get_calculator_description(results))
    fig.suptitle(suptitle)

    for i, n in enumerate(system_sizes):
        rs = list(filter(lambda r: r.calculator.n == n, results))
        computation_times = list([r.computation_time for r in rs])  

        ax = fig.add_subplot(n_rows, n_columns, i + 1)
        ax.plot(computation_times)

        subplot_title = "n = {}".format(n)
        ax.set_title(subplot_title)
    
        # fig.set_xlabel("Run index")
        # fig.set_ylabel("Runtime")

    
def plot_oom_behavior(labels: List[str], system_sizes: List[int], all_properties: List[Calculator], only_stress: List[Calculator], only_stresses: List[Calculator], only_energies_and_forces: List[Calculator], only_energies_and_forces_no_jit: List[Calculator], figsize=(10, 5)):
    bar_width = 0.1
    
    # label locations
    r1 = np.arange(len(all_properties))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]

    plt.figure(figsize=figsize)
    plt.bar(r1, all_properties, width=bar_width, label='All properties')
    plt.bar(r2, only_stress, width=bar_width, label='Only stress')
    plt.bar(r3, only_stresses, width=bar_width, label='Only stresses')
    plt.bar(r4, only_energies_and_forces, width=bar_width, label='Only energies and forces')
    plt.bar(r5, only_energies_and_forces_no_jit, width=bar_width, label='Only energies and forces, no jit')

    plt.title("Maximum number of atoms before going out-of-memory, per calculator")
    plt.xlabel('Calculator implementations', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(all_properties))], labels)
    plt.ylabel('Maximum atom count', fontweight='bold')
    plt.yticks(system_sizes)
    plt.legend()
    plt.show()
    # plt.savefig()
