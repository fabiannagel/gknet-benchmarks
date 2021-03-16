import os
from os import system
from typing import Callable, Iterable, List, Set, Union, Type
import warnings

from jax_md import space
from periodic_general import periodic_general as new_periodic_general
from periodic_general import inverse as new_inverse
from periodic_general import transform as new_transform

from calculators.calculator import Calculator, Result

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pickle
from itertools import groupby


def new_get_displacement(atoms):
    '''what asax.utils.get_displacement() does, only with functions from the new periodic_general()'''
    # TODO: Refactor once new periodic_general() is released

    if not all(atoms.get_pbc()):
        displacement, _ = space.free()
        warnings.warn("Atoms object without periodic boundary conditions passed!")
        return displacement

    cell = atoms.get_cell().array
    inverse_cell = new_inverse(cell)
    displacement_in_scaled_coordinates, _ = new_periodic_general(cell)

    # **kwargs are now used to feed through the box information
    def displacement(Ra: space.Array, Rb: space.Array, **kwargs) -> space.Array:
        Ra_scaled = new_transform(inverse_cell, Ra)
        Rb_scaled = new_transform(inverse_cell, Rb)
        return displacement_in_scaled_coordinates(Ra_scaled, Rb_scaled, **kwargs)

    # TODO: Verify JIT behavior
    return displacement


def generate_system_sizes(z_max: int, unit_cell_size):
    ns = []
    for i in range(z_max):
        n = unit_cell_size * (i+1)**3
        ns.append(n)
    return ns


def create_output_path(runs: int) -> str:
    results_dir = "results/{}_runs/".format(runs)
    output_path = os.path.join(os.getcwd(), results_dir)

    if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
        raise RuntimeError("Output folder {} is not empty".format(output_path))

    os.makedirs(output_path, exist_ok=True)
    return output_path


def persist_results(results: List[Result], runs: int):
    base_path = create_output_path(runs)
    file_name = "results_{}_runs.pickle".format(runs)
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
    

def plot_runtime_variances(results: List[Result], ):
    '''For results of a single calculator, visualize the runtimes over indices separately for each system sizes'''

    if not results:
        raise RuntimeError("Provided result list is empty.")

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

    
