from typing import Callable, Iterable, List, Set
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


def persist_results(results: List[Result], file_name='results.pickle'):
    with open(file_name, 'wb') as handle:
        pickle.dump(results, handle)


def plot_saved_runtimes(file_name: str):
    with open(file_name, 'wb') as handle:
        results = pickle.load(results, handle)
        plot_runtimes('foo', system_sizes, results, 'foo')  

def sort_by_calculator(results: List[Result]) -> List[Result]:
    return sorted(results, key=lambda r: r.calculator.description)

def group_by(iterable: Iterable, key: Callable):
    '''Sorts and groups the iterable by the provided key.'''
    iterable = sorted(iterable, key=key)
    return groupby(iterable, key)


def plot_runtimes(title: str, system_sizes: List[int], results: List[Result], file_name: str):
    fig, ax = plt.subplots()
    runs = []

    for key, results_per_calculator in group_by(results, lambda r: r.calculator.description):
        results_per_calculator = list(results_per_calculator)
        computation_times = []      

        for key, mergeable_results in group_by(results_per_calculator, lambda r: r.n):
           mergeable_results = list(mergeable_results)
           runs.append(len(mergeable_results))
        
           mean_computation_time = np.mean([r.computation_time for r in mergeable_results])   # for all runs of the current system size
           computation_times.append(mean_computation_time)

        ax.plot(system_sizes, computation_times, label=results_per_calculator[0].calculator.description)
       
    if len(set(runs)) > 1:
        raise RuntimeError("Inconsistent number of runs in results")

    ax.set_title("{}\nAverage of {} runs".format(title, runs[0]))
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Computation time [s]")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(file_name)
