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


def group_by_calculator_description(results: List[Result]) -> List[List[Result]]:
    '''Groups the passed list of results by their source Calculator'''
    groups = defaultdict(list)
    for r in results:
        # TODO: Class hash code is a better unique identifier here
        groups[r.calculator.description].append(r)
    return groups


def contains_multiple_runs(results: List[Result]) -> bool:
    '''If the number of runs is not equal to the distinct number of simulated system sizes, there have to be multiple runs.'''
    distinct_system_sizes = set([r.calculator.n for r in results])
    return len(distinct_system_sizes) != len(results)


def group_by_system_size(results: List[Result]) -> List[List[Result]]:
    groups = defaultdict(list)
    for r in results:
        groups[r.calculator.n].append(r)
    return groups


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

    # group by calculator description/identifier
    
        # for each group, group by system size
            # compute averages

        # plot averages

    fig, ax = plt.subplots()

    runs = []

    for key, rs in group_by(results, lambda r: r.calculator.description):
        rs = list(rs)   # all ASE results
        computation_times = []      

        for key, mergeable_results in group_by(rs, lambda r: r.n):
           mergeable_results = list(mergeable_results)
           runs.append(len(mergeable_results))
            # map(lambda r: r.computation_time, mergeable_results)
        
           mean_computation_time = np.mean([r.computation_time for r in mergeable_results])   # for all runs of the current system size
           computation_times.append(mean_computation_time)

        print(len(system_sizes), len(computation_times))
        # plt.plot(system_sizes, computation_times, label=rs[0].calculator.description)
        ax.plot(system_sizes, computation_times, label=rs[0].calculator.description)
       
    if len(set(runs)) > 1:
        raise RuntimeError("Inconsistent number of runs in results")
    runs = runs[0]

    ax.set_title("{}\nAverage of {} runs".format(title, runs))
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Computation time [s]")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(file_name)









def plot_runtimes_old(title: str, system_sizes: List[int], rs: List[Result], file_name: str):
    # group by calculator
    # multiple 


    for results_by_calculator in group_by_calculator_description(rs).values():
        calculator_description = results_by_calculator[0].calculator.description

        if contains_multiple_runs(results_by_calculator):
            runs = group_by_system_size(results_by_calculator)

            for rs in runs.values():
                # TODO: label
                label = rs[0].calculator.description + ", " + str(rs[0].calculator.n)
                plt.plot(system_sizes, [r.computation_time for r in rs], label=label)
                print(rs)
    
            continue

        run_computation_times = [r.computation_time for r in results_by_calculator]
        plt.plot(system_sizes, run_computation_times, label=calculator_description)
    
    plt.title(title)
    plt.xlabel("Number of atoms")
    plt.ylabel("Computation time [s]")
    plt.yscale("log")
    plt.legend()
    plt.savefig(file_name)
