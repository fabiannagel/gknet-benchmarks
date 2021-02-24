from typing import List
import warnings

from jax_md import space
from periodic_general import periodic_general as new_periodic_general
from periodic_general import inverse as new_inverse
from periodic_general import transform as new_transform

from calculators.calculator import Result

import matplotlib.pyplot as plt
from collections import defaultdict



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


def plot_runtimes(title: str, system_sizes: List[int], results: List[Result], file_name: str):
    # group results by calculator description
    groups = defaultdict(list)
    for r in results:
        groups[r.calculator.description].append(r)
    
    for result_group in groups.values():
        calculator_description = result_group[0].calculator.description
        computation_times = [r.computation_time for r in result_group]
        plt.plot(system_sizes, computation_times, label=calculator_description)
        # print(calculator_description)
        # print([r.computation_time for r in result_group])
        # print()
    
    plt.title(title)
    plt.xlabel("Number of atoms")
    plt.ylabel("Computation time [s]")
    plt.legend()
    plt.savefig(file_name)
