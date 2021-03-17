from typing import List
from utils import *
import jax_utils
from jax_utils import XlaMemoryFlag
from calculators.result import Result
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList


def run_benchmark_loop():
    for n in system_sizes:
        print("System size n = {}".format(n))

        # ASE - only to initialize bulk structure
        ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=None, r_onset=None)

        # JAX-MD: stress=True, stresses=True, jit=True
        jmd1 = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)    
        jmd1.warm_up() 
        results.extend(jmd1.calculate(runs))


sigma = 2.0
epsilon = 1.5
runs = 100

system_sizes = generate_system_sizes(z_max=8, unit_cell_size=4)
# system_sizes = [500, 864, 1372, 2048]
results: List[Result] = []

print("Benchmarking system sizes: {}".format(system_sizes))
print("Performing {} run(s) per framework and system size\n".format(runs))

print("JAX Memory Allocation Mode: Default")
jax_utils.reset_memory_allocation_mode()
run_benchmark_loop()

print("JAX Memory Allocation Mode: On Demand + Deallocation")
jax_utils.set_memory_allocation_mode(XlaMemoryFlag.XLA_PYTHON_CLIENT_ALLOCATOR, "platform")
run_benchmark_loop()

persist_results(results, runs, descriptor='memory_modes')