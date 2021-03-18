from typing import List
from utils import *
import jax_utils
from calculators.result import Result
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
import calculators.lennard_jones.pair.jaxmd_lennard_jones_pair as foo


def run_benchmark_loop(system_sizes: List[int]) -> List[Result]:
    results: List[Result] = []

    for n in system_sizes:
        print("System size n = {}".format(n))

        # ASE - initialize bulk structure & run
        ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=None, r_onset=None)
        # ase.warm_up()
        # results.extend(ase.calculate(runs))


        # JAX-MD: stress=True, stresses=True, jit=True
        jmd1 = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)    
        jmd1.warm_up() 
        results.extend(jmd1.calculate(runs))

        # JAX-MD: stress=True, stresses=False, jit=True
        jmd2 = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, stresses=False, adjust_radii=True, jit=True)    
        jmd2.warm_up() 
        results.extend(jmd2.calculate(runs))

        # JAX-MD: stress=False, stresses=False, jit=True
        jmd3 = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=False, stresses=False, adjust_radii=True, jit=True)    
        jmd3.warm_up() 
        results.extend(jmd3.calculate(runs))


        # JAX-MD: stress=False, stresses=False, jit=False
        # jmd_nojit = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=False, stresses=False, adjust_radii=True, jit=False)    
        # results.extend(jmd_nojit.calculate(runs))


        # JAX-MD Neighbor List:     stress=True, stresses=True, jit=True
        # jmd_nl1 = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)    
        # jmd_nl1.warm_up()    
        # results.extend(jmd_nl1.calculate(runs))

        # JAX-MD Neighbor List:     stress=True, stresses=False, jit=True
        # jmd_nl2 = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, stresses=False, adjust_radii=True, jit=True)    
        # jmd_nl2.warm_up()    
        # results.extend(jmd_nl2.calculate(runs))

        # JAX-MD Neighbor List:     stress=False, stresses=False, jit=True
        # jmd_nl3 = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=False, stresses=False, adjust_radii=True, jit=True)    
        # jmd_nl3.warm_up()    
        # results.extend(jmd_nl3.calculate(runs))

    return results


sigma = 2.0
epsilon = 1.5
runs = 100

system_sizes = generate_system_sizes(z_max=8, unit_cell_size=4)
xla_flag = jax_utils.get_memory_allocation_mode()

print("Benchmarking system sizes: {}".format(system_sizes))
print("Performing {} run(s) per framework and system size\n".format(runs))
print("Memory allocation mode: {}{}".format(xla_flag, "\n"))

results = run_benchmark_loop(system_sizes)
persist_results(results, runs, descriptor=xla_flag.value)