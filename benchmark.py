from typing import List
from utils import *
from calculators.calculator import Result
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList

sigma = 2.0
epsilon = 1.5
runs = 100

system_sizes = generate_system_sizes(z_max=8, unit_cell_size=4)
results: List[Result] = []

print("Benchmarking system sizes: {}".format(system_sizes))
print("Performing {} run(s) per framework and system size\n".format(runs))

for n in system_sizes:
    print("System size n = {}".format(n))

    # ASE
    ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=None, r_onset=None)
    results.extend(ase.calculate(runs))

    
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
    jmd_nojit = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=False, stresses=False, adjust_radii=True, jit=False)    
    results.extend(jmd_nojit.calculate(runs))


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

persist_results(results, runs)