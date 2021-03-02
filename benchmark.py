from typing import List
from utils import *
from calculators.calculator import Result
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair

sigma = 2.0
epsilon = 1.5

system_sizes = generate_system_sizes(z_max=8, unit_cell_size=4)
results: List[Result] = []

runs = 100

print("Benchmarking system sizes: {}".format(system_sizes))
print("Performing {} run(s) per framework and system size\n".format(runs))

for n in system_sizes:
    print("System size n =", n)

    # ASE
    ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=None, r_onset=None)
    results.extend(ase.calculate(runs))
    
    # JAX-MD w/ jit
    jmd = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)    
    jmd.warm_up() 
    results.extend(jmd.calculate(runs))

    # JAX-MD w/o jit
    jmd_nojit = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, stresses=True, adjust_radii=True, jit=False)    
    results.extend(jmd_nojit.calculate(runs))
    
    # asax
    asax = AsaxLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True)
    asax.warm_up()
    results.extend(asax.calculate(runs))


persist_results(results, runs)