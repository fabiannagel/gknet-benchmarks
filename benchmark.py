# import sys
# print(sys.path[0])
# del sys.path[0]
# sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')
from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair
from typing import List
from utils import persist_results, plot_runtimes, plot_saved_runtimes
from calculators.calculator import Result
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
# from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair
import pickle

def generate_system_sizes(z_max: int, unit_cell_size):
    ns = []
    for i in range(z_max):
        n = unit_cell_size * (i+1)**3
        ns.append(n)
    return ns


sigma = 2.0
epsilon = 1.5

system_sizes = generate_system_sizes(z_max=4, unit_cell_size=4)
results: List[Result] = []

runs = 4

for n in system_sizes:
    print("System size n =", n)

    # ASE
    ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=None, r_onset=None)
    results.extend(ase.calculate(runs))


    # JAX-MD w/ jit
    jmd = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, adjust_radii=True, jit=True)    
    jmd.warm_up() 
    results.extend(jmd.calculate(runs))

    
    continue
    

    # JAX-MD w/o jit
    jmd_nojit = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, adjust_radii=True, jit=False)    
    results.extend(jmd_nojit.calculate(runs))
    

    # asax
    asax = AsaxLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True)
    asax.calculate()
    for n in range(runs):
        results.append(asax.calculate())


plot_runtimes("Pairwise Lennard-Jones runtimes with increasing system size", system_sizes, results, file_name="pairwise_lj.png")
