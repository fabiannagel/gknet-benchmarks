# import sys
# print(sys.path[0])
# del sys.path[0]
# sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')
from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair
from typing import List
from utils import *
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

system_sizes = generate_system_sizes(z_max=8, unit_cell_size=4)
results: List[Result] = []

runs = 10

print("Benchmarking system sizes: {}".format(system_sizes))
print("Performing {} run(s) per framework and system size\n".format(runs))

for n in system_sizes:
    break

    print("System size n =", n)

    # ASE
    ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=None, r_onset=None)
    results.extend(ase.calculate(runs))


    # JAX-MD w/ jit
    jmd = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, adjust_radii=True, jit=True)    
    jmd.warm_up() 
    results.extend(jmd.calculate(runs))


    # JAX-MD w/o jit
    jmd_nojit = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, adjust_radii=True, jit=False)    
    results.extend(jmd_nojit.calculate(runs))
    

    # asax
    asax = AsaxLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True)
    asax.warm_up()
    results.extend(asax.calculate(runs))


# base_path = "{}_runs-{}_atoms".format(runs, max(system_sizes))
# base_path = "results"
# persist_results(results, base_path + "/results.pickle")

results = load_results_from_pickle("results/results.pickle")
results = list(filter(lambda r: "jit=False" not in r.calculator.description, results))
results = list(filter(lambda r: "ASE" not in r.calculator.description, results))
results = list(filter(lambda r: "ASAX" not in r.calculator.description, results))


plot_runtimes(results=results, 
              plot_title="Pairwise Lennard-Jones runtimes with increasing system size", 
              plot_file_name="results/pickled_lj.png", 
              scatter=True,
              shade_by="minmax")