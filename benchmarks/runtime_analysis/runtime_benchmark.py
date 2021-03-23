import sys
if not '/home/pop518504/git/gknet-benchmarks' in sys.path:
    sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from calculators.GNN.bapst_gnn import BapstGNN
from typing import List
from utils import *
import jax_utils
from calculators.result import Result
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList


def run_jaxmd_pair(ase: AseLennardJonesPair, results: List[Result]):
    # JAX-MD Pair: all properties                       (stress=True, stresses=True, jit=True)
    jmd1 = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)    
    jmd1.warm_up() 
    results.extend(jmd1.calculate(runs))
    
    # JAX-MD Pair: only stress                          (stress=True, stresses=False, jit=True)
    jmd2 = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=False, adjust_radii=True, jit=True)    
    jmd2.warm_up() 
    results.extend(jmd2.calculate(runs))
    
    # JAX-MD Pair: only stresses                        (stress=False, stresses=True, jit=True)
    jmd3 = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=True, adjust_radii=True, jit=True)    
    jmd3.warm_up() 
    results.extend(jmd3.calculate(runs))

    # JAX-MD Pair: only energies and forces             (stress=False, stresses=False, jit=True)
    jmd4 = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=True)    
    jmd4.warm_up() 
    results.extend(jmd4.calculate(runs))
    
    # JAX-MD Pair: only energies and forces, no jit     (stress=False, stresses=False, jit=False)
    jmd_nojit = JmdLennardJonesPair.from_ase_atoms(ase._atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=False)    
    results.extend(jmd_nojit.calculate(runs))


def run_jaxmd_neighbor_list(ase: AseLennardJonesPair, results: List[Result]):
    # JAX-MD Neighbor List: all properties              (stress=True, stresses=True, jit=True)
    jmd_nl1 = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)    
    jmd_nl1.warm_up()    
    results.extend(jmd_nl1.calculate(runs))
    
    # JAX-MD Neighbor List: only stress                 (stress=True, stresses=False, jit=True)
    jmd_nl2 = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=False, adjust_radii=True, jit=True)    
    jmd_nl2.warm_up()    
    results.extend(jmd_nl2.calculate(runs))

    # JAX-MD Neighbor List: only stresses               (stress=False, stresses=True, jit=True)
    jmd_nl3 = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=True, adjust_radii=True, jit=True)    
    jmd_nl3.warm_up()    
    results.extend(jmd_nl3.calculate(runs))
    
    # JAX-MD Neighbor List: only energies and forces    (stress=False, stresses=False, jit=True)
    jmd_nl4 = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=True)    
    jmd_nl4.warm_up()    
    results.extend(jmd_nl4.calculate(runs))


def run_jaxmd_gnn(ase: AseLennardJonesPair, results: List[Result]):
    # JAX-MD GNN: all properties                       (stress=True, stresses=True, jit=True
    gnn1 = BapstGNN.from_ase_atoms(ase._atoms, r_cutoff, r_onset, stress=True, stresses=True, jit=True)
    gnn1.warm_up()
    results.extend(gnn1.calculate(runs))
    
    # JAX-MD GNN: only stress                           (stress=True, stresses=False, jit=True)
    gnn2 = BapstGNN.from_ase_atoms(ase._atoms, r_cutoff, r_onset, stress=True, stresses=False, jit=True)
    gnn2.warm_up()
    results.extend(gnn2.calculate(runs))

    # JAX-MD GNN: only stresses                         (stress=False, stresses=True, jit=True)
    gnn3 = BapstGNN.from_ase_atoms(ase._atoms, r_cutoff, r_onset, stress=False, stresses=True, jit=True)
    gnn3.warm_up()
    results.extend(gnn3.calculate(runs))

    # JAX-MD GNN: only energies and forces              (stress=False, stresses=False, jit=True)
    gnn4 = BapstGNN.from_ase_atoms(ase._atoms, r_cutoff, r_onset, stress=False, stresses=False, jit=True)
    gnn4.warm_up()
    results.extend(gnn4.calculate(runs))


def run_benchmark_loop(system_sizes: List[int]) -> List[Result]:
    results: List[Result] = []
    computed_system_sizes = []

    for n in system_sizes:
        # ASE - initialize bulk structure & run
        ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=r_cutoff, r_onset=r_onset)
        if ase.n in computed_system_sizes:
            print("n={} already computed, skipping.".format(ase.n))
            continue

        if ase.n > n_max:
            print("n={} exceeding n_max={}, aborting.".format(ase.n, n_max))
            break

        print("\nSystem size n = {}\n".format(ase.n))

        computed_system_sizes.append(ase.n)
        ase.warm_up()
        results.extend(ase.calculate(runs))

        run_jaxmd_pair(ase, results)
        run_jaxmd_neighbor_list(ase, results)
        run_jaxmd_gnn(ase, results)

    return results
 

sigma = 3.4
epsilon = 10.42
r_cutoff = 10.54
r_onset = 8

n_min = 2000
n_max = 5500
n_step = 100
system_sizes = generate_system_sizes(start=n_min, stop=n_max, step=n_step)
runs = 1

print("Benchmarking system sizes: {}".format(system_sizes))
print("Performing {} run(s) per framework and system size".format(runs))
print("Memory allocation mode: {}".format(jax_utils.get_memory_allocation_mode()))

results = run_benchmark_loop(system_sizes)
persist_results(results, runs)

# TODO
# According to OOM benchmarks, we should be able to push all systems to n = 5800 (where JAX-MD Pair fails)
# But here, it fails much earlier.
# The only other difference to the OOM benchmark: Multiple runs + warmup

# even w/o warm-up, GNN (all properties) fails already at n = 2000.
# only GNN (no other calculators) still fails at n = 2000 (all properties).


# only thing left: it must be a matter of initialization
# in OOM, we use create_potential() and arbitrary super cells
# here, we use from_ase_atoms() and cubic super cells
# could the custom displacement be the problem? but why would it?

# TODO: maybe use a single strategy to perform benchmarks? cubic super cells is probably more trustworthy?