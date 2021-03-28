import sys
if not '/home/pop518504/git/gknet-benchmarks' in sys.path:
    sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from typing import Dict, List, Tuple
from utils import *
from ase.atoms import Atoms
import jax_utils
from calculators.result import Result
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList
from calculators.GNN.bapst_gnn import BapstGNN


def save_oom_event(reason: str, callable: Callable, calc: Calculator, *args, **kwargs):
    if calc is None:
        calc = callable(*args, **kwargs, skip_initialization=True)  # if OOM during init, create a fake calc object for reference
    
    event = calc, reason
    oom_events.append(event)


def run_and_initialize_expect_oom(callable: Callable, results: List[Result], *args, **kwargs):
    calculator: Calculator = None

    try:
        calculator = callable(*args, **kwargs)
    except RuntimeError:
        save_oom_event("Initialization", callable, None, *args, **kwargs)
        print("OOM during calculator initialization")
        return

    try:
        calculator.warm_up()
    except NotImplementedError:
        pass                    # fine for some calculators.
    except RuntimeError:        # oom during warm-up
        save_oom_event("Warm-up", callable, None, *args, **kwargs)
        print("{} went oom at n={}".format(calculator, calculator.n))
        return    
     
    rs = calculator.calculate(runs)
    if len(calculator._oom_runs) > 0:
        save_oom_event("Skipped run", None, calculator, *args, **kwargs)

    results.extend(rs)


def run_ase(atoms: Atoms, results: List[Result]):
    n = len(atoms)
    if n >= n_max_ase:
        print("n={} exceeding n_max={} for ASE, skipping.".format(n, n_max_ase))
        return

    ase = AseLennardJonesPair.from_ase_atoms(atoms, sigma, epsilon, r_cutoff=r_cutoff, r_onset=r_onset)
    ase.warm_up()
    results.extend(ase.calculate(runs))


def run_jaxmd_pair(atoms: Atoms, results: List[Result]):
    n = len(atoms)

    # JAX-MD Pair: all properties                       (stress=True, stresses=True, jit=True)
    if n < n_max_jaxmd_pair[True, True, True]:
        run_and_initialize_expect_oom(JmdLennardJonesPair.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD Pair, skipping.".format(n, n_max_jaxmd_pair[True, True, True]))


    # JAX-MD Pair: only stress                          (stress=True, stresses=False, jit=True)
    if n < n_max_jaxmd_pair[True, False, True]:
        run_and_initialize_expect_oom(JmdLennardJonesPair.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=False, adjust_radii=True, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD Pair, skipping.".format(n, n_max_jaxmd_pair[True, False, True]))


    # JAX-MD Pair: only stresses                        (stress=False, stresses=True, jit=True)
    if n < n_max_jaxmd_pair[False, True, True]:
        run_and_initialize_expect_oom(JmdLennardJonesPair.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=True, adjust_radii=True, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD Pair, skipping.".format(n, n_max_jaxmd_pair[False, True, True]))


    # JAX-MD Pair: only energies and forces             (stress=False, stresses=False, jit=True)
    if n < n_max_jaxmd_pair[False, False, True]:
        run_and_initialize_expect_oom(JmdLennardJonesPair.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD Pair, skipping.".format(n, n_max_jaxmd_pair[False, False, True]))


    # JAX-MD Pair: only energies and forces, no jit     (stress=False, stresses=False, jit=False)
    if n < n_max_jaxmd_pair[False, False, False]:
        run_and_initialize_expect_oom(JmdLennardJonesPair.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=False)
    else:
        print("n={} exceeding n_max={} for JAX-MD Pair, skipping.".format(n, n_max_jaxmd_pair[False, False, False]))


def run_jaxmd_neighbor_list(atoms: Atoms, results: List[Result]):
    n = len(atoms)

    # JAX-MD Neighbor List: all properties              (stress=True, stresses=True, jit=True)
    if n < n_max_jaxmd_nl[True, True, True]:
        run_and_initialize_expect_oom(JmdLennardJonesNeighborList.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD Neighbor List, skipping.".format(n, n_max_jaxmd_nl[True, True, True]))

    
    # JAX-MD Neighbor List: only stress                 (stress=True, stresses=False, jit=True)
    if n < n_max_jaxmd_nl[True, False, True]:
        run_and_initialize_expect_oom(JmdLennardJonesNeighborList.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=False, adjust_radii=True, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD Neighbor List, skipping.".format(n, n_max_jaxmd_nl[True, False, True]))


    # JAX-MD Neighbor List: only stresses               (stress=False, stresses=True, jit=True)
    if n < n_max_jaxmd_nl[False, True, True]:
        run_and_initialize_expect_oom(JmdLennardJonesNeighborList.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=True, adjust_radii=True, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD Neighbor List, skipping.".format(n, n_max_jaxmd_nl[False, True, True]))

    
    # JAX-MD Neighbor List: only energies and forces    (stress=False, stresses=False, jit=True)
    if n < n_max_jaxmd_nl[False, False, True]:
        run_and_initialize_expect_oom(JmdLennardJonesNeighborList.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD Neighbor List, skipping.".format(n, n_max_jaxmd_nl[False, False, True]))


    # JAX-MD Neighbor List: only energies and forces, no jit    (stress=False, stresses=False, jit=False)
    if n < n_max_jaxmd_nl[False, False, False]:
        run_and_initialize_expect_oom(JmdLennardJonesNeighborList.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=False)
    else:
        print("n={} exceeding n_max={} for JAX-MD Neighbor List, skipping.".format(n, n_max_jaxmd_nl[False, False, False]))


def run_jaxmd_gnn(atoms: Atoms, results: List[Result]):
    n = len(atoms)

    # JAX-MD GNN: all properties                       (stress=True, stresses=True, jit=True
    if n < n_max_jaxmd_gnn[True, True, True]:
        run_and_initialize_expect_oom(BapstGNN.from_ase_atoms, results, atoms, r_cutoff, stress=True, stresses=True, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD GNN, skipping.".format(n, n_max_jaxmd_gnn[True, True, True]))

    
    # JAX-MD GNN: only stress                           (stress=True, stresses=False, jit=True)
    if n < n_max_jaxmd_gnn[True, False, True]:
        run_and_initialize_expect_oom(BapstGNN.from_ase_atoms, results, atoms, r_cutoff, stress=True, stresses=False, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD GNN, skipping.".format(n, n_max_jaxmd_gnn[True, False, True]))


    # JAX-MD GNN: only stresses                         (stress=False, stresses=True, jit=True)
    if n < n_max_jaxmd_gnn[False, True, True]:
        run_and_initialize_expect_oom(BapstGNN.from_ase_atoms, results, atoms, r_cutoff, stress=False, stresses=True, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD GNN, skipping.".format(n, n_max_jaxmd_gnn[False, True, True]))


    # JAX-MD GNN: only energies and forces              (stress=False, stresses=False, jit=True)
    if n < n_max_jaxmd_gnn[False, False, True]:
        run_and_initialize_expect_oom(BapstGNN.from_ase_atoms, results, atoms, r_cutoff, stress=False, stresses=False, jit=True)
    else:
        print("n={} exceeding n_max={} for JAX-MD GNN, skipping.".format(n, n_max_jaxmd_gnn[False, False, True]))


    # JAX-MD GNN: only energies and forces, no jit              (stress=False, stresses=False, jit=False)
    if n < n_max_jaxmd_gnn[False, False, False]:
        run_and_initialize_expect_oom(BapstGNN.from_ase_atoms, results, atoms, r_cutoff, stress=False, stresses=False, jit=False)
    else:
        print("n={} exceeding n_max={} for JAX-MD GNN, skipping.".format(n, n_max_jaxmd_gnn[False, False, False]))


def run_benchmark_loop(super_cells: List[Atoms]) -> List[Result]:
    results: List[Result] = []

    for atoms in super_cells:
        n = len(atoms)
        print("\nSystem size n = {}\n".format(n))

        # run_ase(atoms, results)
        run_jaxmd_pair(atoms, results)
        # run_jaxmd_neighbor_list(atoms, results)
        # run_jaxmd_gnn(atoms, results)

    return results  
 

sigma = 3.4
epsilon = 10.42
r_cutoff = 10.54
r_onset = 8

# n_max for different calculators and parameters (stress, stresses, jit)
n_max_ase = 15360

n_max_jaxmd_pair: Dict[Tuple[bool, bool, bool], int] = {}
n_max_jaxmd_pair[True, True, True] = 6336           # JAX-MD Pair (stress=True, stresses=True, jit=True)                     went OOM at n=6336
n_max_jaxmd_pair[False, True, True] = 6336          # JAX-MD Pair (stress=False, stresses=True, jit=True)                    went OOM at n=6336
n_max_jaxmd_pair[False, False, False] = 6336        # JAX-MD Pair (stress=False, stresses=False, jit=False)                  went OOM at n=6336
n_max_jaxmd_pair[True, False, True] = 13500         # JAX-MD Pair (stress=True, stresses=False, jit=True)                    went OOM at n=13500
n_max_jaxmd_pair[False, False, True] = 15360        # JAX-MD Pair (stress=False, stresses=False, jit=True)                   went OOM at n=15360

n_max_jaxmd_nl: Dict[Tuple[bool, bool, bool], int] = {}
n_max_jaxmd_nl[True, True, True] = 15360            # JAX-MD Neighbor List (stress=True, stresses=True, jit=True)            went OOM at n=15360
n_max_jaxmd_nl[True, False, True] = 15360           # JAX-MD Neighbor List (stress=True, stresses=False, jit=True)           went OOM at n=15360
n_max_jaxmd_nl[False, True, True] = 15360           # JAX-MD Neighbor List (stress=False, stresses=True, jit=True)           went OOM at n=15360
n_max_jaxmd_nl[False, False, True] = 15360          # JAX-MD Neighbor List (stress=False, stresses=False, jit=True)          went OOM at n=15360
n_max_jaxmd_nl[False, False, False] = 15360         # JAX-MD Neighbor List (stress=False, stresses=False, jit=False)         went OOM at n=15360
 
n_max_jaxmd_gnn: Dict[Tuple[bool, bool, bool], int] = {}
n_max_jaxmd_gnn[True, True, True] = 3600            # GNN Neighbor List (stress=True, stresses=True, jit=True)               went OOM at n=3600
n_max_jaxmd_gnn[False, True, True] = 3600           # GNN Neighbor List (stress=False, stresses=True, jit=True)              went OOM at n=3600
n_max_jaxmd_gnn[True, False, True] = 4000           # GNN Neighbor List (stress=True, stresses=False, jit=True)              went OOM at n=4000
n_max_jaxmd_gnn[False, False, False] = 4000         # GNN Neighbor List (stress=False, stresses=False, jit=False)            went OOM at n=4000
n_max_jaxmd_gnn[False, False, True] = 6336          # GNN Neighbor List (stress=False, stresses=False, jit=True)             went OOM at n=6336

super_cells = load_super_cells_from_pickle("/home/pop518504/git/gknet-benchmarks/make_supercells/supercells_100_15360_100.pickle")
runs = 100
oom_events: List[Tuple[Calculator, str]] = []

# print("Benchmarking system sizes: {}".format(system_sizes))
print("Performing {} run(s) per framework and system size".format(runs))
print("Memory allocation mode: {}".format(jax_utils.get_memory_allocation_mode()))

results = run_benchmark_loop(super_cells)
persist_results(results, runs)
persist_oom_events(oom_events)

