import sys
if not '/home/pop518504/git/gknet-benchmarks' in sys.path:
    sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from typing import Dict, List, Tuple, Type
from utils import *
from ase.atoms import Atoms
import jax_utils
from calculators.result import Result
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList
from calculators.GNN.bapst_gnn import BapstGNN


def has_caught_oom(callable: Callable, **kwargs) -> bool:
    # we don't care about skipped runs here, these are tolerated!
    # init_and_warmup_events = filter(lambda ev: ev[2] != "Skipped run", oom_events)
    filtered = filter(lambda ev: ev[0] == callable and ev[1]._stress == kwargs['stress'] and ev[1]._stresses == kwargs['stresses'] and ev[1]._jit == kwargs['jit'], oom_events)
    return len(list(filtered)) >= 1


def save_oom_event(reason: str, callable: Callable, calc: Calculator, *args, **kwargs):
    if calc is None:
        calc = callable(*args, **kwargs, skip_initialization=True)  # if OOM during init, create a fake calc object for reference
    
    event = callable, calc, reason
    oom_events.append(event)


def run_and_initialize_expect_oom(callable: Callable, results: List[Result], *args, **kwargs):
    print("\nRunning {} ({})".format(callable, kwargs))

    if has_caught_oom(callable, **kwargs):
        print("{} ({}) has gone OOM before, skipping.".format(callable, kwargs))
        return

    calculator: Calculator = None

    try:
        print("Phase 1 (init):\t {} ({}), n={}".format(callable, kwargs, len(args[0])))
        print()
        calculator = callable(*args, **kwargs)

    except RuntimeError:
        print("{} ({}) went OOM during calculator initialization".format(callable, kwargs))
        # print("{} went OOM during init (n={})".format(callable, kwargs, len(args[0])))
        save_oom_event("Initialization", callable, None, *args, **kwargs)
        return

    try:
        print("Phase 2 (warmup):\t {} ({}), n={}".format(callable, kwargs, calculator.n))
        print()
        calculator.warm_up()
    except NotImplementedError:
        print("warmup not implemented for {} ({}) - continuing".format(callable, kwargs, calculator.n))
        pass                    # fine for some calculators.
    except RuntimeError:        # oom during warm-up
        # print("{} went oom during warmup at n={}".format(calculator, calculator.n))
        print("{} ({}) went OOM during warm-up (n={})".format(callable, kwargs, calculator.n))
        save_oom_event("Warm-up", callable, calculator, *args, **kwargs)
        return    
     
    try:
        print("Phase 3 (computing):\t {} ({}), n={}".format(callable, kwargs, calculator.n))
        print()
        rs = calculator.calculate(runs)
        results.extend(rs)     # only save results when all runs were successfully performed

    except RuntimeError:
        print("{} ({}) went OOM during property computation (n={})".format(callable, kwargs, calculator.n))
        save_oom_event("Computation", callable, calculator, *args, **kwargs)
        return

    # if calculator._oom_runs > 0:
    #     save_oom_event("Skipped run", None, calculator, *args, **kwargs)
    #     return

    # if calculator._oom_runs == 100:
    #     save_oom_event("Skipped all runs", None, calculator, *args, **kwargs)
    #     return

    # only save results when all runs were successfully performed
    # results.extend(rs)


def run_ase(atoms: Atoms, results: List[Result]):
    ase = AseLennardJonesPair.from_ase_atoms(atoms, sigma, epsilon, r_cutoff=r_cutoff, r_onset=r_onset)
    ase.warm_up()
    results.extend(ase.calculate(runs))


def run_jaxmd_pair(atoms: Atoms, results: List[Result]):
    # JAX-MD Pair: all properties                       (stress=True, stresses=True, jit=True)
    run_and_initialize_expect_oom(JmdLennardJonesPair.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)
    # JAX-MD Pair: only stress                          (stress=True, stresses=False, jit=True)
    run_and_initialize_expect_oom(JmdLennardJonesPair.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=False, adjust_radii=True, jit=True)
    # JAX-MD Pair: only stresses                        (stress=False, stresses=True, jit=True)
    run_and_initialize_expect_oom(JmdLennardJonesPair.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=True, adjust_radii=True, jit=True)
    # JAX-MD Pair: only energies and forces             (stress=False, stresses=False, jit=True)
    run_and_initialize_expect_oom(JmdLennardJonesPair.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=True)
    # JAX-MD Pair: only energies and forces, no jit     (stress=False, stresses=False, jit=False)
    run_and_initialize_expect_oom(JmdLennardJonesPair.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=False)


def run_jaxmd_neighbor_list(atoms: Atoms, results: List[Result]):
    # JAX-MD Neighbor List: all properties              (stress=True, stresses=True, jit=True)
    run_and_initialize_expect_oom(JmdLennardJonesNeighborList.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)
    # JAX-MD Neighbor List: only stress                 (stress=True, stresses=False, jit=True)
    run_and_initialize_expect_oom(JmdLennardJonesNeighborList.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=True, stresses=False, adjust_radii=True, jit=True)
    # JAX-MD Neighbor List: only stresses               (stress=False, stresses=True, jit=True)
    run_and_initialize_expect_oom(JmdLennardJonesNeighborList.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=True, adjust_radii=True, jit=True)
    # JAX-MD Neighbor List: only energies and forces    (stress=False, stresses=False, jit=True)
    run_and_initialize_expect_oom(JmdLennardJonesNeighborList.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=True)
    # JAX-MD Neighbor List: only energies and forces, no jit    (stress=False, stresses=False, jit=False)
    run_and_initialize_expect_oom(JmdLennardJonesNeighborList.from_ase_atoms, results, atoms, sigma, epsilon, r_cutoff, r_onset, stress=False, stresses=False, adjust_radii=True, jit=False)


def run_jaxmd_gnn(atoms: Atoms, results: List[Result]):
    # JAX-MD GNN: all properties                       (stress=True, stresses=True, jit=True
    run_and_initialize_expect_oom(BapstGNN.from_ase_atoms, results, atoms, r_cutoff, stress=True, stresses=True, jit=True)
    # JAX-MD GNN: only stress                           (stress=True, stresses=False, jit=True)
    run_and_initialize_expect_oom(BapstGNN.from_ase_atoms, results, atoms, r_cutoff, stress=True, stresses=False, jit=True)
    # JAX-MD GNN: only stresses                         (stress=False, stresses=True, jit=True)
    run_and_initialize_expect_oom(BapstGNN.from_ase_atoms, results, atoms, r_cutoff, stress=False, stresses=True, jit=True)
    # JAX-MD GNN: only energies and forces              (stress=False, stresses=False, jit=True)
    run_and_initialize_expect_oom(BapstGNN.from_ase_atoms, results, atoms, r_cutoff, stress=False, stresses=False, jit=True)
    # JAX-MD GNN: only energies and forces, no jit              (stress=False, stresses=False, jit=False)
    run_and_initialize_expect_oom(BapstGNN.from_ase_atoms, results, atoms, r_cutoff, stress=False, stresses=False, jit=False)


def run_benchmark_loop(super_cells: List[Atoms]) -> List[Result]:
    results: List[Result] = []

    for atoms in super_cells:
        n = len(atoms)
        print("\nSystem size n = {}\n".format(n))

        # run_ase(atoms, results)
        # run_jaxmd_pair(atoms, results)
        # run_jaxmd_neighbor_list(atoms, results)
        run_jaxmd_gnn(atoms, results)

    return results  
 

sigma = 3.4
epsilon = 10.42
r_cutoff = 10.54
r_onset = 8

super_cells = load_super_cells_from_pickle("/home/pop518504/git/gknet-benchmarks/make_supercells/supercells_108_23328.pickle")

runs = 100
oom_events: List[Tuple[Callable, Calculator, str]] = []

# print("Benchmarking system sizes: {}".format(system_sizes))
print("Performing {} run(s) per framework and system size".format(runs))
print("Memory allocation mode: {}".format(jax_utils.get_memory_allocation_mode()))

results = run_benchmark_loop(super_cells)
persist_results(results, runs)
persist_oom_events(oom_events, "oom_events.pickle")
