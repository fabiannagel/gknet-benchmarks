import sys

if not '/home/pop518504/git/gknet-benchmarks' in sys.path:
    sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from typing import Dict, List, Tuple, Type, Sequence
from utils import *
from ase.atoms import Atoms
import jax_utils
from calculators.result import Result
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList
from calculators.GNN.bapst_gnn import BapstGNN


def get_descriptor_from_callable(create_calculator: Callable[..., Calculator]) -> str:
    return str(create_calculator).split("<class '")[1].replace("'>>", "").split(".")[-1]


def has_caught_oom(create_calculator: Callable[..., Calculator], **kwargs) -> bool:
    # we don't care about skipped runs here, these are tolerated!
    # init_and_warmup_events = filter(lambda ev: ev[2] != "Skipped run", oom_events)

    filtered = filter(lambda ev: ev[0] == create_calculator and ev[1]._stress == kwargs['stress'] and ev[1]._stresses == kwargs['stresses'] and ev[1]._jit == kwargs['jit'], oom_events)
    return len(list(filtered)) >= 1


def save_oom_event(reason: str, create_calculator: Callable[..., Calculator], calc: Calculator, *args, **kwargs):
    # TODO: create_calculator OR calc in one parameter

    if calc is None:
        calc = create_calculator(*args, **kwargs,
                                 skip_initialization=True)  # if OOM during init, create a fake calc object for reference

    event = create_calculator, calc, reason
    oom_events.append(event)


def initialize_calculator(create_calculator: Callable[..., Calculator], *args, **kwargs):
    descriptor = get_descriptor_from_callable(create_calculator)

    try:
        print("Phase 1 (init):\t {} ({}), n={}".format(descriptor, kwargs, len(args[0])))
        print()
        calculator = create_calculator(*args, **kwargs)
        return calculator

    except RuntimeError:
        print("{} ({}) went OOM during calculator initialization".format(descriptor, kwargs))
        save_oom_event("Initialization", create_calculator, None, *args, **kwargs)
        return


def run_calculator_warmup(create_calculator: Callable[..., Calculator], calculator: Calculator, *args, **kwargs):
    if calculator is None:
        return

    descriptor = get_descriptor_from_callable(create_calculator)

    try:
        print("Phase 2 (warmup):\t {} ({}), n={}".format(descriptor, kwargs, calculator.n))
        print()
        calculator.warm_up()
    except NotImplementedError:
        print("warmup not implemented for {} ({}) - continuing".format(descriptor, kwargs, calculator.n))
        pass  # fine for some calculators.
    except RuntimeError:  # oom during warm-up
        print("{} ({}) went OOM during warm-up (n={})".format(descriptor, kwargs, calculator.n))
        save_oom_event("Warm-up", create_calculator, calculator, *args, **kwargs)
        return


def perform_runs(results: List[Result], create_calculator: Callable[..., Calculator], calculator: Calculator, *args, **kwargs):
    if calculator is None:
        return

    descriptor = get_descriptor_from_callable(create_calculator)

    try:
        print("Phase 3 (computing):\t {} ({}), n={}".format(descriptor, kwargs, calculator.n))
        print()
        rs = calculator.calculate(runs)
        results.extend(rs)  # only save results when all runs were successfully performed

    except RuntimeError:
        print("{} ({}) went OOM during property computation (n={})".format(descriptor, kwargs, calculator.n))
        save_oom_event("Computation", create_calculator, calculator, *args, **kwargs)
        return

    # if calculator._oom_runs > 0:
    #     save_oom_event("Skipped run", None, calculator, *args, **kwargs)
    #     return

    # if calculator._oom_runs == 100:
    #     save_oom_event("Skipped all runs", None, calculator, *args, **kwargs)
    #     return

    # only save results when all runs were successfully performed
    # results.extend(rs)


def run_oom_aware(results: List[Result], create_calculator: Callable[..., Calculator], *args, **kwargs):
    descriptor = get_descriptor_from_callable(create_calculator)
    print("\nRunning {} ({})".format(descriptor, kwargs))

    if has_caught_oom(create_calculator, **kwargs):
        print("{} ({}) has gone OOM before, skipping.".format(descriptor, kwargs))
        return

    calculator = initialize_calculator(create_calculator, *args, **kwargs)
    run_calculator_warmup(create_calculator, calculator, *args, **kwargs)
    perform_runs(results, create_calculator, calculator, *args, **kwargs)


def run_jaxmd_neighbor_list(results: List[Result], atoms: Atoms, stress: bool, stresses: bool, jit: bool):
    args = atoms, sigma, epsilon, r_cutoff, r_onset
    kwargs = {'stress': stress, 'stresses': stresses, 'adjust_radii': True, 'jit': jit}
    run_oom_aware(results, JmdLennardJonesNeighborList.from_ase_atoms, *args, **kwargs)


def run_benchmark_loop(super_cells: List[Atoms]) -> List[Result]:
    results: List[Result] = []

    for atoms in super_cells:
        n = len(atoms)
        print("\nSystem size n = {}\n".format(n))

        for stress, stresses, jit in zip(stress_values, stresses_values, jit_values):
            print("stress={}, stresses={}, jit={}".format(stress, stresses, jit))
            run_jaxmd_neighbor_list(results, atoms, stress, stresses, jit)
            break

    return results


stress_values = [True, True, False, False, False]
stresses_values = [True, False, True, False, False]
jit_values = [True, True, True, True, False]

sigma = 3.4
epsilon = 10.42
r_cutoff = 10.54
r_onset = 8

super_cells = load_super_cells_from_pickle(
    "/home/pop518504/git/gknet-benchmarks/make_supercells/supercells_108_23328.pickle")

runs = 100
oom_events: List[Tuple[Callable, Calculator, str]] = []

print("Performing {} run(s) per framework and system size".format(runs))

results = run_benchmark_loop(super_cells)
