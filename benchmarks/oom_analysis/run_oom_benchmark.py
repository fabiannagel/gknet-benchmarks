import sys
if not '/home/pop518504/git/gknet-benchmarks' in sys.path:
    sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from typing import List, Type
from utils import *
import pickle
from calculators.result import Result
from calculators.calculator import Calculator
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList
from calculators.GNN.bapst_gnn import BapstGNN


def run_expect_oom_old(calculator: Calculator):
    try:
        calculator.calculate()
    except RuntimeError:
        oom_calculators.append(calculator)


def run_expect_oom(calculator_class, n: int, stress: bool, stresses: bool, jit: bool):
    if has_caught_oom(calculator_class, stress, stresses, jit):
        print("{} already went OOM before. Skipping.".format(calculator_class))
        return

    calc: Type[Calculator] = None
    try:
        if calculator_class == BapstGNN:
            calc = calculator_class.create_potential(box_size, n, R_scaled=None, r_cutoff=r_cutoff, r_onset=r_onset, stress=stress, stresses=stresses, jit=jit)
        else:
            calc = calculator_class.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=stress, stresses=stresses, jit=jit)
        calc.calculate()
    except RuntimeError:
        if calc is None:
            # in neighbor lists and GNNs, OOM might have occurred during initialization and not the actual computation. Thus, we need to create a data-only instance which we can reference here.
            if calculator_class == BapstGNN:
                calc = calculator_class.create_potential(box_size, n, R_scaled=None, r_cutoff=r_cutoff, r_onset=r_onset, stress=stress, stresses=stresses, jit=jit, skip_initialization=True)
            else:
                calc = calculator_class.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=stress, stresses=stresses, jit=jit, skip_initialization=True)
        oom_calculators.append(calc)


def has_caught_oom(calculator: Calculator, stress: bool, stresses: bool, jit: bool) -> bool:
    filtered = filter(lambda c: type(c) == calculator and c._stress == stress and c._stresses == stresses and c._jit == jit, oom_calculators)
    return len(list(filtered)) == 1


def run_jaxmd_pair(n: int):
    # all properties: stress=True, stresses=True, jit=True
    run_expect_oom(JmdLennardJonesPair, n, stress=True, stresses=True, jit=True)
    # only stress: stress=True, stresses=False, jit=True
    run_expect_oom(JmdLennardJonesPair, n, stress=True, stresses=False, jit=True)
    # only stresses: stress=False, stresses=True, jit=True
    run_expect_oom(JmdLennardJonesPair, n, stress=False, stresses=True, jit=True)
    # neither stress nor stresses: stress=False, stresses=False, jit=True
    run_expect_oom(JmdLennardJonesPair, n, stress=False, stresses=False, jit=True)
    # neither stress nor stresses, also no jit: stress=False, stresses=False, jit=False
    run_expect_oom(JmdLennardJonesPair, n, stress=False, stresses=False, jit=False)


def run_jaxmd_neighbor_list(n: int):
    # all properties: stress=True, stresses=True, jit=True
    run_expect_oom(JmdLennardJonesNeighborList, n, stress=True, stresses=True, jit=True)
    # only stress: stress=True, stresses=False, jit=True
    run_expect_oom(JmdLennardJonesNeighborList, n, stress=True, stresses=False, jit=True)
    # only stresses: stress=False, stresses=True, jit=True
    run_expect_oom(JmdLennardJonesNeighborList, n, stress=False, stresses=True, jit=True)
    # neither stress nor stresses: stress=False, stresses=False, jit=True
    run_expect_oom(JmdLennardJonesNeighborList, n, stress=False, stresses=False, jit=True)
    # neither stress nor stresses, also no jit: stress=False, stresses=False, jit=False
    run_expect_oom(JmdLennardJonesNeighborList, n, stress=False, stresses=False, jit=False)


def run_gnn_neighbor_list(n: int):
    # all properties: stress=True, stresses=True, jit=True
    run_expect_oom(BapstGNN, n, stress=True, stresses=True, jit=True)
    # only stress: stress=True, stresses=False, jit=True
    run_expect_oom(BapstGNN, n, stress=True, stresses=False, jit=True)
    # only stresses: stress=False, stresses=True, jit=True
    run_expect_oom(BapstGNN, n, stress=False, stresses=True, jit=True)
    # neither stress, nor stresses: stress=False, stresses=False, jit=True
    run_expect_oom(BapstGNN, n, stress=False, stresses=False, jit=True)
    # neither stress, nor stresses, also no jit: stress=False, stresses=False, jit=False
    run_expect_oom(BapstGNN, n, stress=False, stresses=False, jit=False)


def run_until_oom(system_sizes: List[int]) -> List[Result]:
    for n in system_sizes:
        print("System size n = {}\n".format(n))
        run_jaxmd_pair(n)
        run_jaxmd_neighbor_list(n)
        run_gnn_neighbor_list(n)


box_size = 100
sigma = 2.0
epsilon = 1.5
r_cutoff = 2
r_onset = 1.5

system_sizes = generate_system_sizes(start=4000, stop=20000, step=6000)
oom_calculators: List[Calculator] = []
run_until_oom(system_sizes)

for calc in oom_calculators:
    print(calc, calc.n)

output_path = "oom_analysis.pickle"
with open(output_path, 'wb') as handle:
    pickle.dump(oom_calculators, handle)
