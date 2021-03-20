import sys
if not '/home/pop518504/git/gknet-benchmarks' in sys.path:
    sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from typing import List
from utils import *
import pickle
from calculators.result import Result
from calculators.calculator import Calculator
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList
from calculators.GNN.bapst_gnn import BapstGNN


def run_expect_oom(calculator: Calculator):
    try:
        calculator.calculate()
        # return False
    except RuntimeError:
        oom_calculators.append(calculator)
        # return True
        

def has_caught_oom(calculator: Calculator, stress: bool, stresses: bool, jit: bool) -> bool:
    # return calculator in [type(c) for c in oom_calculators]
    # return calculator.description in [c.description for c in oom_calculators]
    filtered = filter(lambda c: c._stress == stress and c._stresses == stresses and c._jit == jit, oom_calculators)
    return len(list(filtered)) == 1


def run_jaxmd_pair(n: int):
    # all properties: stress=True, stresses=True, jit=True
    if not has_caught_oom(JmdLennardJonesPair, stress=True, stresses=True, jit=True): 
        jmd1 = JmdLennardJonesPair.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=True, stresses=True, jit=True)
        run_expect_oom(jmd1)

    # only stress: stress=True, stresses=False, jit=True
    if not has_caught_oom(JmdLennardJonesPair, stress=True, stresses=False, jit=True):
        jmd2 = JmdLennardJonesPair.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=True, stresses=False, jit=True)    
        run_expect_oom(jmd2)

    # only stresses: stress=False, stresses=True, jit=True
    if not has_caught_oom(JmdLennardJonesPair, stress=False, stresses=True, jit=True):
        jmd3 = JmdLennardJonesPair.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=False, stresses=True, jit=True)    
        run_expect_oom(jmd3)

    # neither stress nor stresses: stress=False, stresses=False, jit=True
    if not has_caught_oom(JmdLennardJonesPair, stress=False, stresses=False, jit=True):
        jmd4 = JmdLennardJonesPair.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=False, stresses=False, jit=True)    
        run_expect_oom(jmd4)

    # neither stress nor stresses, also no jit: stress=False, stresses=False, jit=False
    if not has_caught_oom(JmdLennardJonesPair, stress=False, stresses=False, jit=False):
        jmd_nojit = JmdLennardJonesPair.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=False, stresses=False, jit=False)    
        run_expect_oom(jmd_nojit)


def run_jaxmd_neighbor_list(n: int):
    # all properties: stress=True, stresses=True, jit=True
    if not has_caught_oom(JmdLennardJonesNeighborList, stress=True, stresses=True, jit=True):
        jmd_nl1 = JmdLennardJonesNeighborList.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=True, stresses=True, jit=True)    
        run_expect_oom(jmd_nl1)

    # only stress: stress=True, stresses=False, jit=True
    if not has_caught_oom(JmdLennardJonesNeighborList, stress=True, stresses=False, jit=True):
        jmd_nl2 = JmdLennardJonesNeighborList.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=True, stresses=False, jit=True)    
        run_expect_oom(jmd_nl2)

    # only stresses: stress=False, stresses=True, jit=True
    if not has_caught_oom(JmdLennardJonesNeighborList, stress=False, stresses=True, jit=True):
        jmd_nl3 = JmdLennardJonesNeighborList.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=False, stresses=True, jit=True)    
        run_expect_oom(jmd_nl3)

    # neither stress nor stresses: stress=False, stresses=False, jit=True
    if not has_caught_oom(JmdLennardJonesNeighborList, stress=False, stresses=False, jit=True):
        jmd_nl4 = JmdLennardJonesNeighborList.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=False, stresses=False, jit=True)    
        run_expect_oom(jmd_nl4)

    # neither stress nor stresses, also no jit: stress=False, stresses=False, jit=False
    if not has_caught_oom(JmdLennardJonesNeighborList, stress=False, stresses=False, jit=False):
        jmd_nl5 = JmdLennardJonesNeighborList.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=False, stresses=False, jit=False)    
        run_expect_oom(jmd_nl5)


def run_gnn_neighbor_list(n: int):
    # all properties: stress=True, stresses=True, jit=True
    if not has_caught_oom(BapstGNN, stress=True, stresses=True, jit=True):
        gnn1 = BapstGNN.create_potential(box_size, n, R_scaled=None, r_cutoff=r_cutoff, r_onset=r_onset, stress=True, stresses=True, jit=True)
        run_expect_oom(gnn1)

    # only stress: stress=True, stresses=False, jit=True
    if not has_caught_oom(BapstGNN, stress=True, stresses=False, jit=True):
        gnn2 = BapstGNN.create_potential(box_size, n, R_scaled=None, r_cutoff=r_cutoff, r_onset=r_onset, stress=True, stresses=False, jit=True)
        run_expect_oom(gnn2)

    # only stresses: stress=False, stresses=True, jit=True
    if not has_caught_oom(BapstGNN, stress=False, stresses=True, jit=True):
        gnn3 = BapstGNN.create_potential(box_size, n, R_scaled=None, r_cutoff=r_cutoff, r_onset=r_onset, stress=False, stresses=True, jit=True)
        run_expect_oom(gnn3)

    # neither stress, nor stresses: stress=False, stresses=False, jit=True
    if not has_caught_oom(BapstGNN, stress=False, stresses=False, jit=True):
        gnn4 = BapstGNN.create_potential(box_size, n, R_scaled=None, r_cutoff=r_cutoff, r_onset=r_onset, stress=False, stresses=False, jit=True)
        run_expect_oom(gnn4)

    # neither stress, nor stresses, also no jit: stress=False, stresses=False, jit=False
    if not has_caught_oom(BapstGNN, stress=False, stresses=False, jit=False):
        gnn5 = BapstGNN.create_potential(box_size, n, R_scaled=None, r_cutoff=r_cutoff, r_onset=r_onset, stress=False, stresses=False, jit=False)
        run_expect_oom(gnn5)


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

system_sizes = generate_system_sizes(start=5700, stop=20000, step=5000)
oom_calculators: List[Calculator] = []
run_until_oom(system_sizes)

for calc in oom_calculators:
    print(calc, calc.n)

output_path = "oom_analysis.pickle"
with open(output_path, 'wb') as handle:
    pickle.dump(oom_calculators, handle)


# persist_results(results, runs)
# print("Runtime error caught")
# jax.profiler.save_device_memory_profile("memory.prof")