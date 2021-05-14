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


def run_expect_oom(calculator_class, ase: AseLennardJonesPair, stress: bool, stresses: bool, jit: bool):
    if has_caught_oom(calculator_class, stress, stresses, jit):
        print("{} already went OOM before. Skipping.".format(calculator_class))
        return

    calc: Type[Calculator] = None
    try:
        if calculator_class == BapstGNN:
            calc = calculator_class.from_ase_atoms(ase._atoms, r_cutoff=r_cutoff, stress=stress, stresses=stresses, jit=jit)
            # calc = calculator_class.create_potential(box_size, n, R_scaled=None, r_cutoff=r_cutoff, r_onset=r_onset, stress=stress, stresses=stresses, jit=jit)
        else:
            calc = calculator_class.from_ase_atoms(ase._atoms, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=stress, stresses=stresses, adjust_radii=True, jit=jit)
            # calc = calculator_class.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=stress, stresses=stresses, jit=jit)
        calc.calculate()
    except RuntimeError:
        if calc is None:
            # in neighbor lists and GNNs, OOM might have occurred during initialization and not the actual computation. Thus, we need to create a data-only instance which we can reference here.
            if calculator_class == BapstGNN:
                calc = calculator_class.from_ase_atoms(ase._atoms, r_cutoff=r_cutoff, stress=stress, stresses=stresses, jit=jit, skip_initialization=True)
                # calc = calculator_class.create_potential(box_size, n, R_scaled=None, r_cutoff=r_cutoff, r_onset=r_onset, stress=stress, stresses=stresses, jit=jit, skip_initialization=True)
            else:
                calc = calculator_class.from_ase_atoms(ase._atoms, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=stress, stresses=stresses, adjust_radii=True, jit=jit, skip_initialization=True)
                # calc = calculator_class.create_potential(box_size, n, R_scaled=None, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, stress=stress, stresses=stresses, jit=jit, skip_initialization=True)
        oom_calculators.append(calc)


def has_caught_oom(calculator: Calculator, stress: bool, stresses: bool, jit: bool) -> bool:
    filtered = filter(lambda c: type(c) == calculator and c._stress == stress and c._stresses == stresses and c._jit == jit, oom_calculators)
    return len(list(filtered)) == 1


def run_jaxmd_pair(ase: AseLennardJonesPair):
    # all properties: stress=True, stresses=True, jit=True
    run_expect_oom(JmdLennardJonesPair, ase, stress=True, stresses=True, jit=True)
    # only stress: stress=True, stresses=False, jit=True
    run_expect_oom(JmdLennardJonesPair, ase, stress=True, stresses=False, jit=True)
    # only stresses: stress=False, stresses=True, jit=True
    run_expect_oom(JmdLennardJonesPair, ase, stress=False, stresses=True, jit=True)
    # neither stress nor stresses: stress=False, stresses=False, jit=True
    run_expect_oom(JmdLennardJonesPair, ase, stress=False, stresses=False, jit=True)
    # neither stress nor stresses, also no jit: stress=False, stresses=False, jit=False
    run_expect_oom(JmdLennardJonesPair, ase, stress=False, stresses=False, jit=False)


def run_jaxmd_neighbor_list(ase: AseLennardJonesPair):
    # all properties: stress=True, stresses=True, jit=True
    run_expect_oom(JmdLennardJonesNeighborList, ase, stress=True, stresses=True, jit=True)
    # only stress: stress=True, stresses=False, jit=True
    run_expect_oom(JmdLennardJonesNeighborList, ase, stress=True, stresses=False, jit=True)
    # only stresses: stress=False, stresses=True, jit=True
    run_expect_oom(JmdLennardJonesNeighborList, ase, stress=False, stresses=True, jit=True)
    # neither stress nor stresses: stress=False, stresses=False, jit=True
    run_expect_oom(JmdLennardJonesNeighborList, ase, stress=False, stresses=False, jit=True)
    # neither stress nor stresses, also no jit: stress=False, stresses=False, jit=False
    run_expect_oom(JmdLennardJonesNeighborList, ase, stress=False, stresses=False, jit=False)


def run_gnn_neighbor_list(ase: AseLennardJonesPair):
    # all properties: stress=True, stresses=True, jit=True
    run_expect_oom(BapstGNN, ase, stress=True, stresses=True, jit=True)
    # only stress: stress=True, stresses=False, jit=True
    run_expect_oom(BapstGNN, ase, stress=True, stresses=False, jit=True)
    # only stresses: stress=False, stresses=True, jit=True
    run_expect_oom(BapstGNN, ase, stress=False, stresses=True, jit=True)
    # neither stress, nor stresses: stress=False, stresses=False, jit=True
    run_expect_oom(BapstGNN, ase, stress=False, stresses=False, jit=True)
    # neither stress, nor stresses, also no jit: stress=False, stresses=False, jit=False
    run_expect_oom(BapstGNN, ase, stress=False, stresses=False, jit=False)


def initialize_with_ase(n: int, computed_system_sizes: List[int]):
    print("Initializing cubic system with n≈{}".format(n))
    ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=None, r_onset=None)
    
    if ase.n in computed_system_sizes:
        print("n={} already computed, skipping.".format(ase.n))
        return None
    if ase.n > n_max:
        print("n={} exceeding n_max={}, aborting.".format(ase.n, n_max))
        return None

    return ase
    

def run_until_oom() -> List[Result]:
    for n in requested_system_sizes:
        ase = initialize_with_ase(n, computed_system_sizes)
        if ase is None: continue
        computed_system_sizes.append(ase.n)
      
        print("System size n = {}\n".format(ase.n))
        run_jaxmd_pair(ase)
        run_jaxmd_neighbor_list(ase)
        run_gnn_neighbor_list(ase)

# new LJ parameters
sigma = 3.4
epsilon = 10.42
r_cutoff = 10.54
r_onset = 8

n_min = 3000
n_max = 25000
n_step = 100

requested_system_sizes = generate_system_sizes(start=n_min, stop=n_max, step=n_step)
computed_system_sizes = []
oom_calculators: List[Calculator] = []
run_until_oom()
for calc in oom_calculators:
    print(calc, calc.n)

oom_result = computed_system_sizes, oom_calculators

output_path = "oom_analysis.pickle"
with open(output_path, 'wb') as handle:
    pickle.dump(oom_result, handle)
