from utils import *
from ase.atoms import Atoms
from calculators.result import Result
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList
from calculators.GNN.bapst_gnn import BapstGNN
import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_log_compiles", False)


class RuntimeBenchmark:
    sigma = 3.4
    epsilon = 10.42
    r_cutoff = 10.54
    r_onset = 8

    stress_values = [True, True, False, False, False]
    stresses_values = [True, False, True, False, False]
    jit_values = [True, True, True, True, False]

    results: List[Result]
    oom_events: List[Tuple[Callable, Calculator, str]]

    def __init__(self, super_cells: List[Atoms], runs=100):
        self.super_cells = super_cells
        self.runs = runs

    def run(self):
        self.results = []
        self.oom_events = []

        for atoms in self.super_cells:
            print("\nSystem size n = {}\n".format(len(atoms)))

            for stress, stresses, jit in zip(self.stress_values, self.stresses_values, self.jit_values):
                print("stress={}, stresses={}, jit={}".format(stress, stresses, jit))
                self._benchmark_lj_pair(atoms, stress, stresses, jit)
                self._benchmark_lj_nl(atoms, stress, stresses, jit)
                self._benchmark_gnn_nl(atoms, stress, stresses, jit)

            break

    def _benchmark_lj_pair(self, atoms: Atoms, stress: bool, stresses: bool, jit: bool):
        args = atoms, self.sigma, self.epsilon, self.r_cutoff, self.r_onset
        kwargs = {'stress': stress, 'stresses': stresses, 'adjust_radii': True, 'jit': jit}
        self._run_oom_aware("JAX-MD Lennard-Jones Pair", JmdLennardJonesPair.from_ase_atoms, *args, **kwargs)

    def _benchmark_lj_nl(self, atoms: Atoms, stress: bool, stresses: bool, jit: bool):
        args = atoms, self.sigma, self.epsilon, self.r_cutoff, self.r_onset
        kwargs = {'stress': stress, 'stresses': stresses, 'adjust_radii': True, 'jit': jit}
        self._run_oom_aware("JAX-MD Lennard-Jones NL", JmdLennardJonesNeighborList.from_ase_atoms, *args, **kwargs)

    def _benchmark_gnn_nl(self, atoms: Atoms, stress: bool, stresses: bool, jit: bool):
        args = atoms, self.r_cutoff
        kwargs = {'stress': stress, 'stresses': stresses, 'jit': jit}
        self._run_oom_aware("JAX-MD GNN", BapstGNN.from_ase_atoms, *args, **kwargs)

    def _run_oom_aware(self, descriptor: str, create_calculator: Callable[..., Calculator], *args, **kwargs):
        print("\nRunning {} ({})".format(descriptor, kwargs))

        if self._has_caught_oom(create_calculator, **kwargs):
            print("{} ({}) has gone OOM before, skipping.".format(descriptor, kwargs))
            return

        calculator = self._initialize_calculator(descriptor, create_calculator, *args, **kwargs)
        self._warmup_calculator(descriptor, create_calculator, calculator, *args, **kwargs)
        self._perform_runs(descriptor, create_calculator, calculator, *args, **kwargs)

    def _initialize_calculator(self, descriptor: str, create_calculator: Callable[..., Calculator], *args, **kwargs):
        try:
            print("Phase 1 (init):\t {} ({}), n={}".format(descriptor, kwargs, len(args[0])))
            print()
            calculator = create_calculator(*args, **kwargs)
            return calculator

        except RuntimeError:
            print("{} ({}) went OOM during calculator initialization".format(descriptor, kwargs))
            self._save_oom_event("Initialization", create_calculator, None, *args, **kwargs)
            return

    def _warmup_calculator(self, descriptor: str, create_calculator: Callable[..., Calculator], calculator: Calculator, *args, **kwargs):
        if calculator is None:
            return

        try:
            print("Phase 2 (warmup):\t {} ({}), n={}".format(descriptor, kwargs, calculator.n))
            print()
            calculator.warm_up()
        except NotImplementedError:
            print("warmup not implemented for {} ({}) - continuing".format(descriptor, kwargs, calculator.n))
            pass  # fine for some calculators.
        except RuntimeError:  # oom during warm-up
            print("{} ({}) went OOM during warm-up (n={})".format(descriptor, kwargs, calculator.n))
            self._save_oom_event("Warm-up", create_calculator, calculator, *args, **kwargs)
            return

    def _perform_runs(self, descriptor: str, create_calculator: Callable[..., Calculator], calculator: Calculator, *args, **kwargs):
        if calculator is None:
            return

        try:
            print("Phase 3 (computing):\t {} ({}), n={}".format(descriptor, kwargs, calculator.n))
            print()
            rs = calculator.calculate(self.runs)
            self.results.extend(rs)  # only save results when all runs were successfully performed

        except RuntimeError:
            print("{} ({}) went OOM during property computation (n={})".format(descriptor, kwargs, calculator.n))
            self._save_oom_event("Computation", create_calculator, calculator, *args, **kwargs)
            return

        # TODO: What about this?

        # if calculator._oom_runs > 0:
        #     save_oom_event("Skipped run", None, calculator, *args, **kwargs)
        #     return

        # if calculator._oom_runs == 100:
        #     save_oom_event("Skipped all runs", None, calculator, *args, **kwargs)
        #     return

        # only save results when all runs were successfully performed
        # results.extend(rs)

    def _has_caught_oom(self, create_calculator: Callable[..., Calculator], **kwargs) -> bool:
        # we don't care about skipped runs here, these are tolerated!
        # init_and_warmup_events = filter(lambda ev: ev[2] != "Skipped run", oom_events)

        filtered = filter(
            lambda ev: ev[0] == create_calculator and ev[1]._stress == kwargs['stress'] and ev[1]._stresses == kwargs[
                'stresses'] and ev[1]._jit == kwargs['jit'], self.oom_events)
        return len(list(filtered)) >= 1

    def _save_oom_event(self, reason: str, create_calculator: Callable[..., Calculator], calc: Calculator, *args, **kwargs):
        # if OOM during init, create a dummy object for reference
        if calc is None:
            calc = create_calculator(*args, **kwargs, skip_initialization=True)

        event = create_calculator, calc, reason
        self.oom_events.append(event)


super_cells = load_super_cells_from_pickle(
    "/home/pop518504/git/gknet-benchmarks/make_supercells/supercells_108_23328.pickle")

benchmark = RuntimeBenchmark(super_cells, runs=100)
benchmark.run()

persist_results(benchmark.results, runs=100, descriptor="float32_benchmarks")
