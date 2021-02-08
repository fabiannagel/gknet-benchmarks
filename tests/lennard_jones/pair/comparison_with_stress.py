import unittest
from tests.comparison_test import ComparisonTest
from unittest.case import skip
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from tests.base_test import BaseTest
import numpy as np
    
class ComparisonWithStress(ComparisonTest, unittest.TestCase):
    _n = 500
    _sigma = 2.0
    _epsilon = 1.5
    _r_cutoff = 11.0
    _r_onset = 6.0

    def __init__(self, methodName: str) -> None:
        self._ase = self._create_ase_calculator()
        self._jmd = self._create_jaxmd_calculator()
        super().__init__(methodName, [self._ase, self._jmd])

    def _create_ase_calculator(self):
        return AseLennardJonesPair.create_equilibrium_potential(self._n, self._sigma, self._epsilon, self._r_cutoff, self._r_onset)


    def _create_jaxmd_calculator(self):
        r_cutoff_adjusted = self._r_cutoff / self._sigma
        r_onset_adjusted = self._r_onset   / self._sigma
        return JmdLennardJonesPair.from_ase_atoms(self._ase._atoms, self._sigma, self._epsilon, r_cutoff_adjusted, r_onset_adjusted, stress=True)


    def setUp(self):
        self._results = [self._ase.calculate(), self._jmd.calculate()]


    @skip
    def test_property_computation_jit_speedup(self):
        """
        Here, we want to test whether the compiler can successfully jit the logic required to compute stresses with JAX-MD.
        There should be a significant speedup after the first call and tracing prints should be removed.
        The latter is probably annoying to use for testing. What is a better way?
        """

        compute_properties = lambda: self._jmd.calculate()
        initial_runtime, jitted_runtimes = self.get_runtimes(compute_properties)

        print(initial_runtime)
        print(jitted_runtimes)
        mean_speedup = self.assert_mean_jit_speedup(initial_runtime, jitted_runtimes, min_speedup_factor=2)
        print("test_property_computation_jit_speedup:\t Average JIT speedup by factor {}", mean_speedup)

        # TODO: How to make these kind of tests more robust?
        # - JIT speedup varies by computation complexity. 500x for pairwise distancesn, ~1.7 for all property computations
        # - Is there a good way to detect whether calls are jitted?
    
    
    
