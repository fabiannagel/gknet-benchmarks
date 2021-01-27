from tests.base_test import BaseTest
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
import numpy as np


class ComparisonWithoutStressTest(BaseTest):

    def __init__(self, methodName: str) -> None:
        super().__init__(methodName)
        
        # 4, 32, 500
        self._n = 500
        self._sigma = 2.0
        self._epsilon = 1.5
        self._r_cutoff = 11.0
        self._r_onset = 6.0
        
        self._ase = AseLennardJonesPair.create_equilibrium_potential(self._n, self._sigma, self._epsilon, self._r_cutoff, self._r_onset)
        
        r_cutoff_adjusted = self._r_cutoff / self._sigma
        r_onset_adjusted = self._r_onset   / self._sigma
        self._jmd = JmdLennardJonesPair.from_ase_atoms(self._ase._atoms, self._sigma, self._epsilon, r_cutoff_adjusted, r_onset_adjusted, stress=False)

    def setUp(self) -> None:
        self._ase_result = self._ase.calculate()
        self._jmd_result = self._jmd.calculate()

    def test_energy_equivalence(self):
        # TODO: Probably due to the neighbor list LJ implementation in ASE
        # print(self._ase_result.energy, self._jmd_result.energy)
        self.assert_all_close(self._ase_result.energy, self._jmd_result.energy)
    
    def test_system_sizes_equivalent(self):
        """ASE minds physical unit cell sizes while JAX-MD does not care. As a result, AseLennardJonesPair() falls back to correct system size when the passed n is not physically realistic."""
        self.assertEqual(self._ase._n, self._jmd._n)

    def test_atom_position_equality(self):
        arrs = [self._ase._R, self._ase._atoms.get_positions(), self._jmd._R]
        self.assert_arrays_all_equal(arrs)

    def test_pairwise_distances_equality(self):
        dR_ase = self._ase.pairwise_distances
        dR_jmd = self._jmd.pairwise_distances
        self.assert_all_close(dR_ase, dR_jmd)
        
    def test_pairwise_distance_jit_speedup(self):
        compute_dR_fn = lambda: self._jmd.pairwise_distances
        runtimes = self.get_runtimes(compute_dR_fn, 10)
        non_jitted_runtime = runtimes[0]        
        jitted_runtimes = runtimes[1:]

        # Is the first function call significantly slower than the rest?
        first_call_slowdown = non_jitted_runtime / np.mean(jitted_runtimes)
        self.assertGreaterEqual(first_call_slowdown, 500)                                # is the first call at least 500x slower than the rest?

        # Disregarding the first call: Does the jitted function's runtime stay consistent?
        max_relative_deviation = np.min(jitted_runtimes) / np.max(jitted_runtimes)      # maximum relative deviation between the fastest and slowest run
        self.assertLessEqual(max_relative_deviation, 0.5)                               # is the fastest run at most 50% quicker than the slowest? 
            
