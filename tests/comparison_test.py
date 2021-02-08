from abc import abstractmethod
from typing import List
from .base_test import BaseTest
from unittest.case import skip
from calculators.calculator import Calculator, Result
import numpy as np
import jax.numpy as jnp
import chex
from .test_utils import *


class ComparisonTest(BaseTest):
    _calculators: List[Calculator]
    _results: List[Result]

    def __init__(self, methodName: str, calculators: List[Calculator]) -> None:
        self._calculators = calculators
        super().__init__(methodName, self._get_stress_implementations())


    def _get_stress_implementations(self):
        ''' Asserts that all passed Calculator instances are in agreement about performing or omitting stress computations. '''
        stress = [c._computes_stress for c in self._calculators]
        if len(set(stress)) > 1:
            raise ValueError("Incosistent state: No agreement among passed calculators for stress computation")
        return all(stress)


    def test_energy_equality(self):
        # TODO: Max error w/ stress is output modulo numerical noise or something.
        total_energies = [r.energy for r in self._results]
        assert_arrays_all_close(total_energies)

    
    def test_system_size_equality(self):
        ''' Assert that all calculators model systems with equal atom count. 
        Background: ASE models physically accurate unit cell sizes while JAX-MD does not care. 
        As a result, ASE might fall back to a realistic atom count when the passed n is not physically correct. '''
        system_sizes = [c.n for c in self._calculators]
        assert_arrays_all_equal(system_sizes)


    def test_atom_position_equality(self):
        atom_positions = [c.R for c in self._calculators]
        assert_arrays_all_equal(atom_positions)


    def test_pairwise_distances_equality(self):
        pairwise_distances = [c.pairwise_distances for c in self._calculators]
        assert_arrays_all_close(pairwise_distances)

    @skip
    def test_pairwise_distance_jit_speedup(self):
        # TODO: Move to unit test base class

        compute_dR_fn = lambda: self._jmd.pairwise_distances
        runtimes = self.get_runtimes(compute_dR_fn)
        non_jitted_runtime = runtimes[0]        
        jitted_runtimes = runtimes[1:]

        # Is the first function call significantly slower than the rest?
        # first_call_slowdown = non_jitted_runtime / np.mean(jitted_runtimes)
        # self.assertGreaterEqual(first_call_slowdown, 500)                                # is the first call at least 500x slower than the rest?
        self.assert_mean_jit_speedup(non_jitted_runtime, jitted_runtimes)

        # Disregarding the first call: Does the jitted function's runtime stay consistent?
        max_relative_deviation = np.min(jitted_runtimes) / np.max(jitted_runtimes)      # maximum relative deviation between the fastest and slowest run
        self.assertLessEqual(max_relative_deviation, 0.5)                               # is the fastest run at most 50% quicker than the slowest? 
    
    @skip
    @chex.variants(with_jit=True, without_jit=True)
    def test_jit_result_correctness(self):       
        # TODO: Move to unit test base class

        # TODO: properties_fn is by default already jitted. is chex still able to test an un-jitted version here?

        jmd_property_fn = self._jax_md._properties_fn
        properties = jmd_property_fn(*self._property_args)

        jmd_var_property_fn = self.variant(jmd_property_fn)
        var_properties = jmd_var_property_fn(*self._property_args)

        self.assertEqual(properties, var_properties)
        
        # TODO: Same for asax


    


       