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
    '''
    Refers to the test case in which the results of multiple calculators are assessed.
    This allows for testing any combination of framework and parameter setting that could be of interest.
    ''' 

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
       