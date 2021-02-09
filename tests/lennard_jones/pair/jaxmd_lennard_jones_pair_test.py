from tests.unit_test import UnitTest
from unittest.case import skip
import unittest 
from tests.base_test import BaseTest
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
import numpy as np
import jax.numpy as jnp
from jax import random
from ... import test_utils


class JmdLennardJonesPairTest(unittest.TestCase):
    _n = 500
    _box_size = 100
    _sigma = 2.0
    _epsilon = 1.5
    _r_cutoff = 11.0
    _r_onset = 6.0

    def __init__(self, methodName: str) -> None:
        ''' Initialize two calculators with the same parameters and equal (randomly generated) atom positions. '''
        super().__init__(methodName)
        key = random.PRNGKey(0)
        R = random.uniform(key, shape=(self._n, 3))        
        self._calculator = JmdLennardJonesPair.create_potential(self._box_size, self._n, R, self._sigma, self._epsilon, self._r_cutoff, self._r_onset, stress=False, displacement_fn=None)
        self._stress_calculator = JmdLennardJonesPair.create_potential(self._box_size, self._n, R, self._sigma, self._epsilon, self._r_cutoff, self._r_onset, stress=True, displacement_fn=None)


    def setUp(self):
        self._result = self._calculator.calculate()
        self._stress_result = self._stress_calculator.calculate()

    # TODO: All of these as chex variants w/ and w/o JIT!


    def test_atom_positions_shape_and_equality(self):
        self.assertEqual(self._calculator.R.shape, (500, 3))
        self.assertEqual(self._stress_calculator.R.shape, (500, 3))
        np.testing.assert_array_equal(self._calculator.R, self._stress_calculator.R)   
        
    
    def test_pairwise_distances_shape_and_equality(self):
        self.assertEqual(self._calculator.pairwise_distances.shape, (500, 500))
        self.assertEqual(self._stress_calculator.pairwise_distances.shape, (500, 500))
        np.testing.assert_array_equal(self._calculator.pairwise_distances, self._stress_calculator.pairwise_distances)
        

    def test_energy_correctness(self):
        # TODO: Verify this?
        expected = 168.32646142635886
        self.assertEqual(self._result.energy, expected)
        self.assertEqual(self._stress_result.energy, expected)


    def test_stress_correctness(self):
        # TODO: Verify this?
        expected = jnp.array([[-1.19595482e-05, 3.20514556e-06, -2.44817042e-06],
                              [3.20514556e-06, -1.19787412e-05, 1.64985829e-06],
                              [-2.44817042e-06, 1.64985829e-06, -3.93428730e-07]])
        np.testing.assert_allclose(self._stress_result.stress, expected)
        


    def test_energies_equality(self):
        self.assertEqual(self._result.energies.shape, (500,))
        self.assertEqual(self._stress_result.energies.shape, (500,))
        np.testing.assert_allclose(self._result.energies, self._stress_result.energies)


    def test_forces_equality(self):
        self.assertEqual(self._result.forces.shape, (500, 3))
        self.assertEqual(self._stress_result.forces.shape, (500, 3))
        np.testing.assert_allclose(self._result.forces, self._stress_result.forces)
        
