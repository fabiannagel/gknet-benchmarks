from unittest.case import skip
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
import numpy as np
import jax.numpy as jnp
from jax import random
import chex
from jax.test_util import check_grads
from jax.config import config
config.update("jax_enable_x64", True)
from ...utils import *


class JmdLennardJonesPairTest(chex.TestCase):
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
        self._calculator = JmdLennardJonesPair.create_potential(self._box_size, self._n, R, self._sigma, self._epsilon, self._r_cutoff, self._r_onset, stress=False, jit=False)
        self._stress_calculator = JmdLennardJonesPair.create_potential(self._box_size, self._n, R, self._sigma, self._epsilon, self._r_cutoff, self._r_onset, stress=True, jit=False)


    def setUp(self):
        self._result = self._calculator.calculate()
        self._stress_result = self._stress_calculator.calculate()


    # TODO: All of these as chex variants w/ and w/o JIT!


    def test_atom_positions_shape_and_equality(self):
        chex.assert_shape(self._calculator.R, (500, 3))
        chex.assert_shape(self._stress_calculator.R, (500, 3))
        np.testing.assert_array_equal(self._calculator.R, self._stress_calculator.R)   
        
    
    def test_pairwise_distances_shape_and_equality(self):
        chex.assert_shape(self._calculator.pairwise_distances, (500, 500))
        chex.assert_shape(self._stress_calculator.pairwise_distances, (500, 500))
        np.testing.assert_array_equal(self._calculator.pairwise_distances, self._stress_calculator.pairwise_distances)
        

    def test_energies_equality(self):
        chex.assert_shape(self._result.energies, (500,))
        chex.assert_shape(self._stress_result.energies, (500,))
        np.testing.assert_allclose(self._result.energies, self._stress_result.energies)


    def test_forces_equality(self):
        chex.assert_shape(self._result.forces, (500, 3))
        chex.assert_shape(self._stress_result.forces, (500, 3))
        np.testing.assert_allclose(self._result.forces, self._stress_result.forces)


    def test_stress_conservation(self):       
        summed_stresses = np.sum(self._stress_result.stresses, axis=0)
        assert_all_close(self._stress_result.stress, summed_stresses)


    @skip    
    def test_forces_correctness(self):
        # TODO: Relative difference is always around 0.99 ??
        total_energy_fn = lambda R: jnp.sum(self._calculator._energy_fn(R))
        check_grads(total_energy_fn, (self._calculator.R,), order=1)#, eps=1E-6)
        # chex.assert_numerical_grads(fn, args, order=1)


    @skip
    def test_stress_correctness(self):
        pass
