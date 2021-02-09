from abc import ABC, abstractmethod
from calculators.calculator import Calculator, Result
from tests.base_test import BaseTest
from unittest.case import skip
import numpy as np
import jax.numpy as jnp
import chex
from .test_utils import *

class UnitTest(BaseTest):
    ''' 
    Refers to the test case in which a single calculator is tested for correctness.
    To allow for multiple parameter settings of the same implementation, multiple Calculator objects of the same class can be passed.
    '''

    
    

    def __init__(self, methodName: str, calculators: List[Calculator], stress: bool) -> None:
        super().__init__(methodName, stress)
        self._calculators = calculators

    # In JAX-MD, do everything with and without JIT

    def test_atom_positions_shape(self):
        shapes = [c.R for c in self._calculators]
        


    def test_pairwise_distances_shape(self):
        pass


    def test_energy_correctness(self):
        # compare the output energy to a ASE hand-computed output that should be constant for the given system
        pass


    def test_forces_correctness(self):
        # same as before, only for forces
        pass


    def test_stress_correctness(self):
        # same as before, only for forces
        pass



    # TODO: Not sure if these make sense. After all, there is no agreement yet on what properties we will finally compute.
         
    # def test_energy_conservation(self):
        # self.assertEqual(self._result.energy, np.sum(self._result.energies))
        # self.assertEqual(self._stress_result.energy, np.sum(self._stress_result.energies))


    # def testForceConservation(self):
        # self.assertEqual(self._result.force, np.sum(self._result.forces))
        # self.assertEqual(self._stress_result.force, np.sum(self._stress_result.forces))

    # def testStressConservation(self):       
        # summed_stresses = np.sum(self._stress_result.stresses, axis=0)
        # np.testing.assert_array_equal(self._stress_result.stress, summed_stresses)
        # self.allClose(summed_stresses, self._result.stress)

  

  
  
  
  
  
  
  
  
  
  
  
  
  
  

  
  
  
  
  
  
  
  
  
  

  
