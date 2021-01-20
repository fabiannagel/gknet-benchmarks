from unittest.case import skip
from tests.base_test import BaseTest
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
import numpy as np

class JmdTest(BaseTest):

    def __init__(self, methodName: str) -> None:
        super().__init__(methodName)
        self._calculator = JmdLennardJonesPair.create_potential(self._box_size, self._n, self._R, self._sigma, self._epsilon, self._r_cutoff, self._r_onset, stress=False)
        self._result = self._calculator.calculate()

        self._stress_calculator = JmdLennardJonesPair.create_potential(self._box_size, self._n, self._R, self._sigma, self._epsilon, self._r_cutoff, self._r_onset, stress=True)
        self._stress_result = self._stress_calculator.calculate()

    def test_energy_conservation(self):
        self.assertEqual(self._result.energy, np.sum(self._result.energies))
        self.assertEqual(self._stress_result.energy, np.sum(self._stress_result.energies))

    def testForceConservation(self):
        self.assertEqual(self._result.force, np.sum(self._result.forces))
        self.assertEqual(self._stress_result.force, np.sum(self._stress_result.forces))
    
    @skip
    def testStressConservation(self):       
        summed_stresses = np.sum(self._stress_result.stresses, axis=0)
        np.testing.assert_array_equal(self._stress_result.stress, summed_stresses)
        # self.allClose(summed_stresses, self._result.stress)

