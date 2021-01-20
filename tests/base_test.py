from unittest import TestCase
import numpy as np
from numpy import testing


class BaseTest(TestCase):
    TOLERANCE = 1E-4

    def __init__(self, methodName: str):
        super().__init__(methodName=methodName)
    
        self._box_size = 100
        self._n = 108
        self._R = self._generate_R()
        self._sigma = 2.0
        self._epsilon = 1.5
        self._r_cutoff = 11.0
        self._r_onset = 6.0

    def _generate_R(self) -> np.ndarray:
        return np.ones(shape=(self._n, 3))
        # return np.random.uniform(size=(self._n, 3)) * self._box_size

    def allClose(self, actual: np.float, desired: np.float, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
        np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)