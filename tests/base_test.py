from unittest import TestCase
import numpy as np
from time import time
import itertools
from typing import Callable, List


class BaseTest(TestCase):

    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def assert_all_close(self, actual: np.float, desired: np.float, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
        np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)
    
    def assert_arrays_all_equal(self, arrs: List[np.array]):
        for arr1, arr2 in itertools.combinations(arrs, 2):
            np.testing.assert_array_equal(arr1, arr2)   

    def get_runtimes(self, fn: Callable, runs: int) -> List[float]:
        runtimes = []
        for i in range(runs):
            start = time()
            fn()
            elapsed = time() - start
            runtimes.append(elapsed)
        
        return runtimes
    