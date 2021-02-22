import numpy as np
from time import time
import itertools
from typing import Callable, List

def assert_all_close(actual: np.float, desired: np.float, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
    np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)


def assert_arrays_all_equal(arrs: List[np.array]):
    for arr1, arr2 in itertools.combinations(arrs, 2):
        np.testing.assert_array_equal(arr1, arr2)   


def assert_arrays_all_close(arrs: List[np.array], rtol=1E-7, atol=0):
    for arr1, arr2 in itertools.combinations(arrs, 2):
        np.testing.assert_allclose(arr1, arr2, rtol, atol)


def get_runtimes(fn: Callable, runs=10) -> List[float]:
    runtimes = []
    for i in range(runs):
        start = time()
        fn()
        elapsed = time() - start
        runtimes.append(elapsed)
    
    initial_runtime = runtimes[0]        
    jitted_runtime = runtimes[1:]
    return initial_runtime, jitted_runtime
    