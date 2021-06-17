import itertools
import time
from abc import ABC, abstractmethod
from statistics import mean
from typing import List

import numpy as np
from ase import Atoms


class MdDriver(ABC):
    _batch_times: List[float]
    _total_simulation_time: float

    def __init__(self, atoms: Atoms, dt: float, batch_size: int):
        self.atoms = atoms
        self.dt = dt
        self.batch_size = batch_size

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def step_times(self) -> List[float]:
        """
        Returns step times as batch_times / batch_size.
        Shape adjusted value repetition such that len(step_times) = steps.
        """
        # TODO: Verify that there are no rounding errors here. alternative:
        # step_times = list(map(lambda bt: bt / float(self.batch_size), self._batch_times))

        step_times = map(lambda bt: bt / self.batch_size, self._batch_times)
        padded_step_times = map(lambda st: itertools.repeat(st, self.batch_size), step_times)
        merged_step_times = itertools.chain(*padded_step_times)
        return list(merged_step_times)

    @property
    def mean_step_time(self) -> float:
        return round(mean(self.step_times), 2)

    @property
    def batch_times(self) -> List[float]:
        """
        Returns elapsed times per simulated batch. Includes write_stress if enabled.
        len(batch_times) = steps / batch_size.
        """
        return self._batch_times

    @property
    def mean_batch_time(self) -> float:
        return round(mean(self.batch_times), 2)

    @property
    def total_simulation_time(self) -> float:
        """Returns the total simulation time (in seconds)."""
        return self._total_simulation_time

    def _create_stress_buffer(self, write_stress: bool, steps: int):
        """
        Creates a tensor to store intermediate stress results for each batch.
        """
        if not write_stress:
            return None

        buffer_depth = int(steps / self.batch_size)
        return np.empty(shape=(buffer_depth, 3, 3))

    @abstractmethod
    def _run_md(self, steps: int, write_stress: bool, verbose: bool):
        pass

    def run(self, steps: int):
        if steps % self.batch_size != 0:
            raise ValueError("steps need to be dividable by batch_size")

        if steps == 0 or self.batch_size == 0:
            raise ValueError("steps and batch_size cannot be 0")

        self._batch_times = []
        start = time.monotonic()
        self._run_md(steps, write_stress=False, verbose=False)
        self._total_simulation_time = round(time.monotonic() - start, 2)
