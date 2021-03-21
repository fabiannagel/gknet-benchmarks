from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional
from ase.atoms import Atoms
from calculators.result import Result
import numpy as np
import time
import itertools


class Calculator(ABC):
    _warmup_time: float = None

    def __init__(self, box: np.ndarray, n: int, R: np.ndarray, computes_stress: bool) -> None:
        self._box = box
        self._n = n     # TODO: What do we need n for? We can derive it from R to avoid inconsistencies
        self._R = R
        self._computes_stress = computes_stress
        

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, *args) -> cls:
        box = atoms.get_cell().array
        R_real = atoms.get_positions()
        return cls(box, len(atoms), R_real, *args)


    @classmethod
    def create_potential(cls, box_size: float, n: int, R: Optional[np.ndarray], *args) -> cls:
        # TODO: Decide if generating our own distances here is a valid use case. Don't we always initialize via ASE in the end?
        if R is None or len(R) == 0:
            R = cls._generate_R(cls, n, box_size)
        box = box_size * np.eye(3)
        return cls(box, n, R, *args)


    @property
    @abstractmethod
    def description(self) -> str:
        """A description string that can be used to outline the functionality of this calculator. Useful for plots etc."""
        pass
    
    @property
    @abstractmethod
    def pairwise_distances(self) -> np.ndarray:     # shape (self._n, 3)    TODO: Dynamic numpy types in Python?
        """Returns a matrix of pairwise atom distances of shape (n, 3)."""
        pass
    
    @property
    def box(self) -> np.ndarray:
        return self._box


    @property
    def n(self) -> int:
        return self._n


    @property
    def R(self):
        return self._R


    @property
    def computes_stress(self) -> bool:
        return self._computes_stress

    
    def _generate_R(self, n: int, scaling_factor: float) -> np.ndarray:
        print("numpy PRNG")
        return np.random.uniform(size=(n, 3)) * scaling_factor


    @abstractmethod
    def _generate_R(self, n: int, scaling_factor: float) -> np.ndarray:
        pass    


    def _time_execution(self, callable: Callable, *args, **kwargs):
        start = time.monotonic()
        return_val = callable(*args, **kwargs)
        elapsed_seconds = time.monotonic() - start   
        return elapsed_seconds, return_val


    @abstractmethod
    def _perform_warm_up(self):
        pass


    def warm_up(self):
        if self._warmup_time:
            raise ValueError("A warm-up has already been performed.")
        
        elapsed_seconds, _ = self._time_execution(self._perform_warm_up)
        self._warmup_time = elapsed_seconds


    @abstractmethod
    def _compute_properties(self) -> Result: 
        pass


    def calculate(self, runs=1) -> List[Result]:
        results = []
        for _ in itertools.repeat(None, runs):       
            elapsed_seconds, r = self._time_execution(self._compute_properties)
            r.computation_time = elapsed_seconds
            results.append(r)
    
        return results
