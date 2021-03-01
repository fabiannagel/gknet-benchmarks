from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional
from ase.atoms import Atoms
import numpy as np
import time
import itertools

@dataclass
class Result():
    calculator: Calculator
    n: int

    energy: float
    energies: np.ndarray
    forces: np.ndarray
    stress: float
    stresses: np.ndarray
    computation_time: float = None


class Calculator(ABC):
    _results: List[Result] = []
    _atoms: Optional[Atoms]
    # _energy_fn: Callable

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
    def results(self) -> List[Result]:
        return self._results

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


    @abstractmethod
    def _compute_properties(self) -> Result: 
        pass


    def calculate(self, runs=1) -> List[Result]:
        results = []
        for _ in itertools.repeat(None, runs):
            start = time.time()
            r = self._compute_properties()
            r.computation_time = time.time() - start   
            results.append(r)
        
        self._results.extend(results)
        return results
