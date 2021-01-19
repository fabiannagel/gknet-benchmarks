from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from ase.atoms import Atoms
import numpy as np
import time

@dataclass
class Result():
    calculator: Calculator
    energies: np.ndarray
    forces: np.ndarray
    stresses: np.ndarray
    computation_time: float = None

    def energy(self) -> float:
        return np.sum(self.energies)

    def force(self) -> float:
        return np.sum(self.forces)

    def stress(self) -> float:
        return np.sum(self.stresses)
    

class Calculator(ABC):
    _runtimes = []

    def __init__(self, box_size: float, n: int, R: np.ndarray) -> None:
        self._box_size = box_size
        self._n = n
        self._R = R
        
    @property
    @abstractmethod
    def description(self) -> str:
        """A description string that can be used to outline the functionality of this calculator"""
        pass

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, *args) -> cls:
        return cls(atoms.get_cell(), len(atoms), atoms.get_positions(), args)

    @classmethod
    def create_potential(cls, box_size: float, n: int, R: np.ndarray, *args) -> cls:
        if R is None or len(R) == 0:
            R = cls._generate_R(cls, n, box_size)
        return cls(box_size, n, R, *args)
    
    def _generate_R(self, n: int, scaling_factor: float) -> np.ndarray:
        print("numpy PRNG")
        return np.random.uniform(size=(n, 3)) * scaling_factor

    @abstractmethod
    def _generate_R(self, n: int, scaling_factor: float) -> np.ndarray:
        pass    

    @abstractmethod
    def _compute_properties(self) -> Result: 
        pass

    def calculate(self) -> Result:
        start = time.time()
        result = self._compute_properties()
        elapsed_seconds = (time.time() - start) / 1000
        self._runtimes.append(elapsed_seconds)

        result.computation_time = elapsed_seconds
        return result


