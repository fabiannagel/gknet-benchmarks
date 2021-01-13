from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from ase.atoms import Atoms
import numpy as np

@dataclass
class Result():
    energies: np.ndarray
    forces: np.ndarray
    stresses: np.ndarray

    def energy(self) -> float:
        return np.sum(self.energies)

    def force(self) -> float:
        return np.sum(self.forces)

    def stress(self) -> float:
        return np.sum(self.stresses)
    
    
class Calculator(ABC):

    def __init__(self, box_size: float, n: int, R: np.ndarray) -> None:
        self._box_size = box_size
        self._n = n
        self._R = R
        
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
    def calculate(self) -> Result: 
        pass
