from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from ase.atoms import Atoms
import numpy as np
import time

@dataclass
class Result():
    calculator: Calculator
    energy: float
    energies: np.ndarray
    force: float
    forces: np.ndarray
    stress: float
    stresses: np.ndarray
    computation_time: float = None

    # TODO: Verify conservational laws on construction

    #def energy(self) -> float:
    #    return np.sum(self.energies)

    #def force(self) -> float:
    #    return np.sum(self.forces)

    #def stress(self) -> float:
    #    return np.sum(self.stresses)
    

class Calculator(ABC):
    _runtimes = []
    _atoms: Optional[Atoms]

    def __init__(self, box: np.array, n: int, R: np.ndarray, computes_stress: bool) -> None:
        self._box = box
        self._n = n
        self._R = R
        self._computes_stress = computes_stress
        

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, *args) -> cls:
        box = atoms.get_cell().array * np.eye(3)
        return cls(box, len(atoms), atoms.get_positions(), *args)


    @classmethod
    def create_potential(cls, box_size: float, n: int, R: np.ndarray, *args) -> cls:
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
    def box(self) -> np.array:
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


    def calculate(self) -> Result:
        start = time.time()
        result = self._compute_properties()
        elapsed_seconds = (time.time() - start) / 1000
        self._runtimes.append(elapsed_seconds)

        result.computation_time = elapsed_seconds
        return result
