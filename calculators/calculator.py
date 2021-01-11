from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    max_r = 10

    def __init__(self, box_size: float, n: int) -> None:
        super().__init__()
        self._box_size = box_size
        self._n = n
        # TODO: Allow passing existing atom coordinates to verify the same computations among different calculators
        self._R = self._generate_R()

    def _generate_R(self) -> np.ndarray:
        print("numpy PRNG")
        return np.random.uniform(size=(self._n, 3)) * self.max_r

    @property
    def box_size(self) -> float:
        return self._box_size

    @property
    def n(self) -> float:
        return self._n

    @property
    def R(self) -> np.ndarray:
        return self._R

    @abstractmethod
    def calculate(self) -> Result: 
        pass
