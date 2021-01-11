from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class Result():
    energies: np.ndarray
    forces: np.ndarray
    stresses: np.ndarray

    def energy(self) -> any:
        return np.sum(self.energies)

    def force(self) -> any:
        return np.sum(self.forces)

    def stress(self) -> any:
        return np.sum(self.stresses)
    
    
class Calculator(ABC):

    def __init__(self, box_size: float, n: int) -> None:
        super().__init__()
        self._box_size = box_size
        self._n = n
        self._R = np.random.uniform(size=(self._n, 3))

    @property
    def box_size(self) -> float:
        return self._box_size

    @property
    def n(self) -> float:
        return self._n

    @property
    def R(self) -> np.ndarray:
        return self._R

    # def compute_energies()
    # def compute_forces()
    # def compute_stresses()

    @abstractmethod
    def calculate(self) -> Result: 
        pass
