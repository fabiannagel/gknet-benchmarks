from dataclasses import dataclass
from calculators.calculator import Calculator
import numpy as np

@dataclass
class Result():
    calculator: Calculator
    n: int
# 
    energy: float
    energies: np.ndarray
    forces: np.ndarray
    stress: float
    stresses: np.ndarray
    computation_time: float = None