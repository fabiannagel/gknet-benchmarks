from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from calculators.calculator import Calculator
    

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



# TODO: JaxResult w/ jnp.ndarray or DeviceArray. What's the difference?