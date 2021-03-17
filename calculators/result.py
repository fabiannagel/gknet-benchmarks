from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from calculators.calculator import Calculator
    

class Result():
    calculator: Calculator
    n: int
# 
    energy: float
    energies: np.ndarray
    forces: np.ndarray
    stress: float
    stresses: np.ndarray
    computation_time: float

    def __init__(self, calculator: Calculator, n: int, energy: float, energies: np.ndarray, forces: np.ndarray, stress: np.ndarray, stresses: np.ndarray):
        self.calculator = calculator
        self.n = n
        self.energy = energy
        self.energies = energies
        self.forces = forces
        self.stress = stress
        self.stresses = stresses


class JaxResult(Result):

    def __init__(self, calculator: Calculator, n: int, energy: jnp.DeviceArray, energies: jnp.DeviceArray, forces: jnp.DeviceArray, stress: jnp.DeviceArray, stresses: jnp.DeviceArray):
        energy.block_until_ready()
        energies.block_until_ready()
        forces.block_until_ready()
        stress.block_until_ready()
        stresses.block_until_ready()
        super().__init__(calculator, n, energy, energies, forces, stress, stresses)