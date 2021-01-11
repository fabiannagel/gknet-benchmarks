from calculators.lennard_jones.lennard_jones_calculator import LennardJonesCalculatorBase
from calculators.calculator import Result

import jax.numpy as jnp
from jax import grad
from jax_md import space, energy

class JmdLennardJonesPair(LennardJonesCalculatorBase):

    def __init__(self, box_size: float, n: int, sigma: float, epsilon: float, r_cutoff: float, r_onset: float) -> None:
        super().__init__(box_size, n, sigma, epsilon, r_cutoff)
        self._r_onset = r_onset
        self._displacement_fn, self._shift_fn = space.free()
        self._energy_fn = energy.lennard_jones_pair(self._displacement_fn, sigma=self._sigma, epsilon=self._epsilon, r_onset=self._r_onset, r_cutoff=self._r_cutoff, per_particle=True)
        self._force_fn = energy.lennard_jones_pair(self._displacement_fn, sigma=self._sigma, epsilon=self._epsilon, r_onset=self._r_onset, r_cutoff=self._r_cutoff, per_particle=False)

        # TODO: Allow overriding CPU-based atom position generation
        # key = random.PRNGKey(0)
        # key, subkey = random.split(key)
        # self._R = random.uniform(subkey, shape=(n, 3))

    def calculate(self) -> Result:
        energies = self._energy_fn(self._R)
        forces = -grad(self._force_fn)(self._R)
        stresses = None
        return Result(energies, forces, stresses)    
