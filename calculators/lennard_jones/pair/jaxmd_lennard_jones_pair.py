from calculators.lennard_jones.lennard_jones_calculator import LennardJonesCalculatorBase
from calculators.calculator import Result

import jax.numpy as jnp
from jax import grad, random, jit
from jax_md import space, energy
from jax.config import config
config.update("jax_enable_x64", True)

class JmdLennardJonesPair(LennardJonesCalculatorBase):

    def __init__(self, box_size: float, n: int, sigma: float, epsilon: float, r_cutoff: float, r_onset: float) -> None:
        super().__init__(box_size, n, sigma, epsilon, r_cutoff)
        self._r_onset = r_onset
        self._displacement_fn, self._shift_fn = space.periodic(self._box_size)
        self._energy_fn = jit(energy.lennard_jones_pair(self._displacement_fn, sigma=self._sigma, epsilon=self._epsilon, r_onset=self._r_onset, r_cutoff=self._r_cutoff, per_particle=True))
        self._force_fn = jit(energy.lennard_jones_pair(self._displacement_fn, sigma=self._sigma, epsilon=self._epsilon, r_onset=self._r_onset, r_cutoff=self._r_cutoff, per_particle=False))

        # TODO: Allow overriding CPU-based atom position generation
        # key = random.PRNGKey(0)
        # key, subkey = random.split(key)
        # self._R = random.uniform(subkey, shape=(n, 3))

    def _generate_R(self) -> jnp.ndarray:
        print("jaxmd PRNG")
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        return random.uniform(subkey, shape=(self._n, 3)) * self.max_r

    @property
    def r_onset(self) -> float:
        return self._r_onset

    def calculate(self) -> Result:
        def wrapped_computation():
            energies = self._energy_fn(self._R)
            forces = -grad(self._force_fn)(self._R)
            stresses = None
            return energies, forces, stresses
        
        energies, forces, stresses = jit(wrapped_computation())
        return Result(energies, forces, stresses)    
