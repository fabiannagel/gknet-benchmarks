from abc import abstractmethod
from unittest import TestCase
import numpy as np
import jax.numpy as jnp
import chex

class BaseTest():
      
    def __init__(self, methodName: str, stress: bool) -> None:
        super().__init__(methodName=methodName)

        if stress:
            # TODO: Use numpy oder JAX.np here? Does it make a difference? Should we decide on the concrete test case (ASE, JAX-MD)?
            strain = jnp.zeros((3, 3))
            self._property_args = strain, self._jmd._R
        else:
            self._property_args = self._jmd._R

    @abstractmethod
    def setUp(self) -> None:
        pass

