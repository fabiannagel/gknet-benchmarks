from calculators.calculator import Calculator

class LennardJonesCalculatorBase(Calculator):

    def __init__(self, box_size: float, n: int, sigma: float, epsilon: float, r_cutoff: float) -> None:
        super().__init__(box_size, n)
        self._sigma = sigma
        self._epsilon = epsilon
        self._r_cutoff = r_cutoff

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def r_cutoff(self) -> float:
        return self._r_cutoff