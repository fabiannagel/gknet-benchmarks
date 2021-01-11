from calculators.lennard_jones.lennard_jones_calculator import LennardJonesCalculatorBase
from calculators.calculator import Result

from ase import Atoms
# from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.calculator import PropertyNotImplementedError


class AseLennardJonesPair(LennardJonesCalculatorBase):
    
    def __init__(self, box_size: float, n: int, sigma: float, epsilon: float, r_cutoff: float) -> None:
        super().__init__(box_size, n, sigma, epsilon, r_cutoff)
        self._atoms = Atoms(positions=self._R)
        self._atoms.calc = LennardJones(sigma=self._sigma, epsilon=self._epsilon, rc=self._r_cutoff)

    def calculate(self) -> Result:
        energies = self._atoms.get_potential_energies()
        forces = self._atoms.get_forces()
        
        # TODO: Is this an error or a result in a particular calculation?
        try:
            stresses = self._atoms.get_stresses()
        except PropertyNotImplementedError:
            stresses = None

        return Result(energies, forces, stresses)
