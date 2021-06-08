from ase_nve import AseNeighborListNve
from ase.atoms import Atoms
from asax.lj import LennardJones


class AsaxNeighborListNve(AseNeighborListNve):

    def __init__(self, atoms: Atoms, dt: float, batch_size: int):
        super().__init__(atoms, dt, batch_size)
        parameters = self.atoms.calc.parameters
        if not parameters:
            raise ValueError("Cannot copy Lennard-Jones parameters from Atoms object")

        sigma = parameters.sigma
        epsilon = parameters.epsilon
        rc = parameters.rc
        ro = parameters.ro
        self.atoms.calc = LennardJones(epsilon, sigma, rc, ro, stress=False)

    @property
    def description(self) -> str:
        return "ASAX"