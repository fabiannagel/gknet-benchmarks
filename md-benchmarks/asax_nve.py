from ase_nve import AseNeighborListNve
from ase.atoms import Atoms
from asax.lj import LennardJones


class AsaxNeighborListNve(AseNeighborListNve):

    def __init__(self, atoms: Atoms, dt: float, batch_size: int):
        super().__init__(atoms, dt, batch_size)
        sigma = 2.0
        epsilon = 1.5
        rc = 10.0
        ro = 6.0
        self.atoms.calc = LennardJones(epsilon, sigma, rc, ro, stress=False)

    @property
    def description(self) -> str:
        return "ASAX"