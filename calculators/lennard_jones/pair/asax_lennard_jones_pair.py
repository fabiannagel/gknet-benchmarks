from calculators.lennard_jones.lennard_jones_calculator import LennardJonesCalculatorBase
from calculators.calculator import Result
from ase import Atoms
from ase.build import bulk
from asax.lj import LennardJones

class AsaxLennardJonesPair(LennardJonesCalculatorBase):

    def __init__(self, box_size: float, n: int, sigma: float, epsilon: float, r_cutoff: float, r_onset: float) -> None:
        super().__init__(box_size, n, sigma, epsilon, r_cutoff)
        self._r_onset = r_onset
        # self._atoms = Atoms(positions=self._R)
        
        self._atoms = bulk("Ar", cubic=True) * [5, 5, 5]
        self._atoms.set_cell(1.05 * self._atoms.get_cell(), scale_atoms=True)
        self._atoms.calc = LennardJones(self._epsilon, self._sigma, self._r_cutoff, self._r_onset, x64=True, stress=True)

    @property
    def r_onset(self) -> float:
        return self._r_onset

    def calculate(self) -> Result:
        energy = self._atoms.calc.get_potential_energy(self._atoms)

        print(energy)

        # atoms = Atoms(positions=[[0, 0, 0], [8, 0, 0]])
        # calc = LennardJones(self._epsilon, self._sigma, self._r_cutoff, self._r_onset, stress=True)
        # energy = calc.get_potential_energy(atoms)
        # print(energy)

        return None
        # return Result([energy], None, None)
    
        # TODO: asax does not output atom-wise energies. do we actually need those?
        # energies = [self._atoms.get_potential_energy()]
        # forces = self._atoms.get_forces()
        # stresses = [self._atoms.get_stress()]
        # return Result(energies, forces, stresses)