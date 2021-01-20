from tests.base_test import BaseTest
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair


class ComparisonWithStressTest(BaseTest):

    def __init__(self, methodName: str) -> None:
        super().__init__(methodName)
        self._initialize_ase()
        self._initialize_asax()
        self._initialize_jmd()

    def _initialize_ase(self):
        r_cutoff_adjusted = self._r_cutoff * self._sigma    # for compatibility: JAX-MD/ASAX does this internally
        self._ase = AseLennardJonesPair.create_potential(self._box_size, self._n, self._R, self._sigma, self._epsilon, r_cutoff_adjusted)
        self._ase_result = self._ase.calculate()

    def _initialize_asax(self):
        self._asax = AsaxLennardJonesPair.create_potential(self._box_size, self._n, self._R, self._sigma, self._epsilon, self._r_cutoff, self._r_onset, stress=True)
        self._asax_result = self._asax.calculate()

    def _initialize_jmd(self):
        self._jmd = JmdLennardJonesPair.create_potential(self._box_size, self._n, self._R, self._sigma, self._epsilon, self._r_cutoff, self._r_onset, stress=True)
        self._jmd_result = self._jmd.calculate()
    
    def test_energy_equivalence(self):

        print(self._ase_result.energy)
        print(self._asax_result.energy)
        print(self._jmd_result.energy)

        self.allClose(self._ase_result.energy, self._asax_result.energy)
        self.allClose(self._ase_result.energy, self._jmd_result.energy)
        self.allClose(self._asax_result.energy, self._jmd_result.energy)
    
        
        
        
