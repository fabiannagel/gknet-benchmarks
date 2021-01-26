import itertools
from typing import List
from unittest import TestCase
from unittest.case import skip
from ase.atoms import Atoms
from jax_md import space
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
import numpy as np


class ComparisonWithoutStressTest(TestCase):

    def __init__(self, methodName: str) -> None:
        super().__init__(methodName)
        
        # 4, 32, 500
        self._n = 500
        self._sigma = 2.0
        self._epsilon = 1.5
        self._r_cutoff = 11.0
        self._r_onset = 6.0
        
        self._ase = AseLennardJonesPair.create_equilibrium_potential(self._n, self._sigma, self._epsilon, self._r_cutoff, self._r_onset)
        
        r_cutoff_adjusted = self._r_cutoff / self._sigma
        r_onset_adjusted = self._r_onset   / self._sigma
        self._jmd = JmdLennardJonesPair.from_ase_atoms(self._ase._atoms, self._sigma, self._epsilon, r_cutoff_adjusted, r_onset_adjusted, stress=False)

    def setUp(self) -> None:
        self._ase_result = self._ase.calculate()
        self._jmd_result = self._jmd.calculate()

    def allClose(self, actual: np.float, desired: np.float, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
        np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)
    
    def arrays_all_equal(self, arrs: List[np.array]):
        for arr1, arr2 in itertools.combinations(arrs, 2):
            np.testing.assert_array_equal(arr1, arr2)   



    def test_energy_equivalence(self):
        # TODO: Probably due to the neighbor list LJ implementation in ASE
        self.allClose(self._ase_result.energy, self._jmd_result.energy)
    
    def test_system_sizes_equivalent(self):
        """ASE minds physical unit cell sizes while JAX-MD does not care. As a result, AseLennardJonesPair() falls back to correct system size when the passed n is not physically realistic."""
        self.assertEqual(self._ase._n, self._jmd._n)

    def test_atom_position_equality(self):
        arrs = [self._ase._R, self._ase._atoms.get_positions(), self._jmd._R]
        self.arrays_all_equal(arrs)

    def test_pairwise_distances_equality(self):
        dR_ase = self._ase.pairwise_distances
        dR_jmd = self._jmd.pairwise_distances
        self.allClose(dR_ase, dR_jmd)
        
        
        
        
        
