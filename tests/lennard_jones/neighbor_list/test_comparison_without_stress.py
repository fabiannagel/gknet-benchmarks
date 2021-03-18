from typing import List
from ase.atoms import Atoms
from calculators.calculator import Calculator
import unittest
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from ...utils import *


class ComparisonWithoutStress(unittest.TestCase):
    _calculators: List[Calculator]
    _n = 500
    _sigma = 2.0
    _epsilon = 1.5
    _r_cutoff: float
    _r_onset: float


    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)


    def _create_ase_calculator(self) -> AseLennardJonesPair:
        return AseLennardJonesPair.create_potential(self._n, self._sigma, self._epsilon, r_cutoff=None, r_onset=None)


    def _create_jaxmd_calculator(self, atoms: Atoms) -> JmdLennardJonesNeighborList:
        return JmdLennardJonesNeighborList.from_ase_atoms(atoms, self._sigma, self._epsilon, self._r_cutoff, self._r_onset, stress=False, stresses=False, adjust_radii=True, jit=True)


    def setUp(self):
        self._ase = self._create_ase_calculator()
        self._r_onset = self._ase.r_onset
        self._r_cutoff = self._ase.r_cutoff
        self._ase_result = self._ase.calculate()[0]

        self._jmd = self._create_jaxmd_calculator(self._ase._atoms)
        self._jmd_result = self._jmd.calculate()[0]
        
        self._calculators = [self._ase, self._jmd]
        self._results = [self._ase_result, self._jmd_result]


    def test_system_size_equality(self):
        ''' Assert that all calculators model systems with equal atom count. 
        Background: ASE models physically accurate unit cell sizes while JAX-MD does not care. 
        As a result, ASE might fall back to a realistic atom count when the passed n is not physically correct. '''
        system_sizes = [c.n for c in self._calculators]
        assert_arrays_all_equal(system_sizes)


    def test_atom_position_equality(self):
        atom_positions = [c.R for c in self._calculators]
        assert_arrays_all_equal(atom_positions)

    
    def test_pairwise_distances_equality(self):
        pairwise_distances = [c.pairwise_distances for c in self._calculators]
        assert_arrays_all_close(pairwise_distances)
    
    
    def test_energy_equality(self):
        total_energy = [r.energy for r in self._results]
        assert_arrays_all_close(total_energy, atol=1E-15)
    
    
    def test_energies_equality(self):
        total_energies = [r.energies for r in self._results]
        assert_arrays_all_close(total_energies, atol=1E-15)
    
    
    def test_forces_equality(self):
        forces = [r.forces for r in self._results]
        assert_arrays_all_close(forces, atol=1E-14)
        