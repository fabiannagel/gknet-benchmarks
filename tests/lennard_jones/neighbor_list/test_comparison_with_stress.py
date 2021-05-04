from calculators.result import Result
from ase.atoms import Atoms
from calculators.calculator import Calculator
import unittest
from unittest.case import skip
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList
from ...utils import *

class ComparisonWithStress(unittest.TestCase):
    _results: List[Result] = []
    _calculators: List[Calculator]
    _n = 500
    _sigma = 2.0
    _epsilon = 1.5
    _r_cutoff: float
    _r_onset: float


    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)


    def setUp(self):
        ase = AseLennardJonesPair.create_potential(self._n, self._sigma, self._epsilon, r_cutoff=None, r_onset=None)
        self._results.extend(ase.calculate())
        r_onset = ase.r_onset
        r_cutoff = ase.r_cutoff
        
        jmd_stress = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, self._sigma, self._epsilon, r_cutoff, r_onset, stress=True, stresses=False, adjust_radii=True, jit=True)
        self._results.extend(jmd_stress.calculate())

        jmd_stresses = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, self._sigma, self._epsilon, r_cutoff, r_onset, stress=False, stresses=True, adjust_radii=True, jit=True)
        self._results.extend(jmd_stresses.calculate())

        jmd_stress_stresses = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, self._sigma, self._epsilon, r_cutoff, r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)
        self._results.extend(jmd_stress_stresses.calculate())

        self._calculators = [ase, jmd_stress, jmd_stresses, jmd_stress_stresses]


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


    def test_energy_conservation(self):
        summed_energy = list(map(lambda r: np.sum(r.energies), self._results))
        computed_energy = list(map(lambda r: r.energy, self._results))
        assert_arrays_all_close([summed_energy, computed_energy], atol=1E-17)
    
    
    def test_energies_equality(self):
        total_energies = [r.energies for r in self._results]
        assert_arrays_all_close(total_energies, atol=1E-15)
    
    
    def test_forces_equality(self):
        forces = [r.forces for r in self._results]
        assert_arrays_all_close(forces, atol=1E-14)


    def test_stress_equality(self):
        results_with_stress = list(filter(lambda r: r.stress is not None, self._results))
        stress = [r.stress for r in results_with_stress]
        assert_arrays_all_close(stress, atol=1E-17)

    
    def test_stress_conservation(self):       
        results_with_stresses = list(filter(lambda r: r.stresses is not None, self._results))        
        summed_stress = list(map(lambda r: np.sum(r.stresses, axis=0), results_with_stresses))
    
        results_with_stress = list(filter(lambda r: r.stress is not None, self._results))
        computed_stress = list(map(lambda r: r.stress, results_with_stress))
        assert_arrays_all_close([summed_stress, computed_stress], atol=1E-17)


    def test_stresses_equality(self):
        results_with_stresses = list(filter(lambda r: r.stresses is not None, self._results))
        stresses = [r.stresses for r in results_with_stresses]
        assert_arrays_all_close(stresses, atol=1E-4)
