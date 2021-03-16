from calculators.lennard_jones.neighbor_list.jaxmd_lennard_jones_neighbor_list import JmdLennardJonesNeighborList
from typing import List
from utils import *
from calculators.calculator import Result
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair

sigma = 2.0
epsilon = 1.5

n = 256

# ASE
ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=None, r_onset=None)
r1 = ase.calculate()[0]


# JAX-MD: stress=True, stresses=True, jit=True
jmd1 = JmdLennardJonesNeighborList.from_ase_atoms(ase._atoms, sigma, epsilon, ase.r_cutoff, ase.r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)    
jmd1.warm_up() 
r2 = jmd1.calculate()[0]


system_sizes = [r.calculator.n for r in [r1, r2]]
print(system_sizes)