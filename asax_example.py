from asax.lj import LennardJones
from ase import Atoms
from ase.build import bulk

sigma = 2.0
epsilon = 1.5
rc = 10.0
ro = 6.0

calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, x64=True, stress=True)
atoms = Atoms(positions=[[0, 0, 0], [8, 0, 0]])
energy = calc.get_potential_energy(atoms)

print(energy)