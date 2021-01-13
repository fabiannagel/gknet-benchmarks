# import sys
# print(sys.path[0])
# del sys.path[0]
# sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from calculators.lennard_jones.lennard_jones_calculator import LennardJonesCalculatorBase
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
# from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
# from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair

def generate_system_sizes(z_max: int, unit_cell_size):
    ns = []
    for i in range(z_max):
        n = unit_cell_size * (i+1)**3
        ns.append(n)
    return ns


box_size = 100
n = 32
sigma = 2.0
epsilon = 1.5
r_cutoff = 11.0
r_onset = 6.0

ase = AseLennardJonesPair.create_potential(box_size, n, [], sigma, epsilon, r_cutoff)
result = ase.calculate()

print(result.energies)


# ase = AseLennardJonesPair(box_size=box_size, n=n, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff)




# jmd = JmdLennardJonesPair(box_size=box_size, n=n, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset)
# jmd._R = ase._R

# asax = AsaxLennardJonesPair(box_size=box_size, n=n, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset)
# asax._R = ase._R
