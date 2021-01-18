# import sys
# print(sys.path[0])
# del sys.path[0]
# sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
# from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair

def generate_system_sizes(z_max: int, unit_cell_size):
    ns = []
    for i in range(z_max):
        n = unit_cell_size * (i+1)**3
        ns.append(n)
    return ns


box_size = 100
n = 40
sigma = 2.0
epsilon = 1.5
r_cutoff = 11.0
r_onset = 6.0

results = []

# ase = AseLennardJonesPair.create_potential(box_size, n, None, sigma, epsilon, r_cutoff)
# results.append(ase.calculate())

jmd = JmdLennardJonesPair.create_potential(box_size, n, None, sigma, epsilon, r_cutoff, r_onset)
results.append(jmd.calculate())

# asax = AsaxLennardJonesPair.create_potential(box_size, n, None, sigma, epsilon, r_cutoff, r_onset)
# results.append(asax.calculate())

