# import sys
# print(sys.path[0])
# del sys.path[0]
# sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.pair.asax_lennard_jones_pair import AsaxLennardJonesPair

box_size = 100
n = 100
# sigma = 3
# epsilon = 0.01
# r_onset = 9.5
# r_cutoff = 10

sigma = 2.0
epsilon = 1.5
r_cutoff = 11.0
r_onset = 6.0

results = []

# ase = AseLennardJonesPair(box_size=box_size, n=n, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff)
# results.append(ase.calculate())
# 
jmd = JmdLennardJonesPair(box_size=box_size, n=n, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset)
# jmd._R = ase._R
results.append(jmd.calculate())

# asax = AsaxLennardJonesPair(box_size=box_size, n=n, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset)
# asax._R = ase._R
# results.append(asax.calculate())

# for r in results:
#     print(r.energy())