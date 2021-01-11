# import sys
# print(sys.path[0])
# del sys.path[0]
# sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from calculators.lennard_jones.pair.jaxmd_lennard_jones_pair import JmdLennardJonesPair
from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair

# calc = AseLennardJonesPair(box_size=10, n=10, sigma=10, epsilon=10, r_cutoff=10)
# calc.calculate()

jmd = JmdLennardJonesPair(box_size=10, n=10, sigma=10, epsilon=10, r_cutoff=10, r_onset=10)
r = jmd.calculate()
print(r)
