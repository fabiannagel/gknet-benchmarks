from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.GNN.bapst_gnn import BapstGNN

sigma = 2.0
epsilon = 1.5
n = 1000

ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=None, r_onset=None)

gnn = BapstGNN.from_ase_atoms(ase._atoms, ase.r_cutoff, ase.r_onset, stress=False, stresses=False, adjust_radii=True, jit=True)
gnn.warm_up()
r = gnn.calculate()[0]

print("Warm-up time: {}".format(r.calculator._warmup_time))
print("Runtime: {}".format(r.computation_time))