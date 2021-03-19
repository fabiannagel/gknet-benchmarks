from calculators.lennard_jones.pair.ase_lennard_jones_pair import AseLennardJonesPair
from calculators.GNN.bapst_gnn import BapstGNN

sigma = 2.0
epsilon = 1.5
n = 500

ase = AseLennardJonesPair.create_potential(n, sigma, epsilon, r_cutoff=None, r_onset=None)

gnn = BapstGNN.from_ase_atoms(ase._atoms, ase.r_cutoff, ase.r_onset, stress=True, stresses=True, adjust_radii=True, jit=True)
gnn.warm_up()
r = gnn.calculate()[0]

print("Energy: {}".format(r.energy))
print("Forces: {}".format(r.forces))
print(r.forces.shape)

print("Stress: {}".format(r.stress))
print(r.stress.shape)

print("Stresses: {}".format(r.stresses))
print(r.stresses.shape)