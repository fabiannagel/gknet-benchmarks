import jax.numpy as np

from jax.api import jit, grad
from jax import lax
from jax import random

from jax_md import space, energy, simulate, minimize, quantity, energy
from periodic_general import periodic_general

key = random.PRNGKey(0)


N = 1024
spatial_dimensions = 3
R = random.uniform(key, (N, spatial_dimensions))

box_size = quantity.box_size_at_number_density(N, 1.2, spatial_dimensions)
box = box_size * np.eye(spatial_dimensions)
displacement_fn, shift_fn = periodic_general(box)


sigma = 2.0
epsilon = 1.5
r_cutoff = 11.0
r_onset = 6.0
energy_fn = energy.lennard_jones_pair(displacement_fn, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset)

def strained_box_energy_fn(epsilon, R):
  return energy_fn(R, box=box + epsilon)

stress_fn = jit(lambda epsilon, R: grad(strained_box_energy_fn)(epsilon, R) / np.linalg.det(box))
# stress_fn = jit(grad(strained_box_energy_fn))

epsilon = np.zeros((3, 3))
energy = strained_box_energy_fn(epsilon, R)
print(energy)

print(energy_fn(R))

stress = stress_fn(epsilon, R)
print(stress)
