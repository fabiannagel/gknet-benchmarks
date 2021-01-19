from jax import random, jit, value_and_grad
import jax.numpy as np
from jax_md import space, energy, quantity
from jax.config import config
config.update("jax_enable_x64", True)
# from asax.utils import get_potential_with_stress

from ase.build import bulk
from asax.lj import LennardJones as jLJ

def get_potential_with_stress(displacement, create_energy_fn):
    
    # why fixed size of 3x3?
    ones = np.eye(N=3, M=3, dtype=np.double)

    def energy_under_strain(R: space.Array, strain: space.Array) -> space.Array:
        def displacement_under_strain(
            Ra: space.Array, Rb: space.Array, **unused_kwargs
        ) -> space.Array:
            transform = ones + strain
            return _transform(transform, displacement(Ra, Rb))

        energy = create_energy_fn(displacement_under_strain)
        return energy(R)

    def energy(R: space.Array) -> space.Array:
        zeros = np.zeros((3, 3), dtype=np.double)
        return value_and_grad(energy_under_strain, argnums=(0, 1))(R, zeros)

    return energy
    # return jit(energy)

def _transform(T: space.Array, v: space.Array) -> space.Array:
    """Apply a linear transformation, T, to a collection of vectors, v.
    Transform is written such that it acts as the identity during gradient
    backpropagation.x
    Args:
      T: Transformation; ndarray(shape=[spatial_dim, spatial_dim]).
      v: Collection of vectors; ndarray(shape=[..., spatial_dim]).
    Returns:
      Transformed vectors; ndarray(shape=[..., spatial_dim]).
    """
    space._check_transform_shapes(T, v)
    return np.dot(v, T)


box_size = 100
n = 50
sigma = 2.0
epsilon = 1.5
r_cutoff = 11.0
r_onset = 6.0

key = random.PRNGKey(0)
key, subkey = random.split(key)
R = random.uniform(subkey, shape=(n, 3)) * box_size


def get_energy(displacement_fn):
    return energy.lennard_jones_pair(displacement_fn, sigma=sigma, epsilon=epsilon, r_onset=r_onset, r_cutoff=r_cutoff, per_particle=False)


displacement_fn, shift_fn = space.periodic(box_size)

atomwise_energy_fn = energy.lennard_jones_pair(displacement_fn, sigma=sigma, epsilon=epsilon, r_onset=r_onset, r_cutoff=r_cutoff, per_particle=True)
total_energy_fn = lambda R: np.sum(atomwise_energy_fn(R))
force_fn = lambda R: quantity.force(total_energy_fn)(R)

stress_fn = get_potential_with_stress(displacement_fn, get_energy)


def compare_energies():
    e1 = total_energy_fn(R)
    e2, grads = stress_fn(R)
    
    print(e1)
    print(e2)
    print(MAD(e1, e2))


def MAD(x, y):
    return np.mean(np.abs(x - y))


compare_energies()


# energy, grads = potential_fn(R)
# grad, stress = grads
# force = -grad
# print(stress)




# atoms = bulk("Ar", cubic=True) * [5, 5, 5]
# atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)
# energy_fn = jLJ(epsilon=epsilon, sigma=sigma, rc=r_cutoff, ro=r_onset, x64=True, stress=True)
# stress = energy_fn.get_stress(R)
