import jax.numpy as jnp
from ase import units
import jax
from jax import grad, jit
from jax_md import space, energy
import jax_utils

jax.config.update("jax_enable_x64", True)
global_dtype = "float32"

# initialize atoms
atoms = jax_utils.initialize_cubic_argon(multiplier=8)
R = jnp.array(atoms.get_positions(wrap=True), dtype=global_dtype)

# setup displacement
box = jnp.array(atoms.get_cell().array, dtype=global_dtype)
displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)

# initialize Lennard-Jones
lj = jax_utils.get_argon_lennard_jones_parameters()
neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(displacement_fn, box,
                                                            sigma=lj['sigma'],
                                                            epsilon=lj['epsilon'],
                                                            r_onset=lj['ro'] / lj['sigma'],
                                                            r_cutoff=lj['rc'] / lj['sigma'],
                                                            dr_threshold=1 * units.Angstrom,
                                                            per_particle=False)
# compute initial neighbor list
neighbors = neighbor_fn(R)

# create strained energy_fn
transform_box_fn = lambda deformation: space.transform(jnp.eye(3, dtype=global_dtype) + (deformation + deformation.T) * 0.5, box)
strained_total_energy_fn = lambda R, deformation, *args, **kwargs: energy_fn(R, *args, box=transform_box_fn(deformation), **kwargs)

# box volume not actually required to be in float32. maybe only relevant for tracing/autodiff?
box_volume = jnp.array(jnp.linalg.det(box), dtype=global_dtype)
# box_volume = jnp.linalg.det(box)
stress_fn = lambda R, deformation, *args, **kwargs: grad(strained_total_energy_fn, argnums=1)(R, deformation, *args, **kwargs) / box_volume

# not actually required to be in float32
deformation = jnp.zeros_like(box, dtype=global_dtype)
# deformation = jnp.zeros_like(box)

# jit everything
energy_fn = jit(energy_fn)
strained_total_energy_fn = jit(strained_total_energy_fn)
stress_fn = jit(stress_fn)

e = energy_fn(R, neighbor=neighbors)
print(e)

e_strained = strained_total_energy_fn(R, deformation, neighbor=neighbors)
print(e_strained)

stress = stress_fn(R, deformation, neighbor=neighbors)
print(stress)