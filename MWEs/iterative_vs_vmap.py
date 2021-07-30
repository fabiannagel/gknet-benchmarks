import timeit
from typing import Callable, Union, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from jax import linear_util as lu
from jax import tree_map, partial, tree_transpose, tree_structure
from jax._src.api import _check_callable, _vjp, _check_input_dtype_jacrev, _check_output_dtype_jacrev, _std_basis, \
    _unravel_array_into_pytree, jacrev
# from jax._src.util import safe_map
from jax._src.util import safe_map
from jax.api_util import argnums_partial
from jax_md import space, energy

import jax_utils

jax.config.update("jax_enable_x64", True)
global_dtype = "float64"


def jacrev_iterative(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           holomorphic: bool = False, allow_int: bool = False) -> Callable:

  _check_callable(fun)

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
    y, pullback = _vjp(f_partial, *dyn_args)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)

    # replace vmap with safe_map
    # jac = vmap(pullback)(_std_basis(y))
    jac = safe_map(pullback, _std_basis(y))

    # don't know what's happening here, but this seems to get rid of the first tensor dimension.
    # jac = jac[0] if isinstance(argnums, int) else jac
    # example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args

    # if we instead use the entire jacobian + all arguments, at least the shapes look correct...
    example_args = dyn_args

    jac = tree_map(partial(_unravel_array_into_pytree, y, 0), jac)

    # ... but JAX crashes when everything is transformed back into a tree
    return tree_transpose(tree_structure(example_args), tree_structure(y), jac)

  return jacfun

# initialize atoms
atoms = jax_utils.initialize_cubic_argon(multiplier=4)
R = jnp.array(atoms.get_positions(wrap=True), dtype=global_dtype)

# setup displacement
box = jnp.array(atoms.get_cell().array, dtype=global_dtype)
displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)

# initialize Lennard-Jones
lj = jax_utils.get_argon_lennard_jones_parameters()
neighbor_fn, atomwise_energy_fn = energy.lennard_jones_neighbor_list(displacement_fn, box,
                                                                     sigma=lj['sigma'],
                                                                     epsilon=lj['epsilon'],
                                                                     r_onset=lj['ro'] / lj['sigma'],
                                                                     r_cutoff=lj['rc'] / lj['sigma'],
                                                                     dr_threshold=1 * units.Angstrom,
                                                                     per_particle=True)

# compute initial neighbor list
neighbors = neighbor_fn(R)

total_energy_fn = lambda R, neighbor: jnp.sum(atomwise_energy_fn(R, neighbor=neighbor))
atomwise_energies = atomwise_energy_fn(R, neighbor=neighbors)

compute_force_contributions_iteratively = lambda: jacrev_iterative(atomwise_energy_fn, argnums=0)(R, neighbor=neighbors)
time_iteratively = timeit.timeit(compute_force_contributions_iteratively, number=5)
print("Iteratively: {} seconds".format(time_iteratively))

compute_force_contributions_vmapped = lambda: jacrev(atomwise_energy_fn, argnums=0)(R, neighbor=neighbors)
time_vmapped = timeit.timeit(compute_force_contributions_vmapped, number=5)
print("Vmapped: {} seconds".format(time_vmapped))

forces_iteratively = compute_force_contributions_iteratively()
forces_vmapped = compute_force_contributions_vmapped()

print(forces_iteratively.shape)
print(forces_vmapped.shape)
# np.testing.assert_array_equal(forces_iteratively, forces_vmapped)
# np.testing.assert_allclose(forces_vmapped, forces_iteratively, atol=1e-5)