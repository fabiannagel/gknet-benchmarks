import timeit
from typing import Callable, Union, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from jax import linear_util as lu
from jax import tree_map, partial, tree_transpose, tree_structure
from jax._src.api import _check_callable, _vjp, _check_input_dtype_jacrev, _check_output_dtype_jacrev, _std_basis, \
    _unravel_array_into_pytree, jacrev, jit
# from jax._src.util import safe_map
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

    # replace vmap with iterative_safe_map
    # jac = vmap(pullback)(_std_basis(y))
    jac = iterative_safe_map(pullback, _std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = tree_map(partial(_unravel_array_into_pytree, y, 0), jac)
    return tree_transpose(tree_structure(example_args), tree_structure(y), jac)

  return jacfun

def iterative_safe_map(f, *args):
  args = list(map(list, args))
  n = len(args[0])
  for arg in args[1:]:
    assert len(arg) == n, 'length mismatch: {}'.format(list(map(len, args)))
  result = list(map(f, *args))

  without_tuples = list(map(lambda r: r[0], result))
  stacked = jnp.stack(without_tuples, axis=0)
  return stacked,


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

compute_force_contributions_iteratively = jit(lambda: jacrev_iterative(atomwise_energy_fn, argnums=0)(R, neighbor=neighbors))
compute_force_contributions_vmapped = jit(lambda: jacrev(atomwise_energy_fn, argnums=0)(R, neighbor=neighbors))
np.testing.assert_allclose(compute_force_contributions_iteratively(), compute_force_contributions_vmapped(), atol=1e-15)

# measure execution time
print("Compute force contribution Jacobians for n = {}".format(len(atoms)))

time_iteratively = timeit.timeit(compute_force_contributions_iteratively, number=10)
print("Iteratively: {} seconds".format(time_iteratively))

time_vmapped = timeit.timeit(compute_force_contributions_vmapped, number=10)
print("Vmapped: {} seconds".format(time_vmapped))

print("\nVmapped version runs {} faster than iterative".format(time_iteratively / time_vmapped))