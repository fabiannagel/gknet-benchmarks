"""
How much would a JAX-based implementation accelerate heat flux computations through vectorization (vmap)?

To provide a rough estimate without actually having to implement the heat flux first, we fall back on a similar problem: Computing force contributions iteratively vs. using vmap.
Essentially, we compute the Jacobian of all atomic energy contributions of shape (n, 3) w.r.t. all atomic positions R of shape (n,).

In JAX, we can do this by calling `jacrev()` which computes the Jacobian using reverse mode and uses vmap under the hood.
In PyTorch, we would have to do this iteratively for each $r \in R$. JAX cannot do this by default (afaik), so we provide an implementation of `jacrev_iterative()`.

This script captures the computing time necessary for multiple runs of computing force contributions, both iteratively and vmapped.
"""


import timeit
from typing import Callable, Union, Sequence, Dict

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
from matplotlib import pyplot as plt

import jax_utils
import utils

jax.config.update("jax_enable_x64", True)
global_dtype = "float32"

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

  # without_tuples = list(map(lambda r: r[0], result))
  # stacked = jnp.stack(without_tuples, axis=0)
  # return stacked,
  return jnp.stack([r[0] for r in result], axis=0),


def initialize_system(multiplier: int):
    # initialize atoms
    atoms = jax_utils.initialize_cubic_argon(multiplier=multiplier)
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

    return R, neighbors, atomwise_energy_fn


multipliers = list(range(15, 20))
runs = 1
runtimes = {'runs': runs}

oom_events = []

for use_jit in [False, True]:

    k_jit = "jit={}".format(use_jit)
    runtimes[k_jit] = {}

    for multiplier in multipliers:

        try:
            R, neighbors, atomwise_energy_fn = initialize_system(multiplier=multiplier)
        except RuntimeError:
            print("OOM during system init")
            break

        print("Compute force contribution Jacobians for n = {}".format(len(R)))

        k_multiplier = "n={}".format(len(R))
        runtimes[k_jit][k_multiplier] = {}

        # compute force contributions iteratively
        if not "{}{}".format(use_jit, "iteratively") in oom_events:
            try:
                compute_force_contributions_iteratively = lambda: jacrev_iterative(atomwise_energy_fn, argnums=0)(R, neighbor=neighbors)
                if use_jit:
                    compute_force_contributions_iteratively = jit(compute_force_contributions_iteratively)

                compute_iteratively_blocked = lambda: compute_force_contributions_iteratively().block_until_ready()
                time_iteratively = timeit.timeit(compute_iteratively_blocked, number=runs)
                runtimes[k_jit][k_multiplier]['iteratively'] = time_iteratively

            except RuntimeError:
                print("jit={}, iteratively went OOM".format(use_jit))
                oom_events.append("{}{}".format(use_jit, "iteratively"))

        else:
            print("jit={}, iteratively went OOM before. skipping...".format(use_jit))

        # compute force contributions vmapped
        if not "{}{}".format(use_jit, "vmapped") in oom_events:

            try:
                compute_force_contributions_vmapped = lambda: jacrev(atomwise_energy_fn, argnums=0)(R, neighbor=neighbors)
                if use_jit:
                    compute_force_contributions_vmapped = jit(compute_force_contributions_vmapped)

                compute_vmapped_blocked = lambda: compute_force_contributions_vmapped().block_until_ready()
                time_vmapped = timeit.timeit(compute_vmapped_blocked, number=runs)
                runtimes[k_jit][k_multiplier]['vmapped'] = time_vmapped

            except RuntimeError:
                print("jit={}, vmapped went OOM".format(use_jit))
                oom_events.append("{}{}".format(use_jit, "vmapped"))

        else:
            print("jit={}, vmapped went OOM before. skipping...".format(use_jit))

        # TODO: Does the assertion fail because of float32?
        # np.testing.assert_allclose(compute_force_contributions_iteratively(), compute_force_contributions_vmapped(), atol=1e-15)

utils.persist(runtimes, 'jacobians_benchmark.pickle')
