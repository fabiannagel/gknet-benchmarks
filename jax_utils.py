from os import environ
from enum import Enum
import warnings
from typing import Dict, List
from jax_md import space
import jax.numpy as jnp
from jax import vmap, random
from periodic_general import periodic_general as new_periodic_general
from periodic_general import inverse as new_inverse
from periodic_general import transform as new_transform


class XlaMemoryFlag(Enum):
    XLA_PYTHON_CLIENT_PREALLOCATE = "XLA_PYTHON_CLIENT_PREALLOCATE"
    XLA_PYTHON_CLIENT_MEM_FRACTION = "XLA_PYTHON_CLIENT_MEM_FRACTION"
    XLA_PYTHON_CLIENT_ALLOCATOR = "XLA_PYTHON_CLIENT_ALLOCATOR" 
    DEFAULT = "DEFAULT"


def get_memory_allocation_mode() -> XlaMemoryFlag:
    active_flags = []

    for f in XlaMemoryFlag:
        try:
            environ[f.name]
            active_flags.append(f)
        except KeyError:
            continue

    if len(active_flags) > 1:
        raise SystemError("Multiple memory allocation modes enabled simultaneously.")
    if not active_flags:
        return XlaMemoryFlag.DEFAULT
    return active_flags[0]


def set_memory_allocation_mode(flag: XlaMemoryFlag, value: str):
    if flag is XlaMemoryFlag.DEFAULT:
        return  # placeholder flag, nothing to do
    if not flag in XlaMemoryFlag:
        raise ValueError("Passed flag is not a valid XLA memory allocation mode.")
    environ[flag.value] = value


def clear_memory_allocation_mode(flag: XlaMemoryFlag):
    if not flag in XlaMemoryFlag:
        raise ValueError("Passed flag is not a valid XLA memory allocation mode.")
    del environ[flag.value]
    

def reset_memory_allocation_mode():
    '''Removes all XLA memory allocation flags to default settings.'''
    for flag in XlaMemoryFlag:
        try:
            del environ[flag.value]
        except KeyError:
            continue


# def get_printable_memory_allocation_modes(modes: Dict[XlaMemoryFlag, str]) -> List[str]:
    # printable = []
    # for k in modes.keys():
        # v = modes[k]
        # printable.append("{}={}".format(k, v))
    # return printable


def compute_pairwise_distances(displacement_fn: space.DisplacementFn, R: jnp.ndarray):
    # displacement_fn takes two vectors Ra and Rb
    # space.map_product() vmaps it twice along rows and columns such that we can input matrices
    dR_dimensionwise_fn = space.map_product(displacement_fn)
    dR_dimensionwise = dR_dimensionwise_fn(R, R)    # ... resulting in 4 dimension-wise distance matrices shaped (n, n, 3)
    # Computing the vector magnitude for every row vector:
    # First, map along the first axis of the initial (n, n, 3) matrix. the "output" will be (n, 3)
    # Secondly, within the mapped (n, 3) matrix, map along the zero-th axis again (one atom).
    # Here, apply the magnitude function for the atom's displacement row vector.
    magnitude_fn = lambda x: jnp.sqrt(jnp.sum(x**2))
    vectorized_fn = vmap(vmap(magnitude_fn, in_axes=0), in_axes=0)
    return vectorized_fn(dR_dimensionwise)


def generate_R(n: int, scaling_factor: float) -> jnp.ndarray:
    # TODO: Build a global service to manage and demand PRNGKeys for JAX-based simulations. if necessary for MD later.
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    return random.uniform(subkey, shape=(n, 3)) * scaling_factor


def new_get_displacement(atoms):
    '''what asax.utils.get_displacement() does, only with functions from the new periodic_general()'''
    # TODO: Refactor once new periodic_general() is released

    if not all(atoms.get_pbc()):
        displacement, _ = space.free()
        warnings.warn("Atoms object without periodic boundary conditions passed!")
        return displacement

    cell = atoms.get_cell().array
    inverse_cell = new_inverse(cell)
    displacement_in_scaled_coordinates, _ = new_periodic_general(cell)

    # **kwargs are now used to feed through the box information
    def displacement(Ra: space.Array, Rb: space.Array, **kwargs) -> space.Array:
        Ra_scaled = new_transform(inverse_cell, Ra)
        Rb_scaled = new_transform(inverse_cell, Rb)
        return displacement_in_scaled_coordinates(Ra_scaled, Rb_scaled, **kwargs)

    return displacement
