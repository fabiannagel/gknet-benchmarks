from ase.atoms import Atoms
from jax_md import space, energy, quantity, smap
import jax.numpy as jnp
from jax import vmap
import ase
from ase.build import bulk
import numpy as np
from asax.utils import get_displacement
from jax.config import config
config.update("jax_enable_x64", True)


atoms: Atoms = bulk('Ar', cubic=True) * [2, 2, 2]
R = atoms.get_positions()   # (4, 4)

dR_ase = atoms.get_all_distances(mic=True)      # Minimum image convention: Doesn't seem to be considered in the calculator - does this matter?
print(dR_ase)
print()

displacement_fn = get_displacement(atoms)
dR_dimensionwise_fn = vmap(vmap(displacement_fn, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=0)
dR_dimensionwise = dR_dimensionwise_fn(R, R)
# print(dimensionwise_displacement)
# print()


# first, map along the first axis of the initial (n, n, 3) matrix. the "output" will be (n, 3)
# secondly, within the mapped (n, 3) matrix, map along the zero-th axis again (one atom).
# here, apply the magnitude function for the atom's displacement vector
magnitude_fn = lambda x: jnp.sqrt(jnp.sum(x**2))
vectorized_fn = vmap(vmap(magnitude_fn, in_axes=0), in_axes=0)
dR_jmd = vectorized_fn(dR_dimensionwise)
print(dR_jmd)

np.testing.assert_allclose(dR_ase, dR_jmd)