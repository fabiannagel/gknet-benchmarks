import warnings

from jax_md import space
from periodic_general import periodic_general as new_periodic_general
from periodic_general import inverse as new_inverse
from periodic_general import transform as new_transform


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

    # TODO: Verify JIT behavior
    return displacement