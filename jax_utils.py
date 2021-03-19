from typing import Callable, Dict, Tuple
from jax import vmap, random
from jax.api import grad, jacfwd, jit
from jax_md import energy
from jax_md.energy import DisplacementFn
from calculators.calculator import Calculator
from os import environ
from enum import Enum
import warnings
from jax_md import space, quantity
import jax.numpy as jnp
from periodic_general import periodic_general as new_periodic_general, transform
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


def jit_if_wanted(do_jit: bool, *args) -> Tuple:

    if not all([callable(a) for a in args]):
        raise ValueError("Expected a list of callables.")

    if not do_jit:
        return args
    
    return tuple([jit(f) for f in args])


PotentialFn = Callable[[space.Array], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, None, None]]
PotentialProperties = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

def get_strained_pair_potential(box: jnp.ndarray, displacement_fn: DisplacementFn, sigma: float, epsilon: float, r_cutoff: float, r_onset: float, compute_stress: bool, compute_stresses: bool) -> PotentialFn:

    def strained_potential_fn(R: space.Array) -> PotentialProperties:
        # 1) Set the box under strain using a symmetrized deformation tensor
        # 2) Override the box in the energy function
        # 3) Derive forces, stress and stresses as gradients of the deformed energy function

        # define a default energy function, an infinitesimal deformation and a function to apply the transformation to the box
        energy_fn = energy.lennard_jones_pair(displacement_fn, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, per_particle=True)                     
        deformation = jnp.zeros_like(box)

        # a function to symmetrize the deformation tensor and apply it to the box
        transform_box_fn = lambda deformation: transform(jnp.eye(3) + (deformation + deformation.T) * 0.5, box) 

        # atomwise and total energy functions that act on the transformed box. same for force, stress and stresses.
        deformation_energy_fn = lambda deformation, R: energy_fn(R, box=transform_box_fn(deformation))
        total_energy_fn = lambda deformation, R: jnp.sum(deformation_energy_fn(deformation, R))            

        force_fn = lambda deformation, R: grad(total_energy_fn, argnums=1)(deformation, R) * -1

        stress = None
        if compute_stress:
            stress_fn = lambda deformation, R: grad(total_energy_fn, argnums=0)(deformation, R) / jnp.linalg.det(box)
            stress = stress_fn(deformation, R)  

        stresses = None
        if compute_stresses:
            stresses_fn = lambda deformation, R: jacfwd(deformation_energy_fn, argnums=0)(deformation, R) / jnp.linalg.det(box)
            stresses = stresses_fn(deformation, R)

        total_energy = total_energy_fn(deformation, R)
        atomwise_energies = deformation_energy_fn(deformation, R)
        forces = force_fn(deformation, R)

        return total_energy, atomwise_energies, forces, stress, stresses

    return strained_potential_fn


def get_unstrained_pair_potential(box: jnp.ndarray, displacement_fn: DisplacementFn, sigma: float, epsilon: float, r_cutoff: float, r_onset: float) -> PotentialFn:

    def unstrained_potential_fn(R: space.Array) -> PotentialProperties:
        energy_fn = energy.lennard_jones_pair(displacement_fn, sigma=sigma, epsilon=epsilon, r_onset=r_onset, r_cutoff=r_cutoff, per_particle=True)       
        total_energy_fn = lambda R: jnp.sum(energy_fn(R))
        forces_fn = quantity.force(total_energy_fn)

        total_energy = total_energy_fn(R)
        atomwise_energies = energy_fn(R)
        forces = forces_fn(R)
        stress, stresses = None, None
        return total_energy, atomwise_energies, forces, stress, stresses

    return unstrained_potential_fn


def get_strained_neighbor_list_potential(energy_fn, neighbors, box: jnp.ndarray, compute_stress: bool, compute_stresses: bool) -> PotentialFn:

    def strained_potential_fn(R: space.Array) -> PotentialProperties:
        # 1) Set the box under strain using a symmetrized deformation tensor
        # 2) Override the box in the energy function
        # 3) Derive forces, stress and stresses as gradients of the deformed energy function
        # define a default energy function, an infinitesimal deformation and a function to apply the transformation to the box
        # energy_fn = energy.lennard_jones_pair(displacement_fn, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, r_onset=r_onset, per_particle=True)                     
        deformation = jnp.zeros_like(box)

        # a function to symmetrize the deformation tensor and apply it to the box
        transform_box_fn = lambda deformation: transform(jnp.eye(3) + (deformation + deformation.T) * 0.5, box) 
        
        # atomwise and total energy functions that act on the transformed box. same for force, stress and stresses.
        deformation_energy_fn = lambda deformation, R, *args, **kwargs: energy_fn(R, box=transform_box_fn(deformation), neighbor=neighbors)
        total_energy_fn = lambda deformation, R, *args, **kwargs: jnp.sum(deformation_energy_fn(deformation, R))            
        force_fn = lambda deformation, R, *args, **kwargs: grad(total_energy_fn, argnums=1)(deformation, R) * -1
        
        stress = None
        if compute_stress:
            stress_fn = lambda deformation, R, *args, **kwargs: grad(total_energy_fn, argnums=0)(deformation, R) / jnp.linalg.det(box)
            stress = stress_fn(deformation, R, neighbor=neighbors)  
        
        stresses = None
        if compute_stresses:
            stresses_fn = lambda deformation, R, *args, **kwargs: jacfwd(deformation_energy_fn, argnums=0)(deformation, R) / jnp.linalg.det(box)
            stresses = stresses_fn(deformation, R, neighbor=neighbors)
        
        total_energy = total_energy_fn(deformation, R, neighbor=neighbors)
        atomwise_energies = deformation_energy_fn(deformation, R, neighbor=neighbors)
        forces = force_fn(deformation, R, neighbor=neighbors)
        
        return total_energy, atomwise_energies, forces, stress, stresses

    return strained_potential_fn


def get_unstrained_neighbor_list_potential(energy_fn, neighbors, box: jnp.ndarray, compute_stress: bool, compute_stresses: bool) -> PotentialFn:

    def unstrained_potential(R: space.Array) -> PotentialProperties:
        total_energy_fn = lambda R, *args, **kwargs: jnp.sum(energy_fn(R, *args, **kwargs))
        forces_fn = quantity.force(total_energy_fn)

        total_energy = total_energy_fn(R, neighbor=neighbors)
        atomwise_energies = energy_fn(R, neighbor=neighbors)
        forces = forces_fn(R, neighbor=neighbors)
        stress, stresses = None, None
        return total_energy, atomwise_energies, forces, stress, stresses

    return unstrained_potential


def get_strained_gnn_potential(energy_fn, neighbors, params, box: jnp.ndarray, compute_stress: bool, compute_stresses: bool) -> PotentialFn:
    
    def strained_potential_fn(R: space.Array) -> PotentialProperties:
        deformation = jnp.zeros_like(box)
        transform_box_fn = lambda deformation: transform(jnp.eye(3) + (deformation + deformation.T) * 0.5, box) 

        total_deformation_energy_fn = lambda params, R, deformation, neighbors: energy_fn(params, R, neighbors, box=transform_box_fn(deformation))
        force_fn = lambda params, R, deformation, neighbors: grad(total_deformation_energy_fn, argnums=1)(params, R, deformation, neighbors) * -1

        # TODO: atom-wise energies + stresses with GNN?
        # fake atomwise energy function from which we can take the jacobian
        atomwise_energy_fn = lambda params, R, deformation, neighbors: jnp.ones((R.shape[0],1)) / total_deformation_energy_fn(params, R, deformation, neighbors)

        total_energy = total_deformation_energy_fn(params, R, deformation, neighbors)
        atomwise_energies = atomwise_energy_fn(params, R, deformation, neighbors)
        forces = force_fn(params, R, deformation, neighbors)

        stress = None
        if compute_stress:
            stress_fn = lambda params, R, deformation, neighbors: grad(total_deformation_energy_fn, argnums=2)(params, R, deformation, neighbors) / jnp.linalg.det(box)
            stress = stress_fn(params, R, deformation, neighbors)

        stresses = None
        if compute_stresses:
            stresses_fn = lambda params, R, deformation, neighbors: jacfwd(atomwise_energy_fn, argnums=2)(params, R, deformation, neighbors) / jnp.linalg.det(box)
            stresses = stresses_fn(params, R, deformation, neighbors) 

        return total_energy, atomwise_energies, forces, stress, stresses

    return strained_potential_fn


def get_unstrained_gnn_potential(energy_fn, neighbors, params, box: jnp.ndarray, compute_stress: bool, compute_stresses: bool) -> PotentialFn:

    def unstrained_potential_fn(R: space.Array) -> PotentialProperties:
        total_energy = energy_fn(params, R, neighbors)

        # TODO: atom-wise energies with GNN?
        # fake atomwise energy function as in strained potential
        atomwise_energy_fn = lambda params, R, neighbors: jnp.ones((R.shape[0],1)) / energy_fn(params, R, neighbors)  
        atomwise_energies = atomwise_energy_fn(params, R, neighbors)

        force_fn = lambda params, R, neighbors, *args, **kwargs: grad(energy_fn, argnums=1)(params, R, neighbors) * -1
        forces = force_fn(params, R, neighbors)

        return total_energy, atomwise_energies, forces, None, None

    return unstrained_potential_fn    

# TODO: JaxCalculator
def get_state(calculator: Calculator) -> Dict:
    # Copy the object's state from self.__dict__ which contains
    # all our instance attributes. Always use the dict.copy()
    # method to avoid modifying the original state.
    state = calculator.__dict__.copy()
    # Remove the unpicklable entries.
    del state['_displacement_fn']
    del state['_potential_fn']
    del state['_R']

    # neighbor list calculator
    if '_energy_fn' in state: del state['_energy_fn']
    if '_neighbor_fn' in state: del state['_neighbor_fn']
    if '_neighbors' in state: del state['_neighbors']
    return state


def set_state(calculator: Calculator, state: Dict):
    # Restore instance attributes (i.e., filename and lineno).
    calculator.__dict__.update(state)
    # Restore the previously opened file's state. To do so, we need to
    # reopen it and read from it until the line count is restored.
    error_fn = lambda *args, **kwargs: print("Pickled instance cannot compute new data")
    calculator._displacement_fn = error_fn
    calculator._potential_fn = error_fn
    calculator._R = error_fn

    calculator._energy_fn = error_fn
    calculator._neighbor_fn = error_fn
    calculator._neighbors = error_fn
