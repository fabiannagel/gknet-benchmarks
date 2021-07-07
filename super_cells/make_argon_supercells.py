import ase.io
from ase.md.velocitydistribution import Stationary, MaxwellBoltzmannDistribution
from typing import List
from ase.atoms import Atoms
from ase.build import bulk
from vibes.helpers.supercell import make_cubic_supercell


def make_cubic_supercells(n_start: int, n_stop: int, n_step: int, temperature_K: float) -> List[Atoms]:
    requested_system_sizes = list(range(n_start, n_stop + n_step, n_step))
    computed_system_sizes: List[int] = []
    supercells: List[Atoms] = []

    for requested_n in requested_system_sizes:
        print("Computing requested n={} ...".format(requested_n))

        atoms = bulk("Ar", cubic=True)
        atoms, _ = make_cubic_supercell(atoms, target_size=requested_n)
        n = len(atoms)

        if n in computed_system_sizes:
            print("n={} already computed, skipping.".format(n))
            continue

        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
        Stationary(atoms)

        computed_system_sizes.append(n)
        supercells.append(atoms)

    return supercells


def create_thermalized_trajectories(n_start: int, n_stop: int, n_step=100, temperature_K=30):
    super_cells = make_cubic_supercells(n_start, n_stop, n_step, temperature_K)

    for atoms in super_cells:
        file_name = "argon_{}k_{}.in".format(temperature_K, len(atoms))
        print("Writing {}".format(file_name))
        ase.io.write(file_name, atoms, velocities=True, format="aims")
