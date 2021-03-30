import sys
if not '/home/pop518504/git/gknet-benchmarks' in sys.path:
    sys.path.insert(0, '/home/pop518504/git/gknet-benchmarks')

from typing import List
from ase.atoms import Atoms
from ase.build import bulk
from vibes.helpers.supercell import make_cubic_supercell
import pickle

def make_cubic_supercells(n_start: int, n_stop: int, n_step: int) -> List[Atoms]:
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

        computed_system_sizes.append(n)
        supercells.append(atoms)

    return supercells


# n_start = 100
# n_stop = 15360
n_start = 23328
n_stop = 30000

n_step = 1000
supercells = make_cubic_supercells(n_start, n_stop, n_step)

output_path = "supercells_{}_{}_{}.pickle".format(n_start, n_stop, n_step)
with open(output_path, 'wb') as handle:
    pickle.dump(supercells, handle)
