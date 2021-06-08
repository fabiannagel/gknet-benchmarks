import numpy as np
from md_driver import MdDriver
from ase_nve import AseNeighborListNve
from jax_nve_nl import JaxmdNeighborListNVE
from ase import Atoms, units
import jax_utils
from jax import config
config.update("jax_enable_x64", True)

def run_ase(atoms: Atoms, dt: float, steps: int, batch_size: int, write_stress: bool, verbose: bool) -> MdDriver:
    md = AseNeighborListNve(atoms, dt, batch_size)
    md.run(steps, write_stress, verbose)
    return md


def run_jaxmd(atoms: Atoms, dt: float, steps: int, batch_size: int, write_stress: bool, verbose: bool) -> MdDriver:
    md = JaxmdNeighborListNVE(atoms, dt, batch_size)
    md.run(steps, write_stress, verbose)
    return md


def run_asax(atoms: Atoms, dt: float, steps: int, batch_size: int, write_stress: bool, verbose: bool) -> MdDriver:
    pass



atoms = jax_utils.initialize_cubic_argon(multiplier=8)
dt = 5 * units.fs
steps = 2500
batch_size = 5
write_stress = False
verbose = False

# ase = run_ase(atoms, dt, steps, batch_size, write_stress, verbose)

# jaxmd = run_jaxmd(atoms, dt, steps, batch_size, write_stress, True)
# print("Total simulation time: {}".format(jaxmd.total_simulation_time))
# print("Average ms/batch: {}".format(np.mean(jaxmd.batch_times)))
# print("Average ms/step: {}".format(np.mean(jaxmd.step_times)))