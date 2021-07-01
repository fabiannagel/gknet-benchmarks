import os
import pickle
import subprocess
import time
from os import path
from typing import List

import ase
from ase.atoms import Atoms


def get_milliseconds(start_time: float, round_decimals=2) -> float:
    return round((time.monotonic() - start_time) * 1000, round_decimals)


def get_current_git_revision() -> str:
    hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return hash.strip().decode('ascii')


def create_output_path(runs: int) -> str:
    results_dir = "results/{}_runs/".format(runs)
    output_path = os.path.join(os.getcwd(), results_dir)

    if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
        raise RuntimeError("Output folder {} is not empty".format(output_path))

    os.makedirs(output_path, exist_ok=True)
    return output_path


def persist(obj, file_path: str):
    path = os.path.dirname(file_path)
    if not os.path.exists(path):
        os.makedirs(path)

    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle)


def load(file_name: str):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


def load_super_cells(base_path: str) -> List[Atoms]:
    files = [path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".in")]
    return sorted([ase.io.read(t) for t in files if path.isfile(t)], key=lambda atoms: len(atoms))
