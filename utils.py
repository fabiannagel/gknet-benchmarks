from vibes.helpers.supercell import make_cubic_supercell
from calculators.calculator import Calculator
import os
from typing import Callable, Iterable, List, Set, Tuple
from calculators.result import Result
import matplotlib.pyplot as plt
import numpy as np
import pickle
from itertools import groupby
from ase.build import bulk
from ase.atoms import Atoms

legend_size = 16


def generate_unit_cells(z_max: int, unit_cell_size):
    ns = []
    for i in range(z_max):
        n = unit_cell_size * (i+1)**3
        ns.append(n)
    return ns


def generate_system_sizes(start: int, stop: int, step=100) -> List[int]:
    return list(range(start, stop+step, step))


def generate_cubic_system_sizes(start: int, stop: int, step=100) -> List[int]:
    system_sizes = []

    for n in range(start, stop+step, step):
        atoms = bulk("Ar", cubic=True)
        atoms, _ = make_cubic_supercell(atoms, target_size=n)
        system_sizes.append(len(atoms))

    return sorted(set(system_sizes))


def create_output_path(runs: int) -> str:
    results_dir = "results/{}_runs/".format(runs)
    output_path = os.path.join(os.getcwd(), results_dir)

    if os.path.exists(output_path) and len(os.listdir(output_path)) > 0:
        raise RuntimeError("Output folder {} is not empty".format(output_path))

    os.makedirs(output_path, exist_ok=True)
    return output_path


def persist_results(results: List[Result], runs: int, descriptor=''):  
    if descriptor:
        file_name = "results_{}_{}_runs.pickle".format(descriptor, runs)
        output_folder_suffix = "{}_{}".format(descriptor, runs)
    else:
        file_name = "results_{}_runs.pickle".format(runs)
        output_folder_suffix = str(runs)

    base_path = create_output_path(output_folder_suffix)
    output_path = os.path.join(base_path, file_name) 

    with open(output_path, 'wb') as handle:
        pickle.dump(results, handle)


def persist_oom_events(events: List[Tuple[Calculator, str]], file_name: str):
    with open(file_name, 'wb') as handle:
        pickle.dump(events, handle)


def load_oom_events_from_pickle(file_path: str):
    with open(file_path, 'rb') as handle:
        oom_events = pickle.load(handle)
    return oom_events


def group_by(iterable: Iterable, key: Callable):
    '''Sorts and groups the iterable by the provided key.'''
    iterable = sorted(iterable, key=key)
    return groupby(iterable, key)


def get_system_sizes(results: List[Result]) -> List[int]:
    unique_ns = set([r.n for r in results])
    return sorted(unique_ns)


def load_results_from_pickle(file_path: str) -> List[Result]:
    with open(file_path, 'rb') as handle:
        results = pickle.load(handle)
    return results


def load_calculators_from_pickle(file_path: str) -> List[Calculator]:
    with open(file_path, 'rb') as handle:
        calculators = pickle.load(handle)
    return calculators


def load_super_cells_from_pickle(file_path: str) -> List[Atoms]:
    with open(file_path, 'rb') as handle:
        super_cells = pickle.load(handle)
    return super_cells


def computed_all_super_cells(results: List[Result], super_cells: Set[int]):
    pass



def plot_runtimes(results: List[Result], 
                  plot_title: str = None,
                  plot_file_name: str = None, 
                  shade_by: str = None,
                  scatter = False,
                  figsize=(20, 10)):

    runs = []
    system_sizes = get_system_sizes(results)
    fig, ax = plt.subplots(figsize=figsize)

    # try to sort by average computation time to match legend order to line order - not working
    # groups = group_by(results, lambda r: r.calculator.description)
    # groups = map(lambda g: list(g[1]), groups)
    # groups = sorted(list(groups), key=lambda rg: np.average([r.computation_time for r in rg]))

    # for results_per_calculator in groups:
    for key, results_per_calculator in group_by(results, lambda r: r.calculator.description):
        # for example: all results of jaxmd pair for all system sizes, 100 runs each
        results_per_calculator = list(results_per_calculator)    
        
        computed_system_sizes = get_system_sizes(results_per_calculator)
        # if computed_system_sizes != system_sizes:
            # oom_n = system_sizes[len(computed_system_sizes)]
            # print("calc went OOM at {}/{} super cells (n={})".format(len(computed_system_sizes), len(system_sizes), oom_n))

        computation_times = []
        standard_deviations = []
        mins = []
        maxs = []

        for key, mergeable_results in group_by(results_per_calculator, lambda r: r.n):
            mergeable_results = list(mergeable_results)
            runs.append(len(mergeable_results))

            mergeable_computation_times = [r.computation_time for r in mergeable_results]
           
            computation_times.append(np.mean(mergeable_computation_times))
            standard_deviations.append(np.std(mergeable_computation_times))      
            mins.append(np.min(mergeable_computation_times))
            maxs.append(np.max(mergeable_computation_times))     
            
            if scatter:
                current_system_sizes = [r.n for r in mergeable_results]
                ax.scatter(current_system_sizes, mergeable_computation_times)
        
        group_label = label_converter(results_per_calculator[0].calculator.description)
        ax.plot(computed_system_sizes, computation_times, label=group_label)
        # plt.errorbar(system_sizes, computation_times, yerr=np.array(computation_times)*10, uplims=True, lolims=True, label='uplims=True, lolims=True')

        if shade_by == 'minmax':
            ax.fill_between(computed_system_sizes, mins, maxs, alpha=0.2)

        elif shade_by == 'std':
            y_start = np.array(computation_times) - np.array(standard_deviations)
            y_end = np.array(computation_times) + np.array(standard_deviations)
            ax.fill_between(computed_system_sizes, y_start, y_end, alpha=0.2)
    
        
    if len(set(runs)) > 1:
        raise RuntimeError("Inconsistent number of runs in results")
        
    if plot_title:
        title = "{}\nAverages of {} runs. {} shading.".format(plot_title, runs[0], shade_by)
        ax.set_title(title)

    ax.set_xlabel("Number of atoms", fontsize=18, fontweight='bold')
    ax.set_xticks(system_sizes)
    ax.set_xticklabels(get_xticklabels(system_sizes))

    # TODO: Font sizes
    ax.set_ylabel("Computation time [s]", fontsize=18, fontweight='bold')
    ax.set_yscale("log")

    # print(handles)
    # print(labels)
    ax.legend(prop={'size': legend_size})
    
    if plot_file_name:
        fig.savefig(plot_file_name)

    return fig, ax


def get_xticklabels(system_sizes: List[int], skip=3) -> List[str]:
    def remove_tick_label(label: str, all_labels: List[str], reduction: int):
        idx = all_labels.index(label)
        if idx % reduction == 0:
            return label
        return ''

    upper_ticks = list(filter(lambda n: n >= 3600, system_sizes))                           # this is where everything is fine
    lower_ticks = list(filter(lambda n: n < 3600, system_sizes))                            # this is where more super cells occur -> overplotting
    lower_ticks = list(map(lambda n: remove_tick_label(n, lower_ticks, skip), lower_ticks))    # skip every 3rd label
    return lower_ticks + upper_ticks


def get_calculator_description(results: List[Result]):
    descriptions = set(map(lambda r: r.calculator.description, results))
    if len(descriptions) > 1:
        raise RuntimeError("Expected only results of a single calculator, got multiple.")
    return list(descriptions)[0]


def contains_multiple_memory_allocation_modes(results: List[Result]) -> bool:
    # Breaking results down by XLA memory allocation modes only makes sense for JAX-MD results
    if not all("JAX-MD" in r.calculator.description for r in results):
        return False

    memory_allocation_modes = set(map(lambda r: r.calculator.memory_allocation_mode, results))
    return len(memory_allocation_modes) > 1   
    

def plot_runtime_variances(results: List[Result], ):
    '''For results of a single calculator, visualize the runtimes over indices separately for each system sizes'''

    if not results:
        raise RuntimeError("Provided result list is empty.")

    if contains_multiple_memory_allocation_modes(results):
        raise RuntimeError("Provided result list was computed with multiple memory allocation modes.")

    system_sizes = sorted(set(map(lambda r: r.calculator.n, results)))

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.15)
    n_columns = 2
    n_rows = int(np.ceil(len(system_sizes) / n_columns))    
    
    suptitle = "Computation time of multiple runs with the same system size\nCalculator: {}".format(get_calculator_description(results))
    fig.suptitle(suptitle)

    for i, n in enumerate(system_sizes):
        rs = list(filter(lambda r: r.calculator.n == n, results))
        computation_times = list([r.computation_time for r in rs])  

        ax = fig.add_subplot(n_rows, n_columns, i + 1)
        ax.plot(computation_times)

        subplot_title = "n = {}".format(n)
        ax.set_title(subplot_title)
    
        # fig.set_xlabel("Run index")
        # fig.set_ylabel("Runtime")

    
def plot_oom_behavior(labels: List[str], system_sizes: List[int], all_properties: List[int], only_stress: List[int], only_stresses: List[int], only_energies_and_forces: List[int], only_energies_and_forces_no_jit: List[int], figsize=(10, 5)):
    
    # n where first OOM happened -> n where last computation successful
    def normalize(oom_system_sizes: List[int]) -> List[int]:
        normalized_system_sizes = []         
        for n_oom in oom_system_sizes:
            idx = system_sizes.index(n_oom)
            normalized_system_sizes.append(system_sizes[idx - 1])

        return normalized_system_sizes

    bar_width = 0.1
    
    # label locations
    r1 = np.arange(len(all_properties))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]

    # 15360 -> 13500
    only_energies_and_forces

    plt.figure(figsize=figsize)
    plt.bar(r1, normalize(all_properties), width=bar_width, label='Energies, Forces, Stress, Stresses')
    plt.bar(r2, normalize(only_stress), width=bar_width, label='Energies, Forces, Stress')
    plt.bar(r3, normalize(only_stresses), width=bar_width, label='Energies, Forces, Stresses')
    plt.bar(r4, normalize(only_energies_and_forces), width=bar_width, label='Energies, Forces')
    plt.bar(r5, normalize(only_energies_and_forces_no_jit), width=bar_width, label='Energies, Forces (jit=False)')

    # plt.title("Maximum number of atoms before going out-of-memory, per calculator")
    plt.xlabel('Calculator implementations', fontsize=18, fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(all_properties))], labels, fontsize=12)

    plt.ylabel('Maximum atom count $n_{max}$', fontsize=18, fontweight='bold')

    yticklabels = get_xticklabels(system_sizes, skip=3)
    yticklabels[1] = ''
    plt.yticks(system_sizes, yticklabels)

    plt.legend(prop={'size': legend_size})
    plt.show()
    # plt.savefig()


def print_oom_behavior_runtime_vs_dedicated(system_sizes: List[int], runtime_results: List[Result], oom_calculators: List[Calculator]):
    '''OOM behavior seems different in runtime benchmarks and the dedicated OOM benchmark. This method plots the last successfully computed system size per calculator for both the runtime and dedicated OOM benchmark.'''
    for key, results_per_calculator in group_by(runtime_results, lambda r: r.calculator.description):
        results_per_calculator = list(results_per_calculator)    
        description = results_per_calculator[0].calculator.description
        n_oom_runtime = max([r.n for r in results_per_calculator])
        
        # print(description)
        if not "ASE" in description:
            # the first system where OOM occurred
            n_oom_dedicated = list(filter(lambda c: c.description == description, oom_calculators))[0].n
            # obtain the last successful system
            n_oom_dedicated = system_sizes[system_sizes.index(n_oom_dedicated) - 1]
            
            # print("{:<70} went runtime OOM at n={}, dedicated OOM at n={}".format(description, n_oom_runtime, n_oom_dedicated))
            print("{:<70} OOM at n={},{} (runtime, dedicated)".format(description, n_oom_runtime, n_oom_dedicated))


def label_converter(calc_description: str) -> str:
    converted = None
    
    if "ASE" in calc_description:
        return "ASE LJ NL: Energies, Forces, Stress, Stresses"
    
    if "JAX-MD Pair" in calc_description:
        converted = "JAX-MD LJ Pair"
    if "JAX-MD Neighbor List" in calc_description:
        converted = "JAX-MD LJ NL"
    if "GNN" in calc_description:
        converted = "JAX-MD GNN"
        
    if "(stress=True, stresses=True, jit=True)" in calc_description:
        return converted + ": Energies, Forces, Stress, Stresses"
    
    if "(stress=True, stresses=False, jit=True)" in calc_description:
        return converted + ": Energies, Forces, Stress"
    
    if "(stress=False, stresses=True, jit=True)" in calc_description:
        return converted + ": Energies, Forces, Stresses"

    if "(stress=False, stresses=False, jit=True)" in calc_description:
        return converted + ": Energies, Forces"
    
    if "(stress=False, stresses=False, jit=False)" in calc_description:
        return converted + ": Energies, Forces (jit=False)"
    
    print(calc_description)




def extract_mean_runtimes(results: List[Result]):
    """
    Should return raw data for plotting:
    
    trace description, system sizes, runtimes (mean of all runs)
    """
    performed_runs = []
    data = {}

    for key, results_per_calculator in group_by(results, lambda r: r.calculator.description):
        results_per_calculator = list(results_per_calculator)    

        print(key)
        
        for key, mergeable_results in group_by(results_per_calculator, lambda r: r.n):
            mergeable_results = list(mergeable_results)
            performed_runs.append(len(mergeable_results))

            calculator_description = results_per_calculator[0].description
            n = mergeable_results[0].n
            mergeable_computation_times = [r.computation_time for r in mergeable_results]

            data[calculator_description][n]['mean_runtime'] = np.mean(mergeable_computation_times)
            data[calculator_description][n]['min_runtime'] = np.min(mergeable_computation_times)
            data[calculator_description][n]['max_runtime'] = np.max(mergeable_computation_times)
            data[calculator_description][n]['std'] = np.std(mergeable_computation_times)
            
        group_label = label_converter(results_per_calculator[0].calculator.description)
        data[calculator_description]['label'] = group_label
        
    # sanity check: all calculators should have consistently performed e.g. 100 runs per system size
    if len(set(performed_runs)) > 1:
        raise RuntimeError("Inconsistent number of runs in results")

    return data        