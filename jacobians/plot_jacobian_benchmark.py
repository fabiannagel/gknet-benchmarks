from typing import Dict
from matplotlib import pyplot as plt

import utils


def plot_jacobian_benchmark(runtimes: Dict):
    plot = lambda x, y, label: plt.plot(x, y, label=label, linestyle='--', marker='o', markersize=5)

    # plot runs with jit=True
    jitted = runtimes['jit=True']
    jitted_multipliers = list(jitted.keys())

    runtime_jitted_iteratively = list(map(lambda m: jitted[m]['iteratively'], jitted_multipliers))
    runtime_jitted_vmapped = list(map(lambda m: jitted[m]['vmapped'], jitted_multipliers))
    runtime_jitted_speedup = list(map(lambda m: jitted[m]['speedup'], jitted_multipliers))

    plot(jitted_multipliers, runtime_jitted_iteratively, "iteratively, jit=True")
    plot(jitted_multipliers, runtime_jitted_vmapped, "vmapped, jit=True")
    # plt.plot(jitted_multipliers, runtime_jitted_speedup)

    # plot runs with jit=False
    non_jitted = runtimes['jit=False']
    non_jitted_multipliers = list(non_jitted.keys())
    runtime_non_jitted_iteratively = list(map(lambda m: non_jitted[m]['iteratively'], non_jitted_multipliers))
    runtime_non_jitted_vmapped = list(map(lambda m: non_jitted[m]['vmapped'], non_jitted_multipliers))
    runtime_non_jitted_speedup = list(map(lambda m: non_jitted[m]['speedup'], non_jitted_multipliers))

    plot(non_jitted_multipliers, runtime_non_jitted_iteratively, "iteratively, jit=False")
    plot(non_jitted_multipliers, runtime_non_jitted_vmapped, "vmapped, jit=False")
    # plt.plot(non_jitted_multipliers, runtime_jitted_speedup)

    plt.title("Computing force contributions: Iterative vs. vmapped ({} runs)".format(runtimes['runs']))
    plt.xlabel("Atom count")
    plt.ylabel("Runtime [seconds]")
    plt.yscale("log")
    plt.legend()
    plt.show()


runtimes = utils.load("jacobians_benchmark.pickle")
plot_jacobian_benchmark(runtimes)