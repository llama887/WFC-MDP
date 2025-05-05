import argparse
import os
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")

from functools import partial

import yaml
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict

from biome_adjacency_rules import create_adjacency_matrix
from evolution import evolve
from tasks.binary_task import binary_percent_water, binary_reward
from wfc_env import WFCWrapper

from tasks.pond_task import pond_reward
from tasks.river_task import river_reward
from tasks.grass_task import grass_reward
from tasks.hill_task import hill_reward
from wfc_env import CombinedReward

FIGURES_DIRECTORY = "figures"
os.makedirs(FIGURES_DIRECTORY, exist_ok=True)


def binary_convergence_over_path_lengths(
    sample_size: int,
    evolution_hyperparameters: dict[str, Any],
    qd: bool,
    hard: bool = False,
) -> None:
    """
    Line plot of how many generations it takes for agents to reach the max
    reward in the binary experiment over various path lengths, plus a bar
    chart showing the fraction of runs that actually converged.

    Parameters
    ----------
    sample_size : int
        Number of evolution runs at each path length.
    evolution_hyperparameters : dict
        Hyperparameters passed through to `evolve(...)`.
    """
    # Constants
    MIN_PATH_LENGTH = 10
    MAX_PATH_LENGTH = 100
    STEP = 10
    MAX_GENERATIONS = 100
    MAP_LENGTH = 15
    MAP_WIDTH = 20

    # Prepare adjacency / tile info
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)

    # Build array of path lengths
    path_lengths = np.arange(MIN_PATH_LENGTH, MAX_PATH_LENGTH + 1, STEP)
    num_lengths = len(path_lengths)

    # Pre‐allocate array for "generations to converge"; initialize to np.nan
    # Shape: (num_lengths, sample_size)
    generations_to_converge = np.full((num_lengths, sample_size), np.nan, dtype=float)

    # --- Run experiments ---
    for idx, path_length in enumerate(path_lengths):
        for sample_idx in range(sample_size):
            print(
                f"Generating agents for path length {path_length} (run {sample_idx + 1}/{sample_size})"
            )
            if not qd:
                env = WFCWrapper(
                    map_length=MAP_LENGTH,
                    map_width=MAP_WIDTH,
                    tile_symbols=tile_symbols,
                    adjacency_bool=adjacency_bool,
                    num_tiles=num_tiles,
                    tile_to_index=tile_to_index,
                    reward=partial(
                        binary_reward, target_path_length=path_length, hard=hard
                    ),
                    deterministic=True,
                )
            else:
                env = WFCWrapper(
                    map_length=MAP_LENGTH,
                    map_width=MAP_WIDTH,
                    tile_symbols=tile_symbols,
                    adjacency_bool=adjacency_bool,
                    num_tiles=num_tiles,
                    tile_to_index=tile_to_index,
                    reward=partial(
                        binary_reward, target_path_length=path_length, hard=hard
                    ),
                    deterministic=True,
                    qd_function=binary_percent_water,
                )

            start_time = time.time()
            _, best_agent, generations, best_agent_rewards, median_agent_rewards = (
                evolve(
                    env=env,
                    generations=MAX_GENERATIONS,
                    population_size=evolution_hyperparameters["population_size"],
                    number_of_actions_mutated_mean=evolution_hyperparameters[
                        "number_of_actions_mutated_mean"
                    ],
                    number_of_actions_mutated_standard_deviation=evolution_hyperparameters[
                        "number_of_actions_mutated_standard_deviation"
                    ],
                    action_noise_standard_deviation=evolution_hyperparameters[
                        "action_noise_standard_deviation"
                    ],
                    survival_rate=evolution_hyperparameters["survival_rate"],
                )
            )
            elapsed = time.time() - start_time
            print(f"Evolution finished in {elapsed:.2f} seconds.")

            # Save performance curves per‐run
            qd_prefix = "qd_" if qd else ""
            qd_label = ", QD" if qd else ""
            hard_prefix = "hard_" if hard else ""
            hard_label = ", hard" if hard else ""
            x_axis = np.arange(1, len(median_agent_rewards) + 1)
            plt.plot(x_axis, best_agent_rewards, label="Best Agent")
            plt.plot(x_axis, median_agent_rewards, label="Median Agent")
            plt.legend()
            plt.title(
                f"Performance (path={path_length}, run={sample_idx}{qd_label}{hard_label})"
            )
            plt.xlabel("Generation")
            plt.ylabel("Reward")
            plt.savefig(
                f"{FIGURES_DIRECTORY}/{qd_prefix}binary{path_length}_{hard_prefix}performance_{sample_idx}.png"
            )
            plt.close()

            # Record generations‐to‐converge or leave as NaN if it never converged
            if best_agent.info.get("achieved_max_reward", False):
                generations_to_converge[idx, sample_idx] = generations

    # Count how many runs actually converged at each path length
    number_converged = np.sum(~np.isnan(generations_to_converge), axis=1)
    convergence_fraction = number_converged / sample_size

    # Determine which path lengths have at least one convergence
    valid = number_converged > 0
    data_valid = generations_to_converge  # full array, NaNs where no convergence

    # Compute mean generations ignoring NaNs
    mean_generations = np.nanmean(data_valid, axis=1)

    # Compute standard deviation and standard errors
    std_dev = np.nanstd(data_valid, axis=1, ddof=0)
    standard_errors = np.zeros_like(std_dev)
    standard_errors[valid] = std_dev[valid] / np.sqrt(number_converged[valid])

    # --- Plot mean ± SEM and convergence fraction with twin axes ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Error‐bar line for mean generations (only where valid)
    ax1.errorbar(
        path_lengths[valid],
        mean_generations[valid],
        yerr=standard_errors[valid],
        fmt="o-",
        capsize=4,
        label="Mean generations to converge",
    )
    ax1.set_xlabel("Desired Path Length")
    ax1.set_ylabel("Mean Generations to Converge")

    # Bar chart for convergence fraction (only where valid)
    bar_width = STEP * 0.8
    ax2.bar(
        path_lengths[valid],
        convergence_fraction[valid],
        width=bar_width,
        alpha=0.3,
        label="Fraction converged",
        align="center",
    )
    ax2.set_ylabel("Fraction of Runs Converged")

    # Ensure the x‐axis shows ticks for all desired path‐lengths
    ax1.set_xticks(path_lengths)

    # Title and combined legend
    qd_label = " (QD)" if qd else ""
    hard_label = " HARD" if hard else ""
    ax1.set_title(f"Convergence Behavior vs. Desired Path Length{qd_label}{hard_label}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    plt.savefig(f"{FIGURES_DIRECTORY}/{qd_prefix}{hard_prefix}convergence_over_path.png")
    plt.close()

def combo_convergence_over_path_lengths(
    sample_size: int,
    evolution_hyperparameters: dict[str, Any],
    qd: bool,
    hard: bool = False,
) -> None:
    """
    Line plot of generations to converge using CombinedReward
    over various path lengths, and bar chart for convergence fraction.
    """
    MIN_PATH_LENGTH = 10
    MAX_PATH_LENGTH = 100
    STEP = 10
    MAX_GENERATIONS = 100
    MAP_LENGTH = 15
    MAP_WIDTH = 20

    from biome_adjacency_rules import create_adjacency_matrix
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)
    path_lengths = np.arange(MIN_PATH_LENGTH, MAX_PATH_LENGTH + 1, STEP)
    generations_to_converge = np.full((len(path_lengths), sample_size), np.nan)

    for idx, path_length in enumerate(path_lengths):
        for sample_idx in range(sample_size):
            print(f"[COMBO] Path {path_length}, Run {sample_idx+1}/{sample_size}")
            reward_fn = CombinedReward([
                partial(binary_reward, target_path_length=path_length, hard=hard),
                pond_reward,
                river_reward
            ])

            env = WFCWrapper(
                map_length=MAP_LENGTH,
                map_width=MAP_WIDTH,
                tile_symbols=tile_symbols,
                adjacency_bool=adjacency_bool,
                num_tiles=num_tiles,
                tile_to_index=tile_to_index,
                reward=reward_fn,
                deterministic=True,
                qd_function=binary_percent_water if qd else None,
            )

            _, best_agent, generations, best_rewards, median_rewards = evolve(
                env=env,
                generations=MAX_GENERATIONS,
                population_size=evolution_hyperparameters["population_size"],
                number_of_actions_mutated_mean=evolution_hyperparameters["number_of_actions_mutated_mean"],
                number_of_actions_mutated_standard_deviation=evolution_hyperparameters["number_of_actions_mutated_standard_deviation"],
                action_noise_standard_deviation=evolution_hyperparameters["action_noise_standard_deviation"],
                survival_rate=evolution_hyperparameters["survival_rate"],
            )

            if best_agent.info.get("achieved_max_reward", False):
                generations_to_converge[idx, sample_idx] = generations

            plt.plot(best_rewards, label="Best Agent")
            plt.plot(median_rewards, label="Median Agent")
            plt.title(f"Combined Reward Performance (path={path_length}, run={sample_idx})")
            plt.xlabel("Generation")
            plt.ylabel("Reward")
            plt.legend()
            plt.savefig(f"{FIGURES_DIRECTORY}/combo_perf_{path_length}_run{sample_idx}.png")
            plt.close()

    num_converged = np.sum(~np.isnan(generations_to_converge), axis=1)
    fraction_converged = num_converged / sample_size
    mean_gens = np.nanmean(generations_to_converge, axis=1)
    std_err = np.nanstd(generations_to_converge, axis=1, ddof=0)
    std_err[num_converged > 0] /= np.sqrt(num_converged[num_converged > 0])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    valid = num_converged > 0

    ax1.errorbar(path_lengths[valid], mean_gens[valid], yerr=std_err[valid], fmt="o-", capsize=4, label="Mean Generations to Converge")
    ax2.bar(path_lengths[valid], fraction_converged[valid], width=STEP * 0.8, alpha=0.3, label="Fraction Converged")

    ax1.set_xlabel("Desired Path Length")
    ax1.set_ylabel("Mean Generations to Converge")
    ax2.set_ylabel("Fraction of Runs Converged")
    ax1.set_xticks(path_lengths)
    ax1.set_title("Convergence vs Path Length (Combined Reward)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIRECTORY}/combo_convergence_over_path.png")
    plt.close()

def collect_convergence_data(task_name: str, reward_fn, hyperparams, qd=False, runs=20):
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)

    MAP_LENGTH = 15
    MAP_WIDTH = 20
    generations_list = []

    for i in range(runs):
        print(f"[{task_name}] Run {i+1}/{runs}")
        env = WFCWrapper(
            map_length=MAP_LENGTH,
            map_width=MAP_WIDTH,
            tile_symbols=tile_symbols,
            adjacency_bool=adjacency_bool,
            num_tiles=num_tiles,
            tile_to_index=tile_to_index,
            reward=reward_fn,
            deterministic=True,
            qd_function=binary_percent_water if qd else None,
        )

        _, best_agent, generations, _, _ = evolve(
            env=env,
            generations=100,
            population_size=hyperparams["population_size"],
            number_of_actions_mutated_mean=hyperparams["number_of_actions_mutated_mean"],
            number_of_actions_mutated_standard_deviation=hyperparams["number_of_actions_mutated_standard_deviation"],
            action_noise_standard_deviation=hyperparams["action_noise_standard_deviation"],
            survival_rate=hyperparams["survival_rate"],
        )

        if best_agent.info.get("achieved_max_reward", False):
            generations_list.append(generations)

    return generations_list


def plot_avg_task_convergence(hyperparams, qd=False):
    task_info = {
        "Pond": pond_reward,
        "River": river_reward,
        "Grass": grass_reward,
        "Hill": hill_reward,
    }

    means = []
    errors = []
    labels = []

    for label, reward_fn in task_info.items():
        gens = collect_convergence_data(label, reward_fn, hyperparams, qd=qd)
        if gens:
            labels.append(label)
            means.append(np.mean(gens))
            errors.append(np.std(gens) / np.sqrt(len(gens)))
        else:
            print(f"[WARNING] No converged runs for {label}.")

    plt.figure(figsize=(10, 5))
    plt.bar(labels, means, yerr=errors, capsize=5, color="mediumpurple")
    plt.title("Mean Generations to Converge per Biome (20 runs)")
    plt.ylabel("Avg. Generations")
    plt.xlabel("Biome")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIRECTORY}/mean_convergence_bar_chart.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plotting WFC Results"
    )
    parser.add_argument(
        "--load-hyperparameters",
        type=str,
        default=None,
        help="Path to a YAML file containing hyperparameters to load.",
    )

    parser.add_argument('--qd', action='store_true', help='Use QD variant of evolution')

    args = parser.parse_args()
    
    if args.load_hyperparameters:
        # --- Load Hyperparameters and Run Evolution ---
        print(f"Loading hyperparameters from: {args.load_hyperparameters}")
        try:
            with open(args.load_hyperparameters, "r") as f:
                hyperparams = yaml.safe_load(f)
            print("Successfully loaded hyperparameters:", hyperparams)


        except FileNotFoundError:
            print(
                f"Error: Hyperparameter file not found at {args.load_hyperparameters}"
            )
            exit(1)
        except Exception as e:
            print(f"Error loading or using hyperparameters: {e}")
            exit(1)

    # ---- BINARY ----
    start_time = time.time()
    binary_convergence_over_path_lengths(20, hyperparams, args.qd)
    elapsed = time.time() - start_time
    print(f"Plotting finished in {elapsed:.2f} seconds.")

    start_time = time.time()
    binary_convergence_over_path_lengths(20, hyperparams, args.qd, True)
    elapsed = time.time() - start_time
    print(f"Plotting finished in {elapsed:.2f} seconds.")

    # ---- COMBO ----
    start_time = time.time()
    combo_convergence_over_path_lengths(20, hyperparams, args.qd)
    print(f"[Combo] Plotting finished in {time.time() - start_time:.2f} seconds.")

    # ---- SUMMARY BAR CHART ----
    plot_avg_task_convergence(hyperparams, args.qd)

