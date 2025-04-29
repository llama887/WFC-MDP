import argparse
import os
import time
from typing import Any

import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt

import numpy as np
import yaml

from biome_adjacency_rules import create_adjacency_matrix
from evolution import evolve
from wfc_env import Task, WFCWrapper

FIGURES_DIRECTORY = "figures"
os.makedirs(FIGURES_DIRECTORY, exist_ok=True)


def binary_convergence_over_path_lengths(
    sample_size: int, evolution_hyperparameters: dict[str, Any], qd: bool
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
    qd : bool
        Determines if to evolve with QD mode. Passed directly into evolve
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
            env = WFCWrapper(
                map_length=MAP_LENGTH,
                map_width=MAP_WIDTH,
                tile_symbols=tile_symbols,
                adjacency_bool=adjacency_bool,
                num_tiles=num_tiles,
                tile_to_index=tile_to_index,
                task=Task.BINARY,
                task_specifications={"target_path_length": int(path_length)},
                deterministic=True,
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
                    qd=qd,
                )
            )
            elapsed = time.time() - start_time
            print(f"Evolution finished in {elapsed:.2f} seconds.")

            # Save performance curves per‐run
            qd_prefix = "qd_" if qd else ""
            qd_label = ", QD" if qd else ""
            x_axis = np.arange(1, len(median_agent_rewards) + 1)
            plt.plot(x_axis, best_agent_rewards, label="Best Agent")
            plt.plot(x_axis, median_agent_rewards, label="Median Agent")
            plt.legend()
            plt.title(f"Performance (path={path_length}, run={sample_idx}{qd_label})")
            plt.xlabel("Generation")
            plt.ylabel("Reward")
            plt.savefig(
                f"{FIGURES_DIRECTORY}/{qd_prefix}binary{path_length}_performance_{sample_idx}.png"
            )
            plt.close()

            # Record generations‐to‐converge or leave as NaN if it never converged
            if best_agent.info.get("achieved_max_reward", False):
                generations_to_converge[idx, sample_idx] = generations


    # Mean generations to converge per path length (preliminary)
    mean_generations = np.nanmean(generations_to_converge, axis=1)
    number_converged = np.sum(~np.isnan(generations_to_converge), axis=1)
    standard_errors = np.nanstd(generations_to_converge, axis=1, ddof=1) / np.sqrt(number_converged)
    convergence_fraction = number_converged / sample_size

    # Mask out path lengths where no runs converged
    valid = number_converged > 0
    mean_generations = mean_generations[valid]
    standard_errors = standard_errors[valid]
    convergence_fraction = convergence_fraction[valid]
    path_lengths = path_lengths[valid]
    

    # --- Plot mean ± SEM and convergence fraction with twin axes ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Error‐bar line for mean generations
    ax1.errorbar(
        path_lengths,
        mean_generations,
        yerr=standard_errors,
        fmt="o-",
        capsize=4,
        label="Mean generations to converge",
    )
    ax1.set_xlabel("Desired Path Length")
    ax1.set_ylabel("Mean Generations to Converge")

    # Bar chart for convergence fraction
    bar_width = STEP * 0.8
    ax2.bar(
        path_lengths,
        convergence_fraction,
        width=bar_width,
        alpha=0.3,
        label="Fraction converged",
        align="center",
    )
    ax2.set_ylabel("Fraction of Runs Converged")

    # Title and combined legend
    qd_label = " (QD)" if qd else ""
    ax1.set_title(f"Convergence Behavior vs. Desired Path Length{qd_label}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    qd_prefix = "qd_" if qd else ""
    fig.tight_layout()
    plt.savefig(f"{FIGURES_DIRECTORY}/{qd_prefix}convergence_over_path.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting WFC Results")
    parser.add_argument(
        "--load-hyperparameters",
        type=str,
        default=None,
        help="Path to a YAML file containing hyperparameters to load.",
    )
    parser.add_argument(
        "--qd",
        action="store_true",
        default=False,
        help="Use QD mode for evolution.",
    )

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

    binary_convergence_over_path_lengths(5, hyperparams, args.qd)
