import argparse
import os
from typing import Any, Callable, Literal

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from functools import partial

import matplotlib.pyplot as plt
import yaml

from assets.biome_adjacency_rules import create_adjacency_matrix
from core.evolution import evolve
from core.wfc_env import CombinedReward, WFCWrapper
from tasks.binary_task import binary_percent_water, binary_reward
from tasks.grass_task import grass_reward
from tasks.pond_task import pond_reward
from tasks.river_task import river_reward

FIGURES_DIRECTORY = "figures"
os.makedirs(FIGURES_DIRECTORY, exist_ok=True)


def collect_single_task_convergence_data(
    task_name: str,
    reward_function: Callable[..., float],
    evolution_hyperparameters: dict[str, Any],
    use_quality_diversity: bool = False,
    runs: int = 20,
    genotype_dimensions: Literal[1, 2] = 1,
) -> list[float]:
    """
    Run `runs` independent evolutions for a single biome-task (pond/river/grass),
    returning the list of generations-to-converge (only those that hit max reward).
    """
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    map_length = 15
    map_width = 20

    generations_list: list[float] = []
    for run_index in range(runs):
        print(f"[{task_name} {str(genotype_dimensions)}d] Run {run_index + 1}/{runs}")
        environment = WFCWrapper(
            map_length=map_length,
            map_width=map_width,
            tile_symbols=tile_symbols,
            adjacency_bool=adjacency_bool,
            num_tiles=len(tile_symbols),
            tile_to_index=tile_to_index,
            reward=reward_function,
            deterministic=True,
            qd_function=binary_percent_water if use_quality_diversity else None,
        )
        _, best_agent, generations, _, _ = evolve(
            env=environment,
            generations=100,
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
            genotype_representation=str(genotype_dimensions) + "d",
        )
        if best_agent.info.get("achieved_max_reward", False):
            generations_list.append(generations)
    return generations_list


def collect_binary_convergence(
    sample_size: int,
    evolution_hyperparameters: dict[str, Any],
    use_quality_diversity: bool = False,
    use_hard_variant: bool = False,
    genotype_dimensions: Literal[1, 2] = 1,
) -> str:
    """
    Run binary-path-length experiments, dump raw generations-to-converge per run+path
    into a CSV, and return the CSV file path.
    """
    minimum_path_length = 10
    maximum_path_length = 100
    step_size = 10
    map_length = 15
    map_width = 20

    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    path_lengths = np.arange(minimum_path_length, maximum_path_length + 1, step_size)

    data_rows: list[dict[str, Any]] = []
    for path_index, desired_path_length in enumerate(path_lengths):
        for run_index in range(sample_size):
            print(
                f"[BINARY] Path {desired_path_length}, Run {run_index + 1}/{sample_size}"
            )
            reward_callable = partial(
                binary_reward,
                target_path_length=desired_path_length,
                hard=use_hard_variant,
            )
<<<<<<< HEAD
            elapsed = time.time() - start_time
            print(f"Evolution finished in {elapsed:.2f} seconds.")

            # Save performance curves per‐run
            qd_prefix = "qd_" if qd else ""
            qd_label = ", QD" if qd else ""
            hard_prefix = "hard_" if hard else ""
            hard_label = ", hard" if hard else ""

            if not os.path.exists(
                f"{AGENT_DIR}/{qd_prefix}{hard_prefix}_binary{path_length}_agent.pkl"
            ) and best_agent.info.get("achieved_max_reward", False):
                with open(
                    f"{AGENT_DIR}/{qd_prefix}{hard_prefix}_binary{path_length}_agent.pkl",
                    "wb",
                ) as f:
                    pickle.dump(best_agent, f)

            # x_axis = np.arange(1, len(median_agent_rewards) + 1)
            # plt.plot(x_axis, best_agent_rewards, label="Best Agent")
            # plt.plot(x_axis, median_agent_rewards, label="Median Agent")
            # plt.legend()
            # plt.title(
            #     f"Performance (path={path_length}, run={sample_idx}{qd_label}{hard_label})"
            # )
            # plt.xlabel("Generation")
            # plt.ylabel("Reward")
            # plt.savefig(
            #     f"{FIGURES_DIRECTORY}/{qd_prefix}binary{path_length}_{hard_prefix}performance_{sample_idx}.png"
            # )
            # plt.close()

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
    plt.savefig(
        f"{FIGURES_DIRECTORY}/{qd_prefix}{hard_prefix}convergence_over_path.png"
    )
    plt.close()


def combo_convergence_over_path_lengths(
    sample_size: int,
    evolution_hyperparameters: dict[str, Any],
    qd: bool,
    second_task: str,
    hard: bool = False,
) -> None:
    """
    Line plot of generations to converge using CombinedReward
    over various path lengths, and bar chart for convergence fraction.
    """
    AGENT_DIR = f"agents_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(AGENT_DIR, exist_ok=True)
    MIN_PATH_LENGTH = 10
    MAX_PATH_LENGTH = 100
    STEP = 10
    MAX_GENERATIONS = 100
    MAP_LENGTH = 15
    MAP_WIDTH = 20

    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)
    path_lengths = np.arange(MIN_PATH_LENGTH, MAX_PATH_LENGTH + 1, STEP)
    generations_to_converge = np.full((len(path_lengths), sample_size), np.nan)

    task_rewards = {"river": river_reward, "pond": pond_reward, "grass": grass_reward}
    second_reward = task_rewards.get(second_task)

    for idx, path_length in enumerate(path_lengths):
        for sample_idx in range(sample_size):
            print(f"[COMBO] Path {path_length}, Run {sample_idx + 1}/{sample_size}")
            reward_fn = CombinedReward(
                [
                    partial(binary_reward, target_path_length=path_length, hard=hard),
                    second_reward,
                ]
            )

            env = WFCWrapper(
                map_length=MAP_LENGTH,
                map_width=MAP_WIDTH,
=======
            environment = WFCWrapper(
                map_length=map_length,
                map_width=map_width,
>>>>>>> 7b60d27 (plotting supports 2d genotype)
                tile_symbols=tile_symbols,
                adjacency_bool=adjacency_bool,
                num_tiles=len(tile_symbols),
                tile_to_index=tile_to_index,
                reward=reward_callable,
                deterministic=True,
                qd_function=binary_percent_water if use_quality_diversity else None,
            )
            _, best_agent, generations, _, _ = evolve(
                env=environment,
                generations=100,
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
                genotype_representation=str(genotype_dimensions) + "d",
            )
            if best_agent.info.get("achieved_max_reward", False):
                gens = generations
            else:
                gens = float("nan")
            data_rows.append(
                {
                    "desired_path_length": desired_path_length,
                    "run_index": run_index + 1,
                    "generations_to_converge": gens,
                }
            )

    prefix = f"{'qd_' if use_quality_diversity else ''}{'hard_' if use_hard_variant else ''}{str(genotype_dimensions)}d_"
    csv_filename = f"{prefix}binary_convergence_over_path.csv"
    csv_path = os.path.join(FIGURES_DIRECTORY, csv_filename)
    pd.DataFrame(data_rows).to_csv(csv_path, index=False)
    print(f"Saved binary convergence raw data to {csv_path}")
    return csv_path


def collect_combo_convergence(
    sample_size: int,
    evolution_hyperparameters: dict[str, Any],
    use_quality_diversity: bool,
    second_task: str,
    use_hard_variant: bool = False,
    genotype_dimensions: Literal[1, 2] = 1,
) -> str:
    """
    Run combined binary+biome experiments, dump raw generations-to-converge per run+path
    into a CSV, and return the CSV file path.
    """
    minimum_path_length = 10
    maximum_path_length = 100
    step_size = 10
    map_length = 15
    map_width = 20

    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    path_lengths = np.arange(minimum_path_length, maximum_path_length + 1, step_size)

    biome_reward_map = {
        "river": river_reward,
        "pond": pond_reward,
        "grass": grass_reward,
    }
    second_reward = biome_reward_map[second_task]

    data_rows: list[dict[str, Any]] = []
    for path_index, desired_path_length in enumerate(path_lengths):
        for run_index in range(sample_size):
            print(
                f"[COMBO] Path {desired_path_length}, Run {run_index + 1}/{sample_size}"
            )
            reward_callable = CombinedReward(
                [
                    partial(
                        binary_reward,
                        target_path_length=desired_path_length,
                        hard=use_hard_variant,
                    ),
                    second_reward,
                ]
            )
            environment = WFCWrapper(
                map_length=map_length,
                map_width=map_width,
                tile_symbols=tile_symbols,
                adjacency_bool=adjacency_bool,
                num_tiles=len(tile_symbols),
                tile_to_index=tile_to_index,
                reward=reward_callable,
                deterministic=True,
                qd_function=binary_percent_water if use_quality_diversity else None,
            )
            _, best_agent, generations, _, _ = evolve(
                env=environment,
                generations=100,
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
                genotype_representation=str(genotype_dimensions) + "d",
            )
            if best_agent.info.get("achieved_max_reward", False):
                gens = generations
            else:
                gens = float("nan")
            data_rows.append(
                {
                    "desired_path_length": desired_path_length,
                    "run_index": run_index + 1,
                    "generations_to_converge": gens,
                }
            )

    prefix = f"{'qd_' if use_quality_diversity else ''}{'hard_' if use_hard_variant else ''}{str(genotype_dimensions)}d_"
    csv_filename = f"{prefix}convergence_over_path.csv"
    csv_path = os.path.join(FIGURES_DIRECTORY, csv_filename)
    pd.DataFrame(data_rows).to_csv(csv_path, index=False)
    print(f"Saved combo convergence raw data to {csv_path}")
    return csv_path


def collect_average_biome_convergence_data(
    evolution_hyperparameters: dict[str, Any],
    use_quality_diversity: bool = False,
    runs: int = 20,
    genotype_dimensions: Literal[1, 2] = 1,
) -> str:
    """
    Collect raw convergence generations for each biome task, save CSV, return its path.
    """
    data_rows: list[dict[str, Any]] = []
    for biome_name, reward_function in [
        ("Pond", pond_reward),
        ("River", river_reward),
        ("Grass", grass_reward),
    ]:
        generations_list = collect_single_task_convergence_data(
            biome_name,
            reward_function,
            evolution_hyperparameters,
            use_quality_diversity=use_quality_diversity,
            runs=runs,
        )
        for run_index, generations in enumerate(generations_list, start=1):
            data_rows.append(
                {
                    "biome": biome_name,
                    "run_index": run_index,
                    "generations_to_converge": generations,
                }
            )
    prefix = f"{'qd_' if use_quality_diversity else ''}{str(genotype_dimensions)}d_"
    csv_filename = f"{prefix}biome_average_convergence.csv"
    csv_path = os.path.join(FIGURES_DIRECTORY, csv_filename)
    pd.DataFrame(data_rows).to_csv(csv_path, index=False)
    print(f"Saved biome average convergence data to {csv_path}")
    return csv_path

def plot_binary_convergence_from_csv(
    csv_file_path: str,
    use_quality_diversity: bool = False,
    use_hard_variant: bool = False,
    output_png_path: str | None = None,
    genotype_dimensions: Literal[1, 2] = 1,
) -> None:
    data_frame = pd.read_csv(csv_file_path)
    valid_frame = data_frame.dropna(subset=["generations_to_converge"])
    statistics = (
        valid_frame
        .groupby("desired_path_length")["generations_to_converge"]
        .agg(
            mean_generation="mean",
            standard_deviation="std",
            count="count"
        )
        .reset_index()
    )
    statistics["standard_error"] = statistics["standard_deviation"] / np.sqrt(statistics["count"])
    total_runs = int(data_frame["run_index"].max())
    statistics["fraction_converged"] = statistics["count"] / total_runs

    fig, ax_left = plt.subplots(figsize=(8, 5))
    ax_right = ax_left.twinx()

    # constants
    X_MIN, X_MAX, STEP = 0, 100, 10
    BAR_WIDTH = STEP * 0.8

    # 1) plot the mean ± stderr
    ax_left.errorbar(
        statistics["desired_path_length"],
        statistics["mean_generation"],
        yerr=statistics["standard_error"],
        fmt="o-",
        capsize=4,
        label="Mean generations to converge"
    )

    # annotate each mean point
    for x, y in zip(statistics["desired_path_length"], statistics["mean_generation"]):
        ax_left.text(x, y, f"{y:.1f}", ha="center", va="bottom")

    # 2) plot the fraction‐converged bars
    ax_right.bar(
        statistics["desired_path_length"],
        statistics["fraction_converged"],
        width=BAR_WIDTH,
        alpha=0.3,
        label="Fraction converged",
        align="center"
    )

    # fix x‐axis and right‐axis limits/ticks
    ax_left.set_xticks(np.arange(X_MIN, X_MAX + 1, STEP))
    ax_left.set_xlim(X_MIN, X_MAX)
    ax_right.set_ylim(0, 1)

    ax_left.set_xlabel("Desired Path Length")
    ax_left.set_ylabel("Mean Generations to Converge")
    ax_right.set_ylabel("Fraction of Runs Converged")

    title = "Convergence Behavior vs Desired Path Length"
    if use_quality_diversity: title += " (QD)"
    if use_hard_variant:       title += " HARD"
    ax_left.set_title(title)

    # combine legends
    hL, lL = ax_left.get_legend_handles_labels()
    hR, lR = ax_right.get_legend_handles_labels()
    ax_left.legend(hL + hR, lL + lR, loc="upper left")

    fig.tight_layout()
    if output_png_path is None:
        prefix = f"{'qd_' if use_quality_diversity else ''}{'hard_' if use_hard_variant else ''}{genotype_dimensions}d_"
        output_png_path = os.path.join(FIGURES_DIRECTORY, f"{prefix}binary_convergence_over_path.png")
    fig.savefig(output_png_path)
    plt.close(fig)
    print(f"Saved binary convergence plot to {output_png_path}")


def plot_combo_convergence_from_csv(
    csv_file_path: str,
    use_quality_diversity: bool,
    second_task: str,
    use_hard_variant: bool = False,
    output_png_path: str | None = None,
    genotype_dimensions: Literal[1, 2] = 1,
) -> None:
    data_frame = pd.read_csv(csv_file_path)
    valid_frame = data_frame.dropna(subset=["generations_to_converge"])
    statistics = (
        valid_frame
        .groupby("desired_path_length")["generations_to_converge"]
        .agg(
            mean_generation="mean",
            standard_deviation="std",
            count="count"
        )
        .reset_index()
    )
    statistics["standard_error"] = statistics["standard_deviation"] / np.sqrt(statistics["count"])
    total_runs = int(data_frame["run_index"].max())
    statistics["fraction_converged"] = statistics["count"] / total_runs

    fig, ax_left = plt.subplots(figsize=(8, 5))
    ax_right = ax_left.twinx()

    # constants
    X_MIN, X_MAX, STEP = 0, 100, 10
    BAR_WIDTH = STEP * 0.8

    ax_left.errorbar(
        statistics["desired_path_length"],
        statistics["mean_generation"],
        yerr=statistics["standard_error"],
        fmt="o-",
        capsize=4,
        label="Mean generations to converge"
    )

    # annotate means
    for x, y in zip(statistics["desired_path_length"], statistics["mean_generation"]):
        ax_left.text(x, y, f"{y:.1f}", ha="center", va="bottom")

    ax_right.bar(
        statistics["desired_path_length"],
        statistics["fraction_converged"],
        width=BAR_WIDTH,
        alpha=0.3,
        label="Fraction converged",
        align="center"
    )

    ax_left.set_xticks(np.arange(X_MIN, X_MAX + 1, STEP))
    ax_left.set_xlim(X_MIN, X_MAX)
    ax_right.set_ylim(0, 1)

    ax_left.set_xlabel("Desired Path Length")
    ax_left.set_ylabel("Mean Generations to Converge")
    ax_right.set_ylabel("Fraction of Runs Converged")

    title = f"Combined Binary + {second_task.capitalize()} Convergence vs Desired Path Length"
    if use_quality_diversity: title += " (QD)"
    if use_hard_variant:       title += " HARD"
    ax_left.set_title(title)

    hL, lL = ax_left.get_legend_handles_labels()
    hR, lR = ax_right.get_legend_handles_labels()
    ax_left.legend(hL + hR, lL + lR, loc="upper left")

    fig.tight_layout()
    if output_png_path is None:
        prefix = f"{'qd_' if use_quality_diversity else ''}{'hard_' if use_hard_variant else ''}{genotype_dimensions}d_{second_task}_combo_"
        output_png_path = os.path.join(FIGURES_DIRECTORY, f"{prefix}convergence_over_path.png")
    fig.savefig(output_png_path)
    plt.close(fig)
    print(f"Saved combo convergence plot to {output_png_path}")


<<<<<<< HEAD
def plot_avg_task_convergence(hyperparams, qd=False):
    task_info = {"Pond": pond_reward, "River": river_reward, "Grass": grass_reward}
=======
def plot_average_biome_convergence_from_csv(
    csv_file_path: str,
    output_png_path: str | None = None,
) -> None:
    """
    Load biome-average convergence CSV, compute mean+SEM, and plot bar chart.
    """
    data_frame = pd.read_csv(csv_file_path)
    statistics = (
        data_frame.groupby("biome")["generations_to_converge"]
        .agg(mean_generation="mean", standard_deviation="std", count="count")
        .reset_index()
    )
    statistics["standard_error"] = statistics["standard_deviation"] / np.sqrt(
        statistics["count"]
    )
>>>>>>> 7b60d27 (plotting supports 2d genotype)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        statistics["biome"],
        statistics["mean_generation"],
        yerr=statistics["standard_error"],
        capsize=5,
        alpha=0.7,
    )
    axis.set_title("Average Generations to Converge per Biome")
    axis.set_xlabel("Biome")
    axis.set_ylabel("Average Generations to Converge")

    figure.tight_layout()
    if output_png_path is None:
        output_png_path = os.path.join(
            FIGURES_DIRECTORY, "biome_average_convergence.png"
        )
    figure.savefig(output_png_path)
    plt.close(figure)
    print(f"Saved biome average convergence plot to {output_png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect and plot WFC convergence experiments"
    )
    parser.add_argument(
        "--load-hyperparameters",
        type=str,
        required=True,
        help="Path to YAML file containing evolution hyperparameters",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["binary_easy", "binary_hard", "river", "pond", "grass", "biomes"],
        required=True,
        help="Which task to run (binary or biome-combo)",
    )
    parser.add_argument(
        "--combo",
        type=str,
        choices=["easy", "hard"],
<<<<<<< HEAD
        help="means that the task will run in combo with binary, specifies easy or hard",
=======
        default="easy",
        help="For non-binary tasks, whether to use the hard binary variant",
    )
    parser.add_argument(
        "--quality-diversity",
        action="store_true",
        help="Use the quality-diversity variant of evolution",
    )
    parser.add_argument(
        "--genotype-dimensions",
        type=int,
        choices=[1, 2],
        default=1,
        help="The dimensions of the genotype representation. 1d or 2d",
>>>>>>> 7b60d27 (plotting supports 2d genotype)
    )
    args = parser.parse_args()

    # Load hyperparameters
    if not os.path.exists(args.load_hyperparameters):
        print(f"Error: hyperparameters file not found at {args.load_hyperparameters}")
        exit(1)
    with open(args.load_hyperparameters, "r") as yaml_file:
        evolution_hyperparameters = yaml.safe_load(yaml_file)
    print("Loaded hyperparameters:", evolution_hyperparameters)

    # Dispatch binary or combo
    if args.task == "binary_easy":
        csv_path = collect_binary_convergence(
            sample_size=20,
            evolution_hyperparameters=evolution_hyperparameters,
            use_quality_diversity=args.quality_diversity,
            use_hard_variant=False,
            genotype_dimensions=args.genotype_dimensions,
        )
        plot_binary_convergence_from_csv(
            csv_file_path=csv_path,
            use_quality_diversity=args.quality_diversity,
            use_hard_variant=False,
            genotype_dimensions=args.genotype_dimensions,
        )

    elif args.task == "binary_hard":
<<<<<<< HEAD
        start_time = time.time()
        binary_convergence_over_path_lengths(20, hyperparams, args.qd, True)
        elapsed = time.time() - start_time
        print(f"Plotting finished in {elapsed:.2f} seconds.")
    elif args.task == "river" and args.combo == "easy":
        start_time = time.time()
        combo_convergence_over_path_lengths(20, hyperparams, args.qd, "river")
        print(f"[river] Plotting finished in {time.time() - start_time:.2f} seconds.")
    elif args.task == "river" and args.combo == "hard":
        start_time = time.time()
        combo_convergence_over_path_lengths(20, hyperparams, args.qd, "river", True)
        print(f"[river] Plotting finished in {time.time() - start_time:.2f} seconds.")
    elif args.task == "pond" and args.combo == "easy":
        start_time = time.time()
        combo_convergence_over_path_lengths(20, hyperparams, args.qd, "pond")
        print(f"[pond] Plotting finished in {time.time() - start_time:.2f} seconds.")
    elif args.task == "pond" and args.combo == "hard":
        start_time = time.time()
        combo_convergence_over_path_lengths(20, hyperparams, args.qd, "pond", True)
        print(f"[pond] Plotting finished in {time.time() - start_time:.2f} seconds.")
    elif args.task == "grass" and args.combo == "easy":
        start_time = time.time()
        combo_convergence_over_path_lengths(20, hyperparams, args.qd, "grass")
        print(f"[grass] Plotting finished in {time.time() - start_time:.2f} seconds.")
    elif args.task == "grass" and args.combo == "hard":
        start_time = time.time()
        combo_convergence_over_path_lengths(20, hyperparams, args.qd, "grass", True)
        print(f"[grass] Plotting finished in {time.time() - start_time:.2f} seconds.")

    # ---- COMBO ----
    # start_time = time.time()
    # combo_convergence_over_path_lengths(20, hyperparams, second_task="pond", qd=args.qd)
    # print(f"[pond] Plotting finished in {time.time() - start_time:.2f} seconds.")

    # start_time = time.time()
    # combo_convergence_over_path_lengths(
    #     20, hyperparams, second_task="pond", qd=args.qd, hard=True
    # )
    # print(f"[pond] Plotting finished in {time.time() - start_time:.2f} seconds.")

    # start_time = time.time()
    # combo_convergence_over_path_lengths(
    #     20, hyperparams, second_task="river", qd=args.qd
    # )
    # print(f"[river] Plotting finished in {time.time() - start_time:.2f} seconds.")

    # start_time = time.time()
    # combo_convergence_over_path_lengths(
    #     20, hyperparams, second_task="river", qd=args.qd, hard=True
    # )
    # print(f"[river] Plotting finished in {time.time() - start_time:.2f} seconds.")

    # start_time = time.time()
    # combo_convergence_over_path_lengths(
    #     20, hyperparams, second_task="grass", qd=args.qd
    # )
    # print(f"[grass] Plotting finished in {time.time() - start_time:.2f} seconds.")

    # start_time = time.time()
    # combo_convergence_over_path_lengths(
    #     20, hyperparams, second_task="grass", qd=args.qd, hard=True
    # )
    # print(f"[grass] Plotting finished in {time.time() - start_time:.2f} seconds.")

    # ---- SUMMARY BAR CHART ----
    plot_avg_task_convergence(hyperparams, args.qd)
=======
        csv_path = collect_binary_convergence(
            sample_size=20,
            evolution_hyperparameters=evolution_hyperparameters,
            use_quality_diversity=args.quality_diversity,
            use_hard_variant=True,
            genotype_dimensions=args.genotype_dimensions,
        )
        plot_binary_convergence_from_csv(
            csv_file_path=csv_path,
            use_quality_diversity=args.quality_diversity,
            use_hard_variant=True,
            genotype_dimensions=args.genotype_dimensions,
        )
    elif args.task == "biomes":
        # Always also produce the summary bar chart
        summary_csv = collect_average_biome_convergence_data(
            evolution_hyperparameters=evolution_hyperparameters,
            use_quality_diversity=args.quality_diversity,
            runs=20,
            genotype_dimensions=args.genotype_dimensions,
        )
        plot_average_biome_convergence_from_csv(csv_file_path=summary_csv)
    else:
        # river, pond, or grass combo
        use_hard = args.combo == "hard"
        csv_path = collect_combo_convergence(
            sample_size=20,
            evolution_hyperparameters=evolution_hyperparameters,
            use_quality_diversity=args.quality_diversity,
            second_task=args.task,
            use_hard_variant=use_hard,
            genotype_dimensions=args.genotype_dimensions,
        )
        plot_combo_convergence_from_csv(
            csv_file_path=csv_path,
            use_quality_diversity=args.quality_diversity,
            second_task=args.task,
            use_hard_variant=use_hard,
            genotype_dimensions=args.genotype_dimensions,
        )
>>>>>>> 7b60d27 (plotting supports 2d genotype)
