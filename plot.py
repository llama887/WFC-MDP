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

from biome_adjacency_rules import create_adjacency_matrix
from evolution import evolve
from tasks.binary_task import binary_percent_water, binary_reward
from tasks.grass_task import grass_reward
from tasks.pond_task import pond_reward
from tasks.river_task import river_reward
from wfc_env import CombinedReward, WFCWrapper

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
    """
    Load binary convergence CSV, compute mean+SEM and fraction, and plot twin‐axis.
    """
    data_frame = pd.read_csv(csv_file_path)
    valid_frame = data_frame.dropna(subset=["generations_to_converge"])

    statistics = (
        valid_frame.groupby("desired_path_length")["generations_to_converge"]
        .agg(mean_generation="mean", standard_deviation="std", count="count")
        .reset_index()
    )
    statistics["standard_error"] = statistics["standard_deviation"] / np.sqrt(
        statistics["count"]
    )
    total_runs = int(data_frame["run_index"].max())
    statistics["fraction_converged"] = statistics["count"] / total_runs

    figure, axis_left = plt.subplots(figsize=(8, 5))
    axis_right = axis_left.twinx()

    axis_left.errorbar(
        statistics["desired_path_length"],
        statistics["mean_generation"],
        yerr=statistics["standard_error"],
        fmt="o-",
        capsize=4,
        label="Mean generations to converge",
    )
    axis_right.bar(
        statistics["desired_path_length"],
        statistics["fraction_converged"],
        width=(
            statistics["desired_path_length"].iloc[1]
            - statistics["desired_path_length"].iloc[0]
        )
        * 0.8,
        alpha=0.3,
        label="Fraction converged",
        align="center",
    )

    axis_left.set_xlabel("Desired Path Length")
    axis_left.set_ylabel("Mean Generations to Converge")
    axis_right.set_ylabel("Fraction of Runs Converged")
    axis_left.set_xticks(statistics["desired_path_length"])

    title = "Convergence Behavior vs Desired Path Length"
    if use_quality_diversity:
        title += " (QD)"
    if use_hard_variant:
        title += " HARD"
    axis_left.set_title(title)

    handles_left, labels_left = axis_left.get_legend_handles_labels()
    handles_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(
        handles_left + handles_right, labels_left + labels_right, loc="upper left"
    )

    figure.tight_layout()
    if output_png_path is None:
        prefix = f"{'qd_' if use_quality_diversity else ''}{'hard_' if use_hard_variant else ''}{genotype_dimensions}d"
        output_png_path = os.path.join(
            FIGURES_DIRECTORY, f"{prefix}binary_convergence_over_path.png"
        )
    figure.savefig(output_png_path)
    plt.close(figure)
    print(f"Saved binary convergence plot to {output_png_path}")


def plot_combo_convergence_from_csv(
    csv_file_path: str,
    use_quality_diversity: bool,
    second_task: str,
    use_hard_variant: bool = False,
    output_png_path: str | None = None,
    genotype_dimensions: Literal[1, 2] = 1,
) -> None:
    """
    Load combo convergence CSV, compute mean+SEM and fraction, and plot twin‐axis.
    """
    data_frame = pd.read_csv(csv_file_path)
    valid_frame = data_frame.dropna(subset=["generations_to_converge"])

    statistics = (
        valid_frame.groupby("desired_path_length")["generations_to_converge"]
        .agg(mean_generation="mean", standard_deviation="std", count="count")
        .reset_index()
    )
    statistics["standard_error"] = statistics["standard_deviation"] / np.sqrt(
        statistics["count"]
    )
    total_runs = int(data_frame["run_index"].max())
    statistics["fraction_converged"] = statistics["count"] / total_runs

    figure, axis_left = plt.subplots(figsize=(8, 5))
    axis_right = axis_left.twinx()

    axis_left.errorbar(
        statistics["desired_path_length"],
        statistics["mean_generation"],
        yerr=statistics["standard_error"],
        fmt="o-",
        capsize=4,
        label="Mean generations to converge",
    )
    axis_right.bar(
        statistics["desired_path_length"],
        statistics["fraction_converged"],
        width=(
            statistics["desired_path_length"].iloc[1]
            - statistics["desired_path_length"].iloc[0]
        )
        * 0.8,
        alpha=0.3,
        label="Fraction converged",
        align="center",
    )

    axis_left.set_xlabel("Desired Path Length")
    axis_left.set_ylabel("Mean Generations to Converge")
    axis_right.set_ylabel("Fraction of Runs Converged")
    axis_left.set_xticks(statistics["desired_path_length"])

    title = f"Combined Binary + {second_task.capitalize()} Convergence vs Desired Path Length"
    if use_quality_diversity:
        title += " (QD)"
    if use_hard_variant:
        title += " HARD"
    axis_left.set_title(title)

    handles_left, labels_left = axis_left.get_legend_handles_labels()
    handles_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(
        handles_left + handles_right, labels_left + labels_right, loc="upper left"
    )

    figure.tight_layout()
    if output_png_path is None:
        prefix = f"{'qd_' if use_quality_diversity else ''}{'hard_' if use_hard_variant else ''}{genotype_dimensions}{second_task}_combo_"
        output_png_path = os.path.join(
            FIGURES_DIRECTORY, f"{prefix}convergence_over_path.png"
        )
    figure.savefig(output_png_path)
    plt.close(figure)
    print(f"Saved combo convergence plot to {output_png_path}")


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
    elif args.task == "biome":
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
