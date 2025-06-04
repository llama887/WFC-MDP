import argparse
import os
import time
from typing import Any, Callable, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from functools import partial

from assets.biome_adjacency_rules import create_adjacency_matrix
from core.evolution import evolve, CrossOverMethod
from core.wfc_env import CombinedReward, WFCWrapper
from tasks.binary_task import binary_percent_water, binary_reward
from tasks.grass_task import grass_reward
from tasks.pond_task import pond_reward
from tasks.river_task import river_reward

FIGURES_DIRECTORY = "figures"
DEBUG_DIRECTORY = os.path.join(FIGURES_DIRECTORY, "debug")
os.makedirs(FIGURES_DIRECTORY, exist_ok=True)
os.makedirs(DEBUG_DIRECTORY, exist_ok=True)


def _generic_convergence_collector(
    loop_keys: list[Any],
    make_reward_fn: Callable[[Any], Callable[..., float]],
    evolution_hyperparameters: dict[str, Any],
    output_csv_prefix: str,
    use_quality_diversity: bool = False,
    genotype_dimensions: Literal[1, 2] = 1,
    is_biome_only: bool = False,
    sample_size: int = 20,
    debug: bool = False
) -> str:
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    map_length, map_width = 15, 20

    # Prepare cross_over_method enum and patience
    raw_xover = evolution_hyperparameters.get("cross_over_method", "ONE_POINT")
    try:
        xover = CrossOverMethod(raw_xover)
    except (ValueError, TypeError):
        xover = CrossOverMethod(int(raw_xover))
    patience = evolution_hyperparameters.get("patience", 100)

    data_rows: list[dict[str, Any]] = []

    for key in loop_keys:
        for run_index in range(1, sample_size + 1):
            reward_callable = make_reward_fn(key)
            env = WFCWrapper(
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

            # Run evolve and record timing
            if debug:
                start_time = time.time()

            pop, best_agent, generations, best_agent_rewards, mean_agent_rewards = evolve(
                env=env,
                generations=evolution_hyperparameters.get("generations", 100),
                population_size=evolution_hyperparameters["population_size"],
                number_of_actions_mutated_mean=evolution_hyperparameters["number_of_actions_mutated_mean"],
                number_of_actions_mutated_standard_deviation=evolution_hyperparameters["number_of_actions_mutated_standard_deviation"],
                action_noise_standard_deviation=evolution_hyperparameters["action_noise_standard_deviation"],
                survival_rate=evolution_hyperparameters["survival_rate"],
                cross_over_method=xover,
                patience=patience,
                qd=use_quality_diversity,
                genotype_representation=f"{genotype_dimensions}d",
            )

            if debug:
                end_time = time.time()
                print(f"[DEBUG] Key={key}, Run={run_index}: Evolve took {end_time - start_time:.2f} sec, stopped at gen {generations}")

            gens = generations if best_agent.info.get("achieved_max_reward", False) else float("nan")
            row = {"run_index": run_index, "generations_to_converge": gens}
            row["biome" if is_biome_only else "desired_path_length"] = key
            data_rows.append(row)

            # Produce debug plot if requested and if we have reward history
            if debug and best_agent_rewards and mean_agent_rewards:
                x_axis = np.arange(1, len(mean_agent_rewards) + 1)
                plt.figure(figsize=(6, 4))
                plt.plot(x_axis, best_agent_rewards, label="Best Agent")
                plt.plot(x_axis, mean_agent_rewards, label="Mean Agent")
                plt.legend()
                plt.xlabel("Generation")
                plt.ylabel("Reward")
                what_str = f"Biom_{key}" if is_biome_only else f"Path_{key}"
                title = f"Gen vs Reward — {what_str} — Run{run_index} — {genotype_dimensions}d"
                plt.title(title)
                filename = f"debug_{output_csv_prefix}{what_str}_run{run_index}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(DEBUG_DIRECTORY, filename))
                plt.close()

    csv_filename = f"{output_csv_prefix}convergence.csv"
    csv_path = os.path.join(FIGURES_DIRECTORY, csv_filename)
    pd.DataFrame(data_rows).to_csv(csv_path, index=False)
    print(f"Saved raw data to {csv_path}")
    return csv_path


def collect_binary_convergence(sample_size, evolution_hyperparameters, use_quality_diversity=False, use_hard_variant=False, genotype_dimensions=1, debug=False):
    path_lengths = list(np.arange(10, 101, 10))
    prefix = f"{'qd_' if use_quality_diversity else ''}{'hard_' if use_hard_variant else ''}{genotype_dimensions}d_binary_"
    def make_reward(path_len): return partial(binary_reward, target_path_length=path_len, hard=use_hard_variant)
    return _generic_convergence_collector(path_lengths, make_reward, evolution_hyperparameters, prefix, use_quality_diversity, genotype_dimensions, is_biome_only=False, sample_size=sample_size, debug=debug)


def collect_combo_convergence(sample_size, evolution_hyperparameters, use_quality_diversity, second_task, use_hard_variant=False, genotype_dimensions=1, debug=False):
    path_lengths = list(np.arange(10, 101, 10))
    prefix = f"{'qd_' if use_quality_diversity else ''}{'hard_' if use_hard_variant else ''}{genotype_dimensions}d_{second_task}_combo_"
    biome_reward_map = {"river": river_reward, "pond": pond_reward, "grass": grass_reward}
    second_reward = biome_reward_map[second_task]
    def make_reward(path_len): return CombinedReward([partial(binary_reward, target_path_length=path_len, hard=use_hard_variant), second_reward])
    return _generic_convergence_collector(path_lengths, make_reward, evolution_hyperparameters, prefix, use_quality_diversity, genotype_dimensions, is_biome_only=False, sample_size=sample_size, debug=debug)


def collect_average_biome_convergence_data(evolution_hyperparameters, use_quality_diversity=False, runs=20, genotype_dimensions=1, debug=False):
    biomes = ["Pond", "River", "Grass"]
    prefix = f"{'qd_' if use_quality_diversity else ''}{genotype_dimensions}d_biome_average_"
    def make_reward(biome): return {"Pond": pond_reward, "River": river_reward, "Grass": grass_reward}[biome]
    return _generic_convergence_collector(biomes, make_reward, evolution_hyperparameters, prefix, use_quality_diversity, genotype_dimensions, is_biome_only=True, sample_size=runs, debug=debug)


def plot_convergence_from_csv(csv_path: str, output_path: str = None, title: str = "", xlabel: str = "desired_path_length"):
    df = pd.read_csv(csv_path)
    df_valid = df.dropna(subset=["generations_to_converge"])
    stats = df_valid.groupby(xlabel)["generations_to_converge"].agg(["mean", "std", "count"]).reset_index()
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    stats["fraction_converged"] = stats["count"] / df["run_index"].max()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.errorbar(stats[xlabel], stats["mean"], yerr=stats["stderr"], fmt="o-", capsize=4, label="Mean generations")
    for x, y in zip(stats[xlabel], stats["mean"]): ax1.text(x, y, f"{y:.1f}", ha="center", va="bottom")
    ax2.bar(stats[xlabel], stats["fraction_converged"], width=8, alpha=0.3, label="Fraction converged")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Mean Generations")
    ax2.set_ylabel("Fraction Converged")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title(title)
    fig.tight_layout()
    if output_path is None:
        output_path = csv_path.replace(".csv", ".png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_average_biome_convergence_from_csv(csv_file_path: str, output_png_path: str = None):
    df = pd.read_csv(csv_file_path)
    stats = df.groupby("biome")["generations_to_converge"].agg(["mean", "std", "count"]).reset_index()
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(stats["biome"], stats["mean"], yerr=stats["stderr"], capsize=4, alpha=0.7)
    ax.set_xlabel("Biome")
    ax.set_ylabel("Mean Generations to Converge")
    ax.set_title("Average Convergence per Biome")
    fig.tight_layout()
    if output_png_path is None:
        output_png_path = csv_file_path.replace(".csv", ".png")
    fig.savefig(output_png_path)
    plt.close(fig)
    print(f"Saved biome convergence plot to {output_png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and plot WFC convergence data")
    parser.add_argument("--load-hyperparameters", type=str, required=True, help="YAML file with evolution hyperparameters")
    parser.add_argument("--task", type=str, choices=["binary_easy", "binary_hard", "river", "pond", "grass", "biomes"], required=True)
    parser.add_argument("--combo", type=str, choices=["easy", "hard"], default="easy")
    parser.add_argument("--quality-diversity", action="store_true", help="Use the QD variant")
    parser.add_argument("--genotype-dimensions", type=int, choices=[1, 2], default=1)
    parser.add_argument("--debug", action="store_true", help="Save per-run debug plots")
    args = parser.parse_args()

    if not os.path.exists(args.load_hyperparameters):
        print(f"Hyperparameters file not found: {args.load_hyperparameters}")
        exit(1)

    with open(args.load_hyperparameters, "r") as f:
        hyperparams = yaml.safe_load(f)

    if args.task == "binary_easy":
        csv_path = collect_binary_convergence(20, hyperparams, args.quality_diversity, False, args.genotype_dimensions, debug=args.debug)
        plot_convergence_from_csv(csv_path, title="Binary Convergence", xlabel="desired_path_length")
    elif args.task == "binary_hard":
        csv_path = collect_binary_convergence(20, hyperparams, args.quality_diversity, True, args.genotype_dimensions, debug=args.debug)
        plot_convergence_from_csv(csv_path, title="Binary Convergence (HARD)", xlabel="desired_path_length")
    elif args.task == "biomes":
        csv_path = collect_average_biome_convergence_data(hyperparams, args.quality_diversity, 20, args.genotype_dimensions, debug=args.debug)
        plot_average_biome_convergence_from_csv(csv_path)
    else:
        use_hard = args.combo == "hard"
        csv_path = collect_combo_convergence(20, hyperparams, args.quality_diversity, args.task, use_hard, args.genotype_dimensions, debug=args.debug)
        title = f"Combo Convergence: {args.task.capitalize()}" + (" HARD" if use_hard else "")
        plot_convergence_from_csv(csv_path, title=title, xlabel="desired_path_length")
