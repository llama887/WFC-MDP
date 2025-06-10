import math
import argparse
import os
from typing import Any, Callable, Literal
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from functools import partial
import yaml

from assets.biome_adjacency_rules import create_adjacency_matrix
from core.evolution import evolve as evolve_standard, CrossOverMethod
from core.fi2pop import evolve as evolve_constrained, EvolutionMode
from core.mcts import run_mcts_until_complete, resume_mcts_search, MCTS
from core.wfc_env import CombinedReward, WFCWrapper
from tasks.binary_task import binary_percent_water, binary_reward
from tasks.grass_task import grass_reward
from tasks.pond_task import pond_reward
from tasks.river_task import river_reward

import matplotlib.pyplot as plt

DEBUG_DIRECTORY = "debug_plots"

def get_figure_directory(method: str) -> str:
    """Returns the output directory for a given method."""
    return {
        "evolution": "figures_evolution",
        "mcts": "figures_mcts",
        "fi2pop": "figures_fi2pop",
        "baseline": "figures_baseline",
    }.get(method, "figures")


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
    raw_cross_over = evolution_hyperparameters.get("cross_over_method", "ONE_POINT")
    try:
        cross_over_method = CrossOverMethod(raw_cross_over)
    except (ValueError, TypeError):
        cross_over_method = CrossOverMethod(int(raw_cross_over))

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

            pop, best_agent, generations, best_agent_rewards, mean_agent_rewards = evolve_standard(
                env=env,
                generations=evolution_hyperparameters.get("generations", 1000),
                population_size=48,  # Fixed population size
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
                cross_over_method=cross_over_method,
                patience=50,  # Fixed patience
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
    fig_dir = get_figure_directory(evolution_hyperparameters.get("method", "evolution"))
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(fig_dir, csv_filename)
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


# --- REMOVED: collect_mcts_binary_convergence, collect_mcts_combo_convergence, collect_mcts_biome_convergence ---

# --- REMOVED: plot_mcts_convergence_from_csv, plot_mcts_biome_convergence_from_csv ---

# ----------- Constrained EA (FI-2Pop/Baseline) helpers -----------

def collect_constrained_binary_convergence(
    mode: EvolutionMode,
    sample_size: int,
    hyperparams: dict,
    use_hard_variant: bool = False,
    debug: bool = False,
) -> str:
    raw_xover = hyperparams.get("cross_over_method", "ONE_POINT")
    try:
        xover = CrossOverMethod(raw_xover)
    except (ValueError, TypeError):
        xover = CrossOverMethod(int(raw_xover))
    path_lengths = list(np.arange(10, 101, 10))
    prefix = f"{'hard_' if use_hard_variant else ''}binary_"
    def make_reward(path_len):
        return partial(binary_reward, target_path_length=path_len, hard=use_hard_variant)
    data_rows = []
    fig_dir = get_figure_directory(mode.value)
    os.makedirs(fig_dir, exist_ok=True)
    for key in path_lengths:
        for run_idx in range(1, sample_size + 1):
            reward_callable = make_reward(key)
            best_agent, final_gen, _, _ = evolve_constrained(
                mode=mode,
                reward_fn=reward_callable,
                task_args={},
                generations=hyperparams.get("generations", 1000),
                population_size=48,
                number_of_actions_mutated_mean=hyperparams[
                    "number_of_actions_mutated_mean"
                ],
                number_of_actions_mutated_standard_deviation=hyperparams[
                    "number_of_actions_mutated_standard_deviation"
                ],
                action_noise_standard_deviation=hyperparams[
                    "action_noise_standard_deviation"
                ],
                survival_rate=hyperparams["survival_rate"],
                cross_over_method=xover,
                cross_or_mutate_proportion=hyperparams.get(
                    "cross_or_mutate_proportion", 0.7
                ),
                patience=50,
            )
            converged = best_agent and best_agent.reward >= 0.0
            gens_to_converge = final_gen if converged else float("nan")
            row = {"run_index": run_idx, "generations_to_converge": gens_to_converge}
            row["desired_path_length"] = key
            data_rows.append(row)
    csv_filename = f"{prefix}convergence.csv"
    csv_path = os.path.join(fig_dir, csv_filename)
    pd.DataFrame(data_rows).to_csv(csv_path, index=False)
    print(f"Saved raw data to {csv_path}")
    return csv_path

def collect_constrained_combo_convergence(
    mode: EvolutionMode,
    sample_size: int,
    hyperparams: dict,
    second_task: str,
    use_hard_variant: bool = False,
    debug: bool = False,
) -> str:
    raw_xover = hyperparams.get("cross_over_method", "ONE_POINT")
    try:
        xover = CrossOverMethod(raw_xover)
    except (ValueError, TypeError):
        xover = CrossOverMethod(int(raw_xover))
    path_lengths = list(np.arange(10, 101, 10))
    prefix = f"{'hard_' if use_hard_variant else ''}{second_task}_combo_"
    biome_reward_map = {"river": river_reward, "pond": pond_reward, "grass": grass_reward}
    second_reward = biome_reward_map[second_task]
    def make_reward(path_len):
        return CombinedReward([
            partial(binary_reward, target_path_length=path_len, hard=use_hard_variant),
            second_reward
        ])
    data_rows = []
    fig_dir = get_figure_directory(mode.value)
    os.makedirs(fig_dir, exist_ok=True)
    for key in path_lengths:
        for run_idx in range(1, sample_size + 1):
            reward_callable = make_reward(key)
            best_agent, final_gen, _, _ = evolve_constrained(
                mode=mode,
                reward_fn=reward_callable,
                task_args={},
                generations=hyperparams.get("generations", 1000),
                population_size=48,
                number_of_actions_mutated_mean=hyperparams[
                    "number_of_actions_mutated_mean"
                ],
                number_of_actions_mutated_standard_deviation=hyperparams[
                    "number_of_actions_mutated_standard_deviation"
                ],
                action_noise_standard_deviation=hyperparams[
                    "action_noise_standard_deviation"
                ],
                survival_rate=hyperparams["survival_rate"],
                cross_over_method=xover,
                cross_or_mutate_proportion=hyperparams.get(
                    "cross_or_mutate_proportion", 0.7
                ),
                patience=50,
            )
            converged = best_agent and best_agent.reward >= 0.0
            gens_to_converge = final_gen if converged else float("nan")
            row = {"run_index": run_idx, "generations_to_converge": gens_to_converge}
            row["desired_path_length"] = key
            data_rows.append(row)
    csv_filename = f"{prefix}convergence.csv"
    csv_path = os.path.join(fig_dir, csv_filename)
    pd.DataFrame(data_rows).to_csv(csv_path, index=False)
    print(f"Saved raw data to {csv_path}")
    return csv_path

def collect_constrained_biome_convergence(
    mode: EvolutionMode,
    hyperparams: dict,
    runs: int = 20,
    debug: bool = False,
) -> str:
    raw_cross_over = hyperparams.get("cross_over_method", "ONE_POINT")
    try:
        cross_over = CrossOverMethod(raw_cross_over)
    except (ValueError, TypeError):
        cross_over = CrossOverMethod(int(raw_cross_over))
    biomes = ["Pond", "River", "Grass"]
    prefix = "biome_"
    def make_reward(biome):
        return {"Pond": pond_reward, "River": river_reward, "Grass": grass_reward}[biome]
    data_rows = []
    fig_dir = get_figure_directory(mode.value)
    os.makedirs(fig_dir, exist_ok=True)
    for key in biomes:
        for run_idx in range(1, runs + 1):
            reward_callable = make_reward(key)
            best_agent, final_gen, _, _ = evolve_constrained(
                mode=mode,
                reward_fn=reward_callable,
                task_args={},
                generations=hyperparams.get("generations", 1000),
                population_size=48,
                number_of_actions_mutated_mean=hyperparams[
                    "number_of_actions_mutated_mean"
                ],
                number_of_actions_mutated_standard_deviation=hyperparams[
                    "number_of_actions_mutated_standard_deviation"
                ],
                action_noise_standard_deviation=hyperparams[
                    "action_noise_standard_deviation"
                ],
                survival_rate=hyperparams["survival_rate"],
                cross_over_method=cross_over,
                cross_or_mutate_proportion=hyperparams.get(
                    "cross_or_mutate_proportion", 0.7
                ),
                patience=50,
            )
            converged = best_agent and best_agent.reward >= 0.0
            gens_to_converge = final_gen if converged else float("nan")
            row = {"run_index": run_idx, "generations_to_converge": gens_to_converge}
            row["biome"] = key
            data_rows.append(row)
    csv_filename = f"{prefix}convergence.csv"
    csv_path = os.path.join(fig_dir, csv_filename)
    pd.DataFrame(data_rows).to_csv(csv_path, index=False)
    print(f"Saved raw data to {csv_path}")
    return csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and plot WFC convergence data")
    parser.add_argument(
        "--method",
        type=str,
        choices=["evolution", "mcts", "fi2pop", "baseline"],
        required=True,
        help="Method to use for convergence testing",
    )
    parser.add_argument(
        "--load-hyperparameters",
        type=str,
        help="YAML file with evolution hyperparameters (required for evolution, fi2pop, baseline)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["binary_easy", "binary_hard", "river", "pond", "grass", "biomes"],
        required=True,
    )
    parser.add_argument("--combo", type=str, choices=["easy", "hard"], default="easy")
    parser.add_argument(
        "--quality-diversity", action="store_true", help="Use the QD variant (evolution only)"
    )
    parser.add_argument(
        "--genotype-dimensions", type=int, choices=[1, 2], default=1
    )
    parser.add_argument("--debug", action="store_true", help="Save per-run debug plots")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of runs to collect per data point.")
    parser.add_argument("--mcts-iterations", type=int, default=1000, help="Number of MCTS iterations to run per session.")
    args = parser.parse_args()

    # --- Argument Validation ---
    hyperparams = {}
    if args.method != "mcts":
        if not args.load_hyperparameters:
            parser.error(
                f"--load-hyperparameters is required for method '{args.method}'"
            )
        if not os.path.exists(args.load_hyperparameters):
            parser.error(f"Hyperparameters file not found: {args.load_hyperparameters}")
        with open(args.load_hyperparameters, "r") as f:
            hyperparams = yaml.safe_load(f)
    elif args.load_hyperparameters:
        print("Warning: --load-hyperparameters is ignored for method 'mcts'")

    # --- Main Dispatcher ---
    if args.method in ["fi2pop", "baseline"]:
        mode = EvolutionMode.FI2POP if args.method == "fi2pop" else EvolutionMode.BASELINE
        if args.task == "binary_easy":
            csv_path = _generic_constrained_ea_collector(
                mode, list(np.arange(10, 101, 10)), lambda p: partial(binary_reward, target_path_length=p, hard=False),
                hyperparams, "binary_", False, args.sample_size, args.debug
            )
            plot_convergence_from_csv(csv_path, title=f"{mode.value.upper()} Binary Convergence")
        elif args.task == "binary_hard":
            csv_path = _generic_constrained_ea_collector(
                mode, list(np.arange(10, 101, 10)), lambda p: partial(binary_reward, target_path_length=p, hard=True),
                hyperparams, "binary_hard_", False, args.sample_size, args.debug
            )
            plot_convergence_from_csv(csv_path, title=f"{mode.value.upper()} Binary Convergence (HARD)")
        elif args.task == "biomes":
            csv_path = _generic_constrained_ea_collector(
                mode, ["Pond", "River", "Grass"], lambda b: {"Pond": pond_reward, "River": river_reward, "Grass": grass_reward}[b],
                hyperparams, "biome_", True, args.sample_size, args.debug
            )
            plot_average_biome_convergence_from_csv(csv_path)
        else: # Combo tasks
            use_hard = args.combo == "hard"
            biome_map = {"river": river_reward, "pond": pond_reward, "grass": grass_reward}
            second_reward = biome_map[args.task]
            def make_reward(p): return CombinedReward([partial(binary_reward, target_path_length=p, hard=use_hard), second_reward])
            csv_path = _generic_constrained_ea_collector(
                mode, list(np.arange(10, 101, 10)), make_reward,
                hyperparams, f"combo_{args.task}{'_hard' if use_hard else ''}_", False, args.sample_size, args.debug
            )
            title = f"{mode.value.upper()} Combo: {args.task.capitalize()}" + (" HARD" if use_hard else "")
            plot_convergence_from_csv(csv_path, title=title)

    elif args.method == "mcts":
        if args.task == "binary_easy":
            path_lengths = list(np.arange(10, 101, 10))
            def make_reward(p): return partial(binary_reward, target_path_length=p, hard=False)
            csv_path = _resumable_mcts_collector(
                path_lengths, make_reward, "mcts_binary_easy", False, args.sample_size, args.mcts_iterations
            )
            plot_convergence_from_csv(csv_path, title="MCTS Binary Convergence (Easy)", xlabel="desired_path_length", y_label="Mean Iterations")

        elif args.task == "binary_hard":
            path_lengths = list(np.arange(10, 101, 10))
            def make_reward(p): return partial(binary_reward, target_path_length=p, hard=True)
            csv_path = _resumable_mcts_collector(
                path_lengths, make_reward, "mcts_binary_hard", False, args.sample_size, args.mcts_iterations
            )
            plot_convergence_from_csv(csv_path, title="MCTS Binary Convergence (Hard)", xlabel="desired_path_length", y_label="Mean Iterations")

        elif args.task == "biomes":
            biomes = ["river", "pond", "grass"]
            biome_map = {"river": river_reward, "pond": pond_reward, "grass": grass_reward}
            def make_reward(b): return biome_map[b]
            csv_path = _resumable_mcts_collector(
                biomes, make_reward, "mcts_biomes", True, args.sample_size, args.mcts_iterations
            )
            plot_average_biome_convergence_from_csv(csv_path, y_label="Mean Iterations to Converge")

        else:  # Combo tasks
            path_lengths = list(np.arange(10, 101, 10))
            use_hard = args.combo == "hard"
            biome_map = {"river": river_reward, "pond": pond_reward, "grass": grass_reward}
            second_reward = biome_map[args.task]
            def make_reward(p): return CombinedReward([partial(binary_reward, target_path_length=p, hard=use_hard), second_reward])
            
            task_prefix = f"mcts_combo_{args.task}{'_hard' if use_hard else ''}"
            csv_path = _resumable_mcts_collector(
                path_lengths, make_reward, task_prefix, False, args.sample_size, args.mcts_iterations
            )
            title = f"MCTS Combo: {args.task.capitalize()}" + (" (Hard)" if use_hard else "")
            plot_convergence_from_csv(csv_path, title=title, xlabel="desired_path_length", y_label="Mean Iterations")

    elif args.method == "evolution":
        if args.task == "binary_easy":
            csv_path = _generic_evolution_collector(
                list(np.arange(10, 101, 10)), lambda p: partial(binary_reward, target_path_length=p, hard=False),
                hyperparams, "binary_easy_", "evolution", args.quality_diversity, args.genotype_dimensions, False, args.sample_size, args.debug
            )
            plot_convergence_from_csv(csv_path, title="Evolution Binary Convergence (Easy)")
        elif args.task == "binary_hard":
            csv_path = _generic_evolution_collector(
                list(np.arange(10, 101, 10)), lambda p: partial(binary_reward, target_path_length=p, hard=True),
                hyperparams, "binary_hard_", "evolution", args.quality_diversity, args.genotype_dimensions, False, args.sample_size, args.debug
            )
            plot_convergence_from_csv(csv_path, title="Evolution Binary Convergence (HARD)")
        elif args.task == "biomes":
            csv_path = _generic_evolution_collector(
                ["Pond", "River", "Grass"], lambda b: {"Pond": pond_reward, "River": river_reward, "Grass": grass_reward}[b],
                hyperparams, "biome_average_", "evolution", args.quality_diversity, args.genotype_dimensions, True, args.sample_size, args.debug
            )
            plot_average_biome_convergence_from_csv(csv_path)
        else: # Combo tasks
            use_hard = args.combo == "hard"
            biome_map = {"river": river_reward, "pond": pond_reward, "grass": grass_reward}
            second_reward = biome_map[args.task]
            def make_reward(p): return CombinedReward([partial(binary_reward, target_path_length=p, hard=use_hard), second_reward])
            csv_path = _generic_evolution_collector(
                list(np.arange(10, 101, 10)), make_reward,
                hyperparams, f"combo_{args.task}{'_hard' if use_hard else ''}_", "evolution", args.quality_diversity, args.genotype_dimensions, False, args.sample_size, args.debug
            )
            title = f"Evolution Combo: {args.task.capitalize()}" + (" HARD" if use_hard else "")
            plot_convergence_from_csv(csv_path, title=title)
