import math
import argparse
import os
import sys
from typing import Any, Callable, Literal
import time
import pickle

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from functools import partial
import yaml

from assets.biome_adjacency_rules import create_adjacency_matrix
from core.evolution import evolve as evolve_standard, CrossOverMethod
from core.fi2pop import evolve as evolve_constrained, EvolutionMode
from core.mcts import resume_mcts_search, MCTS
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


def _generic_evolution_collector(
    loop_keys: list[Any],
    make_reward_fn: Callable[[Any], Callable[..., float]],
    evolution_hyperparameters: dict[str, Any],
    output_csv_prefix: str,
    method: str,
    use_quality_diversity: bool = False,
    genotype_dimensions: Literal[1, 2] = 1,
    is_biome_only: bool = False,
    sample_size: int = 20,
    debug: bool = False,
    no_random_offspring: bool = False  # NEW: Add this parameter
) -> str:
    # NEW: Override random offspring proportion if requested
    if no_random_offspring:
        evolution_hyperparameters["random_offspring"] = 0.0

    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    map_length, map_width = 15, 20

    # NEW: Adjust output directory based on no_random_offspring flag
    base_fig_dir = get_figure_directory(method)
    if method == "evolution":
        fig_dir = os.path.join(base_fig_dir, f"{genotype_dimensions}d")
        if no_random_offspring:
            fig_dir = os.path.join(fig_dir, "0_random_offspring")
    else:
        fig_dir = base_fig_dir
        
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(DEBUG_DIRECTORY, exist_ok=True)
    csv_filename = f"{output_csv_prefix}convergence.csv"
    csv_path = os.path.join(fig_dir, csv_filename)

    data_rows: list[dict[str, Any]] = []

    for key in loop_keys:
        # Get biome-specific hyperparameters if they exist, otherwise use the main ones
        current_hyperparams = evolution_hyperparameters.get(
            str(key), evolution_hyperparameters
        )
        raw_cross_over = current_hyperparams.get("cross_over_method", "ONE_POINT")
        try:
            cross_over_method = CrossOverMethod(raw_cross_over)
        except (ValueError, TypeError):
            cross_over_method = CrossOverMethod(int(raw_cross_over))

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
                generations=current_hyperparams.get("generations", 1000),
                population_size=48,  # Fixed population size
                number_of_actions_mutated_mean=current_hyperparams[
                    "number_of_actions_mutated_mean"
                ],
                number_of_actions_mutated_standard_deviation=current_hyperparams[
                    "number_of_actions_mutated_standard_deviation"
                ],
                action_noise_standard_deviation=current_hyperparams[
                    "action_noise_standard_deviation"
                ],
                survival_rate=current_hyperparams["survival_rate"],
                cross_over_method=cross_over_method,
                patience=50,  # Fixed patience
                qd=use_quality_diversity,
                genotype_representation=f"{genotype_dimensions}d",
                # NEW: Add these hyperparameters to call
                random_offspring_proportion=current_hyperparams.get("random_offspring", 0.1),
                cross_or_mutate_proportion=current_hyperparams.get("cross_or_mutate", 0.7),
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
                plt.plot(x_axis, mean_agent_rewards, label="Mean Elite")
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

    file_exists = os.path.isfile(csv_path)
    new_df = pd.DataFrame(data_rows)
    new_df.to_csv(
        csv_path,
        mode='a',
        index=False,
        header=not file_exists
    )
    if file_exists:
        print(f"Appended {len(new_df)} rows to {csv_path}")
    else:
        print(f"Created new CSV {csv_path} with {len(new_df)} rows")
    return csv_path


def _generic_constrained_ea_collector(
    mode: EvolutionMode,
    loop_keys: list,
    make_reward_fn: callable,
    hyperparameters: dict,
    output_csv_prefix: str,
    is_biome_only: bool,
    sample_size: int,
    debug: bool,
) -> str:
    fig_dir = get_figure_directory(mode.value)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(DEBUG_DIRECTORY, exist_ok=True)
    csv_filename = f"{output_csv_prefix}convergence.csv"
    csv_path = os.path.join(fig_dir, csv_filename)

    data_rows = []
    for key in loop_keys:
        # Get biome-specific hyperparameters if they exist, otherwise use the main ones
        current_hyperparams = hyperparameters.get(str(key), hyperparameters)
        raw_cross_over = current_hyperparams.get("cross_over_method", "ONE_POINT")
        try:
            cross_over_method = CrossOverMethod(raw_cross_over)
        except (ValueError, TypeError):
            cross_over_method = CrossOverMethod(int(raw_cross_over))

        for run_index in range(1, sample_size + 1):
            if debug:
                start_time = time.time()

            reward_callable = make_reward_fn(key)
            best_agent, final_gen, _, _ = evolve_constrained(
                mode=mode,
                reward_fn=reward_callable,
                task_args={},
                generations=current_hyperparams.get("generations", 1000),
                population_size=48,
                number_of_actions_mutated_mean=current_hyperparams[
                    "number_of_actions_mutated_mean"
                ],
                number_of_actions_mutated_standard_deviation=current_hyperparams[
                    "number_of_actions_mutated_standard_deviation"
                ],
                action_noise_standard_deviation=current_hyperparams[
                    "action_noise_standard_deviation"
                ],
                survival_rate=current_hyperparams["survival_rate"],
                cross_over_method=cross_over_method,
                cross_or_mutate_proportion=current_hyperparams.get(
                    "cross_or_mutate_proportion", 0.7
                ),
                patience=50,
            )

            if debug:
                elapsed = time.time() - start_time
                print(f"[{mode.value.upper()}] Key={key}, Run={run_index}: Stopped at gen {final_gen} in {elapsed:.2f}s")

            converged = best_agent and best_agent.reward >= 0.0
            generations_to_converge = final_gen if converged else float("nan")
            row = {"run_index": run_index, "generations_to_converge": generations_to_converge}
            row["biome" if is_biome_only else "desired_path_length"] = key
            data_rows.append(row)

    file_exists = os.path.isfile(csv_path)
    new_df = pd.DataFrame(data_rows)
    new_df.to_csv(
        csv_path,
        mode='a',
        index=False,
        header=not file_exists
    )
    if file_exists:
        print(f"Appended {len(new_df)} rows to {csv_path}")
    else:
        print(f"Created new CSV {csv_path} with {len(new_df)} rows")
    return csv_path


def _resumable_mcts_collector(
    loop_keys: list[Any],
    make_reward_fn: Callable[[Any], Callable[..., float]],
    task_prefix: str,
    is_biome_only: bool,
    sample_size: int,
    max_iterations_per_run: int,
) -> str:
    """
    Collects MCTS data statefully, allowing resumption of incomplete runs.

    This implementation stores each MCTS tree and state in a separate file
    to avoid memory issues with large numbers of runs.

    Returns:
        str: The path to the final summary CSV file.
    """
    fig_dir = get_figure_directory("mcts")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Create tree storage directory
    tree_dir = os.path.join(fig_dir, "trees", task_prefix)
    os.makedirs(tree_dir, exist_ok=True)
    
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    map_length, map_width = 15, 20
    DEFAULT_EXPLORATION_WEIGHT = math.sqrt(2)

    for key in loop_keys:
        for run_index in range(1, sample_size + 1):
            # Create unique filename for this run
            filename = f"{key}_{run_index}.pkl"
            state_path = os.path.join(tree_dir, filename)
            
            if os.path.exists(state_path):
                print(f"Resuming task: Key={key}, Run={run_index}")
                with open(state_path, "rb") as f:
                    state = pickle.load(f)
                mcts_instance = state["mcts_instance"]
                cumulative_iterations = state["cumulative_iterations"]
            else:
                print(f"Starting new task: Key={key}, Run={run_index}")
                reward_callable = make_reward_fn(key)
                env = WFCWrapper(
                    map_length=map_length, map_width=map_width, tile_symbols=tile_symbols,
                    adjacency_bool=adjacency_bool, num_tiles=len(tile_symbols),
                    tile_to_index=tile_to_index, reward=reward_callable, deterministic=True,
                )
                mcts_instance = MCTS(env, exploration_weight=DEFAULT_EXPLORATION_WEIGHT)
                cumulative_iterations = 0
                state = {
                    "mcts_instance": mcts_instance,
                    "cumulative_iterations": cumulative_iterations,
                    "result": None,
                }

            # Run the search for more iterations
            _, _, iterations_in_run = resume_mcts_search(mcts_instance, max_iterations_per_run)

            # Update state
            if iterations_in_run is not None:
                total_iterations = cumulative_iterations + iterations_in_run
                state["result"] = total_iterations
                print(f"Task SOLVED: Key={key}, Run={run_index} at total iterations: {total_iterations}")
            else:
                state["cumulative_iterations"] += max_iterations_per_run
                print(f"Task NOT solved: Key={key}, Run={run_index}. Cumulative iterations now: {state['cumulative_iterations']}")

            # Save updated state
            with open(state_path, "wb") as f:
                pickle.dump(state, f)

    # Generate CSV report by scanning tree directory
    report_rows = []
    for filename in os.listdir(tree_dir):
        if filename.endswith(".pkl"):
            with open(os.path.join(tree_dir, filename), "rb") as f:
                state = pickle.load(f)
            
            # Extract key and run_index from filename
            parts = filename[:-4].split("_")
            key = "_".join(parts[:-1])  # Handle keys with underscores
            run_index = int(parts[-1])
            
            row = {
                "run_index": run_index,
                "iterations_to_converge": state["result"],
            }
            row["biome" if is_biome_only else "desired_path_length"] = key
            report_rows.append(row)

    csv_path = os.path.join(fig_dir, f"{task_prefix}_convergence.csv")
    pd.DataFrame(report_rows).to_csv(csv_path, index=False)
    print(f"Saved final MCTS report to {csv_path}")
    return csv_path


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
    method = evolution_hyperparameters.get("method", "evolution")
    base_fig_dir = get_figure_directory(method)
    fig_dir = os.path.join(base_fig_dir, f"{genotype_dimensions}d") if method == "evolution" else base_fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    
    csv_path = os.path.join(fig_dir, csv_filename)
    file_exists = os.path.isfile(csv_path)
    new_df = pd.DataFrame(data_rows)
    new_df.to_csv(
        csv_path,
        mode='a',
        index=False,
        header=not file_exists
    )
    if file_exists:
        print(f"Appended {len(new_df)} rows to {csv_path}")
    else:
        print(f"Created new CSV {csv_path} with {len(new_df)} rows")
    return csv_path


def collect_binary_convergence(sample_size, evolution_hyperparameters, use_quality_diversity=False, use_hard_variant=False, genotype_dimensions=1, debug=False):
    path_lengths = list(np.arange(10, 101, 10))
    prefix = f"{'qd_' if use_quality_diversity else ''}{'hard_' if use_hard_variant else ''}binary_"
    def make_reward(path_len): return partial(binary_reward, target_path_length=path_len, hard=use_hard_variant)
    return _generic_convergence_collector(path_lengths, make_reward, evolution_hyperparameters, prefix, use_quality_diversity, genotype_dimensions, is_biome_only=False, sample_size=sample_size, debug=debug)


def collect_combo_convergence(sample_size, evolution_hyperparameters, use_quality_diversity, second_task, use_hard_variant=False, genotype_dimensions=1, debug=False):
    path_lengths = list(np.arange(10, 101, 10))
    prefix = f"{'qd_' if use_quality_diversity else ''}{'hard_' if use_hard_variant else ''}{second_task}_combo_"
    biome_reward_map = {"river": river_reward, "pond": pond_reward, "grass": grass_reward}
    second_reward = biome_reward_map[second_task]
    def make_reward(path_len): return CombinedReward([partial(binary_reward, target_path_length=path_len, hard=use_hard_variant), second_reward])
    return _generic_convergence_collector(path_lengths, make_reward, evolution_hyperparameters, prefix, use_quality_diversity, genotype_dimensions, is_biome_only=False, sample_size=sample_size, debug=debug)


def collect_average_biome_convergence_data(evolution_hyperparameters, use_quality_diversity=False, runs=20, genotype_dimensions=1, debug=False):
    biomes = ["Pond", "River"]
    prefix = f"{'qd_' if use_quality_diversity else ''}biome_average_"
    def make_reward(biome): return {"Pond": pond_reward, "River": river_reward, "Grass": grass_reward}[biome]
    return _generic_convergence_collector(biomes, make_reward, evolution_hyperparameters, prefix, use_quality_diversity, genotype_dimensions, is_biome_only=True, sample_size=runs, debug=debug)


def plot_convergence_from_csv(
    csv_path: str,
    output_path: str = None,
    title: str = "",
    xlabel: str = "desired_path_length",
    y_label: str = "Mean Generations"
):
    # 1. Read all runs
    df = pd.read_csv(csv_path)

    # 2. Identify successful convergences
    df_valid = df.dropna(subset=["generations_to_converge"])

    # 3. Compute per-x statistics on converged runs
    stats = (
        df_valid
        .groupby(xlabel)["generations_to_converge"]
        .agg(mean="mean", std="std", successes="count")
    )

    # 4. Compute total runs per x (including non-converged)
    total = df.groupby(xlabel)["run_index"].count().rename("total_runs")

    # 5. Join them together
    stats = stats.join(total)

    # 6. Compute stderr and fraction converged
    stats["stderr"] = stats["std"] / np.sqrt(stats["successes"])
    stats["fraction_converged"] = stats["successes"] / stats["total_runs"]

    # 7. Reset index for plotting
    stats = stats.reset_index()

    # Early exit if nothing to plot
    if stats.empty:
        print(f"No convergence data found in {csv_path}, skipping plot.")
        return

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # a) Errorbar: Mean + stderr
    ax1.errorbar(
        stats[xlabel],
        stats["mean"],
        yerr=stats["stderr"],
        fmt="o-",
        capsize=4,
        label="Mean generations"
    )
    for x, y in zip(stats[xlabel], stats["mean"]):
        ax1.text(x, y, f"{y:.1f}", ha="center", va="bottom")

    # b) Bar: Fraction converged
    ax1.set_xlim(0, 100)
    ax1.set_xticks(np.arange(0, 101, 10))
    bar_width = 9  # 90/(number of bins=10)
    ax2.bar(
        stats[xlabel],
        stats["fraction_converged"],
        width=bar_width,
        alpha=0.3,
        label="Fraction converged"
    )

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y_label)
    ax2.set_ylabel("Fraction Converged")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    ax1.set_title(title)
    fig.tight_layout()

    if output_path is None:
        output_path = csv_path.replace(".csv", ".png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")

def plot_average_biome_convergence_from_csv(csv_file_path: str, output_png_path: str = None, y_label: str = "Mean Generations to Converge"):
    df = pd.read_csv(csv_file_path)
    
    # Determine convergence metric column
    if "generations_to_converge" in df.columns:
        convergence_column = "generations_to_converge"
    elif "iterations_to_converge" in df.columns:
        convergence_column = "iterations_to_converge"
        y_label = "Mean Iterations to Converge"  # Optional: auto-adjust label
    else:
        raise KeyError("Missing convergence column in CSV.")
    
    # Clean invalid runs (non-converged = NaN)
    df_valid = df.dropna(subset=["generations_to_converge"])
    
    # Aggregate statistics
    stats = df_valid.groupby("biome")["generations_to_converge"].agg(["mean", "std", "count"]).reset_index()
    stats["stderr"] = stats["std"] / np.sqrt(stats["count"])
    
    # Compute total runs for normalization (for fraction converged)
    total_runs_per_biome = df.groupby("biome")["run_index"].count().reset_index(name="total_runs")
    stats = stats.merge(total_runs_per_biome, on="biome")
    stats["fraction_converged"] = stats["count"] / stats["total_runs"]

    # Setup dual bar chart
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bar_width = 0.35
    x = np.arange(len(stats["biome"]))

    # Bar 1: Mean generations with error bars
    ax1.bar(x - bar_width/2, stats["mean"], yerr=stats["stderr"], capsize=4, width=bar_width, label=y_label.replace(" to Converge", ""), color='tab:blue', alpha=0.7)
    ax1.set_ylabel(y_label, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Bar 2: Fraction converged
    ax2.bar(x + bar_width/2, stats["fraction_converged"], width=bar_width, label="Fraction Converged", color='tab:orange', alpha=0.5)
    ax2.set_ylabel("Fraction Converged", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats["biome"])
    ax1.set_xlabel("Biome")

    # Title & Layout
    plt.title("Average Convergence per Biome")
    fig.tight_layout()

    # Save
    if output_png_path is None:
        output_png_path = csv_file_path.replace(".csv", ".png")
    fig.savefig(output_png_path)
    plt.close(fig)
    print(f"Saved biome convergence plot to {output_png_path}")

import matplotlib.pyplot as plt
import pandas as pd

def plot_comparison(
    csv_paths: list[str],
    labels: list[str],
    output_path: str | None = None,
    title: str = "Method Comparison",
) -> None:
    import numpy as np
    import pandas as pd
    import itertools
    import matplotlib.pyplot as plt

    def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
        # X axis
        if "biome" in df.columns:
            xcol = "biome"
        elif "desired_path_length" in df.columns:
            xcol = "desired_path_length"
        else:
            raise KeyError("Expected 'biome' or 'desired_path_length' in CSV.")
        # Metric
        if "generations_to_converge" in df.columns:
            metric = "generations_to_converge"
        elif "iterations_to_converge" in df.columns:
            metric = "iterations_to_converge"
        else:
            raise KeyError("Expected 'generations_to_converge' or 'iterations_to_converge' in CSV.")
        return xcol, metric

    all_stats: list[pd.DataFrame] = []
    sample_sizes: dict[str, int] = {}
    global_xcol: str | None = None
    global_metric: str | None = None

    for csv_path, method_name in zip(csv_paths, labels):
        raw = pd.read_csv(csv_path)
        xcol, metric = detect_columns(raw)

        if global_xcol is None:
            global_xcol = xcol
        elif xcol != global_xcol:
            raise ValueError(
                f"Mixed x-axes in one comparison: '{global_xcol}' vs '{xcol}'. "
                "Compare biomes and path-lengths separately."
            )

        if global_metric is None:
            global_metric = metric

        # Total runs per x
        if "run_index" in raw.columns:
            total_runs = raw.groupby(xcol)["run_index"].count().rename("total_runs")
        else:
            total_runs = raw.groupby(xcol).size().rename("total_runs")

        sample_sizes[method_name] = int(total_runs.max())

        valid = raw.dropna(subset=[metric])
        stats = (
            valid.groupby(xcol)[metric]
            .agg(mean="mean", std="std", successes="count")
            .join(total_runs)
        )
        stats["stderr"] = stats["std"] / np.sqrt(stats["successes"])
        stats["fraction_converged"] = stats["successes"] / stats["total_runs"]
        stats = stats.reset_index()
        stats["method"] = method_name
        all_stats.append(stats)

    combined = pd.concat(all_stats, ignore_index=True)

    # X axis handling
    is_numeric_x = np.issubdtype(
        combined[global_xcol].dropna().infer_objects(copy=False).dtype, np.number
    )
    if is_numeric_x:
        x_values = np.array(sorted(combined[global_xcol].unique()))
        pos_map = {v: float(v) for v in x_values}
        step = float(np.min(np.diff(np.sort(x_values)))) if len(x_values) > 1 else 1.0
        total_bar_space = 0.8 * step
    else:
        x_values = list(pd.unique(combined[global_xcol]))
        pos_map = {v: i for i, v in enumerate(x_values)}
        total_bar_space = 0.8

    method_count = len(labels)
    bar_width = total_bar_space / max(method_count, 1)

    fig, left_axis = plt.subplots(figsize=(10, 6))
    right_axis = left_axis.twinx()

    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"])
    method_colors = {m: next(color_cycle)["color"] for m in labels}

    for method_name, group in combined.groupby("method"):
        color = method_colors[method_name]
        group = group.sort_values(global_xcol, key=lambda s: [pos_map[v] for v in s])

        # Fraction converged (lines)
        left_axis.plot(
            [pos_map[v] for v in group[global_xcol]],
            group["fraction_converged"],
            marker="o",
            linestyle="-",
            label=f"{method_name} (n≈{sample_sizes[method_name]})",
            linewidth=2,
            color=color,
        )

        # Mean ± stderr (bars)
        idx = labels.index(method_name)
        offset = (idx - (method_count - 1) / 2) * bar_width
        positions = np.array([pos_map[v] for v in group[global_xcol]]) + offset
        right_axis.bar(
            positions,
            group["mean"],
            yerr=group["stderr"],
            capsize=4,
            width=bar_width,
            alpha=0.6,
            color=color,
        )

    # Labels
    if global_xcol == "biome":
        left_axis.set_xlabel("Biome")
    else:
        left_axis.set_xlabel("Desired Binary Path Length")

    right_label = "Mean Generations" if global_metric == "generations_to_converge" else "Mean Iterations"
    left_axis.set_ylabel("Fraction Converged (lines)")
    right_axis.set_ylabel(right_label)

    # Ticks
    if is_numeric_x:
        left_axis.set_xticks([pos_map[v] for v in x_values])
        left_axis.set_xticklabels([str(v) for v in x_values])
        left_axis.set_xlim(min(pos_map.values()) - 0.5, max(pos_map.values()) + 0.5)
    else:
        left_axis.set_xticks(range(len(x_values)))
        left_axis.set_xticklabels(x_values)

    left_axis.grid(True, linestyle="--", alpha=0.6)
    left_axis.legend(loc="upper left")
    left_axis.set_title(title)
    fig.tight_layout()

    if output_path is None:
        output_path = "comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")

    # ----- ASCII + LaTeX tables -----
    ordered_keys = x_values if not is_numeric_x else sorted(x_values)
    rows, index_tuples = [], []
    for method_name in labels:
        grp = (
            combined[combined["method"] == method_name]
            .set_index(global_xcol)
            .reindex(ordered_keys)
        )

        mean_row = []
        for k in ordered_keys:
            m = grp.at[k, "mean"] if k in grp.index else np.nan
            s = grp.at[k, "stderr"] if k in grp.index else np.nan
            mean_row.append("—" if (pd.isna(m) or pd.isna(s)) else f"{m:.1f}±{s:.1f}")
        rows.append(mean_row)
        index_tuples.append((method_name, f"{right_label} (±stderr)"))

        frac_row = []
        for k in ordered_keys:
            f = grp.at[k, "fraction_converged"] if k in grp.index else np.nan
            frac_row.append("—" if pd.isna(f) else f"{f:.2f}")
        rows.append(frac_row)
        index_tuples.append((method_name, "Fraction Converged"))

    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["Method", "Metric"])
    table_df = pd.DataFrame(rows, index=multi_index, columns=ordered_keys)

    print("\nFull convergence table:\n")
    print(table_df.to_string())

    print("\nLaTeX table:\n")
    print(
        table_df.to_latex(
            caption="Convergence across all methods",
            label="tab:all_convergence",
            na_rep="—",
            escape=False,
        )
    )


    
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
    biomes = ["Pond", "River"]
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
    # --- Comparison mode ---
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Plot multiple CSVs on one comparison chart"
    )
    parser.add_argument(
        "--csv-files",
        nargs="+",
        help="List of convergence CSVs to compare (in same order as --labels)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional custom labels for each CSV (defaults to filename stem)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for comparison PNG (default: comparison.png)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title for the comparison plot"
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default="desired_path_length",
        help="X-axis label for comparison plot"
    )
    parser.add_argument(
        "--y_label",
        dest="y_label",
        type=str,
        default="Mean Generations",
        help="Y-axis label for comparison plot"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["evolution", "mcts", "fi2pop", "baseline"],
        default="evolution",
        help="Method to use for convergence testing",
    )
    parser.add_argument(
        "--no-random-offspring",
        action="store_true",
        help="Set random_offspring proportion to 0 for evolution method"
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
    )
    parser.add_argument("--combo", type=str, choices=["easy", "hard"], default="easy")
    parser.add_argument(
        "--quality-diversity", action="store_true", help="Use the QD variant (evolution only)"
    )
    parser.add_argument(
        "--genotype-dimensions", type=int, choices=[1, 2], default=1
    )
    parser.add_argument("--debug", action="store_true", help="Save per-run debug plots")
    parser.add_argument("--sample-size", type=int, default=40, help="Number of runs to collect per data point.")
    parser.add_argument("--mcts-iterations", type=int, default=1000, help="Number of MCTS iterations to run per session.")
    args = parser.parse_args()

    # --- Handle comparison mode first ---
    if args.compare:
        if not args.csv_files:
            parser.error("--csv-files is required when --compare is set")
        if args.labels and len(args.labels) != len(args.csv_files):
            parser.error("Number of --labels must match number of --csv-files")
            
        labels = args.labels or [
            os.path.basename(p).rsplit(".", 1)[0] 
            for p in args.csv_files
        ]
        
        plot_comparison(
            csv_paths=args.csv_files,
            labels=labels,
            output_path=args.output,
            title=args.title or "Method Comparison",
        )
        sys.exit(0)

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
                mode, ["Pond", "River"], lambda b: {"Pond": pond_reward, "River": river_reward, "Grass": grass_reward}[b],
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
                hyperparams, "binary_easy_", "evolution", args.quality_diversity, args.genotype_dimensions, False, args.sample_size, args.debug,
                no_random_offspring=args.no_random_offspring
            )
            plot_convergence_from_csv(csv_path, title="Evolution Binary Convergence (Easy)")
        elif args.task == "binary_hard":
            csv_path = _generic_evolution_collector(
                list(np.arange(10, 101, 10)), lambda p: partial(binary_reward, target_path_length=p, hard=True),
                hyperparams, "binary_hard_", "evolution", args.quality_diversity, args.genotype_dimensions, False, args.sample_size, args.debug,
                no_random_offspring=args.no_random_offspring
            )
            plot_convergence_from_csv(csv_path, title="Evolution Binary Convergence (HARD)")
        elif args.task == "biomes":
            csv_path = _generic_evolution_collector(
                ["Pond", "River"], lambda b: {"Pond": pond_reward, "River": river_reward, "Grass": grass_reward}[b],
                hyperparams, f"{args.genotype_dimensions}d_biome_average_", "evolution", args.quality_diversity, args.genotype_dimensions, True, args.sample_size, args.debug,
                no_random_offspring=args.no_random_offspring
            )
            plot_average_biome_convergence_from_csv(csv_path)
        else: # Combo tasks
            use_hard = args.combo == "hard"
            biome_map = {"river": river_reward, "pond": pond_reward, "grass": grass_reward}
            second_reward = biome_map[args.task]
            def make_reward(p): return CombinedReward([partial(binary_reward, target_path_length=p, hard=use_hard), second_reward])
            csv_path = _generic_evolution_collector(
                list(np.arange(10, 101, 10)), make_reward,
                hyperparams, f"combo_{args.task}{'_hard' if use_hard else ''}_", "evolution", args.quality_diversity, args.genotype_dimensions, False, args.sample_size, args.debug,
                no_random_offspring=args.no_random_offspring
            )
            title = f"Evolution Combo: {args.task.capitalize()}" + (" HARD" if use_hard else "")
            plot_convergence_from_csv(csv_path, title=title)