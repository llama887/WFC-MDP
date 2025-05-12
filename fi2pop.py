#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import random
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# ----------------------------------------------------------------------------
# Ensure vendored packages on PYTHONPATH
# ----------------------------------------------------------------------------
import sys
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "vendor"))

from biome_adjacency_rules import create_adjacency_matrix
from wfc import load_tile_images
from wfc_env import WFCWrapper, CombinedReward

# ----------------------------------------------------------------------------
# Task callbacks
# ----------------------------------------------------------------------------
from tasks.binary_task import binary_reward, binary_percent_water
from tasks.pond_task   import pond_reward
from tasks.river_task  import river_reward
from tasks.grass_task  import grass_reward
from tasks.hill_task   import hill_reward

# ----------------------------------------------------------------------------
# Prepare figures directory
# ----------------------------------------------------------------------------
FIGURES_DIRECTORY = "figures"
os.makedirs(FIGURES_DIRECTORY, exist_ok=True)

# ----------------------------------------------------------------------------
# WFC environment factory
# ----------------------------------------------------------------------------
ADJ_BOOL, TILE_SYMBOLS, TILE2IDX = create_adjacency_matrix()
NUM_TILES = len(TILE_SYMBOLS)
TILE_IMAGES = load_tile_images()
MAP_LENGTH = 15
MAP_WIDTH = 20

def make_env(reward_callable: Any) -> WFCWrapper:
    return WFCWrapper(
        map_length=MAP_LENGTH,
        map_width=MAP_WIDTH,
        tile_symbols=TILE_SYMBOLS,
        adjacency_bool=ADJ_BOOL,
        num_tiles=NUM_TILES,
        tile_to_index=TILE2IDX,
        reward=reward_callable,
        max_reward=0.0,
        deterministic=True,
        qd_function=None,
        tile_images=None,
        tile_size=32,
        render_mode=None,
    )

# ----------------------------------------------------------------------------
# Genome definition
# ----------------------------------------------------------------------------
class Genome:
    def __init__(self, env: WFCWrapper):
        self.env = env
        self.action_sequence = np.array([
            env.action_space.sample()
            for _ in range(env.map_length * env.map_width)
        ])
        self.reward: float = float("-inf")
        self.violation: int = 1_000_000

    def mutate(self, rate: float = 0.02):
        mask = np.random.rand(len(self.action_sequence)) < rate
        for idx in np.where(mask)[0]:
            self.action_sequence[idx] = self.env.action_space.sample()

    @staticmethod
    def crossover(p1: Genome, p2: Genome) -> Tuple[Genome, Genome]:
        cut = random.randint(1, len(p1.action_sequence) - 1)
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        c1.action_sequence[cut:], c2.action_sequence[cut:] = (
            p2.action_sequence[cut:].copy(),
            p1.action_sequence[cut:].copy(),
        )
        return c1, c2

# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def _count_contradictions(env: WFCWrapper) -> int:
    return sum(
        1
        for row in env.grid
        for cell in row
        if isinstance(cell, set) and len(cell) == 0
    )

def evaluate(env: WFCWrapper, actions: np.ndarray) -> Tuple[float, int]:
    total_reward = 0.0
    info: Dict[str, Any] = {}
    for a in actions:
        _, r, done, trunc, info = env.step(a)
        total_reward += r
        if done or trunc:
            break
    violation = info.get("violations", info.get("contradictions"))
    if violation is None:
        violation = _count_contradictions(env)
    return total_reward, int(violation)

def _parallel_eval(gen: Genome) -> Genome:
    e = copy.deepcopy(gen.env)
    gen.reward, gen.violation = evaluate(e, gen.action_sequence)
    return gen

# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------
def tournament_select(
    pop: List[Genome],
    fitness: List[float],
    k: int,
    n: int
) -> List[Genome]:
    winners: List[Genome] = []
    for _ in range(n):
        best = random.randrange(len(pop))
        for __ in range(1, k):
            cand = random.randrange(len(pop))
            if fitness[cand] > fitness[best]:
                best = cand
        winners.append(copy.deepcopy(pop[best]))
    return winners

# ----------------------------------------------------------------------------
# Mutation rate converter
# ----------------------------------------------------------------------------
def _compute_mutation_rate(hp: dict[str, Any], map_length: int, map_width: int) -> float:
    L = map_length * map_width
    M = hp.get("number_of_actions_mutated_mean", 0)
    return float(M) / L if L > 0 else 0.0

# ----------------------------------------------------------------------------
# FI-2Pop GA: returns best & median histories
# ----------------------------------------------------------------------------
def evolve_fi2pop(
    reward_fn: Any,
    task_args: Dict[str, Any],
    generations: int = 200,
    pop_size: int = 48,
    mutation_rate: float = 0.0559,
    tournament_k: int = 3,
    return_first_gen: bool = False
) -> Tuple[List[Genome], List[Genome], Optional[int], List[float], List[float]]:
    reward_callable = partial(reward_fn, **task_args)
    best_hist: List[float] = []
    median_hist: List[float] = []

    combined = [Genome(make_env(reward_callable)) for _ in range(pop_size*2)]
    with Pool(min(cpu_count(), len(combined))) as P:
        combined = P.map(_parallel_eval, combined)

    feasible   = [g for g in combined if g.violation == 0]
    infeasible = [g for g in combined if g.violation > 0]
    first_gen: Optional[int] = 0 if (feasible and return_first_gen) else None

    # record gen 0
    rewards = [g.reward for g in combined]
    best_hist.append(max(rewards))
    median_hist.append(float(np.median(rewards)))

    for gen in range(1, generations+1):
        with Pool(min(cpu_count(), len(infeasible))) as P:
            infeasible = P.map(_parallel_eval, infeasible)

        newly = [g for g in infeasible if g.violation == 0]
        feasible.extend(newly)
        infeasible = [g for g in infeasible if g.violation > 0]

        if return_first_gen and first_gen is None and feasible:
            first_gen = gen
            break

        combined = feasible + infeasible
        rewards = [g.reward for g in combined]
        best_hist.append(max(rewards))
        median_hist.append(float(np.median(rewards)))

        if max(rewards) >= 0.0:
            if return_first_gen and first_gen is None:
                first_gen = gen
            print(f"[EARLY STOP] reached max reward {max(rewards):.3f} at generation {gen}")
            break

        # breeding
        def breed(pool: List[Genome], key: str) -> List[Genome]:
            fit = [getattr(g, key) for g in pool]
            parents = tournament_select(pool, fit, tournament_k, pop_size)
            kids: List[Genome] = []
            for i in range(0, len(parents), 2):
                c1, c2 = Genome.crossover(parents[i], parents[(i+1)%len(parents)])
                c1.mutate(mutation_rate); c2.mutate(mutation_rate)
                kids.extend([c1, c2])
            return kids

        offspring = breed(feasible, "reward") + breed(infeasible, "violation")
        with Pool(min(cpu_count(), len(offspring))) as P:
            offspring = P.map(_parallel_eval, offspring)

        for g in offspring:
            (feasible if g.violation == 0 else infeasible).append(g)

        feasible   = sorted(feasible,   key=lambda g: g.reward, reverse=True)[:pop_size]
        infeasible = sorted(infeasible, key=lambda g: g.violation)[:pop_size]
        print(f"Gen {gen:04d} | Feas {len(feasible):02d} | Infeas {len(infeasible):02d} | BestReward {feasible[0].reward:.3f}")

    return feasible, infeasible, first_gen, best_hist, median_hist

# ----------------------------------------------------------------------------
# Binary HARD convergence vs path length
# ----------------------------------------------------------------------------
def binary_convergence_over_path_lengths_fi2pop(
    sample_size: int,
    hyperparams: dict[str, Any],
    hard: bool = True,
) -> None:
    MIN_L, MAX_L, STEP = 10, 100, 10
    MAP_L, MAP_W = MAP_LENGTH, MAP_WIDTH
    MAX_G = hyperparams.get("generations", 200)
    pop_size = hyperparams.get("population_size", 48)
    mut_rate = _compute_mutation_rate(hyperparams, MAP_L, MAP_W)
    t_k = hyperparams.get("tournament_k", 3)

    path_lengths = np.arange(MIN_L, MAX_L+1, STEP)
    gens_to_conv = np.full((len(path_lengths), sample_size), np.nan)

    for i, L in enumerate(path_lengths):
        for run in range(sample_size):
            print(f"[BIN-HARD] Path {L}, run {run+1}/{sample_size}")
            reward_fn = partial(binary_reward, target_path_length=L, hard=hard)
            _, _, first_gen, best_hist, med_hist = evolve_fi2pop(
                reward_fn, {}, MAX_G, pop_size, mut_rate, t_k, True
            )
            plt.figure()
            gens = list(range(len(best_hist)))
            plt.plot(gens, best_hist, label="Best Agent")
            plt.plot(gens, med_hist, label="Median Agent")
            plt.title(f"FI-2Pop Binary HARD (path={L}, run={run})")
            plt.xlabel("Generation"); plt.ylabel("Reward"); plt.legend()
            plt.savefig(f"{FIGURES_DIRECTORY}/fi2pop_path{L}_run{run}_history.png")
            plt.close()
            if first_gen is not None:
                gens_to_conv[i, run] = first_gen

    means = np.nanmean(gens_to_conv, axis=1)
    counts = np.sum(~np.isnan(gens_to_conv), axis=1)
    stderr = np.nanstd(gens_to_conv, axis=1)/np.sqrt(np.maximum(counts,1))
    frac = counts/sample_size

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()
    ax1.errorbar(path_lengths, means, yerr=stderr, fmt='o-', capsize=4, label='Mean gens')
    ax2.bar(path_lengths, frac, width=STEP*0.8, alpha=0.3, label='Frac converged')
    ax1.set_xlabel('Desired Path Length'); ax1.set_ylabel('Mean Gens to Converge')
    ax2.set_ylabel('Fraction Converged')
    ax1.set_title('FI-2Pop Binary HARD Convergence')
    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left')
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIRECTORY}/fi2pop_convergence_summary_hard.png"); plt.close()

# ----------------------------------------------------------------------------
# Combo HARD convergence vs path length
# ----------------------------------------------------------------------------
def combo_convergence_over_path_lengths_fi2pop(
    sample_size: int,
    hyperparams: dict[str, Any],
    hard: bool = True,
) -> None:
    MIN_L, MAX_L, STEP = 10, 100, 10
    MAP_L, MAP_W = MAP_LENGTH, MAP_WIDTH
    MAX_G = hyperparams.get("generations", 200)
    pop_size = hyperparams.get("population_size", 48)
    mut_rate = _compute_mutation_rate(hyperparams, MAP_L, MAP_W)
    t_k = hyperparams.get("tournament_k", 3)

    path_lengths = np.arange(MIN_L, MAX_L+1, STEP)
    gens_to_conv = np.full((len(path_lengths), sample_size), np.nan)

    for i, L in enumerate(path_lengths):
        for run in range(sample_size):
            print(f"[COMBO-HARD] Path {L}, run {run+1}/{sample_size}")
            reward_fn = CombinedReward([
                partial(binary_reward, target_path_length=L, hard=hard),
                pond_reward,
                river_reward,
                grass_reward
            ])
            _, _, first_gen, best_hist, med_hist = evolve_fi2pop(
                reward_fn, {}, MAX_G, pop_size, mut_rate, t_k, True
            )
            plt.figure()
            gens = list(range(len(best_hist)))
            plt.plot(gens, best_hist, label="Best Agent")
            plt.plot(gens, med_hist, label="Median Agent")
            plt.title(f"FI-2Pop Combo HARD (path={L}, run={run})")
            plt.xlabel("Generation"); plt.ylabel("Reward"); plt.legend()
            plt.savefig(f"{FIGURES_DIRECTORY}/fi2pop_combo_path{L}_run{run}.png"); plt.close()
            if first_gen is not None:
                gens_to_conv[i, run] = first_gen

    means = np.nanmean(gens_to_conv, axis=1)
    counts = np.sum(~np.isnan(gens_to_conv), axis=1)
    stderr = np.nanstd(gens_to_conv, axis=1)/np.sqrt(np.maximum(counts,1))
    frac = counts/sample_size

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()
    ax1.errorbar(path_lengths, means, yerr=stderr, fmt='o-', capsize=4, label='Mean gens')
    ax2.bar(path_lengths, frac, width=STEP*0.8, alpha=0.3, label='Frac converged')
    ax1.set_xlabel('Desired Path Length'); ax1.set_ylabel('Mean Gens to Converge')
    ax2.set_ylabel('Fraction Converged')
    ax1.set_title('FI-2Pop Combo HARD Convergence')
    h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left')
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIRECTORY}/fi2pop_combo_convergence_summary.png"); plt.close()

# ----------------------------------------------------------------------------
# Mean-convergence bar chart across tasks
# ----------------------------------------------------------------------------
def plot_avg_task_convergence_fi2pop(
    hyperparams: dict[str, Any],
    runs: int = 20
) -> None:
    tasks = {"Pond": pond_reward, "River": river_reward,
             "Grass": grass_reward, "Hill": hill_reward}
    MAP_L, MAP_W = MAP_LENGTH, MAP_WIDTH
    pop_size = hyperparams.get("population_size", 48)
    mut_rate = _compute_mutation_rate(hyperparams, MAP_L, MAP_W)
    t_k = hyperparams.get("tournament_k", 3)

    means, errors, labels = [], [], []
    for label, fn in tasks.items():
        gens_list: List[int] = []
        for i in range(runs):
            print(f"[{label}] run {i+1}/{runs}")
            _, _, first_gen, _, _ = evolve_fi2pop(fn, {},
                hyperparams.get("generations",200), pop_size, mut_rate, t_k, True)
            if first_gen is not None:
                gens_list.append(first_gen)
        if gens_list:
            labels.append(label)
            means.append(np.mean(gens_list))
            errors.append(np.std(gens_list)/np.sqrt(len(gens_list)))

    plt.figure(figsize=(10,5))
    plt.bar(labels, means, yerr=errors, capsize=5)
    plt.title("FI-2Pop Mean Generations to Converge per Task (HARD)")
    plt.xlabel("Task"); plt.ylabel("Avg. Gens to First Feasible")
    plt.tight_layout(); plt.savefig(f"{FIGURES_DIRECTORY}/fi2pop_mean_convergence_bar_chart.png"); plt.close()
    
# ----------------------------------------------------------------------------
# CLI entrypoint (hard-only)
# ----------------------------------------------------------------------------
def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="FI-2Pop HARD-mode convergence sweeps"
    )
    parser.add_argument(
        "-l", "--load-hyperparameters",
        required=True,
        help="Path to YAML file of GA hyperparameters"
    )
    parser.add_argument(
        "-r", "--runs",
        type=int, default=30,
        help="Number of independent trials per path length or per task"
    )
    # flags to select which analysis to run
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Run only the binary-vs-path-length sweep"
    )
    parser.add_argument(
        "--combo",
        action="store_true",
        help="Run only the combo-vs-path-length sweep"
    )
    parser.add_argument(
        "--bar",
        action="store_true",
        help="Run only the per-task mean-convergence bar chart"
    )

    args = parser.parse_args()

    # If no flags are provided, default to running all three
    if not (args.binary or args.combo or args.bar):
        args.binary = args.combo = args.bar = True

    # load hyperparameters
    with open(args.load_hyperparameters) as f:
        hyperparams = yaml.safe_load(f)

    # 1) Binary sweep
    if args.binary:
        print("==> FI-2Pop Binary HARD Convergence")
        binary_convergence_over_path_lengths_fi2pop(
            sample_size=args.runs,
            hyperparams=hyperparams,
            hard=True
        )

    # 2) Combo sweep
    if args.combo:
        print("==> FI-2Pop Combo HARD Convergence")
        combo_convergence_over_path_lengths_fi2pop(
            sample_size=args.runs,
            hyperparams=hyperparams,
            hard=True
        )

    # 3) Per-task bar chart
    if args.bar:
        print("==> FI-2Pop Mean-Convergence Bar Chart")
        plot_avg_task_convergence_fi2pop(
            hyperparams=hyperparams,
            runs=args.runs
        )

    print("Done.")
    
if __name__ == "__main__":
    main()