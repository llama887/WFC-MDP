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
# Dual‐axis summary sweep over all combos & modes
# ----------------------------------------------------------------------------
def summary_sweep(
    hyperparams: dict[str, Any],
    sample_size: int,
    path_lengths: List[int]
) -> None:
    combos = [
        ("Binary",  lambda hard: [partial(binary_reward, target_path_length=L, hard=hard)]),
        ("B+Pond",  lambda hard: [partial(binary_reward, target_path_length=L, hard=hard), pond_reward]),
        ("B+River", lambda hard: [partial(binary_reward, target_path_length=L, hard=hard), river_reward]),
        ("B+Grass", lambda hard: [partial(binary_reward, target_path_length=L, hard=hard), grass_reward]),
        ("B+Hill",  lambda hard: [partial(binary_reward, target_path_length=L, hard=hard), hill_reward]),
    ]
    modes = [("hard", True), ("easy", False)]

    MAP_L, MAP_W = MAP_LENGTH, MAP_WIDTH
    MAX_G   = hyperparams.get("generations",    100)
    pop_size = hyperparams.get("population_size", 96)
    mut_rate = _compute_mutation_rate(hyperparams, MAP_L, MAP_W)
    t_k      = hyperparams.get("tournament_k",    3)

    for combo_label, builder in combos:
        for mode_label, hard_flag in modes:
            gens_to_conv = np.full((len(path_lengths), sample_size), np.nan)

            for i, L in enumerate(path_lengths):
                for run in range(sample_size):
                    print(f"[{combo_label}-{mode_label}] L={L}, run {run+1}/{sample_size}")
                    reward_fn = CombinedReward(builder(hard_flag))
                    _, _, first_gen, _, _ = evolve_fi2pop(
                        reward_fn, {}, MAX_G, pop_size, mut_rate, t_k, True
                    )
                    if first_gen is not None:
                        gens_to_conv[i, run] = first_gen

            means  = np.nanmean(gens_to_conv, axis=1)
            counts = np.sum(~np.isnan(gens_to_conv), axis=1)
            stderr = np.nanstd(gens_to_conv, axis=1)/np.sqrt(np.maximum(counts,1))
            frac   = counts / sample_size

            fig, ax1 = plt.subplots(figsize=(8,5))
            ax2 = ax1.twinx()
            ax1.errorbar(path_lengths, means, yerr=stderr, fmt='o-', capsize=4, label='Mean gens')
            ax2.bar(path_lengths, frac, width=(path_lengths[1]-path_lengths[0])*0.8,
                    alpha=0.3, label='Frac converged')

            ax1.set_xlabel('Desired Path Length')
            ax1.set_ylabel('Mean Gens to Converge')
            ax2.set_ylabel('Fraction Converged')
            ax1.set_title(f'{combo_label} ({mode_label}) Convergence')
            h1,l1 = ax1.get_legend_handles_labels()
            h2,l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1+h2, l1+l2, loc='upper left')

            out = f"{FIGURES_DIRECTORY}/{combo_label}_{mode_label}_summary.png".replace('+','p')
            fig.tight_layout(); fig.savefig(out); plt.close(fig)

# ----------------------------------------------------------------------------
# Green‐bar biome‐only convergence
# ----------------------------------------------------------------------------
def plot_biome_convergence_bar(
    hyperparams: dict[str, Any],
    runs: int = 30,
    hard: bool = True
) -> None:
    tasks = {
        "Pond":  pond_reward,
        "River": river_reward,
        "Grass": grass_reward,
        "Hill":  hill_reward,
    }
    MAP_L, MAP_W = MAP_LENGTH, MAP_WIDTH
    pop_size     = hyperparams.get("population_size", 48)
    mut_rate     = _compute_mutation_rate(hyperparams, MAP_L, MAP_W)
    t_k          = hyperparams.get("tournament_k",    3)
    MAX_G        = hyperparams.get("generations",    200)

    labels, means = [], []
    for name, fn in tasks.items():
        gens_list = []
        print(f"[BIOME-BAR] Task {name} ({'HARD' if hard else 'EASY'})")
        for i in range(runs):
            _, _, first_gen, _, _ = evolve_fi2pop(
                fn, {},
                generations=MAX_G,
                pop_size=pop_size,
                mutation_rate=mut_rate,
                tournament_k=t_k,
                return_first_gen=True
            )
            if first_gen is not None:
                gens_list.append(first_gen)
        if gens_list:
            labels.append(name)
            means.append(np.mean(gens_list))

    plt.figure(figsize=(8,5))
    x = np.arange(len(labels))
    plt.bar(x, means, color='green')
    plt.xticks(x, labels)
    plt.ylabel("Avg Gens to First Feasible")
    plt.title(f"FI-2Pop Biome Convergence ({'HARD' if hard else 'EASY'})")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIRECTORY}/fi2pop_biome_convergence_bar.png")
    plt.close()

# ----------------------------------------------------------------------------
# CLI entrypoint with fine‐grained flags
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FI-2Pop convergence: binary, combos, and bar graph"
    )
    parser.add_argument("-l","--load-hyperparameters", required=True,
                        help="Path to YAML of GA hyperparameters")
    parser.add_argument("-r","--runs", type=int, default=30,
                        help="Trials per path length / per task")
    parser.add_argument("--min-path", type=int, default=10,
                        help="Min target path length")
    parser.add_argument("--max-path", type=int, default=100,
                        help="Max target path length")
    parser.add_argument("--step",    type=int, default=10,
                        help="Step between path-lengths")

    # mutually exclusive easy/hard
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hard", action="store_true", help="Hard mode")
    group.add_argument("--easy", action="store_true", help="Easy/soft mode")

    # analysis flags
    parser.add_argument("--binary",       action="store_true", help="Binary alone")
    parser.add_argument("--binary-pond",  action="store_true", help="Binary + Pond")
    parser.add_argument("--binary-river", action="store_true", help="Binary + River")
    parser.add_argument("--binary-grass", action="store_true", help="Binary + Grass")
    parser.add_argument("--binary-hill",  action="store_true", help="Binary + Hill")
    parser.add_argument("--bar-graph",    action="store_true", help="Green biome-only bar graph")

    args = parser.parse_args()

    # default mode hard
    hard_flag = True if args.hard or not args.easy else False

    # default to all analyses
    if not any([args.binary, args.binary_pond, args.binary_river,
                args.binary_grass, args.binary_hill, args.bar_graph]):
        args.binary = args.binary_pond = args.binary_river = True
        args.binary_grass = args.binary_hill = True
        args.bar_graph = True

    with open(args.load_hyperparameters) as f:
        hyperparams = yaml.safe_load(f)

    path_lengths = list(range(args.min_path, args.max_path+1, args.step))

    # run requested combos
    if args.binary:
        print("==> Running Binary summary")
        summary_sweep(hyperparams, args.runs, path_lengths)

    # bar graph
    if args.bar_graph:
        print("==> Running Biome bar graph")
        plot_biome_convergence_bar(hyperparams, runs=args.runs, hard=hard_flag)

    print("Done. All plots in 'figures/'")

if __name__ == "__main__":
    main()
