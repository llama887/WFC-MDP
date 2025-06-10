#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
import pickle
from enum import Enum

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from scipy.stats import truncnorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pygame
import yaml


class CrossOverMethod(Enum):
    UNIFORM = 0
    ONE_POINT = 1



# ----------------------------------------------------------------------------
# Task callbacks
# ----------------------------------------------------------------------------
from tasks.binary_task import binary_reward
from tasks.grass_task import grass_reward
from tasks.hill_task import hill_reward
from tasks.pond_task import pond_reward
from tasks.river_task import river_reward

from assets.biome_adjacency_rules import create_adjacency_matrix
from wfc import load_tile_images
from wfc_env import CombinedReward, WFCWrapper

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


class PopulationMember:
    def __init__(self, env: WFCWrapper):
        self.env = env
        self.action_sequence = np.array(
            [env.action_space.sample() for _ in range(env.map_length * env.map_width)]
        )
        self.reward: float = float("-inf")
        self.violation: int = sys.maxsize
        self.info: Dict[str, Any] = {}

    def mutate(
        self,
        number_of_actions_mutated_mean: int = 10,
        number_of_actions_mutated_standard_deviation: float = 10,
        action_noise_standard_deviation: float = 0.1,
    ):
        # pick a number of actions to mutate between 0 and len(self.action_sequence) by sampling from normal distribution
        if number_of_actions_mutated_standard_deviation == 0:
            number_of_actions_mutated = number_of_actions_mutated_mean
        else:
            lower_bound = (
                0 - number_of_actions_mutated_mean
            ) / number_of_actions_mutated_standard_deviation
            upper_bound = (
                len(self.action_sequence) - number_of_actions_mutated_mean
            ) / number_of_actions_mutated_standard_deviation
            number_of_actions_mutated = truncnorm.rvs(
                lower_bound,
                upper_bound,
                loc=number_of_actions_mutated_mean,
                scale=number_of_actions_mutated_standard_deviation,
            )
        number_of_actions_mutated = int(
            max(0, min(len(self.action_sequence), number_of_actions_mutated))
        )

        # mutate that number of actions by adding noise sampled from a normal distribution to all values in the action
        mutating_indices = np.random.choice(
            len(self.action_sequence), int(number_of_actions_mutated), replace=False
        )
        noise = np.random.normal(
            0,
            action_noise_standard_deviation,
            size=self.action_sequence[mutating_indices].shape,
        )
        self.action_sequence[mutating_indices] += noise

        # ensure results are between 0 and 1
        self.action_sequence[mutating_indices] = np.clip(
            self.action_sequence[mutating_indices], 0, 1
        )

    @staticmethod
    def crossover(
        p1: "PopulationMember", p2: "PopulationMember", method: CrossOverMethod = CrossOverMethod.ONE_POINT
    ) -> tuple["PopulationMember", "PopulationMember"]:
        if isinstance(method, int):
            method = CrossOverMethod(method)
        seq1 = p1.action_sequence
        seq2 = p2.action_sequence
        length = len(seq1)
        match method:
            case CrossOverMethod.ONE_POINT:
                point = random.randint(1, length - 1)
                child_seq1 = np.concatenate([seq1[:point], seq2[point:]])
                child_seq2 = np.concatenate([seq2[:point], seq1[point:]])
            case CrossOverMethod.UNIFORM:
                mask = np.random.rand(length, 1) < 0.5
                child_seq1 = np.where(mask, seq1, seq2)
                child_seq2 = np.where(mask, seq2, seq1)
            case _:
                raise ValueError(f"Unknown crossover method: {method!r}")

        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        c1.action_sequence, c2.action_sequence = child_seq1.copy(), child_seq2.copy()
        c1.reward, c2.reward = float("-inf"), float("-inf")
        c1.violation, c2.violation = sys.maxsize, sys.maxsize
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


def evaluate(env: WFCWrapper, actions: np.ndarray) -> Tuple[float, int, Dict[str, Any]]:
    total_reward = 0.0
    info: Dict[str, Any] = {}
    env.reset()
    for a in actions:
        _, r, done, trunc, info = env.step(a)
        total_reward += r
        if done or trunc:
            break
    violation = info.get("violations", info.get("contradictions"))
    if violation is None:
        violation = _count_contradictions(env)
    return total_reward, int(violation), info


def _mutate_clone(args):
    member, mean, stddev, noise = args
    member.mutate(mean, stddev, noise)
    return member


def reproduce_pair(
    args: tuple[
        "PopulationMember",  # parent1
        "PopulationMember",  # parent2
        int,  # mean
        float,  # stddev
        float,  # action_noise
        CrossOverMethod,  # method
    ],
) -> tuple["PopulationMember", "PopulationMember"]:
    """
    Given (p1, p2, mean, stddev, noise), perform crossover + mutate
    and return two children.
    """
    p1, p2, mean, stddev, noise, method = args
    c1, c2 = PopulationMember.crossover(p1, p2, method=method)
    c1.mutate(mean, stddev, noise)
    c2.mutate(mean, stddev, noise)
    return c1, c2


def _parallel_eval(gen: PopulationMember) -> PopulationMember:
    e = copy.deepcopy(gen.env)
    gen.reward, gen.violation, gen.info = evaluate(e, gen.action_sequence)
    return gen


def objective(
    trial,
    generations_per_trial: int,
    tasks_list: list[str] = None,
) -> float:
    """Objective function for Optuna hyperparameter optimization."""
    if not tasks_list:
        tasks_list = ["binary_hard"]

    # Suggest new hyperparameters
    hyperparams = {
        "number_of_actions_mutated_mean": trial.suggest_int(
            "number_of_actions_mutated_mean", 1, 200
        ),
        "number_of_actions_mutated_standard_deviation": trial.suggest_float(
            "number_of_actions_mutated_standard_deviation", 0.0, 200.0
        ),
        "action_noise_standard_deviation": trial.suggest_float(
            "action_noise_standard_deviation", 0.01, 1.0, log=True
        ),
        "survival_rate": trial.suggest_float("survival_rate", 0.1, 0.8),
        "cross_over_method": trial.suggest_categorical("cross_over_method", [0, 1]),
        "cross_or_mutate_proportion": trial.suggest_float(
            "cross_or_mutate_proportion", 0.0, 1.0
        ),
    }

    # Build reward function
    reward_funcs = []
    is_combo = len(tasks_list) > 1
    for task in tasks_list:
        if task.startswith("binary_"):
            target_length = 40 if is_combo else 80
            hard = task == "binary_hard"
            reward_funcs.append(
                partial(binary_reward, target_path_length=target_length, hard=hard)
            )
        else:
            reward_funcs.append(globals()[f"{task}_reward"])

    reward_fn = (
        CombinedReward(reward_funcs) if len(reward_funcs) > 1 else reward_funcs[0]
    )

    total_reward = 0
    NUMBER_OF_SAMPLES = 10  # As in evolution.py objective
    for i in range(NUMBER_OF_SAMPLES):
        best_agent, _, _, _ = evolve_fi2pop(
            reward_fn=reward_fn,
            task_args={},
            generations=generations_per_trial,
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
            cross_over_method=CrossOverMethod(hyperparams["cross_over_method"]),
            cross_or_mutate_proportion=hyperparams["cross_or_mutate_proportion"],
            patience=50,
        )
        reward = best_agent.reward if best_agent else float("-inf")
        print(f"Best reward at sample {i + 1}/{NUMBER_OF_SAMPLES}: {reward}")
        total_reward += reward

    return total_reward


def evolve_fi2pop(
    reward_fn: Any,
    task_args: Dict[str, Any],
    generations: int = 100,
    population_size: int = 48,
    number_of_actions_mutated_mean: int = 10,
    number_of_actions_mutated_standard_deviation: float = 10.0,
    action_noise_standard_deviation: float = 0.1,
    survival_rate: float = 0.2,
    cross_over_method: CrossOverMethod = CrossOverMethod.ONE_POINT,
    cross_or_mutate_proportion: float = 0.7,
    patience: int = 50,
) -> Tuple[Optional[PopulationMember], int, List[float], List[float]]:
    reward_callable = partial(reward_fn, **task_args)
    best_hist: List[float] = []
    median_hist: List[float] = []
    best_feasible_agent: Optional[PopulationMember] = None
    best_mean_elite: float | None = None
    patience_counter = 0

    # Initial population
    combined = [PopulationMember(make_env(reward_callable)) for _ in range(population_size * 2)]
    with Pool(min(cpu_count(), len(combined))) as P:
        combined = P.map(_parallel_eval, combined)

    feasible = [g for g in combined if g.violation == 0]
    infeasible = [g for g in combined if g.violation > 0]

    # Truncate initial populations
    feasible = sorted(feasible, key=lambda g: g.reward, reverse=True)[:population_size]
    infeasible = sorted(infeasible, key=lambda g: g.violation)[:population_size]

    if feasible:
        best_feasible_agent = copy.deepcopy(max(feasible, key=lambda g: g.reward))

    # Record gen 0
    all_rewards = [g.reward for g in feasible] + [g.reward for g in infeasible]
    best_hist.append(max(all_rewards) if all_rewards else float("-inf"))
    median_hist.append(float(np.median(all_rewards)) if all_rewards else float("-inf"))

    final_gen = generations
    for gen in range(1, generations + 1):
        # --- 1. Evaluation of newly feasible individuals ---
        # (This step is implicitly handled by the reproduction logic now)
        with Pool(min(cpu_count(), len(infeasible))) as P:
            infeasible = P.map(_parallel_eval, infeasible)

        newly_feasible = [g for g in infeasible if g.violation == 0]
        feasible.extend(newly_feasible)
        infeasible = [g for g in infeasible if g.violation > 0]

        # --- 2. Track global best ---
        if feasible:
            current_best_in_gen = max(feasible, key=lambda g: g.reward)
            if (
                best_feasible_agent is None
                or current_best_in_gen.reward > best_feasible_agent.reward
            ):
                best_feasible_agent = copy.deepcopy(current_best_in_gen)

        # --- 3. Record history ---
        all_rewards = [g.reward for g in feasible] + [g.reward for g in infeasible]
        best_hist.append(max(all_rewards) if all_rewards else float("-inf"))
        median_hist.append(
            float(np.median(all_rewards)) if all_rewards else float("-inf")
        )

        # --- 4. Elitist Selection ---
        num_feasible_survivors = max(2, int(len(feasible) * survival_rate))
        feasible_survivors = sorted(feasible, key=lambda g: g.reward, reverse=True)[
            :num_feasible_survivors
        ]

        num_infeasible_survivors = max(2, int(len(infeasible) * survival_rate))
        infeasible_survivors = sorted(infeasible, key=lambda g: g.violation)[
            :num_infeasible_survivors
        ]

        # --- 5. Early stopping on mean-elite reward of feasible population ---
        if feasible_survivors:
            elite_rewards = [g.reward for g in feasible_survivors]
            mean_elite_val = float(np.mean(elite_rewards))
        else:
            mean_elite_val = float("-inf")

        if best_mean_elite is None or mean_elite_val > best_mean_elite:
            best_mean_elite = mean_elite_val
            patience_counter = 0
        else:
            patience_counter += 1

        achieved_max = best_feasible_agent and best_feasible_agent.reward >= 0.0
        if achieved_max or patience_counter >= patience:
            print(f"[DEBUG] Converged at generation {gen}")
            if achieved_max:
                print(
                    f"[DEBUG] Best agent achieved max reward: {best_feasible_agent.reward}"
                )
            else:
                print(f"[DEBUG] Patience counter reached {patience_counter}")
                print(f"[DEBUG] Best mean-elite reward: {best_mean_elite}")
            final_gen = gen
            break

        # --- 6. Reproduction ---
        offspring = []

        def generate_offspring_from_pool(
            survivors: List[PopulationMember], num_needed: int
        ) -> List[PopulationMember]:
            if not survivors or num_needed <= 0:
                return []

            new_children = []
            n_crossover = int(round(num_needed * cross_or_mutate_proportion))
            n_mutation = num_needed - n_crossover

            # Crossover pairs
            n_pairs = (n_crossover + 1) // 2
            pairs_args = []
            for _ in range(n_pairs):
                p1, p2 = (
                    random.sample(survivors, 2)
                    if len(survivors) >= 2
                    else (survivors[0], survivors[0])
                )
                pairs_args.append(
                    (
                        p1,
                        p2,
                        number_of_actions_mutated_mean,
                        number_of_actions_mutated_standard_deviation,
                        action_noise_standard_deviation,
                        cross_over_method,
                    )
                )
            if pairs_args:
                with Pool(min(cpu_count(), len(pairs_args))) as pool:
                    results = pool.map(reproduce_pair, pairs_args)
                crossover_children = [
                    child for pair in results for child in pair
                ][:n_crossover]
                new_children.extend(crossover_children)

            # Mutation-only children
            mutation_args = [
                (
                    copy.deepcopy(random.choice(survivors)),
                    number_of_actions_mutated_mean,
                    number_of_actions_mutated_standard_deviation,
                    action_noise_standard_deviation,
                )
                for _ in range(n_mutation)
            ]
            if mutation_args:
                with Pool(min(cpu_count(), len(mutation_args))) as pool:
                    mutated = pool.map(_mutate_clone, mutation_args)
                new_children.extend(mutated)

            return new_children

        # Generate offspring for both populations
        n_feasible_offspring = population_size - len(feasible_survivors)
        offspring.extend(
            generate_offspring_from_pool(feasible_survivors, n_feasible_offspring)
        )

        n_infeasible_offspring = population_size - len(infeasible_survivors)
        offspring.extend(
            generate_offspring_from_pool(infeasible_survivors, n_infeasible_offspring)
        )

        if not offspring:
            print(f"Gen {gen:04d} | No offspring to evaluate. Stopping.")
            final_gen = gen
            break

        # --- 7. Evaluate and form next generation ---
        with Pool(min(cpu_count(), len(offspring))) as P:
            offspring = P.map(_parallel_eval, offspring)

        feasible = feasible_survivors + [g for g in offspring if g.violation == 0]
        infeasible = infeasible_survivors + [g for g in offspring if g.violation > 0]

        # Final truncation to maintain population size
        feasible = sorted(feasible, key=lambda g: g.reward, reverse=True)[
            :population_size
        ]
        infeasible = sorted(infeasible, key=lambda g: g.violation)[:population_size]

        best_reward_str = f"{feasible[0].reward:.3f}" if feasible else "N/A"
        print(
            f"Gen {gen:04d} | Feas {len(feasible):02d} | Infeas {len(infeasible):02d} | BestReward {best_reward_str}"
        )

    return best_feasible_agent, final_gen, best_hist, median_hist


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Render best agent (adapted from evolution.py)
# ----------------------------------------------------------------------------
def render_best_agent(
    env: WFCWrapper, best_agent: PopulationMember, tile_images, task_name: str = ""
):
    """Renders the action sequence of the best agent and saves the final map."""
    if not best_agent:
        print("No best agent found to render.")
        return

    pygame.init()
    SCREEN_WIDTH = env.map_width * 32
    SCREEN_HEIGHT = env.map_length * 32
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Best FI-2Pop Evolved WFC Map - {task_name}")
    final_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    env.reset()
    total_reward = 0
    print("Info:", best_agent.info)
    print("Rendering best agent's action sequence...")

    # Re-run the simulation to get the final grid state for rendering
    for action in best_agent.action_sequence:
        _, reward, terminate, truncate, _ = env.step(action)
        total_reward += reward
        if terminate or truncate:
            break

    # Render final state
    final_surface.fill((0, 0, 0))
    for y in range(env.map_length):
        for x in range(env.map_width):
            cell_set = env.grid[y][x]
            if len(cell_set) == 1:
                tile_name = next(iter(cell_set))
                if tile_name in tile_images:
                    final_surface.blit(tile_images[tile_name], (x * 32, y * 32))
            elif len(cell_set) == 0:
                pygame.draw.rect(final_surface, (255, 0, 0), (x * 32, y * 32, 32, 32))

    # --- Draw the path with smooth curves ---
    if "longest_path" in best_agent.info:
        path_indices = best_agent.info["longest_path"]
        if path_indices and len(path_indices) > 1:
            path_points = [
                (idx[1] * 32 + 16, idx[0] * 32 + 16) for idx in path_indices
            ]
            if len(path_points) > 1:
                pygame.draw.lines(final_surface, (255, 0, 0), False, path_points, 3)
            for point in path_points:
                pygame.draw.circle(final_surface, (255, 0, 0), point, 4)

    screen.blit(final_surface, (0, 0))
    pygame.display.flip()

    # Save the final rendered map
    if task_name:
        os.makedirs("wfc_reward_img", exist_ok=True)
        filename = f"wfc_reward_img/fi2pop_{task_name}_{best_agent.reward:.2f}.png"
        pygame.image.save(final_surface, filename)
        print(f"Saved final map to {filename}")

    print(f"Final map reward for the best agent: {total_reward:.4f}")
    print(f"Best agent reward during evolution: {best_agent.reward:.4f}")

    print("Displaying final map for 5 seconds...")
    start_time = time.time()
    while time.time() - start_time < 5:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
    pygame.quit()


# ----------------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evolve WFC agents with FI-2Pop.")
    parser.add_argument(
        "--load-hyperparameters",
        type=str,
        default=None,
        help="Path to a YAML file containing hyperparameters to load.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations to run evolution for.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Number of trials for Optuna hyperparameter search.",
    )
    parser.add_argument(
        "--generations-per-trial",
        type=int,
        default=10,
        help="Number of generations for each Optuna trial.",
    )
    parser.add_argument(
        "--hyperparameter-dir",
        type=str,
        default="hyperparameters",
        help="Directory to save/load hyperparameters.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="best_fi2pop_hyperparameters.yaml",
        help="Filename for the saved hyperparameters YAML.",
    )
    parser.add_argument(
        "--best-agent-pickle",
        type=str,
        help="Path to a pickled agent to load and render.",
    )
    parser.add_argument(
        "--override-patience",
        type=int,
        default=None,
        help="Override the patience setting from YAML.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        choices=["binary_easy", "binary_hard", "river", "pond", "grass", "hill"],
        help="Task(s) to use. For combo tasks, specify multiple --task flags.",
    )
    args = parser.parse_args()
    if not args.task:
        args.task = ["binary_easy"]

    task_rewards = {
        "binary_easy": partial(binary_reward, target_path_length=80),
        "binary_hard": partial(binary_reward, target_path_length=80, hard=True),
        "river": river_reward,
        "pond": pond_reward,
        "grass": grass_reward,
        "hill": hill_reward,
    }

    if len(args.task) == 1:
        selected_reward = task_rewards[args.task[0]]
    else:
        selected_reward = CombinedReward([task_rewards[task] for task in args.task])

    env = make_env(selected_reward)
    hyperparams = {}
    best_agent = None

    if args.load_hyperparameters:
        print(f"Loading hyperparameters from: {args.load_hyperparameters}")
        with open(args.load_hyperparameters, "r") as f:
            hyperparams = yaml.safe_load(f)
        if args.override_patience is not None:
            hyperparams["patience"] = args.override_patience
        print("Successfully loaded hyperparameters:", hyperparams)

        start_time = time.time()
        (
            best_agent,
            generations,
            best_agent_rewards,
            median_agent_rewards,
        ) = evolve_fi2pop(
            reward_fn=selected_reward,
            task_args={},
            generations=args.generations,
            population_size=hyperparams.get("population_size", 48),
            number_of_actions_mutated_mean=hyperparams[
                "number_of_actions_mutated_mean"
            ],
            number_of_actions_mutated_standard_deviation=hyperparams[
                "number_of_actions_mutated_standard_deviation"
            ],
            action_noise_standard_deviation=hyperparams[
                "action_noise_standard_deviation"
            ],
            survival_rate=hyperparams.get("survival_rate", 0.2),
            cross_over_method=CrossOverMethod(
                hyperparams.get("cross_over_method", 1)
            ),
            cross_or_mutate_proportion=hyperparams.get(
                "cross_or_mutate_proportion", 0.7
            ),
            patience=hyperparams.get("patience", 50),
        )
        end_time = time.time()
        print(f"Evolution finished in {end_time - start_time:.2f} seconds.")
        print(f"Evolved for a total of {generations} generations")

        task_str = "_".join(args.task)
        x_axis = np.arange(len(median_agent_rewards))
        plt.plot(x_axis, best_agent_rewards, label="Best Agent Per Generation")
        plt.plot(x_axis, median_agent_rewards, label="Median Agent Per Generation")
        plt.legend()
        plt.title(f"FI-2Pop Performance: {task_str}")
        plt.xlabel("Generations")
        plt.ylabel("Reward")
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/fi2pop_performance_{task_str}.png")
        plt.close()

    elif args.optuna_trials > 0:
        import optuna

        print(f"Running Optuna search for {args.optuna_trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial, args.generations_per_trial, tasks_list=args.task
            ),
            n_trials=args.optuna_trials,
        )
        hyperparams = study.best_params
        print("Best hyperparameters found:", hyperparams)

        os.makedirs(args.hyperparameter_dir, exist_ok=True)
        output_path = os.path.join(args.hyperparameter_dir, args.output_file)
        with open(output_path, "w") as f:
            yaml.dump(hyperparams, f)
        print(f"Saved best hyperparameters to: {output_path}")

    elif args.best_agent_pickle:
        with open(args.best_agent_pickle, "rb") as f:
            best_agent = pickle.load(f)

    if best_agent:
        task_name = "_".join(args.task)
        render_best_agent(env, best_agent, TILE_IMAGES, task_name)

        AGENT_DIR = "agents"
        os.makedirs(AGENT_DIR, exist_ok=True)
        filename = f"{AGENT_DIR}/best_fi2pop_{task_name}_reward_{best_agent.reward:.2f}_agent.pkl"
        with open(filename, "wb") as f:
            pickle.dump(best_agent, f)
        print(f"Saved best agent to {filename}")

    print("Script finished.")


if __name__ == "__main__":
    main()
