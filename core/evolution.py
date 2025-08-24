import os
import sys
from timeit import default_timer as timer

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import math
import os
import pickle
import random
import time
from enum import Enum
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pygame
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import truncnorm
from tqdm import tqdm

from assets.biome_adjacency_rules import create_adjacency_matrix
from core.wfc import (  # We might not need render_wfc_grid if we keep console rendering
    load_tile_images,
)
from core.wfc_env import CombinedReward, WFCWrapper
from tasks.binary_task import binary_percent_water, binary_reward
from tasks.grass_task import grass_reward
from tasks.hill_task import hill_reward
from tasks.pond_task import pond_reward
from tasks.river_task import river_reward


class CrossOverMethod(Enum):
    UNIFORM = 0
    ONE_POINT = 1

def render_boolean_grid(grid_3d, tile_images, all_tiles, tile_size=32):
    """Renders grid state to surface without display.
    grid_3d: 3D numpy array of shape (height, width, num_tiles)
    tile_images: dict mapping tile names to pygame surfaces
    all_tiles: list of tile names (indexable by tile index)
    tile_size: size of each tile in pixels
    """
    height, width, _ = grid_3d.shape
    surface = pygame.Surface((width * tile_size, height * tile_size))
    surface.fill((255, 255, 255))
    for y in range(height):
        for x in range(width):
            cell = grid_3d[y, x]
            if np.sum(cell) == 0:  # Contradiction
                pygame.draw.rect(surface, (255, 0, 0), 
                                (x*tile_size, y*tile_size, tile_size, tile_size))
            elif np.sum(cell) == 1:  # Collapsed
                idx = np.argmax(cell)
                tile_name = all_tiles[idx]
                if tile_name in tile_images:
                    img = tile_images[tile_name]
                    surface.blit(img, (x*tile_size, y*tile_size))
                else:  # Missing tile fallback
                    pygame.draw.rect(surface, (0, 255, 0), 
                                   (x*tile_size, y*tile_size, tile_size, tile_size))
            else:  # Superposition
                pygame.draw.rect(surface, (200, 200, 200), 
                               (x*tile_size, y*tile_size, tile_size, tile_size))
    return surface

# REQUIRED RENDERING UTILITIES
def deepcopy_env_state(env):
    """Safely clones environment state for rendering"""
    return {
        'map_length': env.map_length,
        'map_width': env.map_width,
        'grid': copy.deepcopy(env.grid),
        'all_tiles': copy.copy(env.all_tiles)  # Use all_tiles instead of tile_symbols
    }


def _mutate_clone(args):
    member, mean, stddev, noise = args
    member.mutate(mean, stddev, noise)
    return member


class PopulationMember:
    def __init__(
        self, env: WFCWrapper, genotype_representation: Literal["1d", "2d"] = "1d"
    ):
        self.env: WFCWrapper = copy.deepcopy(env)
        self.env.reset()
        self.reward: float = float("-inf")
        self.genotype_representation: Literal["1d", "2d"] = genotype_representation
        self.action_sequence: np.ndarray = np.array(
            [
                self.env.action_space.sample()
                for _ in range(env.map_length * env.map_width)
            ]
        )
        self.info = {}

    def mutate(
        self,
        number_of_actions_mutated_mean: int = 10,
        number_of_actions_mutated_standard_deviation: float = 10,
        action_noise_standard_deviation: float = 0.1,
    ):
        # pick a number of actions to mutate between 0 and len(self.action_sequence) by sampling from normal distribution
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

    def run_action_sequence(self):
        start_time = timer()
        self.reward = 0
        observation, _ = self.env.reset()
        if self.genotype_representation == "1d":
            for idx, action in enumerate(self.action_sequence):
                _, reward, terminate, truncate, info = self.env.step(action)
                self.reward += reward
                self.info = info
                if terminate or truncate:
                    break
        if self.genotype_representation == "2d":
            truncate = False
            terminate = False
            while not (terminate or truncate):
                next_collapse_x, next_collapse_y = map(int, observation[-2:])
                # print(f"Next collapse: {next_collapse_x}, {next_collapse_y}")
                flattened_index = next_collapse_y * self.env.map_width + next_collapse_x
                observation, reward, terminate, truncate, info = self.env.step(
                    self.action_sequence[flattened_index]
                )
                self.reward += reward
                self.info = info
        end_time = timer()
        self.info["wfc_rollout_time"] = end_time - start_time

    @staticmethod
    def crossover(
        parent1: "PopulationMember",
        parent2: "PopulationMember",
        method: CrossOverMethod = CrossOverMethod.ONE_POINT,
    ) -> tuple["PopulationMember", "PopulationMember"]:
        if isinstance(method, int):
            method = CrossOverMethod(method)
        seq1 = parent1.action_sequence
        seq2 = parent2.action_sequence
        length = len(seq1)
        match method:
            case CrossOverMethod.ONE_POINT:
                # pick a crossover point (not at the extremes)
                point = np.random.randint(1, length)
                # child1 takes seq1[:point] + seq2[point:]
                child_seq1 = np.concatenate([seq1[:point], seq2[point:]])
                # child2 takes seq2[:point] + seq1[point:]
                child_seq2 = np.concatenate([seq2[:point], seq1[point:]])
            case CrossOverMethod.UNIFORM:
                # mask[i,0] says “choose parent1’s action-vector at time i”
                mask = np.random.rand(length, 1) < 0.5

                child_seq1 = np.where(mask, seq1, seq2)
                child_seq2 = np.where(mask, seq2, seq1)

            case _:
                raise ValueError(f"Unknown crossover method: {method!r}")

        # build child objects with fresh deep‐copied envs
        child1 = PopulationMember(parent1.env)
        child2 = PopulationMember(parent2.env)
        # overwrite their action sequences
        child1.action_sequence = child_seq1.copy()
        child2.action_sequence = child_seq2.copy()
        # reset their rewards
        child1.reward = float("-inf")
        child2.reward = float("-inf")

        return child1, child2


def run_member(member: PopulationMember):
    member.env.reset()
    member.run_action_sequence()
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


def evolve(
    env: WFCWrapper,
    generations: int = 100,
    population_size: int = 48,
    number_of_actions_mutated_mean: int = 10,
    number_of_actions_mutated_standard_deviation: float = 10.0,
    action_noise_standard_deviation: float = 0.1,
    survival_rate: float = 0.2,
    cross_over_method: CrossOverMethod = CrossOverMethod.ONE_POINT,
    patience: int = 50,
    qd: bool = False,
    genotype_representation: Literal["1d", "2d"] = "1d",
    cross_or_mutate_proportion: float = 0.7,
    random_offspring_proportion: float = 0.1,
    gen_save_dir: str | None = None,
    tile_images: dict | None = None,
    task_info: str = "",
) -> tuple[
    list[PopulationMember],  # final population
    PopulationMember,  # global best agent
    int,  # generation at which we stopped
    list[float],  # best‐agent reward history
    list[float],  # mean‐elite reward history
]:
    """
    Standard EA if qd=False; QD selection + global reproduction if qd=True.
    Early stopping now depends on the *mean of the elites* (the survivors), not the mean
    of all individuals. Returns:
      1) final population (list of PopulationMember)
      2) global best agent (PopulationMember)
      3) the generation at which we stopped (int)
      4) list of best‐agent rewards by generation (list[float])
      5) list of mean‐elite rewards by generation  (list[float])
    """

    # --- 1) Initialization ---
    population = [
        PopulationMember(env, genotype_representation=genotype_representation)
        for _ in range(population_size)
    ]
    best_agent: PopulationMember | None = None
    best_agent_rewards: list[float] = []
    mean_elite_rewards: list[float] = []
    patience_counter = 0

    # Track the best mean‐elite reward seen so far:
    best_mean_elite: float | None = None

    for gen in tqdm(range(1, generations + 1), desc="Generations"):
        # --- 2) Evaluate entire population (in parallel) ---
        with Pool(min(cpu_count() * 2, len(population))) as pool:
            population = pool.map(run_member, population)


        # --- 3) Gather fitnesses and track best individual ---
        fitnesses = np.array([m.reward for m in population])
        best_idx = int(np.argmax(fitnesses))
        best_reward = float(fitnesses[best_idx])

        # Record the best-agent reward for this generation
        best_agent_rewards.append(best_reward)

        # --- 4) Track global best‐agent (based on max reward) ---
        if best_agent is None or best_reward > best_agent.reward:
            best_agent = copy.deepcopy(population[best_idx])
        # best_agent.env.render_mode = 'human'
        # print(best_agent.env.render())
        # render_best_agent(
        #     best_agent.env, best_agent, tile_images, task_name="Best Agent"
        # )

        rollout_times = np.array([m.info["wfc_rollout_time"] for m in population])
        mean_rollout_time = np.mean(rollout_times)
        print(f"[DEBUG] Mean rollout time: {mean_rollout_time:.2f} seconds")

        # --- 5) Selection step (fitness‐based or QD‐based) ---
        if not qd:
            # Standard EA: pick top‐N by reward
            sorted_pop = sorted(population, key=lambda m: m.reward, reverse=True)
            number_of_surviving_members = max(2, int(population_size * survival_rate))
            survivors = sorted_pop[:number_of_surviving_members]
        else:
            # QD selection: cluster on “qd_score”
            scores = np.array([m.info["qd_score"] for m in population])
            Z = linkage(scores.reshape(-1, 1), method="ward")
            cutoff = np.median(Z[:, 2])
            labels = fcluster(Z, t=cutoff, criterion="distance")

            survivors = []
            for cluster in np.unique(labels):
                members = [
                    population[i] for i, lbl in enumerate(labels) if lbl == cluster
                ]
                members.sort(
                    key=lambda m: m.info.get("qd_score", m.reward), reverse=True
                )
                num_in_cluster = max(1, int(len(members) * survival_rate))
                survivors.extend(members[:num_in_cluster])

            # Ensure at least two survivors overall
            if len(survivors) < 2:
                pop_by_fit = sorted(population, key=lambda m: m.reward, reverse=True)
                survivors = pop_by_fit[:2]

        # Compute mean reward over the elites (survivors)
        elite_rewards = [m.reward for m in survivors]
        mean_elite_val = float(np.mean(elite_rewards))
        assert mean_elite_val <= 0, f"Mean elite reward should be non-positive: {mean_elite_val}"
        mean_elite_rewards.append(mean_elite_val)

        # Image saving per generation (if enabled)\
        if (gen_save_dir and 
            tile_images is not None and
            population[best_idx].reward > -float("inf")):
            assert population[best_idx].reward <= 0, f"Best agent reward should be non-positive: {population[best_idx].reward}\n{population[best_idx].info}"
            
            pygame_initialized = False
            timestamp = int(time.time())
            filename = f"gen_{gen:04d}_{task_info}_reward_{population[best_idx].reward:.2f}_{timestamp}.png"
            full_path = os.path.join(gen_save_dir, filename)
            
            try:
                # Ensure directory exists
                Path(gen_save_dir).mkdir(parents=True, exist_ok=True)
                # Initialize pygame safely
                pygame.init()
                pygame_initialized = True
            except Exception:
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
                pygame.init()
                pygame_initialized = True

                # Get environment state
                env_state = deepcopy_env_state(population[best_idx].env)
                
                # Render using helper function
                surface = render_boolean_grid(
                    env_state['grid'],
                    tile_images,
                    env_state['all_tiles']
                )
                
                # Save image
                pygame.image.save(surface, full_path)
                print(f"Saved generation {gen} image: {full_path}")
                
            except Exception as save_error:
                print(f"Error saving generation {gen} image: {save_error}")
            finally:
                if pygame_initialized:
                    pygame.quit()
                    # Clean up driver setting
                    if 'SDL_VIDEODRIVER' in os.environ:
                        del os.environ['SDL_VIDEODRIVER']
        else:
            print(f"Skipping image save for gen {gen} - tile_images not provided")

        # --- 6) Early stopping on *mean‐elite* reward ---
        if best_mean_elite is None or mean_elite_val > best_mean_elite:
            best_mean_elite = mean_elite_val
            patience_counter = 0
        else:
            patience_counter += 1

        # If someone hit “achieved_max_reward,” stop immediately:
        achieved_max = population[best_idx].info.get("achieved_max_reward", False)
        if achieved_max or patience_counter >= patience:
            print(
                f"[DEBUG] Converged at generation {gen}"
                if achieved_max
                else f"[DEBUG] Stopping early at generation {gen} due to patience."
            )
            print(f"[DEBUG] Best agent reward: {best_agent.reward}")
            print(f"[DEBUG] Mean‐elite reward: {best_mean_elite}")
            print(f"[DEBUG] Patience counter: {patience_counter}")
            return population, best_agent, gen, best_agent_rewards, mean_elite_rewards

        # --- 7) Reproduction step ---
        number_of_surviving_members = len(survivors)
        n_offspring = population_size - number_of_surviving_members
        offspring = []

        # 7a: Generate random agents
        n_random = max(0, int(round(n_offspring * random_offspring_proportion)))
        for _ in range(n_random):
            new_agent = PopulationMember(
                env, genotype_representation=genotype_representation
            )
            offspring.append(new_agent)

        remaining_needed = n_offspring - n_random

        # 7b: Generate crossover and mutation offspring
        if remaining_needed > 0:
            n_crossover = int(round(remaining_needed * cross_or_mutate_proportion))
            n_mutation = remaining_needed - n_crossover

            # Crossover pairs
            n_pairs = math.ceil(n_crossover / 2)
            pairs_args = []
            for _ in range(n_pairs):
                if len(survivors) >= 2:
                    p1, p2 = random.sample(survivors, 2)
                else:
                    p1 = p2 = survivors[0]
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
                n_procs = min(cpu_count() * 2, len(pairs_args))
                with Pool(n_procs) as pool:
                    results = pool.map(reproduce_pair, pairs_args)
                crossover_children = [child for pair in results for child in pair][
                    :n_crossover
                ]
            else:
                crossover_children = []
            offspring.extend(crossover_children)

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
                n_procs = min(cpu_count() * 2, len(mutation_args))
                with Pool(n_procs) as pool:
                    mutated = pool.map(_mutate_clone, mutation_args)
            else:
                mutated = []
            offspring.extend(mutated)

        # Ensure exact population size
        if len(offspring) < n_offspring:
            for _ in range(n_offspring - len(offspring)):
                extra = copy.deepcopy(random.choice(survivors))
                extra.mutate(
                    number_of_actions_mutated_mean,
                    number_of_actions_mutated_standard_deviation,
                    action_noise_standard_deviation,
                )
                offspring.append(extra)

        # --- 8) Form next generation ---
        population = survivors + offspring

    # If we exhaust all generations without early stopping:
    return population, best_agent, generations, best_agent_rewards, mean_elite_rewards


# --- Optuna Objective Function ---


def objective(
    trial: optuna.Trial,
    generations_per_trial: int,
    qd: bool = False,
    tasks_list: list[str] = None,
) -> float:
    """Objective function for Optuna hyperparameter optimization."""

    # Set default task list if not provided
    if not tasks_list:
        tasks_list = ["binary_hard"]

    # Suggest new hyperparameters (fixed population_size=48, patience=50)
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
        "cross_or_mutate": trial.suggest_float("cross_or_mutate", 0.0, 1.0),
        "random_offspring": trial.suggest_float("random_offspring", 0.0, 0.0),
    }
    # Fixed parameters
    population_size = 48
    patience = 50

    # Build reward function with fixed path lengths
    reward_funcs = []
    is_combo = len(tasks_list) > 1

    for task in tasks_list:
        if task.startswith("binary_"):
            # Use fixed path lengths: 80 for standalone, 40 for combos
            target_length = 40 if is_combo else 80
            hard = task == "binary_hard"
            reward_funcs.append(
                partial(
                    binary_reward,
                    target_path_length=target_length,
                    hard=hard,
                )
            )
        else:
            reward_funcs.append(globals()[f"{task}_reward"])

    reward_fn = (
        CombinedReward(reward_funcs) if len(reward_funcs) > 1 else reward_funcs[0]
    )

    # Construct Env
    MAP_LENGTH = 15
    MAP_WIDTH = 20

    total_reward = 0
    num_tiles = len(tile_symbols)

    NUMBER_OF_SAMPLES = 10
    start_time = time.time()
    for i in range(NUMBER_OF_SAMPLES):
        base_env = WFCWrapper(
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

        # Run evolution with suggested hyperparameters
        _, best_agent, _, _, _ = evolve(
            env=base_env,
            generations=generations_per_trial,
            population_size=population_size,  # Fixed 48
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
            patience=patience,  # Fixed 50
            qd=qd,
            cross_or_mutate_proportion=hyperparams["cross_or_mutate"],
            random_offspring_proportion=hyperparams["random_offspring"],
        )
        print(f"Best reward at sample {i + 1}/{NUMBER_OF_SAMPLES}: {best_agent.reward}")
        total_reward += best_agent.reward
    end_time = time.time()

    # Return the best reward but with account for how long it took
    print(f"Total Reward: {total_reward} | Time: {end_time - start_time}")
    return total_reward


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
    pygame.display.set_caption(f"Best Evolved WFC Map - {task_name}")

    # Create a surface for saving the final map
    final_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    env.reset()
    total_reward = 0
    print("Info:", best_agent.info)
    print("Rendering best agent's action sequence...")

    for action in tqdm(best_agent.action_sequence, desc="Rendering Steps"):
        _, reward, terminate, truncate, _ = env.step(action)
        total_reward += reward

        # Clear screen
        screen.fill((0, 0, 0))
        final_surface.fill((0, 0, 0))  # Also clear the final surface

        # Render the current state to both surfaces
        for y in range(env.map_length):
            for x in range(env.map_width):
                cell_vec = env.grid[y][x]
                if cell_vec.sum() == 1:
                    cell_idx = np.argmax(cell_vec)
                    tile_name = env.all_tiles[cell_idx]
                    if tile_name in tile_images:
                        screen.blit(tile_images[tile_name], (x * 32, y * 32))
                        final_surface.blit(tile_images[tile_name], (x * 32, y * 32))
                    else:
                        # Fallback for missing tiles
                        pygame.draw.rect(
                            screen, (255, 0, 255), (x * 32, y * 32, 32, 32)
                        )
                        pygame.draw.rect(
                            final_surface, (255, 0, 255), (x * 32, y * 32, 32, 32)
                        )
                elif not np.any(cell_vec):
                    pygame.draw.rect(screen, (255, 0, 0), (x * 32, y * 32, 32, 32))
                    pygame.draw.rect(
                        final_surface, (255, 0, 0), (x * 32, y * 32, 32, 32)
                    )
                else:  # Superposition
                    pygame.draw.rect(screen, (100, 100, 100), (x * 32, y * 32, 32, 32))
                    pygame.draw.rect(
                        final_surface, (100, 100, 100), (x * 32, y * 32, 32, 32)
                    )

        pygame.display.flip()

        # Capture final frame if this is the last step
        if terminate or truncate:
            break

    # --- Draw the path with smooth curves ---
    if hasattr(best_agent.info, "get") and "longest_path" in best_agent.info:
        path_indices = best_agent.info["longest_path"]
        if path_indices and len(path_indices) > 1:
            print(f"Found path with {len(path_indices)} points")

            # Convert indices to Pygame coordinates (center of each tile)
            path_points = []
            for idx in path_indices:
                if isinstance(idx, (list, tuple, np.ndarray)) and len(idx) >= 2:
                    y, x = idx[0], idx[1]  # Assuming (y, x) format
                    center_x = x * 32 + 16  # Center of the tile
                    center_y = y * 32 + 16
                    path_points.append((center_x, center_y))

            # Draw smooth path using bezier curves if we have enough points
            if len(path_points) >= 3:
                # Create a list of points for smooth curve
                smooth_points = []

                # Add the first point
                smooth_points.append(path_points[0])

                # Add intermediate points with smoothing
                for i in range(1, len(path_points) - 1):
                    prev = path_points[i - 1]
                    curr = path_points[i]
                    next_p = path_points[i + 1]

                    # Calculate control points for bezier curve
                    ctrl1 = ((prev[0] + curr[0]) / 2, (prev[1] + curr[1]) / 2)
                    ctrl2 = ((curr[0] + next_p[0]) / 2, (curr[1] + next_p[1]) / 2)

                    # Generate points along the bezier curve
                    t_values = np.linspace(0, 1, 10)
                    for t in t_values:
                        x = (
                            (1 - t) ** 2 * ctrl1[0]
                            + 2 * (1 - t) * t * curr[0]
                            + t**2 * ctrl2[0]
                        )
                        y = (
                            (1 - t) ** 2 * ctrl1[1]
                            + 2 * (1 - t) * t * curr[1]
                            + t**2 * ctrl2[1]
                        )
                        smooth_points.append((x, y))

                # Add the last point
                smooth_points.append(path_points[-1])

                # Draw the smooth path on both surfaces
                if len(smooth_points) > 1:
                    pygame.draw.lines(screen, (255, 0, 0), False, smooth_points, 3)
                    pygame.draw.lines(
                        final_surface, (255, 0, 0), False, smooth_points, 3
                    )

                    # Draw circles at the original path points
                    for point in path_points:
                        pygame.draw.circle(screen, (255, 0, 0), point, 4)
                        pygame.draw.circle(final_surface, (255, 0, 0), point, 4)
            elif len(path_points) > 1:
                # Fallback to straight lines if not enough points for bezier
                pygame.draw.lines(screen, (255, 0, 0), False, path_points, 3)
                pygame.draw.lines(final_surface, (255, 0, 0), False, path_points, 3)

                for point in path_points:
                    pygame.draw.circle(screen, (255, 0, 0), point, 4)
                    pygame.draw.circle(final_surface, (255, 0, 0), point, 4)

            pygame.display.flip()

    # Save the final rendered map
    if task_name:
        os.makedirs("wfc_reward_img", exist_ok=True)
        filename = f"wfc_reward_img/{task_name}_{best_agent.reward:.2f}.png"
        pygame.image.save(final_surface, filename)
        print(f"Saved final map to {filename}")

    print(f"Final map reward for the best agent: {total_reward:.4f}")
    print(f"Best agent reward during evolution: {best_agent.reward:.4f}")

    if best_agent.reward == 0.0:
        best_agent.info["achieved_max_reward"] = True
        print("Max reward of 0 achieved! Agent truly converged.")
    else:
        best_agent.info["achieved_max_reward"] = False
        print("Max reward NOT achieved. Agent stopped early without solving the task.")

    # Keep the window open for a bit
    print("Displaying final map for 5 seconds...")
    start_time = time.time()
    while time.time() - start_time < 5:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evolve WFC agents with optional hyperparameter tuning."
    )
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
        help="Number of generations to run evolution for (used when loading hyperparameters or after tuning).",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,  # Number of trials for Optuna optimization
        help="Number of trials to run for Optuna hyperparameter search.",
    )
    parser.add_argument(
        "--generations-per-trial",
        type=int,
        default=50,  # Fewer generations during tuning for speed
        help="Number of generations to run for each Optuna trial.",
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
        default="best_hyperparameters.yaml",
        help="Filename for the saved hyperparameters YAML.",
    )
    parser.add_argument(
        "--best-agent-pickle",
        type=str,
        help="Filename for the saved hyperparameters YAML.",
    )
    parser.add_argument(
        "--qd",
        action="store_true",
        default=False,
        help="Use QD mode for evolution.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        choices=["binary_easy", "binary_hard", "river", "pond", "grass", "hill"],
        help="Task(s) to use. For combo tasks, specify multiple --task flags",
    )
    parser.add_argument(
        "--override-patience",
        type=int,
        default=None,
        help="Override the patience setting from YAML.",
    )

    parser.add_argument(
        "--genotype-dimensions",
        type=int,
        choices=[1, 2],
        default=1,
        help="The dimensions of the genotype representation. 1d or 2d",
    )
    parser.add_argument(
        "--save-best-per-gen",
        action="store_true",
        default=False,
        help="Save image of best map per generation",
    )

    args = parser.parse_args()
    if not args.task:
        args.task = ["binary_easy"]

    # Define environment parameters
    MAP_LENGTH = 15
    MAP_WIDTH = 20

    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)

    task_rewards = {
        "binary_easy": partial(binary_reward, target_path_length=20),
        "binary_hard": partial(binary_reward, target_path_length=20, hard=True),
        "river": river_reward,
        "pond": pond_reward,
        "grass": grass_reward,
        "hill": hill_reward,
    }

    if len(args.task) == 1:
        selected_reward = task_rewards[args.task[0]]
    else:
        selected_reward = CombinedReward(
            [task_rewards[task] for task in args.task]
        )  # partial(binary_reward, target_path_length=30),

    # Create the WFC environment instance
    env = WFCWrapper(
        map_length=MAP_LENGTH,
        map_width=MAP_WIDTH,
        tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool,
        num_tiles=num_tiles,
        tile_to_index=tile_to_index,
        reward=selected_reward,
        deterministic=True,
        # qd_function=binary_percent_water if args.qd else None,
    )
    tile_images = load_tile_images()  # Load images needed for rendering later

    hyperparams = {}
    best_agent = None

    if args.load_hyperparameters:
        # --- Load Hyperparameters and Run Evolution ---
        print(f"Loading hyperparameters from: {args.load_hyperparameters}")
        try:
            with open(args.load_hyperparameters, "r") as f:
                hyperparams = yaml.safe_load(f)
                if args.override_patience is not None:
                    hyperparams["patience"] = args.override_patience
            print("Successfully loaded hyperparameters:", hyperparams)

            # Prepare saving best per generation
            gen_save_dir = None
            task_info = ""
            if args.save_best_per_gen:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                task_str = "_".join(args.task)
                if "binary" in task_str and hasattr(selected_reward, "keywords") and "target_path_length" in selected_reward.keywords:
                    task_info = f"{task_str}_tl{selected_reward.keywords['target_path_length']}"
                else:
                    task_info = task_str
                gen_save_dir = f"best_gen_maps/{timestamp}_{task_info.replace('/', '-')}"
            
            print(
                f"Running evolution for {args.generations} generations with loaded hyperparameters..."
            )

        except FileNotFoundError:
            print(
                f"Error: Hyperparameter file not found at {args.load_hyperparameters}"
            )
            exit(1)
        except Exception as e:
            print(f"Error loading or using hyperparameters: {e}")
            exit(1)

        start_time = time.time()
        _, best_agent, generations, best_agent_rewards, mean_agent_rewards = evolve(
            env=env,
            generations=args.generations,
            population_size=48,  # Fixed population size
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
            patience=50,  # Fixed patience
            qd=args.qd,
            genotype_representation=str(args.genotype_dimensions) + "d",
            tile_images=tile_images,
            task_info=task_info,
            gen_save_dir=gen_save_dir,
        )
        end_time = time.time()
        print(f"Evolution finished in {end_time - start_time:.2f} seconds.")
        print(f"Evolved for a total of {generations} generations")
        assert len(best_agent_rewards) == len(mean_agent_rewards)
        task_str = "_".join(args.task)  # Combine task names

        x_axis = np.arange(1, len(mean_agent_rewards) + 1)
        plt.plot(x_axis, best_agent_rewards, label="Best Agent Per Generation")
        plt.plot(x_axis, mean_agent_rewards, label="Median Agent Per Generation")
        plt.legend()
        plt.title(f"Performance Over Generations: {task_str}")
        plt.xlabel("Generations")
        plt.ylabel("Reward")
        plt.savefig(f"agent_performance_over_generations_{task_str}.png")
        plt.close()

    elif not args.best_agent_pickle:
        # --- Run Optuna Hyperparameter Optimization ---
        print(
            f"Running Optuna hyperparameter search for {args.optuna_trials} trials..."
        )
        study = optuna.create_study(direction="maximize")
        start_time = time.time()
        study.optimize(
            lambda trial: objective(
                trial,
                args.generations_per_trial,
                qd=args.qd,
                tasks_list=args.task,
            ),
            n_trials=args.optuna_trials,
        )
        end_time = time.time()
        print(f"Optuna optimization finished in {end_time - start_time:.2f} seconds.")

        hyperparams = study.best_params
        best_value = study.best_value
        print(f"\nBest trial completed with reward: {best_value:.4f}")
        print("Best hyperparameters found:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")

        # Ensure the hyperparameters directory exists
        hyperparam_dir = args.hyperparameter_dir
        os.makedirs(hyperparam_dir, exist_ok=True)
        output_path = os.path.join(hyperparam_dir, args.output_file)

        # Save the best hyperparameters
        print(f"Saving best hyperparameters to: {output_path}")
        try:
            with open(output_path, "w") as f:
                yaml.dump(hyperparams, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving hyperparameters: {e}")

    elif args.best_agent_pickle:
        with open(args.best_agent_pickle, "rb") as f:
            best_agent = pickle.load(f)

    # --- Render the result from the best agent ---
    if best_agent:
        print("\nInitializing Pygame for rendering the best map...")
        pygame.init()
        task_name = "_".join(args.task)

        # Setup environment for rendering
        env.render_mode = "human"
        env.tile_images = tile_images

        os.makedirs("evolution_output", exist_ok=True)
        output_filename = (
            f"evolution_output/{task_name}_reward_{best_agent.reward:.2f}.png"
        )

        # Run the best agent's actions
        observation, _ = env.reset()
        for action in best_agent.action_sequence:
            observation, _, terminated, truncated, _ = env.step(action)
            env.render()
            pygame.display.flip()
            if terminated or truncated:
                break

        # Save the final render
        env.save_render(output_filename)
        print(f"Saved final render to {output_filename}")

        # Display for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            env.render()
            pygame.event.pump()

        env.close()
    else:
        print("\nNo best agent was found during the process.")

    # Save the best agent
    os.makedirs("agents", exist_ok=True)
    if best_agent:
        task_str = "_".join(args.task)
        filename = (
            f"agents/best_evolved_{task_str}_reward_{best_agent.reward:.2f}_agent.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(best_agent, f)
        print(f"Saved best agent to {filename}")

    print("Script finished.")
