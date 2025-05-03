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

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pygame
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import truncnorm
from tqdm import tqdm

from biome_adjacency_rules import create_adjacency_matrix
from tasks.binary_task import binary_percent_water, binary_reward
from tasks.water_biome import water_biome_reward
from tasks.grass_biome import grass_biome_reward
from wfc import (  # We might not need render_wfc_grid if we keep console rendering
    load_tile_images,
    render_wfc_grid,
)
from wfc_env import CombinedReward, WFCWrapper


class CrossOverMethod(Enum):
    UNIFORM = 0
    ONE_POINT = 1


class PopulationMember:
    def __init__(self, env: WFCWrapper):
        self.env: WFCWrapper = copy.deepcopy(env)
        self.env.reset()
        self.reward: float = float("-inf")
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
        self.reward = 0
        for idx, action in enumerate(self.action_sequence):
            _, reward, terminate, truncate, info = self.env.step(action)
            self.reward += reward
            self.info = info
            if terminate or truncate:
                break

    @staticmethod
    def crossover(
        parent1: "PopulationMember",
        parent2: "PopulationMember",
        method: CrossOverMethod = CrossOverMethod.ONE_POINT,
    ) -> tuple["PopulationMember", "PopulationMember"]:
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
    population_size: int = 50,
    number_of_actions_mutated_mean: int = 10,
    number_of_actions_mutated_standard_deviation: float = 10.0,
    action_noise_standard_deviation: float = 0.1,
    survival_rate: float = 0.2,
    cross_over_method: CrossOverMethod = CrossOverMethod.ONE_POINT,
    patience: int = 10,
    qd: bool = False,
) -> tuple[list[PopulationMember], PopulationMember, int, list[float], list[float]]:
    """
    Standard EA if qd=False; QD selection + global reproduction if qd=True.
    """
    # --- Initialization ---
    population = [PopulationMember(env) for _ in range(population_size)]
    best_agent: PopulationMember | None = None
    best_agent_rewards: list[float] = []
    median_agent_rewards: list[float] = []
    patience_counter = 0

    for gen in tqdm(range(1, generations + 1), desc="Generations"):
        # 1) Evaluate entire pop
        with Pool(min(cpu_count() * 2, len(population))) as pool:
            population = pool.map(run_member, population)

        # 2) Gather scores & stats
        #    - fitness-based reward for standard EA
        #    - qd_score for QD clustering
        fitnesses = np.array([m.reward for m in population])
        best_idx = int(np.argmax(fitnesses))
        median_val = float(np.median(fitnesses))

        best_agent_rewards.append(population[best_idx].reward)
        median_agent_rewards.append(median_val)

        # Track global best & early stopping
        if best_agent is None or population[best_idx].reward > best_agent.reward:
            best_agent = copy.deepcopy(population[best_idx])
            patience_counter = 0
        else:
            patience_counter += 1

        if (
            population[best_idx].info.get("achieved_max_reward", False)
            or patience_counter >= patience
        ):
            return population, best_agent, gen, best_agent_rewards, median_agent_rewards

        # 3) Selection
        if not qd:
            # Standard: top‐N by fitness
            sorted_pop = sorted(population, key=lambda m: m.reward, reverse=True)
            number_of_surviving_members = max(2, int(population_size * survival_rate))
            survivors = sorted_pop[:number_of_surviving_members]
        else:
            # QD: cluster on qd_score
            scores = np.array([m.info["qd_score"] for m in population])
            Z = linkage(scores.reshape(-1, 1), method="ward")
            cutoff = np.median(Z[:, 2])
            labels = fcluster(Z, t=cutoff, criterion="distance")

            # pick survivors within each cluster
            survivors = []
            for cluster in np.unique(labels):
                members = [
                    population[i] for i, lbl in enumerate(labels) if lbl == cluster
                ]
                members.sort(
                    key=lambda m: m.info.get("qd_score", m.reward), reverse=True
                )
                number_of_cluster_survivors = max(1, int(len(members) * survival_rate))
                survivors.extend(members[:number_of_cluster_survivors])

            # ensure at least two survivors overall
            if len(survivors) < 2:
                # fallback to best two by fitness
                pop_by_fit = sorted(population, key=lambda m: m.reward, reverse=True)
                survivors = pop_by_fit[:2]

        # 4) Reproduction (global)
        number_of_surviving_members = len(survivors)
        n_offspring = population_size - number_of_surviving_members
        n_pairs = math.ceil(n_offspring / 2)
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

        with Pool(min(cpu_count() * 2, len(pairs_args))) as pool:
            results = pool.map(reproduce_pair, pairs_args)

        # flatten and trim
        offspring = [child for pair in results for child in pair][:n_offspring]

        # 5) Form next gen
        population = survivors + offspring

    # end for
    return population, best_agent, generations, best_agent_rewards, median_agent_rewards


# --- Optuna Objective Function ---


def objective(
    trial: optuna.Trial, task: str, generations_per_trial: int, qd: bool = False
) -> float:
    """Objective function for Optuna hyperparameter optimization."""

    # Suggest hyperparameters
    population_size = trial.suggest_int("population_size", 30, 100)
    number_of_actions_mutated_mean = trial.suggest_int(
        "number_of_actions_mutated_mean", 1, 100
    )
    number_of_actions_mutated_standard_deviation = trial.suggest_float(
        "number_of_actions_mutated_standard_deviation", 1.0, 100.0
    )
    action_noise_standard_deviation = trial.suggest_float(
        "action_noise_standard_deviation", 0.01, 0.8, log=True
    )
    survival_rate = trial.suggest_float("survival_rate", 0.01, 0.99)
    cross_over_method = trial.suggest_categorical("cross_over_method", [0, 1])
    patience = trial.suggest_int("patience", 10, 20)
    # Constuct Env
    MAP_LENGTH = 15
    MAP_WIDTH = 20

    total_reward = 0
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)

    NUMBER_OF_SAMPLES = 10
    start_time = time.time()
    for i in range(NUMBER_OF_SAMPLES):
        match task:
            case "binary":
                target_path_length = random.randint(
                    50, 70
                )  # only focus on the harder problems
                # Create the WFC environment instance
                base_env = WFCWrapper(
                    map_length=MAP_LENGTH,
                    map_width=MAP_WIDTH,
                    tile_symbols=tile_symbols,
                    adjacency_bool=adjacency_bool,
                    num_tiles=num_tiles,
                    tile_to_index=tile_to_index,
                    reward=partial(
                        binary_reward, target_path_length=target_path_length
                    ),
                    deterministic=True,
                    qd_function=binary_percent_water if qd else None,
                )
                print(f"Target Path Length: {target_path_length}")
            case _:
                raise ValueError(f"{task} is not a defined task")

        # Run evolution with suggested hyperparameters
        _, best_agent, _, _, _ = evolve(
            env=base_env,
            generations=generations_per_trial,  # Use fewer generations for faster trials
            population_size=population_size,
            number_of_actions_mutated_mean=number_of_actions_mutated_mean,
            number_of_actions_mutated_standard_deviation=number_of_actions_mutated_standard_deviation,
            action_noise_standard_deviation=action_noise_standard_deviation,
            survival_rate=survival_rate,
            cross_over_method=CrossOverMethod(cross_over_method),
            patience=patience,
            qd=qd,
        )
        print(f"Best reward at sample {i + 1}/{NUMBER_OF_SAMPLES}: {best_agent.reward}")
        total_reward += best_agent.reward
    end_time = time.time()

    # Return the best reward but with account for how long it took
    print(f"Total Reward: {total_reward} | Time: {end_time - start_time}")
    return total_reward - (0.001) * (end_time - start_time)


def render_best_agent(env: WFCWrapper, best_agent: PopulationMember, tile_images):
    """Renders the action sequence of the best agent."""
    if not best_agent:
        print("No best agent found to render.")
        return
    pygame.init()
    SCREEN_WIDTH = env.map_width * 32  # Adjust screen size based on map
    SCREEN_HEIGHT = env.map_length * 32
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Best Evolved WFC Map")

    env.reset()
    total_reward = 0
    print("Rendering best agent's action sequence...")
    for action in tqdm(best_agent.action_sequence, desc="Rendering Steps"):
        _, reward, terminate, truncate, _ = env.step(action)
        total_reward += reward
        render_wfc_grid(env.grid, tile_images, screen=screen)
        pygame.time.delay(5)  # Slightly faster rendering
        if terminate or truncate:
            break

    print(f"Final map reward for the best agent: {total_reward:.4f}")
    print(
        f"Best agent reward during evolution: {best_agent.reward:.4f}"
    )  # Print the reward recorded during evolution

    # Keep the window open for a bit
    print("Displaying final map for 5 seconds...")
    pygame.time.delay(5000)
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
        default=10,  # Fewer generations during tuning for speed
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
        default=["binary_easy"],
        choices=["binary_easy", "binary_hard", "water"],
        help="The task being optimized. Used to pick reward. Pick from: binary_easy, binary_hard, river, pond ect.",
    )

    args = parser.parse_args()

    # Define environment parameters
    MAP_LENGTH = 15
    MAP_WIDTH = 20

    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)

    task_rewards = {
        "binary_easy": partial(binary_reward, target_path_length=50),
        "binary_hard": partial(binary_reward, target_path_length=50, hard=True),
        "water": water_biome_reward,
    }

    # Create the WFC environment instance
    env = WFCWrapper(
        map_length=MAP_LENGTH,
        map_width=MAP_WIDTH,
        tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool,
        num_tiles=num_tiles,
        tile_to_index=tile_to_index,
        reward=CombinedReward(
            [task_rewards[task] for task in args.task]
        ),  # partial(binary_reward, target_path_length=30),
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
            print("Successfully loaded hyperparameters:", hyperparams)

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
        _, best_agent, generations, best_agent_rewards, median_agent_rewards = evolve(
            env=env,
            generations=args.generations,
            population_size=hyperparams["population_size"],
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
            patience=hyperparams["patience"],
            qd=args.qd,
        )
        end_time = time.time()
        print(f"Evolution finished in {end_time - start_time:.2f} seconds.")
        print(f"Evolved for a total of {generations} generations")
        assert len(best_agent_rewards) == len(median_agent_rewards)
        x_axis = np.arange(1, len(median_agent_rewards) + 1)
        plt.plot(x_axis, best_agent_rewards, label="Best Agent Per Generation")
        plt.plot(x_axis, median_agent_rewards, label="Median Agent Per Generation")
        plt.legend()
        plt.title("Agent Performance Over Generations")
        plt.xlabel("Generations")
        plt.ylabel("Reward")
        plt.savefig("agent_performance_over_generations.png")
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
                trial, "binary", args.generations_per_trial, args.qd
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
        render_best_agent(env, best_agent, tile_images)
    else:
        print("\nNo best agent was found during the process.")

    AGENT_DIR = "agents"
    os.makedirs(AGENT_DIR, exist_ok=True)
    # save the best agent in a .pkl file
    with open(
        f"{AGENT_DIR}/best_evolved_{'_'.join([task for task in args.task])}_agent.pkl",
        "wb",
    ) as f:
        pickle.dump(best_agent, f)

    print("Script finished.")
