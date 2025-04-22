import argparse
import copy
import math
import os
import pickle
import random
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import optuna
import pygame
import yaml
from scipy.stats import truncnorm
from tqdm import tqdm

from biome_adjacency_rules import create_adjacency_matrix
from wfc import (  # We might not need render_wfc_grid if we keep console rendering
    load_tile_images,
    render_wfc_grid,
)
from wfc_env import Task, WFCWrapper


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
        method: str = "one_point",
    ) -> tuple["PopulationMember", "PopulationMember"]:
        seq1 = parent1.action_sequence
        seq2 = parent2.action_sequence
        length = len(seq1)

        if method == "one_point":
            # pick a crossover point (not at the extremes)
            point = np.random.randint(1, length)
            # child1 takes seq1[:point] + seq2[point:]
            child_seq1 = np.concatenate([seq1[:point], seq2[point:]])
            # child2 takes seq2[:point] + seq1[point:]
            child_seq2 = np.concatenate([seq2[:point], seq1[point:]])

        elif method == "uniform":
            # for each index, flip a coin to choose parent1 or parent2
            mask = np.random.rand(length) < 0.5
            child_seq1 = np.where(mask, seq1, seq2)
            child_seq2 = np.where(mask, seq2, seq1)
        else:
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
    ],
) -> tuple["PopulationMember", "PopulationMember"]:
    """
    Given (p1, p2, mean, stddev, noise), perform crossover + mutate
    and return two children.
    """
    p1, p2, mean, stddev, noise = args
    c1, c2 = PopulationMember.crossover(p1, p2, method="one_point")
    c1.mutate(mean, stddev, noise)
    c2.mutate(mean, stddev, noise)
    return c1, c2


def evolve(
    env: WFCWrapper,
    generations: int = 100,
    population_size: int = 5,
    number_of_actions_mutated_mean: int = 10,
    number_of_actions_mutated_standard_deviation: float = 10,
    action_noise_standard_deviation: float = 0.1,
    survival_rate: float = 0.2,
    patience: int = 10,
):
    patience_counter = 0
    best_agent: PopulationMember | None = None
    population = [PopulationMember(env) for _ in range(population_size)]
    for generation in tqdm(range(generations), desc="Generations"):
        # Evaluate the entire population in parallel
        with Pool(min(cpu_count(), population_size)) as pool:
            population = pool.map(run_member, population)
        population.sort(key=lambda x: x.reward, reverse=True)
        best = population[0]
        if best_agent is None or best.reward > best_agent.reward:
            best_agent = copy.deepcopy(best)
            # early stopping if max is achieved
            print("info", best_agent.info, "reward", best_agent.reward)
            if best_agent.info.get("achieved_max_reward", False):
                return population, best_agent, generation
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == patience:
                print("Training terminated due to stagnating reward")
                return population, best_agent, generation

        # Determine survivors and reproduce
        n_survivors = max(2, int(population_size * survival_rate))
        survivors = population[:n_survivors]
        offspring: list[PopulationMember] = []
        number_of_offspring_needed = population_size - n_survivors

        pairs_needed = math.ceil(number_of_offspring_needed / 2)

        # Generate exactly the number of pairs required.
        pairs_args = [
            (
                *random.sample(survivors, 2),
                number_of_actions_mutated_mean,
                number_of_actions_mutated_standard_deviation,
                action_noise_standard_deviation,
            )
            for _ in range(pairs_needed)
        ]

        with Pool(cpu_count()) as pool:
            reproduction_results = pool.map(reproduce_pair, pairs_args)

        # Flatten and trim the offspring list
        offspring = [child for pair in reproduction_results for child in pair][
            :number_of_offspring_needed
        ]
        population = survivors + offspring
    # Ensure the best agent is returned even if the population is empty or has issues
    if population:
        population.sort(key=lambda x: x.reward, reverse=True)
        current_best = population[0]
        if best_agent is None or current_best.reward > best_agent.reward:
            best_agent = copy.deepcopy(current_best)
    elif best_agent is None:
        # Handle edge case where population is empty and no best_agent was ever found
        print("Warning: Evolution resulted in an empty population and no best agent.")
        # Optionally return a dummy agent or raise an error
        # For now, returning None as best_agent
        pass

    return population, best_agent, generations


# --- Optuna Objective Function ---


def objective(
    trial: optuna.Trial, base_env: WFCWrapper, generations_per_trial: int
) -> float:
    """Objective function for Optuna hyperparameter optimization."""
    # Suggest hyperparameters
    population_size = trial.suggest_int("population_size", 5, 50)
    number_of_actions_mutated_mean = trial.suggest_int(
        "number_of_actions_mutated_mean", 1, 50
    )
    number_of_actions_mutated_standard_deviation = trial.suggest_float(
        "number_of_actions_mutated_standard_deviation", 1.0, 20.0
    )
    action_noise_standard_deviation = trial.suggest_float(
        "action_noise_standard_deviation", 0.01, 0.5, log=True
    )
    survival_rate = trial.suggest_float("survival_rate", 0.1, 0.8)

    # Run evolution with suggested hyperparameters
    _, best_agent, _ = evolve(
        env=base_env,
        generations=generations_per_trial,  # Use fewer generations for faster trials
        population_size=population_size,
        number_of_actions_mutated_mean=number_of_actions_mutated_mean,
        number_of_actions_mutated_standard_deviation=number_of_actions_mutated_standard_deviation,
        action_noise_standard_deviation=action_noise_standard_deviation,
        survival_rate=survival_rate,
    )

    # Return the reward of the best agent found in this trial
    return best_agent.reward if best_agent else float("-inf")


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
        default=20,  # Number of trials for Optuna optimization
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

    args = parser.parse_args()

    # Define environment parameters (using the same tile set as in our training setup)
    MAP_LENGTH = 15
    MAP_WIDTH = 20

    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)

    # Create the WFC environment instance
    env = WFCWrapper(
        map_length=MAP_LENGTH,
        map_width=MAP_WIDTH,
        tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool,
        num_tiles=num_tiles,
        tile_to_index=tile_to_index,
        task=Task.BINARY,
        task_specifications={"target_path_length": 50},
        deterministic=True,
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
            start_time = time.time()
            _, best_agent, generations = evolve(
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
            )
            end_time = time.time()
            print(f"Evolution finished in {end_time - start_time:.2f} seconds.")
            print(f"Evolved for a total of {generations} generations")

        except FileNotFoundError:
            print(
                f"Error: Hyperparameter file not found at {args.load_hyperparameters}"
            )
            exit(1)
        except Exception as e:
            print(f"Error loading or using hyperparameters: {e}")
            exit(1)

    elif not args.best_agent_pickle:
        # --- Run Optuna Hyperparameter Optimization ---
        print(
            f"Running Optuna hyperparameter search for {args.optuna_trials} trials..."
        )
        study = optuna.create_study(direction="maximize")
        start_time = time.time()
        study.optimize(
            lambda trial: objective(trial, env, args.generations_per_trial),
            n_trials=args.optuna_trials,
            n_jobs=2,  # Use multiple cores for trials if available
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

        # Optional: Run evolution one more time with the best found hyperparameters
        print(
            f"\nRunning final evolution for {args.generations} generations with best hyperparameters..."
        )
        start_time = time.time()
        _, best_agent, _ = evolve(
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
        )
        end_time = time.time()
        print(f"Final evolution finished in {end_time - start_time:.2f} seconds.")
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

    # save the best agent in a .pkl file
    with open("agents/best_evolved_binary_agent.pkl", "wb") as f:
        pickle.dump(best_agent, f)

    print("Script finished.")
