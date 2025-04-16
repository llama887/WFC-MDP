import copy
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pygame
from scipy.stats import truncnorm
from tqdm import tqdm

from biome_adjacency_rules import create_adjacency_matrix
from biome_wfc import (  # We might not need render_wfc_grid if we keep console rendering
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
        for action in self.action_sequence:
            _, reward, _, _, _ = self.env.step(action)
            self.reward += reward

    @staticmethod
    def crossover(
        parent1: "PopulationMember",
        parent2: "PopulationMember",
        method: str = "one_point",
    ) -> tuple["PopulationMember", "PopulationMember"]:
        """
        Create two offspring by combining the parents' action_sequences.

        Args:
            parent1, parent2: the two parents to crossover.
            method: "one_point" or "uniform".

        Returns:
            child1, child2: new PopulationMember instances.
        """
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
):
    best_agent: PopulationMember | None = None
    population = [PopulationMember(env) for _ in range(population_size)]
    for generation in tqdm(range(generations)):
        with Pool(min(cpu_count(), population_size)) as pool:
            population = pool.map(run_member, population)
        population.sort(key=lambda x: x.reward, reverse=True)
        best = population[0]
        print(f"\nGeneration {generation + 1} | Best reward = {best.reward:.4f}")
        if best_agent is None or best.reward > best_agent.reward:
            best_agent = copy.deepcopy(best)

        n_survivors = max(2, int(population_size * survival_rate))
        survivors = population[:n_survivors]
        offspring: list[PopulationMember] = []
        number_of_offspring_needed = population_size - n_survivors

        def parent_pair_generator():
            """Infinite generator of random parent‑pairs + GA params."""
            while True:
                p1, p2 = random.sample(survivors, 2)
                yield (
                    p1,
                    p2,
                    number_of_actions_mutated_mean,
                    number_of_actions_mutated_standard_deviation,
                    action_noise_standard_deviation,
                )

        with Pool(cpu_count()) as pool:
            for child1, child2 in pool.imap_unordered(
                reproduce_pair, parent_pair_generator()
            ):
                offspring.append(child1)
                # only append the second child if we still need more
                if len(offspring) < number_of_offspring_needed:
                    offspring.append(child2)
                # stop as soon as we have enough
                if len(offspring) >= number_of_offspring_needed:
                    pool.terminate()
                    break

        # trim in case we got one extra
        offspring = offspring[:number_of_offspring_needed]

        population = survivors + offspring
    return population, best_agent


if __name__ == "__main__":
    # Use biome_wfc rendering: load tile images (opens a pygame window)
    tile_images = load_tile_images()

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
    )
    import time

    start = time.time()
    number_of_generations = 5
    best_population, best_agent = evolve(env, generations=number_of_generations)
    end = time.time()
    print(
        f"Total time taken: {end - start} seconds over {number_of_generations} | Average time per generation: {(end - start) / number_of_generations}"
    )
    # render the best map
    # Initialize Pygame
    pygame.init()
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 480
    TILE_SIZE = 32

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Evolving WFC")
    best_action_sequence = (
        best_agent.action_sequence if best_agent else best_population[0].action_sequence
    )
    env.reset()
    print("Rendering best")
    total_reward = 0
    for action in best_action_sequence:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        render_wfc_grid(env.grid, tile_images, screen)
        pygame.time.delay(10)
    print(f"Best Reward: {total_reward}")
    pygame.quit()
