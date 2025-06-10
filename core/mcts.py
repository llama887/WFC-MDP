from __future__ import annotations

import argparse
import copy
import math
import multiprocessing
import os
import pickle
import sys
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pygame
import yaml
from pydantic import BaseModel, Field
from tqdm import tqdm
from math import sqrt

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assets.biome_adjacency_rules import create_adjacency_matrix, load_tile_images
from tasks.binary_task import binary_reward
from tasks.grass_task import grass_reward
from tasks.hill_task import hill_reward
from tasks.pond_task import pond_reward
from tasks.river_task import river_reward
from wfc_env import CombinedReward, WFCWrapper


def _simulate_node(node: "Node") -> tuple[float, list[np.ndarray], bool]:
    """
    Run just the simulation/rollout part of MCTS. This is sent to worker processes.
    It does NOT modify the tree (no backpropagation).
    Returns: (reward, action_sequence, achieved_max_flag)
    """
    return node.simulate()


class Action(BaseModel):
    """Represents an action in the MCTS tree with its statistics"""

    action_logits: np.ndarray = Field(
        default_factory=lambda: np.array([]), description="Logits for the action taken"
    )
    visits: int = Field(default=0, description="Number of times this action has been taken")
    total_reward: float = Field(
        default=0.0, description="Total reward accumulated from this action"
    )
    tile_index: int = Field(default=-1, description="Index of the tile this action represents")

    class Config:
        arbitrary_types_allowed = True




class Node:
    """A node in the MCTS tree"""

    def __init__(self, env: WFCWrapper, parent: Node | None = None, action_taken: Action | None = None):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action_taken = action_taken  # Action that led to this node
        self.children: list[Node] = []
        # Available actions are the possible actions from this node's environment state
        self.available_actions: dict[int, Action] = {
            i: Action(action_logits=np.eye(env.num_tiles)[i], tile_index=i)
            for i in range(env.num_tiles)
        }
        self.visits = 0
        self.total_reward = 0.0
        self.is_terminal = False
        self.is_fully_expanded = False

    def uct_score(self, child: "Node", exploration_weight: float) -> float:
        """Calculate the UCT score for a child node"""
        if child.visits == 0:
            return float("inf")

        exploitation = child.total_reward / child.visits
        exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
        return exploitation + exploration

    def best_child(self, exploration_weight: float) -> "Node":
        """Select the child with the highest UCT score"""
        if not self.children:
            raise ValueError("Node has no children")
        return max(self.children, key=lambda child: self.uct_score(child, exploration_weight))

    def expand(self) -> "Node":
        """Expand the node by adding a new child"""
        unexplored_indices = [
            idx for idx in self.available_actions
            if idx not in {child.action_taken.tile_index for child in self.children if child.action_taken}
        ]

        if not unexplored_indices:
            self.is_fully_expanded = True
            return self.best_child(exploration_weight=1.0)

        tile_idx = np.random.choice(unexplored_indices)
        action = self.available_actions[tile_idx]

        child_env = copy.deepcopy(self.env)
        _, _, terminated, truncated, _ = child_env.step(np.array(action.action_logits))

        child = Node(child_env, parent=self, action_taken=action)
        child.is_terminal = terminated or truncated
        self.children.append(child)
        return child

    def simulate(self) -> tuple[float, list[np.ndarray], bool]:
        """Run a simulation from this node to a terminal state"""
        if self.is_terminal:
            return self.env.reward(self.env.grid)[0], [], False

        sim_env = copy.deepcopy(self.env)
        total_reward = 0.0
        action_sequence = []
        terminated, truncated = False, False
        while not (terminated or truncated):
            action_idx = np.random.choice(list(self.available_actions.keys()))
            action = self.available_actions[action_idx].action_logits
            _, reward, terminated, truncated, info = sim_env.step(action)
            total_reward += reward
            action_sequence.append(action)
            if terminated and info.get("achieved_max_reward", False):
                return total_reward, action_sequence, True
        return total_reward, action_sequence, False

    def backpropagate(self, reward: float) -> None:
        """Update statistics for this node and all ancestors"""
        node = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            if node.action_taken:
                node.action_taken.visits += 1
                node.action_taken.total_reward += reward
            node = node.parent


class MCTS:
    """Monte Carlo Tree Search implementation for WFC"""

    def __init__(self, env: WFCWrapper, exploration_weight: float = sqrt(2)):
        self.env = env
        self.exploration_weight = exploration_weight
        self.root = Node(env)
        self.best_action_sequence: list[np.ndarray] = []
        self.best_reward = float("-inf")

    def search(self) -> tuple[Action, list[np.ndarray], bool]:
        """Run the MCTS algorithm and return the best action and sequence"""
        num_simulations = 48  # Hardcoded as per hardware limitations
        num_processes = min(multiprocessing.cpu_count(), num_simulations)

        # 1. Selection Phase: In the main process, select all nodes to be simulated.
        #    This expands the tree and modifies the main tree state.
        nodes_to_simulate = [self.select_node() for _ in range(num_simulations)]

        # 2. Simulation Phase: Run the heavy simulation part in parallel worker processes.
        #    Each worker gets a copy of a node and returns only the simulation result.
        with Pool(processes=num_processes) as pool:
            simulation_outputs: list[tuple[float, list[np.ndarray], bool]] = pool.map(
                _simulate_node, nodes_to_simulate
            )

        # 3. Backpropagation & Processing Phase: Update the main tree in the parent process.
        #    We zip the original nodes with their results, which are in the same order.
        for node, (reward, rollout_sequence, achieved_max_reward) in zip(
            nodes_to_simulate, simulation_outputs
        ):
            # First, update the tree with the result. This is safe as we are in the main process.
            node.backpropagate(reward)

            # Now, process the result to see if we found a solution or a new best reward.
            if achieved_max_reward:
                prefix_actions: list[np.ndarray] = []
                current_node = node
                while current_node.parent is not None:
                    prefix_actions.append(current_node.action_taken.action_logits)
                    current_node = current_node.parent
                prefix_actions.reverse()

                full_sequence = prefix_actions + rollout_sequence
                self.best_action_sequence = full_sequence
                self.best_reward = reward

                # Find the first action taken from the root to return
                if full_sequence:
                    for child in self.root.children:
                        if child.action_taken and np.array_equal(
                            child.action_taken.action_logits, full_sequence[0]
                        ):
                            return child.action_taken, full_sequence, True

            if reward > self.best_reward:
                self.best_reward = reward
                prefix_actions = []
                current_node = node
                while current_node.parent is not None:
                    prefix_actions.append(current_node.action_taken.action_logits)
                    current_node = current_node.parent
                prefix_actions.reverse()
                self.best_action_sequence = prefix_actions + rollout_sequence

        # 4. If no solution was found, return the action leading to the most visited child.
        if not self.root.children:
            # Fallback if the tree was never expanded (e.g., root is terminal)
            action_logits = self.env.action_space.sample()
            return Action(action_logits=action_logits), self.best_action_sequence, False

        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.action_taken, self.best_action_sequence, False

    def select_node(self) -> Node:
        """Select a node to expand using UCT"""
        node = self.root
        while not node.is_terminal and node.is_fully_expanded:
            node = node.best_child(self.exploration_weight)
        if not node.is_terminal and not node.is_fully_expanded:
            return node.expand()
        return node


def render_action_sequence(env: WFCWrapper, action_sequence: list[np.ndarray], tile_images, filename: str) -> None:
    """Render the final state of an action sequence and save to file"""
    env = copy.deepcopy(env)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_mode((1, 1))
    SCREEN_WIDTH, SCREEN_HEIGHT = env.map_width * 32, env.map_length * 32
    final_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    env.reset()
    for action in action_sequence:
        _, _, terminate, truncate, _ = env.step(action)
        if terminate or truncate:
            break
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
    os.makedirs("mcts_output", exist_ok=True)
    output_path = os.path.join("mcts_output", filename)
    pygame.image.save(final_surface, output_path)
    pygame.quit()


def run_mcts_until_complete(env: WFCWrapper, exploration_weight: float, max_iterations: int = 1000) -> tuple[list[np.ndarray] | None, float | None, int | None]:
    """Run MCTS search until a complete solution is found or max iterations are reached."""
    mcts = MCTS(env, exploration_weight)
    test_env = copy.deepcopy(env)
    for i in tqdm(range(max_iterations), desc="MCTS Search Iterations"):
        _, action_sequence, found_max = mcts.search()
        if found_max:
            test_env.reset()
            current_reward = 0
            for action in action_sequence:
                _, reward, terminated, truncated, info = test_env.step(action)
                current_reward += reward
                if terminated or truncated:
                    print(f"Reached terminal/truncated state after {len(action_sequence)} actions")
                    break
            if not info.get("achieved_max_reward", False):
                print(f"WARNING: Expected max reward but not achieved. Final state: terminated={terminated}, achieved_max={info.get('achieved_max_reward', False)}")
            
            # Debugging:
            print(f"Testing best action sequence (found_max={found_max})")
            print(f"  Total reward: {current_reward}")
            print(f"  Info: {info}")
            # Validate reward consistency
            if terminated and info.get("achieved_max_reward", False):
                return action_sequence, current_reward, i
    return None, None, None

    # Load tile images for visualization
    from assets.biome_adjacency_rules import load_tile_images
    tile_images = load_tile_images()

def objective(trial, max_iterations_per_trial: int, tasks_list: list[str]) -> float:
    """Optuna objective: minimize iterations to find a solution."""
    exploration_weight = trial.suggest_float("exploration_weight", 0.1, 3.0)
    reward_funcs = []
    is_combo = len(tasks_list) > 1
    for task in tasks_list:
        if task.startswith("binary_"):
            target_length = 40 if is_combo else 80
            hard = task == "binary_hard"
            reward_funcs.append(partial(binary_reward, target_path_length=target_length, hard=hard))
        else:
            reward_funcs.append(globals()[f"{task}_reward"])
    reward_fn = CombinedReward(reward_funcs) if len(reward_funcs) > 1 else reward_funcs[0]
    MAP_LENGTH, MAP_WIDTH = 15, 20
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)
    env = WFCWrapper(
        map_length=MAP_LENGTH, map_width=MAP_WIDTH, tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool, num_tiles=num_tiles, tile_to_index=tile_to_index,
        reward=reward_fn, deterministic=True,
    )
    iterations_to_converge = []
    NUMBER_OF_SAMPLES = 10
    for _ in range(NUMBER_OF_SAMPLES):
        _, _, iterations = run_mcts_until_complete(env, exploration_weight, max_iterations=max_iterations_per_trial)
        iterations_to_converge.append(iterations if iterations is not None else max_iterations_per_trial)
    return np.mean(iterations_to_converge)


def main():
    parser = argparse.ArgumentParser(description="Run MCTS for WFC.")
    parser.add_argument("--load-hyperparameters", type=str, help="Path to YAML file with MCTS hyperparameters.")
    parser.add_argument("--max-iterations", type=int, default=100, help="Max number of MCTS search attempts.")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Number of trials for Optuna search.")
    parser.add_argument("--iterations-per-trial", type=int, default=50, help="Max search attempts for each Optuna trial.")
    parser.add_argument("--hyperparameter-dir", type=str, default="hyperparameters", help="Directory for hyperparameters.")
    parser.add_argument("--output-file", type=str, default="best_mcts_hyperparameters.yaml", help="Filename for saved hyperparameters.")
    parser.add_argument("--best-sequence-pickle", type=str, help="Path to a pickled sequence to load and render.")
    parser.add_argument("--task", action="append", default=[], choices=["binary_easy", "binary_hard", "river", "pond", "grass", "hill"], help="Task(s) to use.")
    args = parser.parse_args()
    if not args.task:
        args.task = ["binary_easy"]

    task_rewards = {
        "binary_easy": partial(binary_reward, target_path_length=80),
        "binary_hard": partial(binary_reward, target_path_length=80, hard=True),
        "river": river_reward, "pond": pond_reward, "grass": grass_reward, "hill": hill_reward,
    }
    selected_reward = CombinedReward([task_rewards[task] for task in args.task]) if len(args.task) > 1 else task_rewards[args.task[0]]
    MAP_LENGTH, MAP_WIDTH = 15, 20
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)
    env = WFCWrapper(
        map_length=MAP_LENGTH, map_width=MAP_WIDTH, tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool, num_tiles=num_tiles, tile_to_index=tile_to_index,
        reward=selected_reward, deterministic=True,
    )
    tile_images = load_tile_images()
    task_name = "_".join(args.task)

    if args.optuna_trials > 0:
        import optuna
        print(f"Running Optuna search for {args.optuna_trials} trials...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args.iterations_per_trial, args.task), n_trials=args.optuna_trials, n_jobs=1)
        hyperparams = study.best_params
        print("Best hyperparameters found:", hyperparams)
        os.makedirs(args.hyperparameter_dir, exist_ok=True)
        output_path = os.path.join(args.hyperparameter_dir, args.output_file)
        with open(output_path, "w") as f:
            yaml.dump(hyperparams, f)
        print(f"Saved best hyperparameters to: {output_path}")
    elif args.load_hyperparameters:
        print(f"Loading hyperparameters from: {args.load_hyperparameters}")
        with open(args.load_hyperparameters, "r") as f:
            hyperparams = yaml.safe_load(f)
        exploration_weight = hyperparams["exploration_weight"]
        start_time = time.time()
        sequence, reward, iterations = run_mcts_until_complete(env, exploration_weight, max_iterations=args.max_iterations)
        end_time = time.time()
        print(f"MCTS search finished in {end_time - start_time:.2f} seconds.")
        if sequence:
            print(f"Complete solution found at iteration {iterations} with reward {reward:.4f}")
            AGENT_DIR = "agents"
            os.makedirs(AGENT_DIR, exist_ok=True)
            filename = f"{AGENT_DIR}/best_mcts_{task_name}_reward_{reward:.2f}_sequence.pkl"
            with open(filename, "wb") as f:
                pickle.dump(sequence, f)
            print(f"Saved best sequence to {filename}")
            render_action_sequence(env, sequence, tile_images, f"mcts_solution_{task_name}.png")
        else:
            print("No complete solution found within the maximum iterations.")
    elif args.best_sequence_pickle:
        with open(args.best_sequence_pickle, "rb") as f:
            sequence = pickle.load(f)
        print(f"Loaded sequence from {args.best_sequence_pickle}")
        render_action_sequence(env, sequence, tile_images, f"rendered_{os.path.basename(args.best_sequence_pickle)}.png")
    else:
        parser.error("Please specify a mode: --optuna-trials, --load-hyperparameters, or --best-sequence-pickle")
    print("Script finished.")


if __name__ == "__main__":
    main()

