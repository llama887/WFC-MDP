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
from core.wfc_env import CombinedReward, WFCWrapper


def _simulate_node(node: "Node") -> tuple[float, list[np.ndarray], bool]:
    """Run just the simulation/rollout part of MCTS in worker processes."""
    return node.simulate()


class Action(BaseModel):
    """Represents an action in the MCTS tree with its statistics"""
    action_logits: np.ndarray = Field(default_factory=lambda: np.array([]))
    visits: int = Field(default=0)
    total_reward: float = Field(default=0.0)
    tile_index: int = Field(default=-1)

    class Config:
        arbitrary_types_allowed = True


class Node:
    """A node in the MCTS tree"""
    def __init__(self, env: WFCWrapper, parent: Node | None = None, action_taken: Action | None = None):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action_taken = action_taken
        self.children: list[Node] = []
        self.available_actions: dict[int, Action] = {
            i: Action(action_logits=np.eye(env.num_tiles)[i], tile_index=i)
            for i in range(env.num_tiles)
        }
        self.visits = 0
        self.total_reward = 0.0
        self.is_terminal = False
        self.is_fully_expanded = False

    def uct_score(self, child: "Node", exploration_weight: float) -> float:
        if child.visits == 0:
            return float("inf")
        exploitation = child.total_reward / child.visits
        exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
        return exploitation + exploration

    def best_child(self, exploration_weight: float) -> "Node":
        if not self.children:
            raise ValueError("Node has no children")
        return max(self.children, key=lambda child: self.uct_score(child, exploration_weight))

    def expand(self) -> "Node":
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
        num_simulations = 48
        num_processes = min(multiprocessing.cpu_count(), num_simulations)

        nodes_to_simulate = [self.select_node() for _ in range(num_simulations)]

        with Pool(processes=num_processes) as pool:
            simulation_outputs = pool.map(_simulate_node, nodes_to_simulate)

        for node, (reward, rollout_sequence, achieved_max_reward) in zip(
            nodes_to_simulate, simulation_outputs
        ):
            node.backpropagate(reward)

            if achieved_max_reward:
                prefix_actions = []
                current_node = node
                while current_node.parent is not None:
                    prefix_actions.append(current_node.action_taken.action_logits)
                    current_node = current_node.parent
                prefix_actions.reverse()

                full_sequence = prefix_actions + rollout_sequence
                self.best_action_sequence = full_sequence
                self.best_reward = reward

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

        if not self.root.children:
            action_logits = self.env.action_space.sample()
            return Action(action_logits=action_logits), self.best_action_sequence, False

        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.action_taken, self.best_action_sequence, False

    def select_node(self) -> Node:
        node = self.root
        while not node.is_terminal and node.is_fully_expanded:
            node = node.best_child(self.exploration_weight)
        if not node.is_terminal and not node.is_fully_expanded:
            return node.expand()
        return node


def run_mcts_search(
    env: WFCWrapper,
    exploration_weight: float,
    max_iterations: int = 1000,
    patience: int = 50,
) -> tuple[list[np.ndarray] | None, float, int]:
    """Run MCTS search with early stopping"""
    mcts = MCTS(env, exploration_weight)
    best_reward = float("-inf")
    best_sequence = None
    patience_counter = 0
    
    for i in tqdm(range(max_iterations), desc="MCTS Search"):
        _, sequence, found_max = mcts.search()
        
        if found_max:
            return sequence, mcts.best_reward, i
        
        if mcts.best_reward > best_reward:
            best_reward = mcts.best_reward
            best_sequence = sequence
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    return best_sequence, best_reward, max_iterations


def objective(trial, max_iterations_per_trial: int, tasks_list: list[str]) -> float:
    """Optuna objective function for MCTS hyperparameter tuning"""
    exploration_weight = trial.suggest_float("exploration_weight", 0.1, 3.0)
    
    # Build reward function
    is_combo = len(tasks_list) > 1
    reward_funcs = []
    for task in tasks_list:
        if task.startswith("binary_"):
            target_length = 40 if is_combo else 80
            hard = task == "binary_hard"
            reward_funcs.append(partial(binary_reward, target_path_length=target_length, hard=hard))
        else:
            reward_funcs.append(globals()[f"{task}_reward"])
            
    reward_fn = CombinedReward(reward_funcs) if len(reward_funcs) > 1 else reward_funcs[0]
    
    # Create environment
    MAP_LENGTH, MAP_WIDTH = 15, 20
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)
    env = WFCWrapper(
        map_length=MAP_LENGTH,
        map_width=MAP_WIDTH,
        tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool,
        num_tiles=num_tiles,
        tile_to_index=tile_to_index,
        reward=reward_fn,
        deterministic=True,
    )
    
    # Run multiple samples
    iterations_to_converge = []
    NUMBER_OF_SAMPLES = 10
    for _ in range(NUMBER_OF_SAMPLES):
        _, _, iterations = run_mcts_search(env, exploration_weight, max_iterations_per_trial)
        iterations_to_converge.append(iterations if iterations is not None else max_iterations_per_trial)
    
    return np.mean(iterations_to_converge)


def resume_mcts_search(mcts_instance: "MCTS", max_iterations: int) -> tuple[list[np.ndarray] | None, float | None, int | None]:
    """
    Resumes an MCTS search on an existing MCTS object for a given number of iterations.

    Args:
        mcts_instance (MCTS): The MCTS object to resume the search on.
        max_iterations (int): The maximum number of additional search iterations to perform.

    Returns:
        A tuple containing:
        - The best action sequence if a solution is found, otherwise None.
        - The best reward if a solution is found, otherwise None.
        - The number of iterations within this run it took to find the solution, otherwise None.
    """
    for i in tqdm(range(max_iterations), desc="Resuming MCTS Search", leave=False):
        _, action_sequence, found_max = mcts_instance.search()
        if found_max:
            # Validate that the found sequence actually achieves the max reward
            test_env = copy.deepcopy(mcts_instance.env)
            test_env.reset()
            current_reward = 0.0
            for action in action_sequence:
                _, reward, terminated, truncated, info = test_env.step(action)
                current_reward += reward
                if terminated or truncated:
                    break

            if info.get("achieved_max_reward", False):
                return action_sequence, current_reward, i + 1
            else:
                # This can happen if a rollout spuriously reported max reward.
                # We should continue searching.
                print("Warning: MCTS reported a solution, but validation failed. Continuing search.")

    return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Run MCTS for WFC map generation.")
    parser.add_argument(
        "--load-hyperparameters",
        type=str,
        default=None,
        help="Path to YAML file with hyperparameters",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of MCTS iterations to run",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Number of Optuna trials for hyperparameter search",
    )
    parser.add_argument(
        "--generations-per-trial",
        type=int,
        default=10,
        help="Generations per Optuna trial",
    )
    parser.add_argument(
        "--hyperparameter-dir",
        type=str,
        default="hyperparameters",
        help="Directory for hyperparameters",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="best_mcts_hyperparameters.yaml",
        help="Output file for hyperparameters",
    )
    parser.add_argument(
        "--best-agent-pickle",
        type=str,
        help="Path to pickle file with best agent",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        choices=["binary_easy", "binary_hard", "river", "pond", "grass", "hill"],
        help="Tasks to optimize for",
    )
    parser.add_argument(
        "--override-patience",
        type=int,
        default=None,
        help="Override patience parameter",
    )
    args = parser.parse_args()
    
    if not args.task:
        args.task = ["binary_easy"]

    # Create environment
    MAP_LENGTH, MAP_WIDTH = 15, 20
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)
    
    # Build reward function
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

    env = WFCWrapper(
        map_length=MAP_LENGTH,
        map_width=MAP_WIDTH,
        tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool,
        num_tiles=num_tiles,
        tile_to_index=tile_to_index,
        reward=selected_reward,
        deterministic=True,
    )
    tile_images = load_tile_images()
    env.tile_images = tile_images

    if args.optuna_trials > 0:
        import optuna
        
        print(f"Running Optuna optimization for {args.optuna_trials} trials...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, args.generations_per_trial, args.task),
            n_trials=args.optuna_trials,
        )
        
        hyperparams = study.best_params
        print("Best hyperparameters:", hyperparams)
        
        os.makedirs(args.hyperparameter_dir, exist_ok=True)
        output_path = os.path.join(args.hyperparameter_dir, args.output_file)
        with open(output_path, "w") as f:
            yaml.dump(hyperparams, f)
        print(f"Saved hyperparameters to {output_path}")
        
    elif args.best_agent_pickle:
        with open(args.best_agent_pickle, "rb") as f:
            best_sequence = pickle.load(f)
        
        # Create output directory
        os.makedirs("mcts_output", exist_ok=True)
        
        # Calculate total reward for the sequence
        test_env = copy.deepcopy(env)
        test_env.reset()
        total_reward = 0
        for action in best_sequence:
            _, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        # Save the rendered output
        output_filename = f"mcts_output/{'_'.join(args.task)}_steps_{len(best_sequence)}_reward_{total_reward:.2f}.png"
        env.render_mode = "human"
        observation, _ = env.reset()
        for action in best_sequence:
            observation, _, terminated, truncated, _ = env.step(action)
            env.render()
            if terminated or truncated:
                break
        
        env.save_render(output_filename)
        print(f"Saved rendered output to {output_filename}")
        time.sleep(5)
        env.close()
        
    else:
        # Default run mode
        exploration_weight = sqrt(2)  # Default value
        if args.load_hyperparameters:
            with open(args.load_hyperparameters, "r") as f:
                hyperparams = yaml.safe_load(f)
            exploration_weight = hyperparams["exploration_weight"]
            if args.override_patience is not None:
                hyperparams["patience"] = args.override_patience
        
        start_time = time.time()
        best_sequence, best_reward, generations = run_mcts_search(
            env,
            exploration_weight,
            max_iterations=args.generations,
            patience=args.override_patience or 50,
        )
        end_time = time.time()
        
        print(f"MCTS completed in {end_time - start_time:.2f} seconds")
        print(f"Best reward: {best_reward}")
        print(f"Generations: {generations}")
        
        if best_sequence:
            # Save the sequence
            AGENT_DIR = "agents"
            os.makedirs(AGENT_DIR, exist_ok=True)
            task_str = "_".join(args.task)
            filename = f"{AGENT_DIR}/best_mcts_{task_str}_reward_{best_reward:.2f}_sequence.pkl"
            with open(filename, "wb") as f:
                pickle.dump(best_sequence, f)
            print(f"Saved best sequence to {filename}")
            
            # Save the rendered output
            os.makedirs("mcts_output", exist_ok=True)
            output_filename = f"mcts_output/{task_str}_steps_{len(best_sequence)}_reward_{best_reward:.2f}.png"
            env.render_mode = "human"
            observation, _ = env.reset()
            for action in best_sequence:
                observation, _, terminated, truncated, _ = env.step(action)
                env.render()
                if terminated or truncated:
                    break
            
            env.save_render(output_filename)
            print(f"Saved rendered output to {output_filename}")
            time.sleep(5)
            env.close()

    print("Script finished.")


if __name__ == "__main__":
    main()