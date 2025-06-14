from __future__ import annotations

import argparse
import copy
import math
import os
import pickle
import sys
import time
from functools import partial

import numpy as np
import yaml
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


# Action class removed for memory optimization.


class Node:
    __slots__ = (
        "parent", "action_index", "children", 
        "visits", "total_reward", "untried_actions"
    )
    
    def __init__(self, parent: "Node" | None, action_index: int | None, num_tiles: int):
        self.parent = parent
        self.action_index = action_index
        self.children: list[Node] = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = set(range(num_tiles)) if parent is None else set()

    def uct_value(self, parent_visits: int, exploration_weight: float) -> float:
        if self.visits == 0:
            return float("inf")
        return (self.total_reward / self.visits) + exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )


class MCTS:
    def __init__(self, env: WFCWrapper, exploration_weight: float = math.sqrt(2)):
        self.root_env = copy.deepcopy(env)
        self.exploration_weight = exploration_weight
        self.root = Node(parent=None, action_index=None, num_tiles=env.num_tiles)
        self.best_action_sequence: list[int] = []
        self.best_reward = float("-inf")

    def _replay_env(self, node: Node) -> WFCWrapper:
        env = copy.deepcopy(self.root_env)
        action_sequence = []
        current = node
        while current.parent is not None:
            action_sequence.append(current.action_index)
            current = current.parent
        action_sequence.reverse()
        
        for act_idx in action_sequence:
            logits = np.eye(env.num_tiles)[act_idx]
            env.step(logits)
        return env

    def select_node(self) -> Node:
        node = self.root
        while node.untried_actions == set() and node.children:
            node = max(node.children, key=lambda c: c.uct_value(node.visits, self.exploration_weight))
        return node

    def expand(self, node: Node) -> Node:
        act_idx = next(iter(node.untried_actions))
        node.untried_actions.remove(act_idx)
        child = Node(parent=node, action_index=act_idx, num_tiles=self.root_env.num_tiles)
        node.children.append(child)
        return child

    def simulate(self, node: Node) -> tuple[float, list[int], bool]:
        env = self._replay_env(node)
        total_reward = 0.0
        action_sequence = []
        achieved_max = False
        
        while True:
            choice = np.random.choice(env.num_tiles)
            logits = np.eye(env.num_tiles)[choice]
            _, reward, terminated, truncated, info = env.step(logits)
            total_reward += reward
            action_sequence.append(choice)
            
            if terminated or truncated:
                achieved_max = info.get("achieved_max_reward", False)
                break
                
        return total_reward, action_sequence, achieved_max

    def backpropagate(self, node: Node, reward: float) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def search(self) -> tuple[int | None, list[int], bool]:
        num_simulations = 48
        leaf_nodes = [self.select_node() for _ in range(num_simulations)]
        
        # Process simulations sequentially (parallel removed for simplicity)
        results = []
        for node in leaf_nodes:
            if node.untried_actions:
                node = self.expand(node)
            results.append(self.simulate(node))
        
        # Backpropagate results
        for node, (reward, rollout_sequence, achieved_max) in zip(leaf_nodes, results):
            self.backpropagate(node, reward)
            
            if achieved_max:
                prefix_actions = []
                current = node
                while current.parent is not None:
                    prefix_actions.append(current.action_index)
                    current = current.parent
                prefix_actions.reverse()
                
                full_sequence = prefix_actions + rollout_sequence
                self.best_action_sequence = full_sequence
                self.best_reward = reward
                return full_sequence[0], full_sequence, True
                
            elif reward > self.best_reward:
                self.best_reward = reward
                prefix_actions = []
                current = node
                while current.parent is not None:
                    prefix_actions.append(current.action_index)
                    current = current.parent
                prefix_actions.reverse()
                self.best_action_sequence = prefix_actions + rollout_sequence
        
        if not self.root.children:
            return None, self.best_action_sequence, False
            
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.action_index, self.best_action_sequence, False


def run_mcts_search(
    env: WFCWrapper,
    exploration_weight: float,
    max_iterations: int = 1000,
    patience: int = 50,
) -> tuple[list[int] | None, float, int]:
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


def resume_mcts_search(mcts_instance: MCTS, max_iterations: int) -> tuple[list[int] | None, float | None, int | None]:
    for i in tqdm(range(max_iterations), desc="Resuming MCTS Search", leave=False):
        _, action_sequence, found_max = mcts_instance.search()
        if found_max:
            return action_sequence, mcts_instance.best_reward, i + 1
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
            # Convert tile indices to action logits for saving
            action_logits_sequence = [
                np.eye(env.num_tiles)[idx] for idx in best_sequence
            ]
            AGENT_DIR = "agents"
            os.makedirs(AGENT_DIR, exist_ok=True)
            task_str = "_".join(args.task)
            filename = f"{AGENT_DIR}/best_mcts_{task_str}_reward_{best_reward:.2f}_sequence.pkl"
            with open(filename, "wb") as f:
                pickle.dump(action_logits_sequence, f)
            print(f"Saved best sequence to {filename}")
            
            # Save the rendered output
            os.makedirs("mcts_output", exist_ok=True)
            output_filename = f"mcts_output/{task_str}_steps_{len(best_sequence)}_reward_{best_reward:.2f}.png"
            env.render_mode = "human"
            observation, _ = env.reset()
            for idx in best_sequence:
                action = np.eye(env.num_tiles)[idx]
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
