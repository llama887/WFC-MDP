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

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assets.biome_adjacency_rules import create_adjacency_matrix, load_tile_images
from tasks.binary_task import binary_reward
from tasks.grass_task import grass_reward
from tasks.hill_task import hill_reward
from tasks.pond_task import pond_reward
from tasks.river_task import river_reward
from wfc_env import CombinedReward, WFCWrapper


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


class MCTSConfig(BaseModel):
    """Configuration for the MCTS algorithm"""

    exploration_weight: float = Field(
        default=1.0, description="Exploration weight for UCT calculation"
    )
    # num_simulations is always 48, hardcoded everywhere else.


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

    def __init__(self, env: WFCWrapper, config: MCTSConfig = MCTSConfig()):
        self.env = env
        self.config = config
        self.root = Node(env)
        self.best_action_sequence: list[np.ndarray] = []
        self.best_reward = float("-inf")

    def _run_simulation(self, node: Node) -> tuple[float, list[np.ndarray], bool, Node]:
        """Run a single simulation from a node"""
        reward, action_sequence, achieved_max_reward = node.simulate()
        node.backpropagate(reward)
        return reward, action_sequence, achieved_max_reward, node

    def search(self) -> tuple[Action, list[np.ndarray], bool]:
        """Run the MCTS algorithm and return the best action and sequence"""
        num_simulations = 48  # Hardcoded as per hardware limitations
        num_processes = min(multiprocessing.cpu_count(), num_simulations)
        with Pool(processes=num_processes) as pool:
            nodes = [self.select_node() for _ in range(num_simulations)]
            simulation_results = pool.map(self._run_simulation, nodes)

        for reward, rollout_sequence, achieved_max_reward, node in simulation_results:
            if achieved_max_reward:
                prefix_actions: list[np.ndarray] = []
                n = node
                while n.parent is not None:
                    prefix_actions.append(n.action_taken.action_logits)
                    n = n.parent
                prefix_actions.reverse()
                full_sequence = prefix_actions + rollout_sequence
                self.best_action_sequence = full_sequence
                self.best_reward = reward
                for child in self.root.children:
                    if child.action_taken and np.array_equal(child.action_taken.action_logits, full_sequence[0]):
                        return child.action_taken, full_sequence, True

            if reward > self.best_reward:
                self.best_reward = reward
                prefix_actions = []
                n = node
                while n.parent is not None:
                    prefix_actions.append(n.action_taken.action_logits)
                    n = n.parent
                prefix_actions.reverse()
                self.best_action_sequence = prefix_actions + rollout_sequence

        if not self.root.children:
            action_logits = self.env.action_space.sample()
            return Action(action_logits=action_logits), self.best_action_sequence, False

        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.action_taken, self.best_action_sequence, False

    def select_node(self) -> Node:
        """Select a node to expand using UCT"""
        node = self.root
        while not node.is_terminal and node.is_fully_expanded:
            node = node.best_child(self.config.exploration_weight)
        if not node.is_terminal and not node.is_fully_expanded:
            return node.expand()
        return node


def run_mcts_until_complete(env: WFCWrapper, config: MCTSConfig, max_iterations: int = 1000) -> tuple[list[np.ndarray] | None, float | None, int | None]:
    """Run MCTS search until a complete solution is found or max iterations are reached."""
    mcts = MCTS(env, config)
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
    hyperparams = {
        "exploration_weight": trial.suggest_float("exploration_weight", 0.1, 3.0),
    }
    config = MCTSConfig(**hyperparams)
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
        _, _, iterations = run_mcts_until_complete(env, config, max_iterations=max_iterations_per_trial)
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
        config = MCTSConfig(**hyperparams)
        print("Successfully loaded hyperparameters:", hyperparams)
        start_time = time.time()
        sequence, reward, iterations = run_mcts_until_complete(env, config, max_iterations=args.max_iterations)
        end_time = time.time()
        print(f"MCTS search finished in {end_time - start_time:.2f} seconds.")
        if sequence:
            print(f"Complete solution found at iteration {iterations} with reward {reward:.4f}")
            AGENT_DIR = "agents"
            task_str = "_".join(args.task)
            filename = f"{AGENT_DIR}/best_mcts_{task_str}_reward_{sequence:.2f}_sequence.pkl"
            with open(filename, "wb") as f:
                pickle.dump(sequence, f)
            print(f"Saved best sequence to {filename}")
            os.makedirs("mcts_output", exist_ok=True)

            output_filename = f"mcts_output/{task_str}_steps_{len(sequence)}_reward_{reward:.2f}.png"
            env.render_mode = "human"
            observation, _ = env.reset()
            for action in sequence:
                observation, _, terminated, truncated, _ = env.step(action)
                env.render()
                if terminated or truncated:
                    break
            
            env.save_render(output_filename)
            print(f"Saved rendered output to {output_filename}")
            time.sleep(5)
            env.close()
        else:
            print("No complete solution found within the maximum iterations.")
    elif args.best_sequence_pickle:
        with open(args.best_sequence_pickle, "rb") as f:
            sequence = pickle.load(f)
        print(f"Loaded sequence from {args.best_sequence_pickle}")

        os.makedirs("mcts_output", exist_ok=True)
        test_env = copy.deepcopy(env)
        test_env.reset()
        total_reward = 0
        for action in sequence:
            _, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        output_filename = f"mcts_output/{'_'.join(args.task)}_steps_{len(sequence)}_reward_{total_reward:.2f}.png"
        env.render_mode = "human"
        observation, _ = env.reset()
        for action in sequence:
            observation, _, terminated, truncated, _ = env.step(action)
            env.render()
            if terminated or truncated:
                break
        
        env.save_render(output_filename)
        print(f"Saved rendered output to {output_filename}")
        time.sleep(5)
        env.close()
    else:
        parser.error("Please specify a mode: --optuna-trials, --load-hyperparameters, or --best-sequence-pickle")
    print("Script finished.")


if __name__ == "__main__":
    main()
