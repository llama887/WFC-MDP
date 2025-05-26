import copy
import math
import os
import sys
from functools import partial
from multiprocessing import Pool
from typing import Any, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from wfc_env import WFCWrapper

# ---------------------- MCTS Node Class ----------------------


class MonteCarloTreeSearchNode:
    """A single node in an MCTS tree."""

    def __init__(
        self,
        env_state: Any,
        parent_node: Optional["MonteCarloTreeSearchNode"] = None,
        action_taken: Optional[np.ndarray] = None,
        untried_actions: Optional[List[np.ndarray]] = None,
    ):
        self.env_state: Any = env_state  # Deep-copied environment at this node
        self.parent_node: Optional["MonteCarloTreeSearchNode"] = parent_node
        self.action_taken: Optional[np.ndarray] = action_taken
        self.children_nodes: List["MonteCarloTreeSearchNode"] = []
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.untried_actions: List[np.ndarray] = (
            [] if untried_actions is None else untried_actions.copy()
        )

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child_by_uct(
        self, exploration_constant: float
    ) -> "MonteCarloTreeSearchNode":
        """
        UCT = (value_sum / visit_count) + exploration_constant * sqrt(ln(parent.visit_count) / visit_count)
        """
        uct_scores: List[float] = []
        for child_node in self.children_nodes:
            exploitation_term = child_node.value_sum / (child_node.visit_count + 1e-8)
            exploration_term = exploration_constant * math.sqrt(
                math.log(self.visit_count + 1) / (child_node.visit_count + 1e-8)
            )
            uct_scores.append(exploitation_term + exploration_term)
        best_index: int = int(np.argmax(uct_scores))
        return self.children_nodes[best_index]


# ---------------------- MCTS Search Core ----------------------


def build_single_mcts_tree(
    env_copy: WFCWrapper,
    total_iterations: int,
    exploration_constant: float,
    action_score_scale: float,
) -> MonteCarloTreeSearchNode:
    """
    Build an MCTS tree starting from env_copy, return only the root node.
    """
    env_copy.reset()
    num_tiles: int = env_copy.num_tiles
    discrete_action_list: List[np.ndarray] = [
        np.eye(num_tiles, dtype=np.float32)[i] * action_score_scale
        for i in range(num_tiles)
    ]

    root_node = MonteCarloTreeSearchNode(
        env_state=copy.deepcopy(env_copy),
        untried_actions=discrete_action_list,
    )

    for _ in range(total_iterations):
        # Selection
        current_node = root_node
        while current_node.is_fully_expanded() and current_node.children_nodes:
            current_node = current_node.best_child_by_uct(exploration_constant)

        # Expansion
        if current_node.untried_actions:
            chosen_action = current_node.untried_actions.pop()
            next_state = copy.deepcopy(current_node.env_state)
            _, _, terminated, truncated, _ = next_state.step(chosen_action)
            new_child = MonteCarloTreeSearchNode(
                env_state=next_state,
                parent_node=current_node,
                action_taken=chosen_action,
                untried_actions=discrete_action_list,
            )
            current_node.children_nodes.append(new_child)
            current_node = new_child

        # Simulation
        rollout_state = copy.deepcopy(current_node.env_state)
        accumulated_reward: float = 0.0
        while True:
            random_action = discrete_action_list[np.random.randint(num_tiles)]
            _, reward, done, trunc, _ = rollout_state.step(random_action)
            accumulated_reward += reward
            if done or trunc:
                break

        # Backpropagation
        node_to_update = current_node
        while node_to_update is not None:
            node_to_update.visit_count += 1
            node_to_update.value_sum += accumulated_reward
            node_to_update = node_to_update.parent_node

    return root_node


# ---------------------- Worker and Parallel Planner ----------------------


def collect_root_statistics(
    args: Tuple[Any, int, float, float],
) -> List[Tuple[int, int, float]]:
    """
    Run a single-threaded MCTS and return root children stats.
    """
    env_instance, iterations, uct_constant, scale = args
    root_node = build_single_mcts_tree(env_instance, iterations, uct_constant, scale)
    stats: List[Tuple[int, int, float]] = []
    for child in root_node.children_nodes:
        action_index: int = int(np.argmax(child.action_taken))
        stats.append((action_index, child.visit_count, child.value_sum))
    return stats


def run_parallel_mcts_full_action_sequence(
    environment: Any,
    total_search_iterations: int = 1000,
    num_parallel_workers: int = 4,
    exploration_constant: float = 1.4,
    action_score_scale: float = 100.0,
) -> List[np.ndarray]:
    """
    Perform root-parallel MCTS planning for the full action sequence.
    """
    iterations_per_worker: int = total_search_iterations // num_parallel_workers
    num_tiles: int = environment.num_tiles
    discrete_action_list: List[np.ndarray] = [
        np.eye(num_tiles, dtype=np.float32)[i] * action_score_scale
        for i in range(num_tiles)
    ]

    best_action_sequence: List[np.ndarray] = []
    progress_bar = tqdm(total=environment.max_steps, desc="Planning steps")
    is_done: bool = False

    while not is_done:
        worker_arguments = [
            (
                copy.deepcopy(environment),
                iterations_per_worker,
                exploration_constant,
                action_score_scale,
            )
            for _ in range(num_parallel_workers)
        ]
        visit_counter: List[int] = [0] * num_tiles
        value_accumulator: List[float] = [0.0] * num_tiles

        with Pool(processes=num_parallel_workers) as process_pool:
            for worker_result in process_pool.imap_unordered(
                collect_root_statistics, worker_arguments
            ):
                for tile_index, visit_count, value_sum in worker_result:
                    visit_counter[tile_index] += visit_count
                    value_accumulator[tile_index] += value_sum

        selected_action_index: int = int(np.argmax(visit_counter))
        selected_action: np.ndarray = discrete_action_list[selected_action_index]
        best_action_sequence.append(selected_action)

        _, _, terminated, truncated, _ = environment.step(selected_action)
        is_done = terminated or truncated
        progress_bar.update(1)

    progress_bar.close()
    return best_action_sequence


# ---------------------- Script Entrypoint ----------------------

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wfc_env import WFCWrapper

from assets.biome_adjacency_rules import create_adjacency_matrix
from tasks.binary_task import binary_reward

if __name__ == "__main__":
    map_length: int = 15
    map_width: int = 20
    adjacency_matrix, tile_symbol_list, symbol_to_index_map = create_adjacency_matrix()
    total_tile_types: int = len(tile_symbol_list)

    environment_instance = WFCWrapper(
        map_length=map_length,
        map_width=map_width,
        tile_symbols=tile_symbol_list,
        adjacency_bool=adjacency_matrix,
        num_tiles=total_tile_types,
        tile_to_index=symbol_to_index_map,
        reward=partial(binary_reward, target_path_length=80),
        deterministic=True,
    )

    environment_instance.reset()

    action_plan = run_parallel_mcts_full_action_sequence(
        environment=environment_instance,
        total_search_iterations=2000,
        num_parallel_workers=10,
        exploration_constant=1.0,
        action_score_scale=100.0,
    )

    print(
        f"Planned {len(action_plan)} steps. First tile index: {np.argmax(action_plan[0])}"
    )
