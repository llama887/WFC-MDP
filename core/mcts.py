import argparse
import copy
import math
import os
import random
import sys
import time
import yaml
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, List, Optional, Tuple

import numpy as np
import pygame
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wfc_env import WFCWrapper
from wfc import load_tile_images, find_lowest_entropy_cell
from assets.biome_adjacency_rules import create_adjacency_matrix


# ---------------------- MCTS Node Class ----------------------

class MonteCarloTreeSearchNode:
    """A single node in an MCTS tree."""

    def __init__(
        self,
        env_state: Any,
        parent_node: Optional["MonteCarloTreeSearchNode"] = None,
        action_taken: Optional[np.ndarray] = None,
        untried_actions: Optional[List[np.ndarray]] = None,
        state_hash: Optional[str] = None,
        use_smart_logic: bool = True,
        discrete_action_list: Optional[List[np.ndarray]] = None,
    ):
        self.env_state: Any = env_state  # Deep-copied environment at this node
        self.parent_node: Optional["MonteCarloTreeSearchNode"] = parent_node
        self.action_taken: Optional[np.ndarray] = action_taken
        self.children_nodes: List["MonteCarloTreeSearchNode"] = []
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.state_hash = state_hash  # For node reuse
        self.use_smart_logic = use_smart_logic
        
        # Handle untried actions based on smart vs original logic
        if untried_actions is None:
            if use_smart_logic:
                # Smart logic: Find lowest entropy cell and get legal actions
                pos = find_lowest_entropy_cell(
                    self.env_state.grid, 
                    deterministic=self.env_state.deterministic
                )
                if pos is None:
                    self.untried_actions = []
                else:
                    y, x = pos
                    allowed = self.env_state.grid[y][x]
                    # Map symbols to action indices
                    self.untried_actions = [
                        self.env_state.tile_to_index[t] for t in allowed
                    ]
            else:
                # Original logic: Use all discrete actions
                if discrete_action_list is not None:
                    self.untried_actions = list(range(len(discrete_action_list)))
                else:
                    self.untried_actions = []
        else:
            self.untried_actions = untried_actions.copy()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child_by_uct(
        self, exploration_constant: float, temperature: float = 1.0, use_temperature: bool = True
    ) -> "MonteCarloTreeSearchNode":
        """
        UCT with optional temperature scaling and stability improvements.
        UCT = (value_sum / visit_count) + exploration_constant * sqrt(ln(parent.visit_count) / visit_count)
        """
        if not self.children_nodes:
            return None
            
        uct_scores: List[float] = []
        
        if use_temperature:
            # Smart logic: Temperature-based UCT with stability improvements
            min_visits = 1  # Minimum visit count for stability
            
            for child_node in self.children_nodes:
                # Ensure minimum visits for stability
                child_visits = max(child_node.visit_count, min_visits)
                
                # Q-value with temperature scaling
                q_value = child_node.value_sum / child_visits
                
                # Exploration term with parent visits
                parent_visits = max(self.visit_count, 1)
                exploration_term = exploration_constant * math.sqrt(
                    math.log(parent_visits) / child_visits
                )
                
                # Apply temperature scaling for better exploration control
                score = q_value + exploration_term
                uct_scores.append(score / temperature)
                
            # Softmax selection with temperature
            if temperature < 1e-6:
                # Greedy selection
                best_index = int(np.argmax(uct_scores))
            else:
                # Probabilistic selection based on temperature
                exp_scores = np.exp(np.array(uct_scores) - np.max(uct_scores))
                probs = exp_scores / np.sum(exp_scores)
                best_index = np.random.choice(len(self.children_nodes), p=probs)
        else:
            # Original logic: Standard UCT formula
            for child_node in self.children_nodes:
                exploitation_term = child_node.value_sum / (child_node.visit_count + 1e-8)
                exploration_term = exploration_constant * math.sqrt(
                    math.log(self.visit_count + 1) / (child_node.visit_count + 1e-8)
                )
                uct_scores.append(exploitation_term + exploration_term)
            best_index = int(np.argmax(uct_scores))
            
        return self.children_nodes[best_index]


# ---------------------- MCTS Search Core ----------------------

class MCTS:
    """Monte Carlo Tree Search with smart vs original logic control."""
    
    def __init__(
        self,
        root_env: WFCWrapper,
        c_puct: float = 1.4,
        max_simulations: int = None,
        min_visit_improve: float = 1e-3,
        max_no_improve_iters: int = 5000,
        rollout_count: int = 1,
        rollout_policy: str = "random",  # "random", "entropy", "greedy"
        action_score_scale: float = 100.0,
        verbose: bool = True,
        reuse_tree: bool = False,
        temperature: float = 1.0,
        temperature_decay: float = 0.99,
        env_cache_size: int = 1000,
        # New flags for controlling smart vs original logic
        uniform_cell_selection: bool = False,
        allow_illegal_tiles: bool = False,
        no_reuse_tree: bool = False,
        fixed_uct: bool = False,
    ):
        # Deep-copy a clean environment for search
        self.root_env = copy.deepcopy(root_env)
        self.c_puct = c_puct
        self.max_simulations = max_simulations
        self.min_visit_improve = min_visit_improve
        self.max_no_improve_iters = max_no_improve_iters
        self.rollout_count = rollout_count
        self.rollout_policy = rollout_policy
        self.action_score_scale = action_score_scale
        self.verbose = verbose
        self.reuse_tree = reuse_tree and not no_reuse_tree  # Can be overridden by flag
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        
        # Smart vs Original logic flags
        self.uniform_cell_selection = uniform_cell_selection
        self.allow_illegal_tiles = allow_illegal_tiles
        self.no_reuse_tree = no_reuse_tree
        self.fixed_uct = fixed_uct
        self.use_smart_logic = not (uniform_cell_selection and allow_illegal_tiles and no_reuse_tree and fixed_uct)
        
        # Initialize discrete action list
        self.num_tiles = root_env.num_tiles
        self.discrete_action_list = [
            np.eye(self.num_tiles, dtype=np.float32)[i] * action_score_scale
            for i in range(self.num_tiles)
        ]
        
        # Create root node with appropriate logic
        if self.use_smart_logic and not uniform_cell_selection and not allow_illegal_tiles:
            # Smart logic: Find untried actions based on current state
            self.root = MonteCarloTreeSearchNode(
                env_state=self.root_env,
                untried_actions=None,  # Will be computed in __init__
                use_smart_logic=True,
                discrete_action_list=self.discrete_action_list
            )
        else:
            # Original logic: All actions are initially untried
            self.root = MonteCarloTreeSearchNode(
                env_state=self.root_env,
                untried_actions=self.discrete_action_list,
                use_smart_logic=False,
                discrete_action_list=self.discrete_action_list
            )
        
        # Node cache for tree reuse (disabled if no_reuse_tree flag is set)
        self.node_cache = {} if (reuse_tree and not no_reuse_tree) else None
        
        # Environment cache for efficient cloning (disabled if no_reuse_tree flag is set)
        self.env_cache = {} if not no_reuse_tree else None
        self.env_cache_size = env_cache_size

    def _get_state_hash(self, env_state: WFCWrapper) -> str:
        """Generate a hash for the environment state for caching."""
        if self.no_reuse_tree:
            return None
        # Simple hash based on grid state
        grid_str = str(env_state.grid)
        return hash(grid_str)
    
    def _clone_env(self, env_state: WFCWrapper) -> WFCWrapper:
        """Efficient environment cloning with optional caching."""
        if self.no_reuse_tree or self.env_cache is None:
            # Original logic: Always deep copy without caching
            return copy.deepcopy(env_state)
            
        # Smart logic: Use caching for efficiency
        state_hash = self._get_state_hash(env_state)
        
        if state_hash in self.env_cache:
            return copy.deepcopy(self.env_cache[state_hash])
        
        cloned = copy.deepcopy(env_state)
        
        # Manage cache size
        if len(self.env_cache) >= self.env_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.env_cache))
            del self.env_cache[oldest_key]
        
        self.env_cache[state_hash] = cloned
        return copy.deepcopy(cloned)

    def _get_next_cell_position(self, env_state: WFCWrapper) -> Optional[Tuple[int, int]]:
        """Get next cell position using either entropy-based or uniform selection."""
        if self.uniform_cell_selection:
            # Original logic: Random cell selection
            available_cells = []
            for y in range(env_state.map_length):
                for x in range(env_state.map_width):
                    if isinstance(env_state.grid[y][x], (list, set)) and len(env_state.grid[y][x]) > 1:
                        available_cells.append((y, x))
            return random.choice(available_cells) if available_cells else None
        else:
            # Smart logic: Lowest entropy cell selection
            return find_lowest_entropy_cell(
                env_state.grid, 
                deterministic=env_state.deterministic
            )

    def _get_available_actions(self, env_state: WFCWrapper, position: Optional[Tuple[int, int]]) -> List[int]:
        """Get available actions using either legal-only or all-tiles logic."""
        if self.allow_illegal_tiles:
            # Original logic: Try all possible tiles
            return list(range(self.num_tiles))
        else:
            # Smart logic: Only legal tiles based on current constraints
            if position is None:
                return []
            y, x = position
            allowed = env_state.grid[y][x]
            if isinstance(allowed, str):
                return []  # Already collapsed
            return [env_state.tile_to_index[t] for t in allowed]

    def search(self):
        """Run MCTS search with early stopping."""
        best_mean = -float('inf')
        no_improve = 0
        sims_done = 0

        # Create simulation range
        sim_range = range(self.max_simulations) if self.max_simulations is not None else iter(int, 1)
        
        for _ in sim_range:
            sims_done += 1
            
            # MCTS phases
            node = self._select(self.root)
            node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

            # Early stopping check
            if self.max_simulations is None and self.root.children_nodes:
                top = max(
                    self.root.children_nodes,
                    key=lambda c: c.value_sum / (c.visit_count + 1e-8)
                )
                mean_r = top.value_sum / (top.visit_count + 1e-8)
                
                if mean_r - best_mean > self.min_visit_improve:
                    best_mean = mean_r
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve >= self.max_no_improve_iters:
                    if self.verbose:
                        print(f"[MCTS] Converged after {sims_done} sims; best mean {best_mean:.4f}")
                    break
            
            # Progress logging
            if self.verbose and sims_done % 1000 == 0 and self.root.children_nodes:
                top = max(
                    self.root.children_nodes,
                    key=lambda c: c.value_sum / (c.visit_count + 1e-8)
                )
                action_idx = np.argmax(top.action_taken) if top.action_taken is not None else -1
                print(
                    f"[MCTS] Sims {sims_done}, top act {action_idx}, "
                    f"visits {top.visit_count}, avg_r {top.value_sum/top.visit_count:.4f}"
                )

        return self._extract_best_sequence()

    def _select(self, node: MonteCarloTreeSearchNode) -> MonteCarloTreeSearchNode:
        """Select leaf node using UCT with optional temperature."""
        current_temperature = self.temperature if not self.fixed_uct else 1.0
        
        while (node.is_fully_expanded() and 
               node.children_nodes and
               self._get_next_cell_position(node.env_state) is not None):
            node = node.best_child_by_uct(
                self.c_puct, 
                current_temperature, 
                use_temperature=not self.fixed_uct
            )
            if not self.fixed_uct:
                current_temperature *= self.temperature_decay  # Decay temperature as we go deeper
            
        return node

    def _expand(self, node: MonteCarloTreeSearchNode) -> MonteCarloTreeSearchNode:
        """Expand by trying an untried action with smart vs original logic."""
        if not node.untried_actions:
            return node
            
        if self.use_smart_logic and not self.allow_illegal_tiles:
            # Smart logic: Pop an untried action index and convert to action
            action_idx = node.untried_actions.pop(0)
            action = self.discrete_action_list[action_idx]
        else:
            # Original logic: Pop an action directly or get action by index
            if isinstance(node.untried_actions[0], np.ndarray):
                action = node.untried_actions.pop()
                action_idx = np.argmax(action)
            else:
                action_idx = node.untried_actions.pop()
                action = self.discrete_action_list[action_idx]
        
        # Create new environment and apply action
        new_env = self._clone_env(node.env_state)
        try:
            _, _, terminated, truncated, _ = new_env.step(action)
            
            # Generate state hash for potential reuse
            state_hash = self._get_state_hash(new_env) if (self.reuse_tree and not self.no_reuse_tree) else None
            
            # Check if we've seen this state before
            if self.reuse_tree and not self.no_reuse_tree and state_hash and state_hash in self.node_cache:
                # Reuse existing node
                child = self.node_cache[state_hash]
                child.parent_node = node  # Update parent
            else:
                # Create new child node with appropriate logic
                if self.use_smart_logic and not self.allow_illegal_tiles:
                    child = MonteCarloTreeSearchNode(
                        env_state=new_env,
                        parent_node=node,
                        action_taken=action,
                        untried_actions=None,  # Will be computed in __init__
                        state_hash=state_hash,
                        use_smart_logic=True,
                        discrete_action_list=self.discrete_action_list
                    )
                else:
                    child = MonteCarloTreeSearchNode(
                        env_state=new_env,
                        parent_node=node,
                        action_taken=action,
                        untried_actions=self.discrete_action_list,
                        state_hash=state_hash,
                        use_smart_logic=False,
                        discrete_action_list=self.discrete_action_list
                    )
                
                if self.reuse_tree and not self.no_reuse_tree and state_hash and self.node_cache is not None:
                    self.node_cache[state_hash] = child
                    
            node.children_nodes.append(child)
            return child
        except Exception:
            # Contradiction - try next action
            return self._expand(node)

    def _single_rollout(self, env_state: WFCWrapper) -> float:
        """Perform a single rollout with configurable policy."""
        sim_env = self._clone_env(env_state)
        accumulated_reward = 0.0
        
        try:
            while True:
                pos = self._get_next_cell_position(sim_env)
                if pos is None:
                    break
                    
                available_actions = self._get_available_actions(sim_env, pos)
                if not available_actions:
                    break
                
                # Select action based on rollout policy
                if self.rollout_policy == "random" or self.allow_illegal_tiles:
                    # Random action from available tiles (original logic when allow_illegal_tiles is True)
                    tile_idx = random.choice(available_actions)
                    
                elif self.rollout_policy == "entropy" and not self.allow_illegal_tiles:
                    # Entropy-weighted selection (prefer lower entropy options)
                    # Calculate entropy reduction for each choice
                    entropy_scores = []
                    for tile_idx in available_actions:
                        # Simulate the propagation effect (simplified)
                        entropy_score = 1.0 / (len(available_actions) + 1)  # Simple heuristic
                        entropy_scores.append(entropy_score)
                    
                    # Convert to probabilities (higher score = higher probability)
                    probs = np.array(entropy_scores)
                    probs = probs / probs.sum()
                    tile_idx = np.random.choice(available_actions, p=probs)
                    
                elif self.rollout_policy == "greedy" and not self.allow_illegal_tiles:
                    # Greedy selection based on local constraints
                    y, x = pos
                    best_score = -1
                    best_tile = available_actions[0]
                    
                    for tile_idx in available_actions:
                        score = 0
                        tile_symbol = sim_env.tile_symbols[tile_idx]
                        
                        # Check adjacent cells
                        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < sim_env.map_length and 0 <= nx < sim_env.map_width:
                                neighbor_options = sim_env.grid[ny][nx]
                                if isinstance(neighbor_options, str):
                                    # Already collapsed
                                    neighbor_idx = sim_env.tile_to_index[neighbor_options]
                                    if sim_env.adjacency_bool[tile_idx][neighbor_idx]:
                                        score += 1
                                else:
                                    # Count compatible options
                                    for n_tile in neighbor_options:
                                        n_idx = sim_env.tile_to_index[n_tile]
                                        if sim_env.adjacency_bool[tile_idx][n_idx]:
                                            score += 0.1
                        
                        if score > best_score:
                            best_score = score
                            best_tile = tile_idx
                    
                    tile_idx = best_tile
                else:
                    # Default to random
                    tile_idx = random.choice(available_actions)
                
                action = self.discrete_action_list[tile_idx]
                
                _, reward, done, trunc, _ = sim_env.step(action)
                accumulated_reward += reward
                
                if done or trunc:
                    break
                    
            return accumulated_reward
        except Exception:
            return -1e6  # Penalty for invalid states

    def _simulate(self, node: MonteCarloTreeSearchNode) -> float:
        """Simulate with optional parallel rollouts."""
        if self.rollout_count <= 1:
            return self._single_rollout(node.env_state)
            
        # Parallel rollouts
        with Pool(min(self.rollout_count, cpu_count())) as pool:
            rewards = pool.map(
                self._single_rollout, 
                [node.env_state] * self.rollout_count
            )
        return float(np.mean(rewards))

    def _backpropagate(self, node: MonteCarloTreeSearchNode, reward: float):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += reward
            node = node.parent_node

    def _extract_best_sequence(self) -> List[np.ndarray]:
        """Extract best action sequence from root to terminal."""
        sequence = []
        node = self.root
        
        while True:
            pos = self._get_next_cell_position(node.env_state)
            if pos is None or not node.children_nodes:
                break
                
            # Select best child by average reward
            node = max(
                node.children_nodes,
                key=lambda c: c.value_sum / (c.visit_count + 1e-8)
            )
            if node.action_taken is not None:
                sequence.append(node.action_taken)
                
        return sequence


# ---------------------- Parallel MCTS Functions ----------------------

def collect_root_statistics(
    args: Tuple[Any, int, float, float, dict],
) -> List[Tuple[int, int, float]]:
    """Run a single-threaded MCTS and return root children stats."""
    env_instance, iterations, uct_constant, scale, flags = args
    
    mcts = MCTS(
        root_env=env_instance,
        c_puct=uct_constant,
        max_simulations=iterations,
        action_score_scale=scale,
        verbose=False,  # Disable verbose in parallel workers
        **flags  # Pass through all the smart vs original logic flags
    )
    
    # Run search
    _ = mcts.search()
    
    # Collect statistics
    stats: List[Tuple[int, int, float]] = []
    for child in mcts.root.children_nodes:
        if child.action_taken is not None:
            action_index: int = int(np.argmax(child.action_taken))
            stats.append((action_index, child.visit_count, child.value_sum))
    return stats


def run_parallel_mcts_full_action_sequence(
    environment: Any,
    total_search_iterations: int = 1000,
    num_parallel_workers: int = 4,
    exploration_constant: float = 1.4,
    action_score_scale: float = 100.0,
    verbose: bool = True,
    **mcts_flags
) -> List[np.ndarray]:
    """Perform root-parallel MCTS planning for the full action sequence."""
    iterations_per_worker: int = total_search_iterations // num_parallel_workers
    num_tiles: int = environment.num_tiles
    discrete_action_list: List[np.ndarray] = [
        np.eye(num_tiles, dtype=np.float32)[i] * action_score_scale
        for i in range(num_tiles)
    ]

    best_action_sequence: List[np.ndarray] = []
    progress_bar = tqdm(total=environment.max_steps, desc="Planning steps") if verbose else None
    is_done: bool = False

    while not is_done:
        worker_arguments = [
            (
                copy.deepcopy(environment),
                iterations_per_worker,
                exploration_constant,
                action_score_scale,
                mcts_flags,
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
        
        if progress_bar:
            progress_bar.update(1)

    if progress_bar:
        progress_bar.close()
        
    return best_action_sequence


# ---------------------- Visualization Functions ----------------------

def replay_with_pygame(env: WFCWrapper, sequence: List[np.ndarray], delay: float = 0.05):
    """Step through the sequence, rendering each collapse via pygame."""
    env.reset()
    env.render()
    pygame.display.flip()
    pygame.event.pump()
    time.sleep(delay)

    for action in sequence:
        env.step(action)
        env.render()
        pygame.display.flip()
        pygame.event.pump()
        time.sleep(delay)

    time.sleep(1.0)


# ---------------------- Script Entrypoint ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS for Wave Function Collapse")
    parser.add_argument("--task", choices=["binary", "pond", "river"], required=True)
    parser.add_argument("--map-width", type=int, default=20)
    parser.add_argument("--map-height", type=int, default=15)
    parser.add_argument("--load-hyperparameters", type=str,
                        help="YAML containing hyperparameters")
    parser.add_argument("--c-puct", type=float, default=1.4)
    parser.add_argument("--max-sims", type=int, default=None)
    parser.add_argument("--min-imp", type=float, default=1e-3)
    parser.add_argument("--max-no-imp", type=int, default=5000)
    parser.add_argument("--rollout-policy", choices=["random", "entropy", "greedy"], 
                        default="random", help="Rollout policy for simulations")
    parser.add_argument("--reuse-tree", action="store_true",
                        help="Enable tree reuse across states")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for UCT selection")
    parser.add_argument("--temperature-decay", type=float, default=0.99,
                        help="Temperature decay rate")
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--total-iterations", type=int, default=2000)
    parser.add_argument("--action-scale", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true",
                        help="Visualize collapse process with Pygame")
    parser.add_argument("--use-parallel", action="store_true",
                        help="Use parallel MCTS instead of single tree")
    parser.add_argument("--rollout-count", type=int, default=1)
    
    # ============= NEW FLAGS FOR SMART VS ORIGINAL LOGIC =============
    parser.add_argument("--uniform-cell-selection", action="store_true",
                        help="Use random cell selection instead of lowest entropy (reverts to original logic)")
    parser.add_argument("--allow-illegal-tiles", action="store_true", 
                        help="Try all tiles, not just legal ones (reverts to original logic)")
    parser.add_argument("--no-reuse-tree", action="store_true",
                        help="Disable environment reuse and caching (reverts to original logic)")
    parser.add_argument("--fixed-uct", action="store_true",
                        help="Disable temperature-scaling in UCT formula (reverts to original logic)")
    
    args = parser.parse_args()

    # Load hyperparameters from YAML if provided
    if args.load_hyperparameters:
        with open(args.load_hyperparameters, 'r') as f:
            cfg = yaml.safe_load(f)
        c_puct = cfg.get("c_puct", args.c_puct)
        max_sims = cfg.get("max_simulations", args.max_sims)
        min_imp = cfg.get("min_visit_improve", args.min_imp)
        max_no_imp = cfg.get("max_no_improve_iters", args.max_no_imp)
        rollout_count = cfg.get("rollout_count", args.rollout_count)
        rollout_policy = cfg.get("rollout_policy", args.rollout_policy)
        reuse_tree = cfg.get("reuse_tree", args.reuse_tree)
        temperature = cfg.get("temperature", args.temperature)
        temperature_decay = cfg.get("temperature_decay", args.temperature_decay)
        parallel_workers = cfg.get("parallel_workers", args.parallel_workers)
        total_iterations = cfg.get("total_iterations", args.total_iterations)
        action_scale = cfg.get("action_score_scale", args.action_scale)
        # New flags
        uniform_cell_selection = cfg.get("uniform_cell_selection", args.uniform_cell_selection)
        allow_illegal_tiles = cfg.get("allow_illegal_tiles", args.allow_illegal_tiles)
        no_reuse_tree = cfg.get("no_reuse_tree", args.no_reuse_tree)
        fixed_uct = cfg.get("fixed_uct", args.fixed_uct)
    else:
        c_puct = args.c_puct
        max_sims = args.max_sims
        min_imp = args.min_imp
        max_no_imp = args.max_no_imp
        rollout_count = args.rollout_count
        rollout_policy = args.rollout_policy
        reuse_tree = args.reuse_tree
        temperature = args.temperature
        temperature_decay = args.temperature_decay
        parallel_workers = args.parallel_workers
        total_iterations = args.total_iterations
        action_scale = args.action_scale
        # New flags
        uniform_cell_selection = args.uniform_cell_selection
        allow_illegal_tiles = args.allow_illegal_tiles
        no_reuse_tree = args.no_reuse_tree
        fixed_uct = args.fixed_uct

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Print configuration summary
    print("=== MCTS Configuration ===")
    print(f"Smart vs Original Logic Flags:")
    print(f"  Uniform cell selection: {uniform_cell_selection} {'(ORIGINAL)' if uniform_cell_selection else '(SMART)'}")
    print(f"  Allow illegal tiles: {allow_illegal_tiles} {'(ORIGINAL)' if allow_illegal_tiles else '(SMART)'}")
    print(f"  No reuse tree: {no_reuse_tree} {'(ORIGINAL)' if no_reuse_tree else '(SMART)'}")
    print(f"  Fixed UCT: {fixed_uct} {'(ORIGINAL)' if fixed_uct else '(SMART)'}")
    
    all_original = uniform_cell_selection and allow_illegal_tiles and no_reuse_tree and fixed_uct
    all_smart = not (uniform_cell_selection or allow_illegal_tiles or no_reuse_tree or fixed_uct)
    
    if all_original:
        print("🔄 Running in FULL ORIGINAL MODE")
    elif all_smart:
        print("🧠 Running in FULL SMART MODE")
    else:
        print("🔀 Running in HYBRID MODE")

    # Build adjacency & tile mappings
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)

    # Dynamically import the task's reward function
    task_mod = __import__(f"tasks.{args.task}_task", fromlist=[f"{args.task}_reward"])
    reward_fn = getattr(task_mod, f"{args.task}_reward")

    # Create environment for search
    search_env = WFCWrapper(
        map_length=args.map_height,
        map_width=args.map_width,
        tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool,
        num_tiles=num_tiles,
        tile_to_index=tile_to_index,
        reward=reward_fn,
        render_mode=None,
        deterministic=True
    )

    search_env.reset()

    # Prepare MCTS flags for passing to functions
    mcts_flags = {
        'uniform_cell_selection': uniform_cell_selection,
        'allow_illegal_tiles': allow_illegal_tiles,
        'no_reuse_tree': no_reuse_tree,
        'fixed_uct': fixed_uct,
        'reuse_tree': reuse_tree,
        'temperature': temperature,
        'temperature_decay': temperature_decay,
        'rollout_count': rollout_count,
        'rollout_policy': rollout_policy,
        'min_visit_improve': min_imp,
        'max_no_improve_iters': max_no_imp,
    }

    # Run MCTS
    if args.use_parallel:
        print(f"Running parallel MCTS with {parallel_workers} workers...")
        best_seq = run_parallel_mcts_full_action_sequence(
            environment=search_env,
            total_search_iterations=total_iterations,
            num_parallel_workers=parallel_workers,
            exploration_constant=c_puct,
            action_score_scale=action_scale,
            verbose=True,
            **mcts_flags
        )
    else:
        print("Running single-tree MCTS...")
        agent = MCTS(
            root_env=search_env,
            c_puct=c_puct,
            max_simulations=max_sims,
            action_score_scale=action_scale,
            verbose=True,
            **mcts_flags
        )
        best_seq = agent.search()

    print(f"Planned {len(best_seq)} steps.")

    # Render or save result
    if args.render:
        # Load images and build a fresh env for rendering
        tile_images = load_tile_images()
        render_env = WFCWrapper(
            map_length=args.map_height,
            map_width=args.map_width,
            tile_symbols=tile_symbols,
            adjacency_bool=adjacency_bool,
            num_tiles=num_tiles,
            tile_to_index=tile_to_index,
            reward=reward_fn,
            tile_images=tile_images,
            render_mode="human",
        )

        pygame.init()
        screen_w = render_env.map_width * render_env.tile_size
        screen_h = render_env.map_length * render_env.tile_size
        screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("MCTS WFC Collapse")

        replay_with_pygame(render_env, best_seq, delay=0.1)

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            time.sleep(0.1)
    else:
        tile_images = load_tile_images()
        final_env = WFCWrapper(
            map_length=args.map_height,
            map_width=args.map_width,
            tile_symbols=tile_symbols,
            adjacency_bool=adjacency_bool,
            num_tiles=num_tiles,
            tile_to_index=tile_to_index,
            reward=reward_fn,
            tile_images=tile_images,
            render_mode=None
        )
        final_env.reset()
        for action in best_seq:
            final_env.step(action)
        final_env.save_map("mcts_result.png")
        print("Saved result to mcts_result.png")