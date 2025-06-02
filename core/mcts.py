from __future__ import annotations
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import numpy as np
from copy import deepcopy
from pydantic import BaseModel, Field
import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from wfc_env import WFCWrapper
from assets.biome_adjacency_rules import create_adjacency_matrix
from tasks.binary_task import binary_reward
import pygame

class Action(BaseModel):
    """Represents an action in the MCTS tree with its statistics"""
    action_logits: np.ndarray = Field(default_factory=lambda: np.array([]), description="Logits for the action taken")
    visits: int = Field(default=0, description="Number of times this action has been taken")
    total_reward: float = Field(default=0.0, description="Total reward accumulated from this action")
    tile_index: int = Field(default=-1, description="Index of the tile this action represents")
    
    class Config:
        arbitrary_types_allowed = True

class MCTSConfig(BaseModel):
    """Configuration for the MCTS algorithm"""
    exploration_weight: float = Field(default=1.0, description="Exploration weight for UCT calculation")
    num_simulations: int = Field(default=20, description="Number of simulations to run")

class Node:
    """A node in the MCTS tree"""
    def __init__(self, env: WFCWrapper, parent: Node | None=None, action_taken: Action | None=None):
        self.env = deepcopy(env)
        self.parent = parent
        self.action_taken = action_taken  # Action that led to this node
        self.children: list[Node] = []
        # Available actions are the possible actions from this node's environment state
        self.available_actions: dict[int, Action] = {i: Action(action_logits=np.eye(env.num_tiles)[i], tile_index=i) 
                                                      for i in range(env.num_tiles)}
        self.visits = 0
        self.total_reward = 0.0
        self.is_terminal = False
        self.is_fully_expanded = False
    
    def uct_score(self, child: 'Node', exploration_weight: float) -> float:
        """Calculate the UCT score for a child node"""
        # Avoid division by zero
        if child.visits == 0:
            return float('inf')
        
        # Exploitation term
        exploitation = child.total_reward / child.visits
        
        # Exploration term
        exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
        
        return exploitation + exploration
    
    def best_child(self, exploration_weight: float) -> 'Node':
        """Select the child with the highest UCT score"""
        if not self.children:
            raise ValueError("Node has no children")
        
        return max(self.children, key=lambda child: self.uct_score(child, exploration_weight))
    
    def expand(self) -> 'Node':
        """Expand the node by adding a new child"""
        
        # Choose an unexplored action
        unexplored = [idx for idx, action in self.available_actions.items() 
                     if all(not child.action_taken or child.action_taken.tile_index != idx 
                           for child in self.children)]
        
        if not unexplored:
            self.is_fully_expanded = True
            return self.best_child(exploration_weight=1.0)
        
        # Select a random unexplored action
        tile_idx = unexplored[np.random.randint(0, len(unexplored))]
        action = self.available_actions[tile_idx]
        
        # Create a new child node
        child_env = deepcopy(self.env)
        obs, reward, terminated, truncated, info = child_env.step(np.array(action.action_logits))
        
        child = Node(child_env, parent=self, action_taken=action)
        child.is_terminal = terminated or truncated
        
        self.children.append(child)
        return child
    
    def simulate(self) -> tuple[float, list[np.ndarray], bool]:
        """Run a simulation from this node to a terminal state
        
        Returns:
            - total_reward: Total reward accumulated during the simulation
            - action_sequence: List of actions taken during the simulation
            - achieved_max_reward: Whether the maximum reward was achieved
        """
        if self.is_terminal:
            # Get the final reward from the environment
            return self.env.reward(self.env.grid)[0]
        
        # Make a copy of the environment to avoid modifying the original
        sim_env = deepcopy(self.env)
        terminated = False
        truncated = False
        total_reward = 0.0
        action_sequence = []
        # Run random actions until terminal state
        while not (terminated or truncated):
            # pick a random action from available actions
            action_idx = np.random.choice(list(self.available_actions.keys()))
            action = self.available_actions[action_idx].action_logits
            
            # Take a step in the environment
            _, reward, terminated, truncated, info = sim_env.step(action)
            total_reward += reward
            action_sequence.append(action)
            if terminated and info.get("achieved_max_reward", False):
                assert total_reward == 0, f"Total reward is {total_reward} while achieved_max_reward is {info.get("achieved_max_reward", False)}, expected 0 for max reward"
                assert sim_env.deterministic, "Expected deterministic environment for MCTS simulation"
                print("Simulation info:", info)
                return total_reward, action_sequence, True
        
        return total_reward, action_sequence, False
    
    def backpropagate(self, reward: float) -> None:
        """Update statistics for this node and all ancestors"""
        node = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            
            # Also update the action that led to this node
            if node.parent is not None and node.action_taken is not None:
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
        self.best_reward = float('-inf')                                                                                                                                                                                             
                                                                                                                                                                                                                                     
    def _run_simulation(self, node: Node) -> tuple[float, list[np.ndarray], bool, Node]:
        """Run a single simulation from a node
        
        Returns:
            - reward: The reward from the simulation
            - action_sequence: The action sequence from the simulation
            - achieved_max_reward: Whether the maximum reward was achieved
            - node: The node that was simulated
        """
        # Simulation
        reward, action_sequence, achieved_max_reward = node.simulate()
        if achieved_max_reward:
            assert reward == 0, f"Expected reward to be 0 for max reward, got {reward}"
            print("Parallel simulations found a complete solution with reward 0")

        # Backpropagation
        node.backpropagate(reward)
        
        return reward, action_sequence, achieved_max_reward, node
    
    def search(self) -> tuple[Action, list[np.ndarray], bool]:
        """Run the MCTS algorithm and return the best action and sequence

        Returns:
            - best_action: The best action to take next
            - best_action_sequence: The full action sequence (from root → terminal) found during search
            - found_max: Whether a max-reward (complete) solution was found
        """
        # Determine number of processes to use
        num_processes = min(multiprocessing.cpu_count(), self.config.num_simulations)

        # Run simulations in parallel
        with Pool(processes=num_processes) as pool:
            nodes = [self.select_node() for _ in range(self.config.num_simulations)]
            simulation_results = pool.map(self._run_simulation, nodes)

        # Process results
        for reward, rollout_sequence, achieved_max_reward, node in simulation_results:
            if achieved_max_reward:
                # We expect reward == 0 for a complete solution
                assert reward == 0, f"Expected reward to be 0 for max reward, got {reward}"

                # 1) Reconstruct the prefix (actions from root → this node)
                prefix_actions: list[np.ndarray] = []
                n = node
                while n.parent is not None:
                    # Each node.action_taken.action_logits is the array that transitioned from parent→n
                    prefix_actions.append(n.action_taken.action_logits)
                    n = n.parent
                prefix_actions.reverse()  # now in order from root to the parent of 'node'

                # 2) Concatenate the prefix + rollout to form the full sequence
                full_sequence = prefix_actions + rollout_sequence

                # 3) Record the best sequence and return
                self.best_action_sequence = full_sequence
                self.best_reward = reward

                # Identify which immediate child of root corresponds to full_sequence[0]
                for child in self.root.children:
                    if (
                        child.action_taken
                        and np.array_equal(child.action_taken.action_logits, full_sequence[0])
                    ):
                        return child.action_taken, full_sequence, True

            # If this rollout's reward is better than our current best, update best_action_sequence
            if reward > self.best_reward:
                self.best_reward = reward

                # Reconstruct prefix for this node (same as above)
                prefix_actions: list[np.ndarray] = []
                n = node
                while n.parent is not None:
                    prefix_actions.append(n.action_taken.action_logits)
                    n = n.parent
                prefix_actions.reverse()

                self.best_action_sequence = prefix_actions + rollout_sequence

        # If no children were ever expanded, pick a random action
        if not self.root.children:
            action_logits = self.env.action_space.sample()
            return Action(action_logits=action_logits), self.best_action_sequence, False

        # Otherwise, choose the child with the highest visit count
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.action_taken, self.best_action_sequence, False
   
    def select_node(self) -> Node:
        """Select a node to expand using UCT"""
        node = self.root
        depth = 0
        
        # Traverse the tree until we find a node that is not fully expanded
        # or until we reach a terminal state or maximum depth
        while not node.is_terminal and node.is_fully_expanded:
            node = node.best_child(self.config.exploration_weight)
            depth += 1
        
        # If the node is not fully expanded, expand it
        if not node.is_terminal and not node.is_fully_expanded:
            return node.expand()
        
        return node

def render_action_sequence(env: WFCWrapper, action_sequence: list[np.ndarray], tile_images) -> None:
    env = deepcopy(env)  # Ensure we don't modify the original environment
    pygame.init()
    SCREEN_WIDTH = env.map_width * 32
    SCREEN_HEIGHT = env.map_length * 32
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Best Map")

    # Create a surface for saving the final map
    final_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    env.reset()
    total_reward = 0
    print("Rendering action sequence...")

    for action in tqdm(action_sequence, desc="Rendering Steps"):
        _, reward, terminate, truncate, _ = env.step(action)
        total_reward += reward

        # Clear screen
        screen.fill((0, 0, 0))
        final_surface.fill((0, 0, 0))  # Also clear the final surface

        # Render the current state to both surfaces
        for y in range(env.map_length):
            for x in range(env.map_width):
                cell_set = env.grid[y][x]
                if len(cell_set) == 1:  # Collapsed cell
                    tile_name = next(iter(cell_set))
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
                elif len(cell_set) == 0:  # Contradiction
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

                                                                                                                                                                                                                 
                                                                                                                                                                                                                                     
# Function to run MCTS until we have a complete solution
def run_mcts_until_complete(env: WFCWrapper, mcts: MCTS, max_iterations:int=1000):
    """
    Run MCTS search repeatedly until we have a complete solution or reach max iterations
    
    Args:
        env: The WFC environment
        mcts: The MCTS instance
        max_iterations: Maximum number of MCTS search iterations
        
    Returns:
        best_action_sequence: The best action sequence found
        total_reward: The total reward of the best sequence
    """
    best_action_sequence = []
    total_reward = 0
    
    # Clone the environment to avoid modifying the original
    test_env = deepcopy(env)
    assert test_env.deterministic, "Expected deterministic environment for MCTS search"
    
    for i in tqdm(range(max_iterations), desc="MCTS Iterations"):
        # Run the search
        _, action_sequence, found_max = mcts.search()
        
        # If we found a solution with early stopping, return it
        if found_max:
            assert len(action_sequence) > 0, "Action sequence should not be empty"
            # Test the action sequence
            info = {}   
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
                import ipdb
                ipdb.set_trace()
            
            # Debugging:
            print(f"Testing best action sequence (found_max={found_max})")
            print(f"  Total reward: {current_reward}")
            print(f"  Info: {info}")
            # Validate reward consistency
            if terminated and info.get("achieved_max_reward", False):
                if current_reward != 0:
                    print(f"WARNING: Max reward achieved but total reward is {current_reward} (expected 0)")
                    print("This indicates a potential bug in the reward calculation or environment logic")
                else:
                    print(f"Found complete solution with reward {current_reward}")
                return action_sequence, current_reward
            else:
                print(f"WARNING: Expected max reward but not achieved. Final state: terminated={terminated}, achieved_max={info.get('achieved_max_reward', False)}")
        
        # If we didn't find a complete solution, reset and try again
        mcts = MCTS(env)
    
    print(f"Failed to find complete solution after {max_iterations} iterations")
    return best_action_sequence, total_reward

# Define environment parameters
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
    reward=partial(binary_reward, target_path_length=20, hard=True),
    deterministic=True,
    # qd_function=binary_percent_water if args.qd else None,
)
                                                                                                                                                                                                            
env.reset()                                                                                                                                                                                                                          
                                                                                                                                                                                                                                     
# Create MCTS instance                                                                                                                                                                                                               
mcts = MCTS(env)    

env.reset() 

# Run MCTS until we have a complete solution
best_action_sequence, total_reward = run_mcts_until_complete(env, mcts)

# Load tile images for visualization
from assets.biome_adjacency_rules import load_tile_images
tile_images = load_tile_images()

# Render the best action sequence if we found one
if best_action_sequence:
    print(f"Rendering best solution with reward {total_reward}")
    render_action_sequence(env, best_action_sequence, tile_images)
else:
    print("No complete solution found")


