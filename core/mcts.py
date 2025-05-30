import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from copy import deepcopy
from pydantic import BaseModel, Field

from .wfc_env import WFCWrapper

class Action(BaseModel):
    """Represents an action in the MCTS tree with its statistics"""
    action_logits: List[float] = Field(default_factory=list, description="Logits for the action taken")
    visits: int = Field(default=0, description="Number of times this action has been taken")
    total_reward: float = Field(default=0.0, description="Total reward accumulated from this action")
    tile_index: int = Field(default=-1, description="Index of the tile this action represents")

class MCTSConfig(BaseModel):
    """Configuration for the MCTS algorithm"""
    exploration_weight: float = Field(default=1.0, description="Exploration weight for UCT calculation")
    num_simulations: int = Field(default=100, description="Number of simulations to run")
    max_depth: int = Field(default=50, description="Maximum depth of the search tree")

class Node:
    """A node in the MCTS tree"""
    def __init__(self, env: WFCWrapper, parent=None, action_taken=None):
        self.env = deepcopy(env)
        self.parent = parent
        self.action_taken = action_taken  # Action that led to this node
        self.children: List[Node] = []
        self.available_actions: Dict[int, Action] = {}  # Maps tile index to Action
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
        # Get the current state observation
        obs = self.env.get_observation()
        
        # Find the lowest entropy cell position
        pos_tuple = self.env.find_lowest_entropy_cell(self.env.grid, deterministic=True)
        if pos_tuple is None:
            self.is_terminal = True
            return self
        
        # Get possible tiles for this cell
        x, y = pos_tuple
        possible_tiles = list(self.env.grid[y][x])
        
        # Create actions for each possible tile if not already created
        if not self.available_actions:
            for tile in possible_tiles:
                tile_idx = self.env.tile_to_index.get(tile)
                if tile_idx is not None and tile_idx not in self.available_actions:
                    # Create action with one-hot encoding (only one 1.0, rest 0.0)
                    action_logits = [0.0] * self.env.num_tiles
                    action_logits[tile_idx] = 1.0
                    
                    self.available_actions[tile_idx] = Action(
                        action_logits=action_logits,
                        visits=0,
                        total_reward=0.0,
                        tile_index=tile_idx
                    )
        
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
    
    def simulate(self) -> float:
        """Run a simulation from this node to a terminal state"""
        if self.is_terminal:
            # Get the final reward from the environment
            return self.env.reward(self.env.grid)[0]
        
        # Make a copy of the environment to avoid modifying the original
        sim_env = deepcopy(self.env)
        terminated = False
        truncated = False
        total_reward = 0.0
        
        # Run random actions until terminal state
        while not (terminated or truncated):
            # Sample a random action
            action = sim_env.action_space.sample()
            
            # Take a step in the environment
            _, reward, terminated, truncated, _ = sim_env.step(action)
            total_reward += reward
        
        return total_reward
    
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
    
    def search(self) -> Action:
        """Run the MCTS algorithm and return the best action"""
        for _ in range(self.config.num_simulations):
            # Selection and expansion
            node = self.select_node()
            
            # Simulation
            reward = node.simulate()
            
            # Backpropagation
            node.backpropagate(reward)
        
        # Return the action with the most visits
        if not self.root.children:
            # If no children, return a random action
            action_logits = self.env.action_space.sample()
            return Action(action_logits=action_logits.tolist())
        
        best_child = max(self.root.children, key=lambda child: child.visits)
        return best_child.action_taken
    
    def select_node(self) -> Node:
        """Select a node to expand using UCT"""
        node = self.root
        depth = 0
        
        # Traverse the tree until we find a node that is not fully expanded
        # or until we reach a terminal state or maximum depth
        while not node.is_terminal and node.is_fully_expanded and depth < self.config.max_depth:
            node = node.best_child(self.config.exploration_weight)
            depth += 1
        
        # If the node is not fully expanded, expand it
        if not node.is_terminal and not node.is_fully_expanded:
            return node.expand()
        
        return node
