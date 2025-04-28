import random
from enum import Enum, auto
from typing import Any

import numpy as np
import gymnasium as gym  # Use Gymnasium
import pygame
from gymnasium import spaces
import os
from typing import Dict, Optional

# Import functions from biome_wfc instead of fast_wfc
from wfc import (  # We might not need render_wfc_grid if we keep console rendering
    biome_wfc_step, 
    find_lowest_entropy_cell, 
    initialize_wfc_grid
)
from tasks.binary_task import calc_longest_path, calc_num_regions, grid_to_binary_map
from biome_adjacency_rules import load_tile_images, create_adjacency_matrix


class Task(Enum):
    # TODO: replace place holder biomes with real biome specifications
    WATER = auto()
    BIOME2 = auto()
    BINARY = auto()


def grid_to_array(
    grid: list[list[set[str]]],
    tile_symbols: list[str],
    tile_to_index: dict[str, int],
    map_length: int,
    map_width: int,
) -> np.ndarray:
    """Converts the list-of-sets grid to a flattened numpy array for the observation."""
    arr = np.empty((map_length, map_width), dtype=np.float32)
    num_tiles = len(tile_symbols)
    for y in range(map_length):
        for x in range(map_width):
            cell_set = grid[y][x]
            num_options = len(cell_set)
            if num_options == 1:
                # Collapsed cell
                tile_name = next(iter(cell_set))
                idx = tile_to_index.get(tile_name, -1)  # Get index from map
                if idx != -1 and num_tiles > 1:
                    arr[y, x] = idx / (num_tiles - 1)
                elif idx != -1 and num_tiles == 1:
                    arr[y, x] = 0.0  # Handle single tile case
                else:
                    arr[y, x] = -1.0  # Should not happen if tile_to_index is correct
            elif num_options == 0:
                # Contradiction cell
                arr[y, x] = -1.0  # Use a different value for contradiction? Or stick to -1? Let's use -1.
                arr[y, x] = -1.0  # Use a different value for contradiction? Or stick to -1? Let's use -1.
            else:
                # Undecided cell
                arr[y, x] = -1.0
    return arr.flatten()

def get_dominant_biome(grid: list[list[set[str]]]) -> str:
    """Enhanced biome detection with specific thresholds for ponds and rivers."""
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr", 
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }
    
    # Count water cells and shore patterns
    water_cells = 0
    shore_cells = 0
    for row in grid:
        for cell in row:
            if len(cell) == 1:
                tile = next(iter(cell)).lower()
                if tile in water_tiles:
                    water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1
    
    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return "unknown"
    
    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0
    
    # River detection - requires continuous flow and appropriate water ratio
    has_flow = check_continuous_flow(grid, water_tiles, 'horizontal') or \
               check_continuous_flow(grid, water_tiles, 'vertical')
    
    if has_flow and 0.2 <= water_ratio <= 0.4:
        return "river"
    elif water_ratio >= 0.45 and shore_ratio <= 0.2:
        return "pond"
    return "unknown"

def check_continuous_flow(grid: list[list[set[str]]], water_tiles: set[str], direction: str) -> bool:
    """Check if there's a continuous water path across the map in specified direction."""
    if direction == 'horizontal':
        # Check from left to right
        for y in range(len(grid)):
            if has_water_path(grid, (0, y), (len(grid[0])-1, y), water_tiles):
                return True
    else:
        # Check from top to bottom
        for x in range(len(grid[0])):
            if has_water_path(grid, (x, 0), (x, len(grid)-1), water_tiles):
                return True
    return False

def has_water_path(grid: list[list[set[str]]], start: tuple, end: tuple, water_tiles: set[str]) -> bool:
    """Check if there's a continuous water path between two points."""
    from collections import deque
    
    # Convert grid to binary water map
    water_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1 and next(iter(grid[y][x])).lower() in water_tiles:
                water_map[y, x] = True
    
    if not water_map[start[1], start[0]] or not water_map[end[1], end[0]]:
        return False
    
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-way connectivity
    
    while queue:
        current = queue.popleft()
        if current == end:
            return True
            
        for dx, dy in directions:
            x, y = current[0] + dx, current[1] + dy
            if (0 <= x < len(grid[0]) and 0 <= y < len(grid) and 
                water_map[y, x] and (x, y) not in visited):
                visited.add((x, y))
                queue.append((x, y))
    
    return False

# wfc_next_collapse_position is replaced by find_lowest_entropy_cell from biome_wfc
def compute_reward(
    grid: list[list[set[str]]],
    task: Task,
    tile_symbols: list[str],
    tile_to_index: dict[str, int],
    terminated: bool,
    truncated: bool
) -> float:
    """Computes the reward based task

    Binary Task: reward is based on regions connectivity and how far we are from the target path length.
    Regions: +100 reward if there's a single connected empty region; else -100.
    Path: Scales linearly up to +100 when the longest path increases by at least 20 tiles.
    """
    if truncated:
        return -1000.0 
    if not terminated:
        return 0.0

    match task:
        case Task.BINARY:
            TARGET_PATH_LENGTH = 50
            binary_map = grid_to_binary_map(grid)
            regions = calc_num_regions(binary_map)
            current_path = calc_longest_path(binary_map)

            region_reward = 100.0 if regions == 1 else -100.0

            if current_path >= TARGET_PATH_LENGTH:
                path_reward = 100.0
            else:
                path_reward = current_path - TARGET_PATH_LENGTH
            return region_reward + path_reward

        case Task.WATER | Task.BIOME2:
            water_tiles = {
                "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
                "water_bl", "water_b", "water_br", "shore_tl", "shore_tr", 
                "shore_bl", "shore_br", "shore_lr", "shore_rl"
            }
            
            water_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
            water_cells = 0
            shore_cells = 0
            pure_water_cells = 0
            
            for y in range(len(grid)):
                for x in range(len(grid[0])):
                    if len(grid[y][x]) == 1:
                        tile = next(iter(grid[y][x])).lower()
                        if tile in water_tiles:
                            water_map[y, x] = True
                            water_cells += 1
                            if tile == "water":
                                pure_water_cells += 1
                            if "shore" in tile:
                                shore_cells += 1
            
            total_cells = len(grid) * len(grid[0])
            if total_cells == 0:
                return 0.0
            
            water_ratio = water_cells / total_cells
            pure_water_ratio = pure_water_cells / total_cells
            shore_ratio = shore_cells / water_cells if water_cells > 0 else 0
            
            # Check biome type
            biome = get_dominant_biome(grid)
            
            if biome == "river":
                # River scoring
                has_flow = check_continuous_flow(grid, water_tiles, 'horizontal') or \
                          check_continuous_flow(grid, water_tiles, 'vertical')
                flow_score = 1.0 if has_flow else 0.0
                coverage_score = max(0.0, 1.0 - abs(water_ratio - 0.3) * 3.33)
                regions = calc_num_regions(water_map.astype(np.int8))
                connected_score = 1.0 / (regions ** 0.5)
                combined = (0.4 * flow_score + 0.3 * coverage_score + 0.3 * connected_score)
                return float(combined * 100 * 1.5)  # Bonus for rivers
            
            elif biome == "pond":
                # Pond scoring - prioritize high water concentration
                coverage_score = max(0.0, 1.0 - abs(pure_water_ratio - 0.5) * 2.0)  # Target 50% pure water
                shore_penalty = max(0.0, 1.0 - shore_ratio * 5.0)  # Penalize shore tiles
                combined = (0.7 * coverage_score + 0.3 * shore_penalty)
                return float(combined * 100 * 1.2)  # Smaller bonus for ponds
            
            # Default scoring for other cases
            coverage_score = max(0.0, 1.0 - abs(water_ratio - 0.35) * 2.86)
            return float(coverage_score * 100)
        case _:
            # TODO: incorporate biome rewards
            return 0
        
def find_edge_water_cells(grid: list[list[set[str]]], water_tiles: set[str], edge: str) -> list[tuple]:
    """Find water cells along a specific edge of the grid."""
    edge_cells = []
    if edge == 'left':
        for y in range(len(grid)):
            if len(grid[y][0]) == 1 and next(iter(grid[y][0])).lower() in water_tiles:
                edge_cells.append((0, y))
    elif edge == 'right':
        for y in range(len(grid)):
            if len(grid[y][-1]) == 1 and next(iter(grid[y][-1])).lower() in water_tiles:
                edge_cells.append((len(grid[0])-1, y))
    elif edge == 'top':
        for x in range(len(grid[0])):
            if len(grid[0][x]) == 1 and next(iter(grid[0][x])).lower() in water_tiles:
                edge_cells.append((x, 0))
    elif edge == 'bottom':
        for x in range(len(grid[0])):
            if len(grid[-1][x]) == 1 and next(iter(grid[-1][x])).lower() in water_tiles:
                edge_cells.append((x, len(grid)-1))
    return edge_cells

class WFCWrapper(gym.Env):
    """
    Gymnasium Environment for Wave Function Collapse controlled by an RL agent.
    Gymnasium Environment for Wave Function Collapse with graphical rendering.
    Gymnasium Environment for Wave Function Collapse with graphical rendering.
    Observation: Flattened grid state + normalized coordinates of the next cell to collapse.
                 Grid cells: Value is index/(num_tiles-1) if collapsed, -1.0 if undecided.
    Action: A vector of preferences (logits) for each tile type for the selected cell.
    Reward: Sparse reward given only at the end of the episode.
            + Scaled reward (0 to 100) based on proximity to target tile count for successful termination.
            - 1000 for truncation (contradiction or max steps).
            0 otherwise.
    Termination: Grid is fully collapsed (all cells have exactly one possibility).
    Truncation: A contradiction occurs during propagation OR max steps reached.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}  # Add metadata, adjust FPS

    def __init__(
        self,
        map_length: int,
        map_width: int,
        tile_symbols: list[str],
        adjacency_bool: np.ndarray,
        num_tiles: int,
        tile_to_index: dict[str, int],
        task: Task,
        deterministic: bool,
        tile_images: Optional[Dict[str, pygame.Surface]] = None,
        tile_size: int = 32,
        render_mode: Optional[str] = None,
        task_specifications: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()  # Call parent constructor
        self.all_tiles = tile_symbols
        self.adjacency = adjacency_bool
        self.num_tiles = num_tiles
        self.map_length: int = map_length
        self.map_width: int = map_width
        self.tile_to_index = tile_to_index  # Store tile_to_index
        self.tile_images = tile_images
        self.tile_size = tile_size
        self.render_mode = render_mode
        self.dominant_biome = "unknown"  # Track dominant biome
        self.deterministic = deterministic

        # Initialize pygame if we have tile images and human rendering
        if self.tile_images is not None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.map_width * self.tile_size, self.map_length * self.tile_size))
            pygame.display.set_caption("WFC Environment")

        self.grid = initialize_wfc_grid(self.map_width, self.map_length, self.all_tiles)
        
        # Keep a way to reset easily if needed, maybe store initial args?
        # Or just call initialize_wfc_grid again in reset.
        # Action space: Agent outputs preferences (logits) for each tile type.
        # Needs to be float32 for SB3.
        self.action_space: spaces.Box = spaces.Box(
            low=0, high=1, shape=(self.num_tiles,), dtype=np.float32
        )

        # Observation space: Flattened map + normalized coordinates of the next cell to collapse
        # Map values range from -1 (undecided) to 1 (max index / max index).
        # Coordinates range from 0 to 1. Needs to be float32 for SB3.
        self.observation_space: spaces.Box = spaces.Box(
            low=-1.0,   # Lower bound changed due to -1 for undecided cells
            high=1.0,
            shape=(self.map_length * self.map_width + 2,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.max_steps = self.map_length * self.map_width + 10
        self.task = task

    def get_observation(self) -> np.ndarray:
        """Constructs the observation array (needs to be float32)."""
        # Convert the list-of-sets grid to the flat numpy array format
        map_flat = grid_to_array(
            self.grid,
            self.all_tiles,
            self.tile_to_index,
            self.map_length,
            self.map_width,
        )
        # Find the next cell to collapse using the function from biome_wfc
        pos_tuple = find_lowest_entropy_cell(
            self.grid, deterministic=self.deterministic
        )  # Returns (x, y) or None

        # Handle case where grid is fully collapsed (pos_tuple is None)
        if pos_tuple is None:
            # If fully collapsed or contradiction, position is irrelevant for next step
            pos_array = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # Normalize the collapse position (x, y) to be between 0 and 1
            x, y = pos_tuple
            norm_x = x / (self.map_width - 1) if self.map_width > 1 else 0.0
            norm_y = y / (self.map_length - 1) if self.map_length > 1 else 0.0
            pos_array = np.array([norm_x, norm_y], dtype=np.float32)

        # Ensure final observation is float32
        return np.concatenate([map_flat, pos_array]).astype(np.float32)

    def save_completed_map(self, reward_val: float):
        """Saves the completed map as an image with reward and biome in the filename."""
        if not os.path.exists("wfc_reward_img"):
            os.makedirs("wfc_reward_img")
        
        # Get dominant biome
        self.dominant_biome = get_dominant_biome(self.grid)
        
        filename = f"wfc_reward_img/{self.dominant_biome}_reward_{reward_val:.1f}%.png"
        
        # Create a surface to render the map
        surface = pygame.Surface(
            (self.map_width * self.tile_size, self.map_length * self.tile_size))
        surface.fill((0, 0, 0))
        
        for y in range(self.map_length):
            for x in range(self.map_width):
                cell_set = self.grid[y][x]
                if len(cell_set) == 1:
                    tile_name = next(iter(cell_set))
                    if tile_name in self.tile_images:
                        surface.blit(
                            self.tile_images[tile_name],
                            (x * self.tile_size, y * self.tile_size)
                        )
        
        pygame.image.save(surface, filename)
        print(f"Saved completed map to {filename}")

    def step(self, action: np.ndarray):
        """Performs one step of the WFC process based on the agent's action."""
        info = {}
        self.current_step += 1

        # Ensure action is float32 numpy array
        action = np.asarray(action, dtype=np.float32)

        # Convert action (potentially logits) to a probability distribution using softmax
        # Improve numerical stability by subtracting the max before exponentiating
        action_exp = np.exp(action - np.max(action))
        action_probs = action_exp / (np.sum(action_exp) + 1e-8) # Add epsilon for stability
        action_probs = action_exp / (np.sum(action_exp) + 1e-8) # Add epsilon for stability

        # action_probs are already float32, biome_wfc_step expects list or numpy array
        # No need to convert to float64 unless biome_wfc specifically requires it (it doesn't seem to)

        # Call the biome_wfc_step function
        # It modifies the grid in-place and returns terminated/truncated status
        # Note: biome_wfc_step expects action_probs, not logits
        self.grid, terminated, truncated = biome_wfc_step(
            self.grid,  # The list-of-sets grid
            self.adjacency,  # Adjacency rules (numpy bool array)
            self.all_tiles,  # List of tile symbols
            self.tile_to_index,  # Tile symbol to index map
            action_probs,  # Action probabilities from agent
            deterministic=self.deterministic,
        )

        # Check for truncation due to reaching max steps
        if not terminated and not truncated and self.current_step >= self.max_steps:
            # print(f"Max steps reached ({self.current_step}), truncating.") # Debug print
            truncated = True
            terminated = False  # Cannot be both terminated and truncated

        # Calculate reward using the updated grid and initial longest path
        reward_val = (
            compute_reward(
                self.grid,
                self.task,
                self.all_tiles,
                self.tile_to_index,
                terminated,
                truncated
            )
            if terminated
            else (-1000 if truncated else 0)
        )

        # Get the next observation
        observation = self.get_observation()
        info = {
            "steps": self.current_step,
            "terminated_reason": "completed" if terminated else None,
            "truncated_reason": (
                "contradiction" if self.current_step < self.max_steps else "max_steps"
            ) if truncated else None,
        }

        if terminated and self.tile_images is not None:
            self.save_completed_map(reward_val)

        if self.render_mode == "human":
            self.render()
        return observation, reward_val, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)  # Handle seeding correctly via Gymnasium Env
        random.seed(seed)
        np.random.seed(seed)
        # Re-initialize the grid using the function from biome_wfc
        self.grid = initialize_wfc_grid(self.map_width, self.map_length, self.all_tiles)
        self.current_step = 0
        # Compute and store initial longest path
        observation = self.get_observation()
        
        if self.render_mode == "human" and self.tile_images is not None:
            self.render()
            
        return observation, {}

    def render(self):
        """Renders the current grid state to the console."""
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            if self.tile_images is not None:
                # Graphical rendering with tile images
                self.screen.fill((0, 0, 0))  # Clear screen
                font = pygame.font.SysFont(None, 20)
                
                for y in range(self.map_length):
                    for x in range(self.map_width):
                        cell_set = self.grid[y][x]
                        num_options = len(cell_set)
                        
                        if num_options == 1:
                            # Draw the collapsed tile
                            tile_name = next(iter(cell_set))
                            if tile_name in self.tile_images:
                                self.screen.blit(
                                    self.tile_images[tile_name],
                                    (x * self.tile_size, y * self.tile_size)
                                )
                        elif num_options == 0:
                            # Draw contradiction (red)
                            pygame.draw.rect(
                                self.screen,
                                (255, 0, 0),
                                (x * self.tile_size, y * self.tile_size, 
                                 self.tile_size, self.tile_size)
                            )
                        else:
                            # Draw superposition (gray with number of options)
                            shade = min(255, 50 + 205 * (1 - len(cell_set)/self.num_tiles))
                            pygame.draw.rect(
                                self.screen,
                                (shade, shade, shade),
                                (x * self.tile_size, y * self.tile_size, 
                                 self.tile_size, self.tile_size)
                            )
                            # Display number of remaining options
                            text = font.render(str(num_options), True, (255, 255, 255))
                            self.screen.blit(
                                text,
                                (x * self.tile_size + 5, y * self.tile_size + 5)
                            )
                
                pygame.display.flip()
            else:
                # Fallback to console rendering
                print(f"--- Step: {self.current_step} ---")
                for y in range(self.map_length):
                    row_str = ""
                    for x in range(self.map_width):
                        cell_set = self.grid[y][x]
                        num_options = len(cell_set)
                        if num_options == 1:
                            tile_name = next(iter(cell_set))
                            row_str += tile_name + " "
                        elif num_options == self.num_tiles:
                            row_str += "? "
                        elif num_options == 0:
                            row_str += "! "
                        else:
                            row_str += f"{len(cell_set)} "
                    print(row_str.strip())
                print("-" * (self.map_width * 2))
        else:
            pass

    def close(self):
        """Cleans up any resources used by the environment."""
        if hasattr(self, 'screen'):
            pygame.quit()
        if hasattr(self, 'screen'):
            pygame.quit()

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 480
    TILE_SIZE = 32

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Evolving WFC")
    import os

    # Create output directory if it doesn't exist
    os.makedirs("wfc_reward_img", exist_ok=True)

    # Define environment parameters
    MAP_LENGTH = 15
    MAP_WIDTH = 20

    from biome_adjacency_rules import create_adjacency_matrix

    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    tile_images = load_tile_images()  # Load tile images
    num_tiles = len(tile_symbols)

    # Create the WFC environment instance with graphical rendering
    env = WFCWrapper(
        map_length=MAP_LENGTH,
        map_width=MAP_WIDTH,
        tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool,
        num_tiles=num_tiles,
        tile_to_index=tile_to_index,
        tile_images=tile_images,
        render_mode="human",
        # task=Task.BINARY,
        task=Task.WATER,
        deterministic=False,
    )

    # Reset the environment
    obs, info = env.reset()
    running = True

    while running:
        # Sample a random action
        action = env.action_space.sample()
        obs, reward_val, terminated, truncated, info = env.step(action)
        env.render()

        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if terminated or truncated:
            print(f"WFC ({'completed' if terminated else 'failed'}) with reward: {reward_val:.1f}")
            obs, info = env.reset()

    env.close()
