import random
from typing import Any, Callable

import numpy as np
import gymnasium as gym  # Use Gymnasium
import pygame
from gymnasium import spaces

# Import functions from biome_wfc instead of fast_wfc
from wfc import (  # We might not need render_wfc_grid if we keep console rendering
    biome_wfc_step,
    find_lowest_entropy_cell,
    initialize_wfc_grid,
)


class CombinedReward:
    """
    A picklable callable that sums multiple reward functions.
    """

    def __init__(self, funcs):
        # funcs is a list of callables, e.g. partials or top‐level functions
        self.funcs = funcs

    def __call__(self, *args, **kwargs):
        total_reward = 0.0
        merged_info = {}
        for fn in self.funcs:
            out = fn(*args, **kwargs)
            # Expect either a float or a tuple (reward, info)
            if isinstance(out, tuple):
                r, info = out
                total_reward += r
                # merge infos (later functions overwrite earlier keys)
                merged_info.update(info)
            else:
                # fall back: single‐value reward
                total_reward += out
        return total_reward, merged_info


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
        reward: Callable[[list[list[set[str]]]], tuple[float, dict[str, Any]]],
        deterministic: bool = True,
        qd_function: Callable[[list[list[set[str]]]], float] | None = None,
        tile_images: dict[str, pygame.Surface] | None = None,
        tile_size: int = 32,
        render_mode: str | None = None,
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
        self.reward = reward
        self.qd_function = qd_function
        self.tile_size = tile_size
        self.tile_images = tile_images
        self.render_mode = render_mode

        # Initialize pygame if we have tile images and human rendering
        if self.tile_images is not None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.map_width * self.tile_size, self.map_length * self.tile_size)
            )
            pygame.display.set_caption("WFC Environment")

        self.grid = initialize_wfc_grid(self.map_width, self.map_length, self.all_tiles)

        # Initial grid state using the function from biome_wfc
        # self.grid will hold the current state (list of lists of sets)
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
        if terminated:
            reward, info = self.reward(self.grid)
            if self.qd_function is not None:
                qd_score = self.qd_function(self.grid)
                info["qd_score"] = qd_score
        elif truncated:
            reward = -1000
        else:
            reward = 0

        # if reward != 0:
        #     print(reward)
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
                                    (x * self.tile_size, y * self.tile_size),
                                )
                        elif num_options == 0:
                            # Draw contradiction (red)
                            pygame.draw.rect(
                                self.screen,
                                (255, 0, 0),
                                (
                                    x * self.tile_size,
                                    y * self.tile_size,
                                    self.tile_size,
                                    self.tile_size,
                                ),
                            )
                        else:
                            # Draw superposition (gray with number of options)
                            shade = min(
                                255, 50 + 205 * (1 - len(cell_set) / self.num_tiles)
                            )
                            pygame.draw.rect(
                                self.screen,
                                (shade, shade, shade),
                                (
                                    x * self.tile_size,
                                    y * self.tile_size,
                                    self.tile_size,
                                    self.tile_size,
                                ),
                            )
                            # Display number of remaining options
                            text = font.render(str(num_options), True, (255, 255, 255))
                            self.screen.blit(
                                text, (x * self.tile_size + 5, y * self.tile_size + 5)
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
        if hasattr(self, "screen"):
            pygame.quit()
        if hasattr(self, "screen"):
            pygame.quit()
            pygame.quit()
