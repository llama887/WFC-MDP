from collections import deque

import gymnasium as gym  # Use Gymnasium
import numpy as np
from gymnasium import spaces

# Import functions from biome_wfc instead of fast_wfc
from biome_wfc import (  # We might not need render_wfc_grid if we keep console rendering
    biome_wfc_step,
    find_lowest_entropy_cell,
    initialize_wfc_grid,
    load_tile_images,
    render_wfc_grid,
)


def grid_to_binary_map(grid: list[list[set[str]]]) -> np.ndarray:
    """Converts the WFC grid into a binary map.
    Empty cells (0) are those whose single tile name starts with 'sand' or 'path',
    solid cells (1) are everything else.
    """
    height = len(grid)
    width = len(grid[0])
    binary_map = np.ones((height, width), dtype=np.int32)  # default solid (1)
    for y in range(height):
        for x in range(width):
            cell = grid[y][x]
            if len(cell) == 1:
                tile_name = next(iter(cell))
                if tile_name.startswith("sand") or tile_name.startswith("path"):
                    binary_map[y, x] = 0  # empty
                else:
                    binary_map[y, x] = 1  # solid
            else:
                binary_map[y, x] = 1
    return binary_map


def calc_num_regions(binary_map: np.ndarray) -> int:
    """Counts connected regions of empty cells (value 0) using flood-fill."""
    h, w = binary_map.shape
    visited = np.zeros((h, w), dtype=bool)
    num_regions = 0

    def neighbors(y, x):
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx

    for y in range(h):
        for x in range(w):
            if binary_map[y, x] == 0 and not visited[y, x]:
                num_regions += 1
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if visited[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    for ny, nx in neighbors(cy, cx):
                        if binary_map[ny, nx] == 0 and not visited[ny, nx]:
                            stack.append((ny, nx))
    return num_regions


def calc_longest_path(binary_map: np.ndarray) -> int:
    """Computes the longest shortest path among all empty cells (value 0) using BFS."""
    h, w = binary_map.shape

    def bfs(start_y, start_x):
        visited = -np.ones((h, w), dtype=int)
        q = deque()
        visited[start_y, start_x] = 0
        q.append((start_y, start_x))
        max_dist = 0
        while q:
            y, x = q.popleft()
            d = visited[y, x]
            max_dist = max(max_dist, d)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w:
                    if binary_map[ny, nx] == 0 and visited[ny, nx] == -1:
                        visited[ny, nx] = d + 1
                        q.append((ny, nx))
        return max_dist

    overall_max = 0
    for y in range(h):
        for x in range(w):
            if binary_map[y, x] == 0:
                overall_max = max(overall_max, bfs(y, x))
    return overall_max


def grid_to_array(
    grid: list[list[set[str]]],  # Grid is now list of lists of sets
    tile_symbols: list[str],  # Renamed for consistency
    tile_to_index: dict[str, int],  # Need mapping
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
                arr[
                    y, x
                ] = -2.0  # Use a different value for contradiction? Or stick to -1? Let's use -1.
                arr[y, x] = -1.0
            else:
                # Undecided cell
                arr[y, x] = -1.0
    return arr.flatten()


# wfc_next_collapse_position is replaced by find_lowest_entropy_cell from biome_wfc


def compute_reward(grid: list[list[set[str]]], target_path_length: int) -> float:
    """Computes the reward based on regions connectivity and how far we are from the target path length.
    Regions: +100 reward if there's a single connected empty region; else -100.
    Path: Scales linearly up to +100 when the longest path increases by at least 20 tiles.
    """
    binary_map = grid_to_binary_map(grid)
    regions = calc_num_regions(binary_map)
    current_path = calc_longest_path(binary_map)

    region_reward = 100.0 if regions == 1 else -100.0

    if current_path >= target_path_length:
        path_reward = 100.0
    else:
        path_reward = 100.0 / (abs(target_path_length - current_path) + 1)
    return region_reward + path_reward


# # Target subarray to count
# target = np.array([0, 1, 0])


# # Count occurrences of the target subarray along the last dimension
# matches = np.all(array == target, axis=-1)
class WFCWrapper(gym.Env):
    """
    Gymnasium Environment for Wave Function Collapse controlled by an RL agent.

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
        tile_to_index: dict[str, int],  # Add tile_to_index
        target_path_length: int = 50,
    ):
        super().__init__()  # Call parent constructor
        self.all_tiles = tile_symbols
        self.adjacency = adjacency_bool
        self.num_tiles = num_tiles
        self.map_length: int = map_length
        self.map_width: int = map_width
        self.tile_to_index = tile_to_index  # Store tile_to_index

        # Initial grid state using the function from biome_wfc
        # self.grid will hold the current state (list of lists of sets)
        self.grid = initialize_wfc_grid(self.map_width, self.map_length, self.all_tiles)
        # Keep a way to reset easily if needed, maybe store initial args?
        # Or just call initialize_wfc_grid again in reset.

        # Action space: Agent outputs preferences (logits) for each tile type.
        # Needs to be float32 for SB3.
        self.action_space: spaces.Box = spaces.Box(
            low=-1, high=1, shape=(self.num_tiles,), dtype=np.float32
        )

        # Observation space: Flattened map + normalized coordinates of the next cell to collapse
        # Map values range from -1 (undecided) to 1 (max index / max index).
        # Coordinates range from 0 to 1. Needs to be float32 for SB3.
        self.observation_space: spaces.Box = spaces.Box(
            low=-1.0,  # Lower bound changed due to -1 for undecided cells
            high=1.0,
            shape=(self.map_length * self.map_width + 2,),
            dtype=np.float32,
        )
        self.current_step = 0
        # Set a maximum number of steps to prevent infinite loops if termination fails
        self.max_steps = self.map_length * self.map_width + 10  # Allow some buffer
        self.target_path_length = target_path_length

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
        pos_tuple = find_lowest_entropy_cell(self.grid)  # Returns (x, y) or None

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

    def step(self, action: np.ndarray):
        """Performs one step of the WFC process based on the agent's action."""
        self.current_step += 1

        # Ensure action is float32 numpy array
        action = np.asarray(action, dtype=np.float32)

        # Convert action (potentially logits) to a probability distribution using softmax
        # Improve numerical stability by subtracting the max before exponentiating
        action_exp = np.exp(action - np.max(action))
        action_probs = action_exp / (
            np.sum(action_exp) + 1e-8
        )  # Add epsilon for stability

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
            deterministic=False,  # Use stochastic collapse during training
        )

        # Check for truncation due to reaching max steps
        if not terminated and not truncated and self.current_step >= self.max_steps:
            # print(f"Max steps reached ({self.current_step}), truncating.") # Debug print
            truncated = True
            terminated = False  # Cannot be both terminated and truncated

        # Calculate reward using the updated grid and initial longest path
        reward = (
            compute_reward(self.grid, self.target_path_length)
            if terminated
            else 0
            if not truncated
            else -1000
        )

        if reward != 0:
            print(reward)
        # Get the next observation
        observation = self.get_observation()
        info = {}  # Provide additional info if needed (e.g., current step count)
        info["steps"] = self.current_step
        if terminated:
            info["terminated_reason"] = "completed"
        if truncated:
            info["truncated_reason"] = (
                "contradiction" if self.current_step < self.max_steps else "max_steps"
            )

        # Ensure reward is float
        reward = float(reward)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)  # Handle seeding correctly via Gymnasium Env
        # Re-initialize the grid using the function from biome_wfc
        self.grid = initialize_wfc_grid(self.map_width, self.map_length, self.all_tiles)
        self.current_step = 0
        # Compute and store initial longest path
        observation = self.get_observation()
        info = {}  # Can provide initial info if needed
        # print("Environment Reset") # Debug print
        return observation, info

    def render(self, mode="human"):
        """Renders the current grid state to the console."""
        if mode == "human":
            print(f"--- Step: {self.current_step} ---")
            for y in range(self.map_length):
                row_str = ""
                for x in range(self.map_width):
                    cell_set = self.grid[y][x]
                    num_options = len(cell_set)
                    if num_options == 1:
                        # Collapsed cell
                        tile_name = next(iter(cell_set))
                        row_str += tile_name + " "
                    elif num_options == self.num_tiles:
                        # Not touched yet (all possibilities)
                        row_str += "? "
                    elif num_options == 0:
                        # Contradiction
                        row_str += "! "
                    else:
                        # Undecided cell (superposition)
                        row_str += ". "
                print(row_str.strip())
            print("-" * (self.map_width * 2))
        else:
            # Handle other modes or just pass as per gym interface
            # return super().render(mode=mode) # Use this if inheriting from gym.Env directly
            pass  # No other render modes implemented

    def close(self):
        """Cleans up any resources used by the environment."""
        # No specific resources to clean up in this case
        pass


if __name__ == "__main__":
    import numpy as np
    import pygame

    # Use biome_wfc rendering: load tile images (opens a pygame window)
    tile_images = load_tile_images()

    # Define environment parameters (using the same tile set as in our training setup)
    MAP_LENGTH = 15
    MAP_WIDTH = 20

    from biome_adjacency_rules import create_adjacency_matrix

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
    )

    # Reset the environment to its initial state
    obs, info = env.reset()

    running = True
    while running:
        # Sample a random action (agent's output) from the environment's action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Instead of console rendering, call the biome_wfc rendering (using pygame)
        render_wfc_grid(env.grid, tile_images)
        # pygame.time.delay(1)  # Delay for visualization (in milliseconds)

        # Process pygame events for window closure
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if terminated or truncated:
            print(
                "WFC completed successfully."
                if terminated
                else "WFC failed (contradiction)."
            )
            obs, info = env.reset()

    pygame.quit()
    exit(0)
