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


def fake_reward(
    grid: list[list[set[str]]],  # Grid is now list of lists of sets
    tile_symbols: list[str],  # Use symbols list
    tile_to_index: dict[str, int],  # Use the mapping
    terminated: bool,
    truncated: bool,
) -> float:
    final_reward = 0
    return final_reward


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

        # Calculate reward using the updated grid (list of sets)
        reward = fake_reward(
            self.grid, self.all_tiles, self.tile_to_index, terminated, truncated
        )

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
    MAP_LENGTH = 12
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
        pygame.time.delay(200)  # Delay for visualization (in milliseconds)

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
