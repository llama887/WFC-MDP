import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from fast_wfc import fast_wfc_collapse_step


def grid_to_array(
    grid: np.ndarray,
    all_tiles: list[str],
    map_length: int,
    map_width: int,
) -> np.ndarray:
    arr = np.empty((map_length, map_width), dtype=np.float32)
    for y in range(map_length):
        for x in range(map_width):
            possibilities = grid[y, x, :]
            if np.count_nonzero(possibilities) == 1:
                idx = int(np.argmax(possibilities))
                arr[y, x] = idx / (len(all_tiles) - 1)
            else:
                arr[y, x] = 0.5
    return arr.flatten()


def wfc_next_collapse_position(grid: np.ndarray) -> tuple[int, int]:
    min_options = float("inf")
    best_cell = (0, 0)
    map_length, map_width, _ = grid.shape
    for y in range(map_length):
        for x in range(map_width):
            options = np.count_nonzero(grid[y, x, :])
            if options > 1 and options < min_options:
                min_options = options
                best_cell = (x, y)
    return best_cell


def fake_reward(grid: np.ndarray, num_tiles: int) -> float:
    desired_X_count = 20
    x_array = np.zeros(num_tiles)
    x_array[1] = 1

    matches = np.all(grid == x_array, axis=-1)
    count = np.sum(matches)
    return float(-abs(desired_X_count - count))


# # Target subarray to count
# target = np.array([0, 1, 0])

# # Count occurrences of the target subarray along the last dimension
# matches = np.all(array == target, axis=-1)
# count = np.sum(matches)


class WFCWrapper(gym.Env):
    def __init__(
        self,
        map_length: int,
        map_width: int,
        tile_symbols,
        adjacency_bool,
        num_tiles,
    ):
        # Use the fast implementation variables from fast_wfc.py:
        self.all_tiles = tile_symbols  # use the precomputed tile order
        self.adjacency = adjacency_bool  # Numpy boolean array with compatibility info
        self.num_tiles = num_tiles
        self.map_length: int = map_length
        self.map_width: int = map_width
        # Initialize grid as a NumPy boolean array (all possibilities True)
        self.grid = np.ones(
            (self.map_length, self.map_width, self.num_tiles), dtype=bool
        )
        self.action_space: spaces.Box = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_tiles,), dtype=np.float32
        )
        self.observation_space: spaces.Box = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.map_length * self.map_width + 2,),
            dtype=np.float32,
        )

    def get_observation(self) -> np.ndarray:
        map_flat = grid_to_array(
            self.grid, self.all_tiles, self.map_length, self.map_width
        )
        pos = wfc_next_collapse_position(self.grid)
        # Normalize the collapse position: x-coordinate divided by (map_width-1), y by (map_length-1)
        pos_array = np.array(
            [pos[0] / (self.map_width - 1), pos[1] / (self.map_length - 1)],
            dtype=np.float32,
        )
        return np.concatenate([map_flat, pos_array])

    def step(self, action):
        # action is a float vector; convert to np.ndarray for fast_wfc functions.
        action_vector = np.array(action, dtype=np.float64)
        self.grid, terminate, truncate = fast_wfc_collapse_step(
            self.grid,
            self.map_width,
            self.map_length,
            self.num_tiles,
            self.adjacency,
            action_vector,
        )
        reward = fake_reward(self.grid, self.num_tiles) if terminate else 0.0
        reward = -10000 if truncate else reward
        info = {}
        return self.get_observation(), reward, terminate, truncate, info

    def reset(self, seed=0):
        self.grid = np.ones(
            (self.map_length, self.map_width, self.num_tiles), dtype=bool
        )
        return self.get_observation(), {}

    def render(self, mode="human"): ...


if __name__ == "__main__":
    PAC_TILES = {
        " ": {
            "edges": {"U": "OPEN", "R": "OPEN", "D": "OPEN", "L": "OPEN"},
        },
        "X": {
            "edges": {"U": "OPEN", "R": "OPEN", "D": "OPEN", "L": "OPEN"},
        },
        "═": {
            "edges": {"U": "OPEN", "R": "LINE", "D": "OPEN", "L": "LINE"},
        },
        "║": {
            "edges": {"U": "LINE", "R": "OPEN", "D": "LINE", "L": "OPEN"},
        },
        "╔": {
            "edges": {"U": "OPEN", "R": "LINE", "D": "LINE", "L": "OPEN"},
        },
        "╗": {
            "edges": {"U": "OPEN", "R": "OPEN", "D": "LINE", "L": "LINE"},
        },
        "╚": {
            "edges": {"U": "LINE", "R": "LINE", "D": "OPEN", "L": "OPEN"},
        },
        "╝": {
            "edges": {"U": "LINE", "R": "OPEN", "D": "OPEN", "L": "LINE"},
        },
    }

    # Opposite directions used for edge compatibility.
    OPPOSITE_DIRECTION = {"U": "D", "D": "U", "L": "R", "R": "L"}
    DIRECTIONS = ["U", "R", "D", "L"]

    # Create a fixed order for tiles and a mapping from symbol to index.
    tile_symbols = list(PAC_TILES.keys())
    num_tiles = len(tile_symbols)
    tile_to_index = {s: i for i, s in enumerate(tile_symbols)}

    # ---------------------------------------------------------------------------------------
    # 2. PRECOMPUTE ADJACENCY MATRIX (as a Boolean NumPy array)
    # ---------------------------------------------------------------------------------------

    # Build a boolean array of shape (num_tiles, 4, num_tiles). For each tile index i
    # and direction d (0:U, 1:R, 2:D, 3:L), a True value for index j indicates that tile j
    # is allowed as a neighbor.
    adjacency_bool = np.zeros((num_tiles, 4, num_tiles), dtype=np.bool_)

    for i, tile_a in enumerate(tile_symbols):
        for d, direction in enumerate(DIRECTIONS):
            for j, tile_b in enumerate(tile_symbols):
                edge_a = PAC_TILES[tile_a]["edges"][direction]
                edge_b = PAC_TILES[tile_b]["edges"][OPPOSITE_DIRECTION[direction]]
                if edge_a == edge_b:
                    adjacency_bool[i, d, j] = True

    # Create an instance of the environment using
    env = WFCWrapper(
        map_length=12,
        map_width=20,
        tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool,
        num_tiles=num_tiles,
    )

    # Check if the environment follows the Gym interface.
    check_env(env, warn=True)
    print("Environment check passed!")

    # Create and train a PPO model.
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=10000)

    # Save the trained model.
    model.save("ppo_wfc")
    print("Training complete and model saved as 'ppo_wfc'")
