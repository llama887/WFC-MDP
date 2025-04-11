import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fast_wfc import fast_wfc_collapse_step, tile_symbols, adjacency_bool, num_tiles


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


class WFCWrapper(gym.Env):
    def __init__(
        self,
        map_length: int,
        map_width: int,
        tile_defs: dict[str, dict[str, object]],
    ):
        # Use the fast implementation variables from fast_wfc.py:
        self.all_tiles = tile_symbols  # use the precomputed tile order
        self.adjacency = adjacency_bool  # Numpy boolean array with compatibility info
        self.num_tiles = num_tiles
        self.map_length: int = map_length
        self.map_width: int = map_width
        # Initialize grid as a NumPy boolean array (all possibilities True)
        self.grid = np.ones((self.map_length, self.map_width, self.num_tiles), dtype=bool)
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
        self.grid, truncate = fast_wfc_collapse_step(
            self.grid,
            self.map_width,
            self.map_length,
            self.num_tiles,
            self.adjacency,
            action_vector,
        )
        reward = 0  # todo: assign reward if needed
        terminate = False  # update termination condition as needed
        info = {}
        return self.get_observation(), reward, terminate, truncate, info

    def reset(self, seed=0):
        self.grid = np.ones((self.map_length, self.map_width, self.num_tiles), dtype=bool)
        return self.get_observation(), {}

    def render(self, mode="human"): ...


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    from wfc_pacman_tiles import PAC_TILES

    # Create an instance of the environment using PAC_TILES.
    env = WFCWrapper(map_length=12, map_width=20, tile_defs=PAC_TILES)

    # Check if the environment follows the Gym interface.
    check_env(env, warn=True)
    print("Environment check passed!")

    # Create and train a PPO model.
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Save the trained model.
    model.save("ppo_wfc")
    print("Training complete and model saved as 'ppo_wfc'")
