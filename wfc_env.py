import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wfc_pacman_tiles import build_pacman_adjacency, wfc_collapse


def grid_to_array(
    grid: list[list[set[str]]],
    all_tiles: list[str],
    map_length: int,
    map_width: int,
) -> np.ndarray:
    arr = np.empty((map_length, map_width), dtype=np.float32)
    for y in range(map_length):
        for x in range(map_width):
            cell = grid[y][x]
            if len(cell) == 1:
                # Get normalized tile index: index / (num_tiles -1)
                tile = next(iter(cell))
                idx = all_tiles.index(tile)
                arr[y, x] = idx / (len(all_tiles) - 1)
            else:
                # Uncollapsed cells get a fixed uncertainty value (e.g. 0.5)
                arr[y, x] = 0.5
    return arr.flatten()


def wfc_next_collapse_position(grid: list[list[set[str]]]) -> tuple[int, int]:
    min_options = float("inf")
    best_cell = (0, 0)
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if len(cell) > 1 and len(cell) < min_options:
                min_options = len(cell)
                best_cell = (x, y)
    return best_cell


class WFCWrapper(gym.Env):
    def __init__(
        self,
        tile_count: int,
        map_length: int,
        map_width: int,
        tile_defs: dict[str, dict[str, object]],
    ):
        self.all_tiles = list(tile_defs.keys())
        # Build the adjacency rules from the tile defs:
        self.adjacency = build_pacman_adjacency(tile_defs)
        # Use grid (list-of-list of sets) as internal state (grid indexed by [y][x])
        self.grid: list[list[set[str]]] = [
            [set(self.all_tiles) for _ in range(map_width)] for _ in range(map_length)
        ]
        self.tile_count: int = tile_count
        self.map_length: int = map_length
        self.map_width: int = map_width
        # Remove the torch.Tensor state; we no longer use self.current_map.
        self.action_space: spaces.Box = spaces.Box(
            low=0.0, high=1.0, shape=(self.tile_count,), dtype=np.float32
        )
        # Remove the torch.Tensor state; we no longer use self.current_map.
        self.observation_space: spaces.Dict = spaces.Box(
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
        # Convert the action (a float vector) into a dict: {tile: weight}
        action_dict = {tile: float(val) for tile, val in zip(self.all_tiles, action)}
        best_cell = wfc_next_collapse_position(self.grid)
        self.grid, truncate = wfc_collapse(
            self.grid, best_cell, self.adjacency, action_dict
        )
        reward = 0  # todo: assign reward if needed
        terminate = False  # update termination condition as needed
        info = {}
        return self.get_observation(), reward, terminate, truncate, info

    def reset(self):
        self.grid = [
            [set(self.all_tiles) for _ in range(self.map_width)]
            for _ in range(self.map_length)
        ]
        return self.get_observation(), {}

    def render(self, mode="human"): ...
