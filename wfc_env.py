import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from wfc_pacman_tiles import wfc_collapse


def wfc_next_collapse_position(map: torch.Tensor) -> tuple[int, int]:
    # returns the next x, y position that will be collpased, used as a observation
    # should be deterministic (if the map is emtpy maybe pick the bottom corner or smth, ties should also be broken deterministically)
    return 1, 1


class GymWrapperEnv(gym.Env):
    def __init__(
        self,
        tile_count: int,
        map_length: int,
        map_width: int,
        tile_defs: dict[str, dict[str, object]],
    ):
        all_tiles = list(tile_defs.keys())
        self.grid: list[list[set[str]]] = [
            [set(all_tiles) for _ in range(map_length)] for _ in range(map_width)
        ]
        self.tile_count: int = tile_count
        self.map_length: int = map_length
        self.map_width: int = map_width
        self.current_map: torch.Tensor = torch.zeros(
            (self.map_length, self.map_width), dtype=torch.int8
        )
        self.action_space: spaces.Box = spaces.Box(
            low=0.0, high=1.0, shape=(self.tile_count,), dtype=np.float32
        )
        self.observation_space: spaces.Dict = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.map_length * self.map_width + 2,),
            dtype=np.float32,
        )

    def get_observation(self) -> spaces.Dict:
        map_flat = (
            self.current_map.numpy().flatten().astype(np.float32)
        )  # shape: (map_length * map_width,)
        pos = wfc_next_collapse_position(
            self.current_map
        )  # assume returns (x, y) or [x, y]
        pos_array = np.array(pos, dtype=np.float32)  # shape: (2,)
        return np.concatenate([map_flat, pos_array])

    def step(self, action):
        truncate = False
        terminate = False
        info = {}
        self.current_map, truncate = wfc_collapse(self.current_map, action)
        reward = 0  # todo: rewards
        return (
            self.get_observation(),
            reward,
            terminate,
            truncate,
            info,
        )

    def reset(self):
        self.current_map = torch.zeros(
            (self.map_length, self.map_width), dtype=torch.int8
        )
        return self.get_observation(), {}

    def render(self, mode="human"): ...
    def render(self, mode="human"): ...
