import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


def wfc_next_collapse_position(map: torch.Tensor) -> tuple[int, int]:
    # returns the next x, y position that will be collpased, used as a observation
    # should be deterministic (if the map is emtpy maybe pick the bottom corner or smth, ties should also be broken deterministically)
    return 1, 1


def wfc_collapse(
    map: torch.Tensor, action: spaces.Box, deterministic: bool = True
) -> torch.Tensor:
    # returns the new map after collpasing based on the given action
    # if deterministic is false then collpase by sampling from softmax, else use argmax
    return torch.zeros(1, 1)


def wfc_is_terminated(map: torch.Tensor) -> bool:
    # not sure if w need this, this was originally going to be used for if the map does not collpase properly and we get stuckin an impossible state but we can also implement backtracking for the rl solution
    # cannot backtracking is harder in evolution bc there is a set action sequence
    return False


class GymWrapperEnv(gym.Env):
    def __init__(
        self,
        tile_count: int,
        map_length: int,
        map_width: int,
    ):
        self.tile_count: int = tile_count
        self.map_length: int = map_length
        self.map_width: int = map_width
        self.current_map: torch.Tensor = torch.zeros(
            (self.map_length, self.map_width), dtype=torch.int8
        )
        self.action_space: spaces.Box = spaces.Box(
            low=0.0, high=1.0, shape=(self.tile_count,), dtype=np.float32
        )
        self.observation_space: spaces.Dict = spaces.Dict(  # we talked about flattening this, that has not been implemented yet
            {
                "map": spaces.MultiDiscrete(
                    np.full(
                        (self.map_length, self.map_width), tile_count, dtype=np.int8
                    )
                ),
                "position": spaces.Tuple(
                    (spaces.Discrete(self.map_length), spaces.Discrete(self.map_width))
                ),
            }
        )

    def get_observation(self) -> spaces.Dict:
        # this will be flattened
        return {
            "map": self.current_map.numpy(),
            "position": wfc_next_collapse_position(self.current_map),
        }

    def step(self, action):
        truncate = False
        info = {}
        self.current_map = wfc_collapse(self.current_map, action)
        reward = 0  # todo: rewards
        return self.get_observation(), reward, wfc_is_terminated(self.current_map), truncate, info

    def reset(self):
        self.current_map = torch.zeros(
            (self.map_length, self.map_width), dtype=torch.int8
        )
        return self.get_observation(), {}

    def render(self, mode="human"): ...
