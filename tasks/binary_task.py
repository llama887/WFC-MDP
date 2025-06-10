from typing import Any

import numpy as np

from .utils import (
    calc_longest_path,
    calc_num_regions,
    grid_to_binary_map,
    percent_target_tiles_excluding_excluded_tiles,
)

MAX_BINARY_REWARD = 0


def binary_reward(
    grid: list[list[set[str]]], target_path_length: int, passable_mask: np.ndarray, hard: bool = False,
) -> tuple[float, dict[str, Any]]:
    # binary_map = grid_to_binary_map(
    #     grid,
    #     lambda tile_name: tile_name.startswith("sand") or tile_name.startswith("path"),
    # )
    binary_map = ~np.any(grid * passable_mask[None, None], axis=2)
    number_of_regions = calc_num_regions(binary_map)
    current_path_length, longest_path = calc_longest_path(binary_map)

    region_reward = 1 - number_of_regions
    if not hard:
        path_reward = (
            0
            if current_path_length >= target_path_length
            else current_path_length - target_path_length
        )
    else:
        # hard reward requires getting the EXACT path length
        path_reward = -abs(target_path_length - current_path_length)

    info = {
        "number_of_regions": number_of_regions,
        "path_length": current_path_length,
        "longest_path": longest_path,
    }
    return (region_reward + path_reward, info)


def binary_percent_water(grid: list[list[set[str]]]) -> float:
    """Calculates the percentage of water tiles in the grid excluding the path tiles."""
    return percent_target_tiles_excluding_excluded_tiles(
        grid,
        lambda tile_name: tile_name.startswith("water")
        or tile_name.startswith("shore"),
        lambda tile_name: tile_name.startswith("sand") or tile_name.startswith("path"),
    )
