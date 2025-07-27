import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any

import numpy as np
from numpy.typing import NDArray

from assets.biome_adjacency_rules import create_adjacency_matrix
from tasks.utils import (
    calc_longest_path,
    calc_num_regions,
    percent_target_tiles_excluding_excluded_tiles,
)

MAX_BINARY_REWARD = 0

adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()

num_tiles = len(tile_symbols)
PASSABLE_MASK: NDArray = np.zeros(num_tiles, dtype=bool)
for tile_name in tile_symbols:
    if tile_name.startswith("sand") or tile_name.startswith("path"):
        PASSABLE_MASK[tile_to_index[tile_name]] = True


def binary_reward(
    grid: NDArray,
    target_path_length: int,
    hard: bool = False,
) -> tuple[float, dict[str, Any]]:
    binary_map = ~np.any(grid * PASSABLE_MASK[None, None], axis=2)
    number_of_regions = calc_num_regions(binary_map)
    current_path_length, longest_path = calc_longest_path(binary_map)

    region_reward = min(1 - number_of_regions, 0)
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
    assert region_reward + path_reward <= 0, info
    return (region_reward + path_reward, info)


def binary_percent_water(grid: list[list[set[str]]]) -> float:
    """Calculates the percentage of water tiles in the grid excluding the path tiles."""
    return percent_target_tiles_excluding_excluded_tiles(
        grid,
        lambda tile_name: tile_name.startswith("water")
        or tile_name.startswith("shore"),
        lambda tile_name: tile_name.startswith("sand") or tile_name.startswith("path"),
    )
