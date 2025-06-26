import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from numpy.typing import NDArray

from assets.biome_adjacency_rules import create_adjacency_matrix
from tasks.utils import percent_target_tiles_excluding_excluded_tiles

adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
num_tiles = len(tile_symbols)

SAND_PATH_MASK = np.zeros(num_tiles, dtype=bool)
GRASS_TARGET_MASK = np.zeros(num_tiles, dtype=bool)
FLOWER_TARGET_MASK = np.zeros(num_tiles, dtype=bool)
WATER_MASK = np.zeros(num_tiles, dtype=bool)
HILL_MASK = np.zeros(num_tiles, dtype=bool)

for idx, tile_name in enumerate(tile_symbols):
    if tile_name.startswith("sand") or tile_name.startswith("path"):
        SAND_PATH_MASK[idx] = True
    if "grass" in tile_name:
        GRASS_TARGET_MASK[idx] = True
    if tile_name == "flower":
        FLOWER_TARGET_MASK[idx] = True
    if tile_name.startswith("water") or tile_name.startswith("shore"):
        WATER_MASK[idx] = True
    if "hill" in tile_name:
        HILL_MASK[idx] = True


def grass_reward(grid: NDArray) -> tuple[float, dict[str, any]]:
    """Improved reward calculation using mask operations"""
    water_count = np.sum(grid * WATER_MASK[None, None, :])
    hill_count = np.sum(grid * HILL_MASK[None, None, :])

    grass_percent = percent_target_tiles_excluding_excluded_tiles(
        grid, GRASS_TARGET_MASK, SAND_PATH_MASK
    )

    flower_percent = percent_target_tiles_excluding_excluded_tiles(
        grid, FLOWER_TARGET_MASK, SAND_PATH_MASK
    )

    # Calculate rewards based on thresholds
    grass_reward_val = min(0, grass_percent - 0.2) * 10
    flower_reward_val = min(0, flower_percent - 0.2) * 10
    penalty = water_count + hill_count

    total_reward = grass_reward_val + flower_reward_val - penalty

    return total_reward, {
        "water_count": water_count,
        "hill_count": hill_count,
        "grass_percent": grass_percent,
        "flower_percent": flower_percent,
        "grass_biome_reward": total_reward,
    }


def classify_grass_biome(counts: dict, grass_cells: int) -> str:
    """Returns 'grassgrass', 'meadow', or 'unknown' — not used in scoring."""
    if grass_cells == 0:
        return "unknown"

    flower_ratio = counts["flower"] / grass_cells
    grass_ratio = grass_cells / sum(counts.values())

    if flower_ratio <= 0.15:
        return "grassgrass"
    elif flower_ratio > 0.15:
        return "meadow"
    return "unknown"


# def grass_reward(grid: np.ndarray) -> tuple[float, dict[str, any]]:
#     biome = "grass"
#     grid = np.array(grid)
#     if isinstance(grid.flat[0], set):
#         grid = np.vectorize(lambda cell: next(iter(cell)) if isinstance(cell, set) else cell)(grid)

#     map_height, map_width = grid.shape
#     total_tiles = map_height * map_width
#     tile_counts = Counter(grid.flatten())
#     tile_ratios = {symbol: count / total_tiles for symbol, count in tile_counts.items()}

#     target_ratios = {
#         "G": 0.8,  # Grass
#         "F": 0.1,  # Flower
#         "P": 0.1,  # Path
#     }

# grass_count = tile_counts.get("G", 0)
# grass_coverage = grass_count / total_tiles

# structure_matrix = (grid == "G").astype(np.uint8)
# labeled_array, num_regions = label(structure_matrix)
# continuity_score = 1.0 if num_regions == 1 else max(0, 1.0 - (num_regions - 1) * 0.2)

# distribution_score = 1.0 - np.mean([
#     abs(tile_ratios.get(symbol, 0) - target_ratio)
#     for symbol, target_ratio in target_ratios.items()
# ])

# penalty = 0

# if not 0.7 <= grass_coverage <= 0.9:
#     penalty += abs(grass_coverage - 0.8) * 50

# if num_regions > 2:
#     penalty += (num_regions - 2) * 15
# penalty += (1.0 - distribution_score) * 10

# total_score = -penalty

# return min(total_score, 0.0), {
#     "biome": biome,
#     "coverage": grass_coverage,
#     "continuity": continuity_score,
#     "distribution": distribution_score,
#     "ratios": tile_ratios,
#     "num_regions": num_regions,
#     "reward": min(total_score, 0.0),
# }
