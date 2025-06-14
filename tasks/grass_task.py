from typing import Any
import numpy as np
from .utils import count_tiles, percent_target_tiles_excluding_excluded_tiles

MAX_GRASS_REWARD = 0


def grass_reward(grid: list[list[set[str]]]) -> tuple[float, dict[str, Any]]:
    """Calculates the grass biome score based on the grid (now a list of lists of sets)."""
    water_count = count_tiles(
        grid,
        lambda x: isinstance(x, str) and (x.startswith("water") or x.startswith("shore")),
    ) 
    # print(f"Water tiles counted: {water_count}")

    hill_count = count_tiles(
        grid,
        lambda x: isinstance(x, str) and ("hill" in x),
    )
    # print(f"Hill tiles counted: {hill_count}")

    TARGET_GRASS_PERCENT = 0.2
    grass_percent = percent_target_tiles_excluding_excluded_tiles(
        grid,
        lambda x: isinstance(x, str) and ("grass" in x),
        lambda x: isinstance(x, str) and (x.startswith("path") or x.startswith("sand")),
    )
    # print(f"Grass percentage: {grass_percent}")

    TARGET_FLOWER_PERCENT = 0.2
    flower_percent = percent_target_tiles_excluding_excluded_tiles(
        grid,
        lambda x: isinstance(x, str) and (x == "flower"),
        lambda x: isinstance(x, str) and (x.startswith("path") or x.startswith("sand")),
    )
    grass_reward = (
        0
        if grass_percent >= TARGET_GRASS_PERCENT
        else grass_percent - TARGET_GRASS_PERCENT
    )
    flower_reward = (
        0
        if flower_percent >= TARGET_FLOWER_PERCENT
        else flower_percent - TARGET_FLOWER_PERCENT
    )

    total_reward = -water_count + -hill_count + grass_reward + flower_reward

    info = {
        "water_count": water_count,
        "hill_count": hill_count,
        "grass_percent": grass_percent,
        "flower_percent": flower_percent,
    }

    return (total_reward, info)


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

# from typing import Any
# import numpy as np
# from .utils import count_tiles, percent_target_tiles_excluding_excluded_tiles

# MAX_GRASS_REWARD = 0


# def grass_reward(grid: np.ndarray) -> tuple[float, dict[str, Any]]:
#     """Calculates the grass biome score based on the grid (now a list of lists of sets).""" 
    
#     water_count = count_tiles(
#         grid,
#         lambda x: isinstance(x, str) and (x.startswith("water") or x.startswith("shore")),
#     ) 
#     print(f"Water tiles counted: {water_count}")

#     hill_count = count_tiles(
#         grid,
#         lambda x: isinstance(x, str) and ("hill" in x),
#     )
#     print(f"Hill tiles counted: {hill_count}")

#     TARGET_GRASS_PERCENT = 0.2
#     grass_percent = percent_target_tiles_excluding_excluded_tiles(
#         grid,
#         lambda x: isinstance(x, str) and ("grass" in x),
#         lambda x: isinstance(x, str) and (x.startswith("path") or x.startswith("sand")),
#     )
#     print(f"Grass percentage: {grass_percent}")

#     TARGET_FLOWER_PERCENT = 0.2
#     flower_percent = percent_target_tiles_excluding_excluded_tiles(
#         grid,
#         lambda x: isinstance(x, str) and (x == "flower"),
#         lambda x: isinstance(x, str) and (x.startswith("path") or x.startswith("sand")),
#     )
#     grass_reward = (
#         0
#         if grass_percent >= TARGET_GRASS_PERCENT
#         else grass_percent - TARGET_GRASS_PERCENT
#     )
#     flower_reward = (
#         0
#         if flower_percent >= TARGET_FLOWER_PERCENT
#         else flower_percent - TARGET_FLOWER_PERCENT
#     )

#     total_reward = -water_count + -hill_count + grass_reward + flower_reward

#     info = {
#         "water_count": water_count,
#         "hill_count": hill_count,
#         "grass_percent": grass_percent,
#         "flower_percent": flower_percent,
#     }

#     return (total_reward, info)


# def classify_grass_biome(counts: dict, grass_cells: int) -> str:
#     """Returns 'grassgrass', 'meadow', or 'unknown' — not used in scoring."""
#     if grass_cells == 0:
#         return "unknown"

#     flower_ratio = counts["flower"] / grass_cells
#     grass_ratio = grass_cells / sum(counts.values())

#     if flower_ratio <= 0.15:
#         return "grassgrass"
#     elif flower_ratio > 0.15:
#         return "meadow"
#     return "unknown"


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