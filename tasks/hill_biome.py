import numpy as np
from typing import List, Set, Tuple, Dict
from .utils import calc_num_regions

__all__ = ["hill_biome_reward"]

def is_rectangle(mask: np.ndarray) -> bool:
    """Check if the True area in the mask forms a clean rectangle."""
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return False
    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()
    submask = mask[min_y:max_y + 1, min_x:max_x + 1]
    return np.all(submask) 


def hill_biome_reward(grid: List[List[Set[str]]]) -> Tuple[float, Dict[str, float]]:
    """
    Computes a reward for generating natural, extendable, rectangular hill biomes.
    - Encourages edge hill tiles to dominate
    - Favors rectangular, continuous structures
    - Rewards extended hill walls and flower-filled centers

    Args:
        grid (List[List[Set[str]]]): WFC tile grid with sets of tile symbols.

    Returns:
        Tuple[float, Dict[str, float]]: Final score and metric breakdown.
    """
    corner_tiles = {"grass_hill_tl", "grass_hill_tr", "grass_hill_bl", "grass_hill_br"}
    edge_tiles = {"grass_hill_t", "grass_hill_b", "grass_hill_l", "grass_hill_r"}
    flower_tile = "flower"
    hill_tiles = edge_tiles.union(corner_tiles)

    height, width = len(grid), len(grid[0])
    hill_map = np.zeros((height, width), dtype=np.int8)
    flower_map = np.zeros_like(hill_map)
    edge_matrix = np.full((height, width), "", dtype=object)

    counts = {
        "hill_total": 0,
        "edge_tiles": 0,
        "corner_tiles": 0,
        "flower_tiles": 0,
        "extendable_runs": 0
    }

    for y in range(height):
        for x in range(width):
            cell = grid[y][x]
            if len(cell) != 1:
                continue
            tile = next(iter(cell)).lower()
            if tile in hill_tiles:
                hill_map[y, x] = 1
                counts["hill_total"] += 1
                if tile in edge_tiles:
                    counts["edge_tiles"] += 1
                    edge_matrix[y, x] = tile
                else:
                    counts["corner_tiles"] += 1
            elif tile == flower_tile:
                flower_map[y, x] = 1
                counts["flower_tiles"] += 1

    if counts["hill_total"] == 0:
        return 0.0, {"biome": "invalid"}

    total_cells = height * width
    hill_ratio = counts["hill_total"] / total_cells
    edge_ratio = counts["edge_tiles"] / counts["hill_total"]

    num_regions = calc_num_regions(hill_map)
    continuity_score = 1.0 / (num_regions ** 0.5) if num_regions > 0 else 0.0

    for y in range(height):
        run = 1
        last = None
        for x in range(width):
            t = edge_matrix[y, x]
            if t == last and t != "":
                run += 1
            else:
                if run >= 2:
                    counts["extendable_runs"] += 1
                run = 1
                last = t
        if run >= 2:
            counts["extendable_runs"] += 1

    for x in range(width):
        run = 1
        last = None
        for y in range(height):
            t = edge_matrix[y, x]
            if t == last and t != "":
                run += 1
            else:
                if run >= 2:
                    counts["extendable_runs"] += 1
                run = 1
                last = t
        if run >= 2:
            counts["extendable_runs"] += 1

    max_runs = height + width
    extendability_score = min(counts["extendable_runs"] / max_runs, 1.0)

    flower_score = counts["flower_tiles"] / total_cells
    is_rectangular = is_rectangle(hill_map.astype(bool))
    shape_bonus = 1.0 if is_rectangular else 0.0

    score = (
        0.3 * edge_ratio +
        0.25 * continuity_score +
        0.2 * extendability_score +
        0.15 * flower_score +
        0.1 * shape_bonus
    ) * 100

    return float(score), {
        "biome": "hill",
        "coverage": hill_ratio,
        "edge_ratio": edge_ratio,
        "continuity": continuity_score,
        "extendability": extendability_score,
        "flower_fill": flower_score,
        "rectangle_shape": is_rectangular,
        "num_regions": num_regions,
        "hill_tiles": counts["hill_total"]
    }
