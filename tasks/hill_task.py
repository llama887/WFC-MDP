import numpy as np
from typing import List, Set, Tuple, Dict
from collections import Counter
from scipy.ndimage import label, find_objects

__all__ = ["hill_reward"]

def is_rectangle(mask: np.ndarray) -> bool:
    """Check if the True area in the mask forms a clean rectangle."""
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return False
    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()
    submask = mask[min_y:max_y + 1, min_x:max_x + 1]
    return np.all(submask)


def hill_reward(grid: np.ndarray) -> tuple[float, dict[str, any]]:
    biome = "hill"
    grid = np.array(grid)
    if isinstance(grid.flat[0], set):
        grid = np.vectorize(lambda cell: next(iter(cell)) if isinstance(cell, set) else cell)(grid)

    map_height, map_width = grid.shape
    total_tiles = map_height * map_width
    tile_counts = Counter(grid.flatten())

    hill_count = tile_counts.get("H", 0)
    hill_ratio = hill_count / total_tiles
    ideal_hill_ratio = 0.5
    if not 0.4 <= hill_ratio <= 0.6:
        ratio_penalty = abs(hill_ratio - 0.5) * 50
    else:
        ratio_penalty = 0

    structure_matrix = (grid == "H").astype(np.uint8)
    labeled_array, num_regions = label(structure_matrix)
    continuity_penalty = max(0, (num_regions - 1) * 20)

    flower_count = tile_counts.get("F", 0)
    flower_ratio = flower_count / total_tiles
    flower_penalty = abs(flower_ratio - 0.1) * 30

    # rectangle shape reward (bonus)
    bounding_boxes = find_objects(labeled_array)
    rectangle_bonus = 0
    for region_slice in bounding_boxes:
        if region_slice:
            height = region_slice[0].stop - region_slice[0].start
            width = region_slice[1].stop - region_slice[1].start
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio <= 1.5:
                rectangle_bonus += 10

    total_penalty = ratio_penalty + continuity_penalty + flower_penalty - rectangle_bonus
    total_score = -total_penalty  

    return min(total_score, 0.0), {
        "biome": biome,
        "hill_ratio": hill_ratio,
        "num_regions": num_regions,
        "flower_ratio": flower_ratio,
        "rectangle_bonus": rectangle_bonus,
        "reward": min(total_score, 0.0),
    }
