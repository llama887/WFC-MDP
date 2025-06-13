from typing import Any
import numpy as np
from collections import Counter
from scipy.ndimage import label, find_objects

__all__ = ["hill_reward"]
MAX_HILL_REWARD = 0


def hill_reward(
    grid: list[list[set[str]]], target_hill_ratio: float = 0.5, target_flower_ratio: float = 0.1, ideal_clusters: int = 5, hard: bool = False,
) -> tuple[float, dict[str, Any]]:
    np_grid = np.array(grid)
    if isinstance(np_grid.flat[0], set):
        np_grid = np.vectorize(lambda cell: next(iter(cell)) if isinstance(cell, set) else cell)(np_grid)

    map_height, map_width = np_grid.shape
    total_tiles = map_height * map_width
    tile_counts = Counter(np_grid.flatten())

    hill_count = tile_counts.get("H", 0)
    hill_ratio = hill_count / total_tiles
    flower_count = tile_counts.get("F", 0)
    flower_ratio = flower_count / total_tiles

    if hard:
        hill_ratio_reward = -abs(hill_ratio - target_hill_ratio) * 100
    else:
        hill_ratio_reward = min(hill_ratio - target_hill_ratio, 0) * 100

    if hard:
        flower_reward = -abs(flower_ratio - target_flower_ratio) * 30
    else:
        flower_reward = min(flower_ratio - target_flower_ratio, 0) * 30

    # Penalize central concentration
    structure_matrix = (np_grid == "H").astype(np.uint8)
    labeled_array, num_regions = label(structure_matrix)

    # Compute distance from center for each hill tile
    yy, xx = np.where(structure_matrix)
    center_y, center_x = map_height / 2, map_width / 2
    distances = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2) if yy.size > 0 else np.array([0])
    max_distance = np.sqrt(center_y ** 2 + center_x ** 2)
    spread_ratio = np.mean(distances) / max_distance if yy.size > 0 else 0
    center_reward = (spread_ratio - 0.5) * 50

    # Cluster count reward
    if hard:
        cluster_reward = -abs(num_regions - ideal_clusters) * 20
    else:
        if num_regions < 3:
            cluster_reward = (num_regions - 3) * 20
        elif num_regions > 10:
            cluster_reward = (10 - num_regions) * 20
        else:
            cluster_reward = 0

    # Rectangle detection (flat hill clusters are bad)
    bounding_boxes = find_objects(labeled_array)
    rectangle_penalty = 0
    for region_slice in bounding_boxes:
        if region_slice:
            height = region_slice[0].stop - region_slice[0].start
            width = region_slice[1].stop - region_slice[1].start
            aspect_ratio = max(width / height, height / width) if height and width else 10
            if aspect_ratio <= 1.5:  # compact cluster (almost square or rectangle)
                rectangle_penalty += 10  # discourage too rectangular blocks

    # Minor reward for presence of scattered hills (non-rectangular regions)
    scatter_bonus = max(num_regions - 2, 0) * 2

    total_reward = (
        hill_ratio_reward
        + flower_reward
        + center_reward
        + cluster_reward
        - rectangle_penalty
        + scatter_bonus
    )

    info = {
        "biome": "hill",
        "hill_ratio": round(hill_ratio, 3),
        "flower_ratio": round(flower_ratio, 3),
        "num_regions": num_regions,
        "spread_ratio": round(spread_ratio, 3),
        "rectangle_penalty": rectangle_penalty,
        "scatter_bonus": scatter_bonus,
        "target_hill_ratio": target_hill_ratio,
        "target_flower_ratio": target_flower_ratio,
        "ideal_clusters": ideal_clusters,
    }

    return min(total_reward, MAX_HILL_REWARD), info


# import numpy as np
# from collections import Counter
# from scipy.ndimage import label, find_objects

# __all__ = ["hill_reward"]

# def is_rectangle(mask: np.ndarray) -> bool:
#     """Check if the True area in the mask forms a clean rectangle."""
#     ys, xs = np.where(mask)
#     if ys.size == 0 or xs.size == 0:
#         return False
#     min_y, max_y = ys.min(), ys.max()
#     min_x, max_x = xs.min(), xs.max()
#     submask = mask[min_y:max_y + 1, min_x:max_x + 1]
#     return np.all(submask)


# def hill_reward(grid: np.ndarray) -> tuple[float, dict[str, any]]:
#     biome = "hill"
#     grid = np.array(grid)
#     if isinstance(grid.flat[0], set):
#         grid = np.vectorize(lambda cell: next(iter(cell)) if isinstance(cell, set) else cell)(grid)

#     map_height, map_width = grid.shape
#     total_tiles = map_height * map_width
#     tile_counts = Counter(grid.flatten())

#     hill_count = tile_counts.get("H", 0)
#     hill_ratio = hill_count / total_tiles
#     ideal_hill_ratio = 0.5
#     hill_ratio_penalty = abs(hill_ratio - ideal_hill_ratio) * 100  # strong bias toward 50%

#     # Penalize central concentration
#     structure_matrix = (grid == "H").astype(np.uint8)
#     labeled_array, num_regions = label(structure_matrix)

#     # Compute distance from center for each hill tile
#     yy, xx = np.where(structure_matrix)
#     center_y, center_x = map_height / 2, map_width / 2
#     distances = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
#     if distances.size > 0:
#         avg_distance_from_center = np.mean(distances)
#         max_distance = np.sqrt(center_y ** 2 + center_x ** 2)
#         spread_ratio = avg_distance_from_center / max_distance
#         center_penalty = (1 - spread_ratio) * 50  # prefer spread-out hills
#     else:
#         center_penalty = 50

#     # Prefer a moderate number of hill clusters (e.g., 3–10)
#     ideal_clusters = 5
#     continuity_penalty = abs(num_regions - ideal_clusters) * 20

#     # Rectangle detection (flat hill clusters are bad)
#     bounding_boxes = find_objects(labeled_array)
#     rectangle_penalty = 0
#     for region_slice in bounding_boxes:
#         if region_slice:
#             height = region_slice[0].stop - region_slice[0].start
#             width = region_slice[1].stop - region_slice[1].start
#             aspect_ratio = max(width / height, height / width) if height and width else 10
#             if aspect_ratio <= 1.5:  # compact cluster (almost square or rectangle)
#                 rectangle_penalty += 10  # discourage too rectangular blocks

#     # Minor reward for presence of scattered hills (non-rectangular regions)
#     scatter_bonus = (num_regions if num_regions > 2 else 0) * 2

#     # Optional secondary tile (flowers, decoration)
#     flower_count = tile_counts.get("F", 0)
#     flower_ratio = flower_count / total_tiles
#     flower_penalty = abs(flower_ratio - 0.1) * 30

#     total_penalty = (
#         hill_ratio_penalty
#         + center_penalty
#         + continuity_penalty
#         + rectangle_penalty
#         + flower_penalty
#         - scatter_bonus
#     )
#     total_score = -total_penalty
#     return min(total_score, 0.0), {
#         "biome": biome,
#         "hill_ratio": round(hill_ratio, 3),
#         "num_regions": num_regions,
#         "flower_ratio": round(flower_ratio, 3),
#         "rectangle_penalty": rectangle_penalty,
#         "center_penalty": round(center_penalty, 3),
#         "scatter_bonus": scatter_bonus,
#         "reward": min(total_score, 0.0),
#     }
