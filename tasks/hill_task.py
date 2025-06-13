import numpy as np
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
    hill_ratio_penalty = abs(hill_ratio - ideal_hill_ratio) * 100  # strong bias toward 50%

    # Penalize central concentration
    structure_matrix = (grid == "H").astype(np.uint8)
    labeled_array, num_regions = label(structure_matrix)

    # Compute distance from center for each hill tile
    yy, xx = np.where(structure_matrix)
    center_y, center_x = map_height / 2, map_width / 2
    distances = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    if distances.size > 0:
        avg_distance_from_center = np.mean(distances)
        max_distance = np.sqrt(center_y ** 2 + center_x ** 2)
        spread_ratio = avg_distance_from_center / max_distance
        center_penalty = (1 - spread_ratio) * 50  # prefer spread-out hills
    else:
        center_penalty = 50

    # Prefer a moderate number of hill clusters (e.g., 3–10)
    ideal_clusters = 5
    continuity_penalty = abs(num_regions - ideal_clusters) * 20

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
    scatter_bonus = (num_regions if num_regions > 2 else 0) * 2

    # Optional secondary tile (flowers, decoration)
    flower_count = tile_counts.get("F", 0)
    flower_ratio = flower_count / total_tiles
    flower_penalty = abs(flower_ratio - 0.1) * 30

    total_penalty = (
        hill_ratio_penalty
        + center_penalty
        + continuity_penalty
        + rectangle_penalty
        + flower_penalty
        - scatter_bonus
    )
    total_score = -total_penalty
    return min(total_score, 0.0), {
        "biome": biome,
        "hill_ratio": round(hill_ratio, 3),
        "num_regions": num_regions,
        "flower_ratio": round(flower_ratio, 3),
        "rectangle_penalty": rectangle_penalty,
        "center_penalty": round(center_penalty, 3),
        "scatter_bonus": scatter_bonus,
        "reward": min(total_score, 0.0),
    }