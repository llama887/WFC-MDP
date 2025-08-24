import numpy as np
from typing import Any
from scipy.ndimage import label, find_objects
from .utils import grid_to_binary_map, percent_target_tiles_excluding_excluded_tiles

__all__ = ["hill_reward"]

from assets.biome_adjacency_rules import create_adjacency_matrix
_, tile_symbols, _ = create_adjacency_matrix()
num_tiles = len(tile_symbols)

# Define masks for river tasks
WATER_SHORE_MASK = np.zeros(num_tiles, dtype=bool)
PURE_WATER_MASK = np.zeros(num_tiles, dtype=bool)
SAND_PATH_MASK = np.zeros(num_tiles, dtype=bool)
HILL_MASK = np.zeros(num_tiles, dtype=bool)

for idx, tile_name in enumerate(tile_symbols):
    if tile_name.startswith("water") or tile_name.startswith("shore"):
        WATER_SHORE_MASK[idx] = True
    if tile_name == "water":
        PURE_WATER_MASK[idx] = True
    if tile_name.startswith("sand") or tile_name.startswith("path"):
        SAND_PATH_MASK[idx] = True
    if "hill" in tile_name:
        HILL_MASK[idx] = True

def is_rectangle(mask: np.ndarray) -> bool:
    """Check if the True area in the mask forms a clean rectangle."""
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return False
    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()
    submask = mask[min_y:max_y + 1, min_x:max_x + 1]
    return np.all(submask)


def hill_reward(grid: np.ndarray) -> tuple[float, dict[str, Any]]:
    """
    Hill biome reward using one-hot channel masks (shape H x W x C).
    Mirrors the structure of grass_reward:
      - uses mask ops, no string grids
      - returns non-positive reward (penalties)
      - provides diagnostics
    """

    # Basic counts from masks
    water_count: float = float(np.sum(grid * WATER_SHORE_MASK[None, None, :]))
    hill_count: float = float(np.sum(grid * HILL_MASK[None, None, :]))

    # Binary map of hill tiles (H x W), robust to channels
    hill_binary: np.ndarray = grid_to_binary_map(grid, HILL_MASK)
    map_height: int
    map_width: int
    map_height, map_width = hill_binary.shape

    # Percent of tiles that are hills (exclude nothing for the main ratio)
    hill_percent: float = percent_target_tiles_excluding_excluded_tiles(
        grid, HILL_MASK, SAND_PATH_MASK
    )



    # --- Objectives/penalties (all become <= 0 total) ---

    # 1) Target ~50% hills overall
    hill_penalty = max(0.0, 50.0 - hill_percent)

    # 2) Prefer spread-out hills (discourage central clumping)
    if hill_count > 0:
        yy, xx = np.nonzero(hill_binary)
        center_y: float = (map_height - 1) / 2.0
        center_x: float = (map_width - 1) / 2.0
        distances: np.ndarray = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        avg_distance_from_center: float = float(np.mean(distances))
        max_distance: float = float(np.sqrt(center_y**2 + center_x**2)) if (center_y or center_x) else 1.0
        spread_ratio: float = (avg_distance_from_center / max_distance) if max_distance > 0 else 1.0
        center_penalty: float = (1.0 - spread_ratio) * 50.0  # higher penalty if concentrated near center
    else:
        center_penalty = 50.0  # no hills at all → penalize

    # 3) Prefer a moderate number of clusters (connected components)
    labeled_array, num_regions = label(hill_binary.astype(np.uint8))
    ideal_clusters: int = 5
    continuity_penalty: float = abs(int(num_regions) - ideal_clusters) * 20.0

    # 4) Discourage too-rectangular clusters (blocky hills)
    rectangle_penalty: float = 0.0
    for region_slice in find_objects(labeled_array):
        if region_slice is None:
            continue
        height: int = region_slice[0].stop - region_slice[0].start
        width: int = region_slice[1].stop - region_slice[1].start
        if height == 0 or width == 0:
            continue
        aspect_ratio: float = max(width / height, height / width)
        if aspect_ratio <= 1.5:  # compact/boxy
            rectangle_penalty += 10.0

    # 5) Minor reward for more, smaller clusters (non-rectangular spread)
    scatter_bonus: float = (float(num_regions) * 2.0) if num_regions > 2 else 0.0


    # 7) Penalize water presence in a hill biome
    water_penalty: float = water_count  # 1:1 per tile; tune if needed

    # Sum penalties (negative reward); ensure ≤ 0
    total_penalty: float = (
        hill_penalty
        + center_penalty
        + continuity_penalty
        + rectangle_penalty
        + water_penalty
        - scatter_bonus
    )
    total_reward: float = -float(total_penalty)

    assert total_reward <= 0, {
        "hill_percent": hill_percent,
        "hill_penalty": hill_penalty,
        "num_regions": int(num_regions),
        "water_count": water_count,
        "hill_count": hill_count,
        "rectangle_penalty": rectangle_penalty,
        "center_penalty": center_penalty,
        "scatter_bonus": scatter_bonus,
        "hill_biome_reward": total_reward,
    }

    return total_reward, {
        "biome": "hill",
        "hill_percent": hill_percent,
        "num_regions": int(num_regions),
        "water_count": water_count,
        "hill_count": hill_count,
        "rectangle_penalty": rectangle_penalty,
        "center_penalty": center_penalty,
        "scatter_bonus": scatter_bonus,
        "hill_biome_reward": total_reward,
    }
