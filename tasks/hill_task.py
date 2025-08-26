import numpy as np
from typing import Any, Dict, Tuple
from scipy.ndimage import label
from .utils import grid_to_binary_map

__all__ = ["hill_reward"]

from assets.biome_adjacency_rules import create_adjacency_matrix
_, tile_symbols, _ = create_adjacency_matrix()
num_tiles = len(tile_symbols)

# Build masks
WATER_SHORE_MASK = np.zeros(num_tiles, dtype=bool)
PURE_WATER_MASK = np.zeros(num_tiles, dtype=bool)
SAND_PATH_MASK = np.zeros(num_tiles, dtype=bool)
HILL_MASK = np.zeros(num_tiles, dtype=bool)

for index, tile_name in enumerate(tile_symbols):
    if tile_name.startswith("water") or tile_name.startswith("shore"):
        WATER_SHORE_MASK[index] = True
    if tile_name == "water":
        PURE_WATER_MASK[index] = True
    if tile_name.startswith("sand") or tile_name.startswith("path"):
        SAND_PATH_MASK[index] = True
    if "hill" in tile_name:
        HILL_MASK[index] = True

# 4-neighborhood connectivity (no diagonals)
FOUR_CONN = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=np.uint8)


def _count_rocks_2x2(hill_binary: np.ndarray) -> int:
    """Count 2x2 windows that are entirely hill."""
    height, width = hill_binary.shape
    if height < 2 or width < 2:
        return 0
    window_sum = (
        hill_binary[:-1, :-1]
        + hill_binary[1:, :-1]
        + hill_binary[:-1, 1:]
        + hill_binary[1:, 1:]
    )
    return int(np.sum(window_sum == 4))


def _count_enclosed_areas_as_hills(hill_binary: np.ndarray) -> int:
    """
    Count 'hills' as any enclosed area of non-hill tiles that does NOT touch the border.
    This is the number of background components (4-connected) not touching the outer frame.
    """
    background_binary = (hill_binary == 0).astype(np.uint8)
    labeled_background, num_background = label(background_binary, structure=FOUR_CONN)
    if num_background == 0:
        return 0

    height, width = hill_binary.shape
    touch_mask = np.zeros(num_background + 1, dtype=bool)
    # Mark background components that touch the border
    for (ys, xs) in [
        (0, slice(None)),               # top row
        (height - 1, slice(None)),      # bottom row
        (slice(None), 0),               # left col
        (slice(None), width - 1)        # right col
    ]:
        border_labels = labeled_background[ys, xs]
        touch_mask[border_labels] = True

    # Exclude label 0 (background of labeling) & those touching the border
    enclosed_count = 0
    for label_id in range(1, num_background + 1):
        if not touch_mask[label_id]:
            enclosed_count += 1
    return enclosed_count





def hill_reward(grid: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Hill biome reward with MAX = 0. Penalties:
      - Rock penalty:               -1 for each 2x2 all-hill block
      - Hill quota penalty:         -1 for each missing enclosed area until we reach 5 total
                                    (a 'hill' is ANY enclosed area of non-hill fully surrounded by hills)
      - Water/shore penalty:        -1 per water or shore tile
      - Elevation penalty:          -1 per level below 2, where elevation level is the maximum
                                    number of nested rings ('hill on a hill' means depth ≥ 2)
    """
    # Binary hill map from one-hot channels
    hill_binary: np.ndarray = grid_to_binary_map(grid, HILL_MASK).astype(np.uint8)

    # Counts
    num_rocks_2x2: int = _count_rocks_2x2(hill_binary)
    num_enclosed_areas: int = _count_enclosed_areas_as_hills(hill_binary)
    # max_nesting_depth: int = _max_ring_nesting_depth(hill_binary)

    # Water/shore tiles directly from one-hot grid
    water_or_shore_count: int = int(np.sum(grid * WATER_SHORE_MASK[None, None, :]))

    MAX_ROCKS_ALLOWED: int = 5          # cap for 2x2 hill blocks
    MIN_HILLS_REQUIRED: int = 3        # minimum enclosed areas required

    rock_penalty: float = float(max(0, num_rocks_2x2 - MAX_ROCKS_ALLOWED))
    hill_quota_penalty: float = float(max(0, MIN_HILLS_REQUIRED - num_enclosed_areas))
    water_shore_penalty: float = float(water_or_shore_count)
    # elevation_level: int = int(min(2, max_nesting_depth)) 
    # elevation_penalty: float = float(max(0, 2 - elevation_level))

    total_penalty: float = (
        5 * rock_penalty
        + 2 * hill_quota_penalty
        + water_shore_penalty
        # + elevation_penalty
    )
    total_reward: float = -total_penalty


    diagnostics: dict[str, Any] = {
        "biome": "hill",
        "num_rocks_2x2": num_rocks_2x2,
        "num_enclosed_areas": num_enclosed_areas,
        # "max_nesting_depth": max_nesting_depth,
        # "elevation_level": elevation_level,
        "water_or_shore_count": water_or_shore_count,
        "rock_penalty": rock_penalty,
        "hill_quota_penalty": hill_quota_penalty,
        "water_shore_penalty": water_shore_penalty,
        # "elevation_penalty": elevation_penalty,
        "hill_biome_reward": total_reward,
    }

    # Enforce invariant
    assert total_reward <= 0, diagnostics

    return total_reward, diagnostics
