import numpy as np
from typing import Any
from .utils import calc_num_regions, calc_longest_path, grid_to_binary_map, percent_target_tiles_excluding_excluded_tiles

# Get tile definitions from biome rules
from assets.biome_adjacency_rules import create_adjacency_matrix
_, tile_symbols, _ = create_adjacency_matrix()
num_tiles = len(tile_symbols)

# Define masks for river tasks
WATER_SHORE_MASK = np.zeros(num_tiles, dtype=bool)
PURE_WATER_MASK = np.zeros(num_tiles, dtype=bool)
SAND_PATH_MASK = np.zeros(num_tiles, dtype=bool)

for idx, tile_name in enumerate(tile_symbols):
    if tile_name.startswith("water") or tile_name.startswith("shore"):
        WATER_SHORE_MASK[idx] = True
    if tile_name == "water":
        PURE_WATER_MASK[idx] = True
    if tile_name.startswith("sand") or tile_name.startswith("path"):
        SAND_PATH_MASK[idx] = True

def river_reward(grid: np.ndarray) -> tuple[float, dict[str, Any]]:
    water_binary_map = grid_to_binary_map(grid, WATER_SHORE_MASK)
    land_binary_map = grid_to_binary_map(grid, ~WATER_SHORE_MASK)

    water_regions = calc_num_regions(water_binary_map)
    land_regions = calc_num_regions(land_binary_map)
    path_length, longest_path = calc_longest_path(water_binary_map)
    pure_water_count = np.sum(grid * PURE_WATER_MASK[None, None, :])

    # Apply rewards/penalties
    DESIRED_RIVER_LENGTH = 35
    region_penalty = 1 - water_regions
    path_penalty = path_length - DESIRED_RIVER_LENGTH if path_length < DESIRED_RIVER_LENGTH else 0
    land_penalty = 3 - land_regions if land_regions > 3 else 0

    total_reward = region_penalty + path_penalty + land_penalty - pure_water_count

    return total_reward, {
        "number_of_river_regions": water_regions,
        "river_length": path_length,
        "longest_river_path": longest_path,
        "number_of_water_centers": pure_water_count,
        "number_of_land_regions": land_regions
    }


def get_river_biome(grid: list[list[set[str]]]) -> str:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    water_cells = 0
    shore_cells = 0
    pure_water_cells = 0

    for row in grid:
        for cell in row:
            if len(cell) == 1:
                tile = next(iter(cell)).lower()
                if tile in water_tiles:
                    water_cells += 1
                    if tile == "water":
                        pure_water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1

    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return "unknown"

    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

    has_flow = measure_river_flow(grid, water_tiles) >= min(len(grid), len(grid[0])) * 0.75

    if has_flow and 0.2 <= water_ratio <= 0.4 and shore_ratio <= 0.3:
        return "river"
    return "unknown"

def measure_river_flow(grid: list[list[set[str]]], water_tiles: set[str]) -> float:
    h, w = len(grid), len(grid[0])
    water_map = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        for x in range(w):
            cell = grid[y][x]
            if len(cell) == 1 and next(iter(cell)).lower() in water_tiles:
                water_map[y, x] = True

    def measure_flow(axis: int) -> float:
        max_flow = 0
        if axis == 0:  # horizontal
            for y in range(h):
                current = 0
                for x in range(w):
                    if water_map[y, x]:
                        current += 1
                        max_flow = max(max_flow, current)
                    else:
                        current = 0
        else:  # vertical
            for x in range(w):
                current = 0
                for y in range(h):
                    if water_map[y, x]:
                        current += 1
                        max_flow = max(max_flow, current)
                    else:
                        current = 0
        return max_flow

    horizontal_flow = measure_flow(0)
    vertical_flow = measure_flow(1)
    
    return max(horizontal_flow, vertical_flow)

# def river_reward(grid: list[list[set[str]]]) -> tuple[float, dict]:
#     water_tiles = {
#         "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
#         "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
#         "shore_bl", "shore_br", "shore_lr", "shore_rl"
#     }

#     h, w = len(grid), len(grid[0])
#     water_map = np.zeros((h, w), dtype=bool)
#     shore_map = np.zeros((h, w), dtype=bool)
    
#     for y in range(h):
#         for x in range(w):
#             cell = grid[y][x]
#             if len(cell) == 1:
#                 tile = next(iter(cell)).lower()
#                 if tile in water_tiles:
#                     water_map[y, x] = True
#                     if "shore" in tile:
#                         shore_map[y, x] = True

#     # empty-grid guard
#     if h == 0 or w == 0:
#         return -float('inf'), {}

#     water_ratio = percent_target_tiles_excluding_excluded_tiles(
#         grid,
#         is_target_tiles=lambda t: t.lower() in water_tiles,
#         exclude_prefixes=["path"],
#     )
    
#     shore_ratio = (shore_map.sum() / water_map.sum()) if water_map.sum() > 0 else 0.0

#     flow_length = measure_river_flow(grid, water_tiles)
#     max_dimension = max(h, w)
#     flow_quality = flow_length / max_dimension if max_dimension > 0 else 0.0

#     regions = calc_num_regions(water_map.astype(np.int8))

#     def largest_cluster(wmap: np.ndarray) -> int:
#         visited = np.zeros_like(wmap, dtype=bool)
#         best = 0
#         for y in range(h):
#             for x in range(w):
#                 if wmap[y,x] and not visited[y,x]:
#                     size = 0
#                     dq = deque([(y,x)])
#                     visited[y,x] = True
#                     while dq:
#                         cy, cx = dq.popleft()
#                         size += 1
#                         for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
#                             ny, nx = cy+dy, cx+dx
#                             if (0 <= ny < h and 0 <= nx < w and 
#                                 wmap[ny,nx] and not visited[ny,nx]):
#                                 visited[ny,nx] = True
#                                 dq.append((ny,nx))
#                     best = max(best, size)
#         return best

#     cluster_size = largest_cluster(water_map)

#     # Calculate interior water tiles
#     interior = 0
#     for y in range(1, h-1):
#         for x in range(1, w-1):
#             if water_map[y, x] and all(
#                 water_map[y+dy, x+dx] for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]
#             ):
#                 interior += 1
#     interior_ratio = (interior / water_map.sum()) if water_map.sum() > 0 else 0.0

#     # Reward parameters
#     IDEAL_WATER_RATIO_MIN = 0.2
#     IDEAL_WATER_RATIO_MAX = 0.4
#     IDEAL_SHORE_RATIO = 0.3
#     IDEAL_REGIONS = 1
#     MIN_FLOW_QUALITY = 0.75
#     FLOW_BONUS = 20

#     # Calculate penalties and bonuses - now positive values mean better
#     # Water ratio penalty: how close we are to ideal range
#     if water_ratio < IDEAL_WATER_RATIO_MIN:
#         water_penalty = (water_ratio - IDEAL_WATER_RATIO_MIN) * 100  # negative
#     elif water_ratio > IDEAL_WATER_RATIO_MAX:
#         water_penalty = (IDEAL_WATER_RATIO_MAX - water_ratio) * 100  # negative
#     else:
#         # Within ideal range - bonus based on how centered we are
#         center = (IDEAL_WATER_RATIO_MIN + IDEAL_WATER_RATIO_MAX) / 2
#         water_penalty = -abs(water_ratio - center) * 50  # small penalty for not being centered

#     # Shore ratio penalty: how close we are to ideal
#     shore_penalty = -abs(shore_ratio - IDEAL_SHORE_RATIO) * 100
    
#     # Region penalty: how close to 1 region we are
#     region_penalty = -abs(regions - IDEAL_REGIONS) * 50
    
#     # Flow quality: bonus for good flow, penalty for bad
#     flow_penalty = -50 if flow_quality < MIN_FLOW_QUALITY else 0
#     flow_bonus = FLOW_BONUS * flow_quality if flow_quality >= MIN_FLOW_QUALITY else 0
    
#     # Cluster size bonus: bigger is better
#     cluster_bonus = cluster_size * 0.5
    
#     # Interior bonus: more interior water is better
#     interior_bonus = interior_ratio * 20

#     # Combine all components - now we sum them (they're all negative or positive)
#     total_reward = (
#         water_penalty
#         + shore_penalty
#         + region_penalty
#         + flow_penalty
#         + flow_bonus
#         + cluster_bonus
#         + interior_bonus
#     )

#     # Clamp at 0 (perfect score) - but now higher is better
#     total_reward = min(total_reward, 0.0)

#     return total_reward, {
#         "water_ratio": round(water_ratio, 3),
#         "shore_ratio": round(shore_ratio, 3),
#         "flow_quality": round(flow_quality, 3),
#         "regions": regions,
#         "cluster_size": cluster_size,
#         "interior_ratio": round(interior_ratio, 3),
#         "reward": total_reward,
#     }
