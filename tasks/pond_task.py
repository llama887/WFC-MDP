from typing import Any
import numpy as np
from .utils import (
    calc_num_regions,
    percent_target_tiles_excluding_excluded_tiles,
    grid_to_binary_map,
    calc_longest_path,
)

MAX_POND_REWARD = 0


def pond_reward(
    grid: list[list[set[str]]], target_water_percent: float = 25.0, target_center_percent: float = 11.0, max_water_regions: int = 3, max_land_regions: int = 5, max_water_path: int = 25, hard: bool = False,
) -> tuple[float, dict[str, Any]]:
    percent_water = percent_target_tiles_excluding_excluded_tiles(
        grid,
        is_target_tiles=lambda tile_name: tile_name.startswith("water"),
        is_excluded_tiles=lambda tile_name: tile_name.startswith("sand") or tile_name.startswith("path")
    ) * 100

    percent_water_center = percent_target_tiles_excluding_excluded_tiles(
        grid,
        is_target_tiles=lambda tile_name: tile_name == "water",
        is_excluded_tiles=lambda tile_name: tile_name.startswith("sand") or tile_name.startswith("path")
    ) * 100

    if hard:
        water_reward = -abs(percent_water - target_water_percent)
        center_reward = -abs(percent_water_center - target_center_percent) * 3
    else:
        water_reward = min(percent_water - target_water_percent, 0)
        center_reward = min(percent_water_center - target_center_percent, 0) * 3

    water_binary_map = grid_to_binary_map(
        grid,
        lambda tile_name: tile_name.startswith("water") or tile_name.startswith("shore"),
    )
    number_of_water_regions = calc_num_regions(water_binary_map)
    region_reward = max(max_water_regions - number_of_water_regions, 0) - max(number_of_water_regions - max_water_regions, 0)

    water_path_length, _ = calc_longest_path(water_binary_map)
    path_reward = -max(water_path_length - max_water_path, 0)

    land_binary_map = grid_to_binary_map(
        grid,
        lambda tile_name: not (tile_name.startswith("water") or tile_name.startswith("shore")),
    )
    number_of_land_regions = calc_num_regions(land_binary_map)
    land_reward = -max(number_of_land_regions - max_land_regions, 0)

    total_reward = water_reward + center_reward + region_reward + land_reward + path_reward

    info = {
        "biome": "pond",
        "percent_water": percent_water,
        "percent_water_center": percent_water_center,
        "number_of_water_regions": number_of_water_regions,
        "water_path_length": water_path_length,
        "number_of_land_regions": number_of_land_regions,
        "target_water_percent": target_water_percent,
        "target_center_percent": target_center_percent,
        "max_water_regions": max_water_regions,
        "max_land_regions": max_land_regions,
        "max_water_path": max_water_path,
        "water_reward": water_reward,
        "center_reward": center_reward,
        "region_reward": region_reward,
        "land_reward": land_reward,
        "path_reward": path_reward,
    }

    return min(total_reward, MAX_POND_REWARD), info


def get_pond_biome(grid: list[list[set[str]]]) -> str:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r", "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",  "shore_bl", "shore_br", "shore_lr", "shore_rl", "pond"
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
                    if tile == "water" or tile == "pond":
                        pure_water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1

    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return "unknown"

    water_ratio = water_cells / total_cells
    pure_ratio = pure_water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

    if water_ratio >= 0.4 and pure_ratio >= 0.3 and shore_ratio <= 0.2:
        return "pond"
    return "unknown"


def measure_pond_flow(water_map: np.ndarray, direction: str) -> int:
    max_length = 0
    if direction == 'horizontal':
        for y in range(water_map.shape[0]):
            current = 0
            for x in range(water_map.shape[1]):
                if water_map[y, x]:
                    current += 1
                    max_length = max(max_length, current)
                else:
                    current = 0
    else:
        for x in range(water_map.shape[1]):
            current = 0
            for y in range(water_map.shape[0]):
                if water_map[y, x]:
                    current += 1
                    max_length = max(max_length, current)
                else:
                    current = 0
    return max_length

# import numpy as np
# from .utils import calc_num_regions, percent_target_tiles_excluding_excluded_tiles, grid_to_binary_map, calc_longest_path

# def pond_reward(grid: list[list[set[str]]]) -> tuple[float, dict]:
#     percent_water = percent_target_tiles_excluding_excluded_tiles(grid, is_target_tiles=lambda tile_name: tile_name.startswith("water"), is_excluded_tiles=lambda tile_name: tile_name.startswith("sand") or tile_name.startswith("path"),)
#     percent_water *= 100
#     TARGET_PERCENT_WATER = 25
#     percent_water_reward = 0
#     if percent_water < TARGET_PERCENT_WATER:
#         percent_water_reward = percent_water - TARGET_PERCENT_WATER

#     percent_water_center = percent_target_tiles_excluding_excluded_tiles(grid, is_target_tiles=lambda tile_name: tile_name == "water", is_excluded_tiles=lambda tile_name: tile_name.startswith("sand") or tile_name.startswith("path"),)
#     percent_water_center *= 100
#     TARGET_PERCENT_WATER_CENTER = 11
#     percent_water_center_reward = 0
#     if percent_water_center < TARGET_PERCENT_WATER_CENTER:
#         percent_water_center_reward = percent_water_center - TARGET_PERCENT_WATER_CENTER
        
#     water_binary_map = grid_to_binary_map(
#         grid,
#         lambda tile_name: tile_name.startswith("water") or tile_name.startswith("shore"),
#     )
#     DESIRED_MAX_WATER_REGIONS = 3
#     number_of_water_regions = calc_num_regions(water_binary_map)
#     water_region_reward = 0
#     if number_of_water_regions > DESIRED_MAX_WATER_REGIONS:
#         water_region_reward = 1 - number_of_water_regions
#     water_path_length, _ = calc_longest_path(water_binary_map)
#     DESIRED_MAX_WATER_PATH_LENGTH = 25
#     water_path_reward = 0
#     if water_path_length > DESIRED_MAX_WATER_PATH_LENGTH:
#         water_path_reward = DESIRED_MAX_WATER_PATH_LENGTH - water_path_length

#     land_binary_map = grid_to_binary_map(
#         grid,
#         lambda tile_name: not (tile_name.startswith("water") or tile_name.startswith("shore")),
#     )
#     number_of_land_regions = calc_num_regions(land_binary_map)
#     land_region_reward = 0
#     DESIRED_MAX_LAND_REGIONS = 5
#     if number_of_land_regions > DESIRED_MAX_LAND_REGIONS:
#         land_region_reward = DESIRED_MAX_LAND_REGIONS - number_of_land_regions
#     # land_region_reward = 0

#     return percent_water_reward + 3 * percent_water_center_reward + water_region_reward + land_region_reward + water_path_reward, {"percent_water": percent_water, "percent_water_reward": percent_water_reward, "percent_water_center_reward": percent_water_center_reward,"percent_water_center": percent_water_center, "number_of_water_regions": number_of_water_regions, "water_region_reward": water_region_reward, "number_of_land_regions": number_of_land_regions, "land_region_reward": land_region_reward, "water_path_length": water_path_length, "water_path_reward": water_path_reward,}
    
    
# def get_pond_biome(grid: list[list[set[str]]]) -> str:
#     water_tiles = {
#         "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
#         "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
#         "shore_bl", "shore_br", "shore_lr", "shore_rl",
#         "pond"
#     }

#     water_cells = 0
#     shore_cells = 0
#     pure_water_cells = 0
    
#     for row in grid:
#         for cell in row:
#             if len(cell) == 1:
#                 tile = next(iter(cell)).lower()
#                 if tile in water_tiles:
#                     water_cells += 1
#                     if tile == "water" or tile == "pond":
#                         pure_water_cells += 1
#                     if "shore" in tile:
#                         shore_cells += 1

#     total_cells = len(grid) * len(grid[0])
#     if total_cells == 0:
#         return "unknown"

#     water_ratio = water_cells / total_cells
#     pure_ratio = pure_water_cells / total_cells
#     shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

#     if water_ratio >= 0.4 and pure_ratio >= 0.3 and shore_ratio <= 0.2:
#         return "pond"
#     return "unknown"

# def measure_pond_flow(water_map: np.ndarray, direction: str) -> int:
#     max_length = 0
#     if direction == 'horizontal':
#         for y in range(water_map.shape[0]):
#             current = 0
#             for x in range(water_map.shape[1]):
#                 if water_map[y, x]:
#                     current += 1
#                     max_length = max(max_length, current)
#                 else:
#                     current = 0
#     else:
#         for x in range(water_map.shape[1]):
#             current = 0
#             for y in range(water_map.shape[0]):
#                 if water_map[y, x]:
#                     current += 1
#                     max_length = max(max_length, current)
#                 else:
#                     current = 0
#     return max_length

# # def pond_reward(grid: list[list[set[str]]]) -> tuple[float, dict]:
# #     water_tiles = {
# #         "water","water_tl","water_tr","water_t","water_l","water_r",
# #         "water_bl","water_b","water_br",
# #         "shore_tl","shore_tr","shore_bl","shore_br","shore_lr","shore_rl",
# #         "pond"
# #     }

# #     h, w = len(grid), len(grid[0])
# #     water_map = np.zeros((h, w), dtype=bool)
# #     for y in range(h):
# #         for x in range(w):
# #             cell = grid[y][x]
# #             if len(cell)==1 and next(iter(cell)).lower() in water_tiles:
# #                 water_map[y, x] = True

# #     # empty‐grid guard
# #     if h==0 or w==0:
# #         return -float("inf"), {}

# #     water_ratio = percent_target_tiles_excluding_excluded_tiles(
# #         grid,
# #         is_target_tiles=lambda t: t.lower() in water_tiles,
# #         exclude_prefixes=["path"],
# #     )

# #     def measure_run(mask: np.ndarray, axis: int) -> int:
# #         best = 0
# #         if axis == 0:
# #             for row in mask:
# #                 run = 0
# #                 for v in row:
# #                     run = run + 1 if v else 0
# #                     best = max(best, run)
# #         else:
# #             for col in mask.T:
# #                 run = 0
# #                 for v in col:
# #                     run = run + 1 if v else 0
# #                     best = max(best, run)
# #         return best

# #     flow_length = max(measure_run(water_map, 0), measure_run(water_map, 1))

# #     regions = calc_num_regions(water_map.astype(np.int8))

# #     def largest_cluster(wmap: np.ndarray) -> int:
# #         visited = np.zeros_like(wmap, dtype=bool)
# #         best = 0
# #         for yy in range(h):
# #             for xx in range(w):
# #                 if wmap[yy,xx] and not visited[yy,xx]:
# #                     size = 0
# #                     dq = deque([(yy,xx)])
# #                     visited[yy,xx] = True
# #                     while dq:
# #                         cy, cx = dq.popleft(); size += 1
# #                         for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
# #                             ny, nx = cy+dy, cx+dx
# #                             if (
# #                                 0 <= ny < h and 0 <= nx < w
# #                                 and wmap[ny,nx] and not visited[ny,nx]
# #                             ):
# #                                 visited[ny,nx] = True
# #                                 dq.append((ny,nx))
# #                     best = max(best, size)
# #         return best

# #     cluster_size = largest_cluster(water_map)

# #     total_water = int(water_map.sum())
# #     interior = 0
# #     for y in range(1, h-1):
# #         for x in range(1, w-1):
# #             if water_map[y, x] and all(
# #                 water_map[y+dy, x+dx] for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]
# #             ):
# #                 interior += 1
# #     interior_ratio = (interior / total_water) if total_water > 0 else 0.0

# #     MIN_WATER_RATIO    = 0.3
# #     MAX_FLOW_LENGTH    = 5    
# #     IDEAL_REGIONS      = 1
# #     IDEAL_INTERIOR     = 0.5  

# #     water_penalty    = -(MIN_WATER_RATIO - water_ratio) * 100 if water_ratio < MIN_WATER_RATIO else 0
# #     flow_penalty     = -max(0, flow_length - MAX_FLOW_LENGTH) * 30
# #     region_penalty   = -abs(regions - IDEAL_REGIONS) * 500
# #     interior_penalty = -abs(interior_ratio - IDEAL_INTERIOR) * 100
# #     cluster_bonus    = cluster_size * 2
# #     presence_bonus   = water_ratio * 50

# #     hill_ratio = percent_target_tiles_excluding_excluded_tiles(
# #         grid,
# #         is_target_tiles=lambda t: t.lower().startswith("hill"),
# #         exclude_prefixes=["path"]
# #     )
# #     hill_penalty = -hill_ratio * 200

# #     # --- combine & clamp so perfect achievable = 0 ---
# #     raw = (
# #         water_penalty
# #       + flow_penalty
# #       + region_penalty
# #       + interior_penalty
# #       + cluster_bonus
# #       + presence_bonus
# #       + hill_penalty
# #     )
# #     total_reward = min(raw, 0.0)

# #     return total_reward, {
# #         "water_ratio":    round(water_ratio, 3),
# #         "flow_length":    flow_length,
# #         "regions":        regions,
# #         "interior_ratio": round(interior_ratio, 3),
# #         "cluster_size":   cluster_size,
# #         "hill_ratio":     round(hill_ratio, 3),
# #         "reward":         total_reward,
# #     }


# # def has_water_path(
# #     grid: list[list[set[str]]], start: tuple, end: tuple, water_tiles: set[str]
# # ) -> bool:
# #     """Check if there's a continuous water path between two points."""
# #     from collections import deque

# #     water_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
# #     for y in range(len(grid)):
# #         for x in range(len(grid[0])):
# #             if len(grid[y][x]) == 1 and next(iter(grid[y][x])).lower() in water_tiles:
# #                 water_map[y, x] = True

# #     if not water_map[start[1], start[0]] or not water_map[end[1], end[0]]:
# #         return False

# #     visited = set()
# #     queue = deque([start])
# #     visited.add(start)

# #     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# #     while queue:
# #         current = queue.popleft()
# #         if current == end:
# #             return True

# #         for dx, dy in directions:
# #             x, y = current[0] + dx, current[1] + dy
# #             if (0 <= x < len(grid[0]) and 0 <= y < len(grid) 
# #                 and water_map[y, x] and (x, y) not in visited):
# #                 visited.add((x, y))
# #                 queue.append((x, y))

# #     return False