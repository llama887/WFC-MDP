import numpy as np
from .utils import calc_num_regions

def get_river_biome(grid: list[list[set[str]]]) -> str:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    water_cells = 0
    shore_cells = 0
    for row in grid:
        for cell in row:
            if len(cell) == 1:
                tile = next(iter(cell)).lower()
                if tile in water_tiles:
                    water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1

    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return "unknown"

    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

    has_flow = check_river_flow(grid, water_tiles, "horizontal") or \
               check_river_flow(grid, water_tiles, "vertical")

    if has_flow and 0.2 <= water_ratio <= 0.4 and shore_ratio <= 0.3:
        return "river"
    return "unknown"

def check_river_flow(
    grid: list[list[set[str]]], water_tiles: set[str], direction: str
) -> bool:
    if direction == "horizontal":
        for y in range(len(grid)):
            if has_water_path(grid, (0, y), (len(grid[0]) - 1, y), water_tiles):
                return True
    else:
        for x in range(len(grid[0])):
            if has_water_path(grid, (x, 0), (x, len(grid) - 1), water_tiles):
                return True
    return False

def river_reward(grid: list[list[set[str]]]) -> float:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    water_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
    water_cells = 0
    shore_cells = 0

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1:
                tile = next(iter(grid[y][x])).lower()
                if tile in water_tiles:
                    water_map[y, x] = True
                    water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1

    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return -float('inf')

    water_ratio = water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

    has_horizontal_flow = check_river_flow(grid, water_tiles, "horizontal")
    has_vertical_flow = check_river_flow(grid, water_tiles, "vertical")
    has_flow = has_horizontal_flow or has_vertical_flow

    regions = calc_num_regions(water_map.astype(np.int8))

    IDEAL_WATER_RATIO_MIN = 0.2
    IDEAL_WATER_RATIO_MAX = 0.4
    IDEAL_SHORE_RATIO = 0.3
    IDEAL_REGIONS = 1

    flow_penalty = 0 if has_flow else -100
    
    if water_ratio < IDEAL_WATER_RATIO_MIN:
        water_penalty = (IDEAL_WATER_RATIO_MIN - water_ratio) * -200
    elif water_ratio > IDEAL_WATER_RATIO_MAX:
        water_penalty = (water_ratio - IDEAL_WATER_RATIO_MAX) * -200
    else:
        water_penalty = 0
    
    shore_penalty = max(0, (shore_ratio - IDEAL_SHORE_RATIO)) * -100
    region_penalty = abs(regions - IDEAL_REGIONS) * -50
    flow_bonus = 20 if (has_horizontal_flow and has_vertical_flow) else 0
    total_reward = (flow_penalty + water_penalty + shore_penalty + region_penalty + flow_bonus)

    return min(total_reward, 0)

def has_water_path(
    grid: list[list[set[str]]], start: tuple, end: tuple, water_tiles: set[str]
) -> bool:
    """Check if there's a continuous water path between two points."""
    from collections import deque

    water_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1 and next(iter(grid[y][x])).lower() in water_tiles:
                water_map[y, x] = True

    if not water_map[start[1], start[0]] or not water_map[end[1], end[0]]:
        return False

    visited = set()
    queue = deque([start])
    visited.add(start)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current = queue.popleft()
        if current == end:
            return True

        for dx, dy in directions:
            x, y = current[0] + dx, current[1] + dy
            if (0 <= x < len(grid[0]) and 0 <= y < len(grid) 
                and water_map[y, x] and (x, y) not in visited):
                visited.add((x, y))
                queue.append((x, y))

    return False