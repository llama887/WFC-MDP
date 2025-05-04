import numpy as np
from .utils import calc_num_regions

def get_pond_biome(grid: list[list[set[str]]]) -> str:
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

def pond_reward(grid: list[list[set[str]]]) -> float:
    water_tiles = {
        "water", "water_tl", "water_tr", "water_t", "water_l", "water_r",
        "water_bl", "water_b", "water_br", "shore_tl", "shore_tr",
        "shore_bl", "shore_br", "shore_lr", "shore_rl"
    }

    water_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
    water_cells = 0
    shore_cells = 0
    pure_water_cells = 0

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1:
                tile = next(iter(grid[y][x])).lower()
                if tile in water_tiles:
                    water_map[y, x] = True
                    water_cells += 1
                    if tile == "water":
                        pure_water_cells += 1
                    if "shore" in tile:
                        shore_cells += 1

    total_cells = len(grid) * len(grid[0])
    if total_cells == 0:
        return -float('inf')

    water_ratio = water_cells / total_cells
    pure_ratio = pure_water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

    flow_length = max(
        measure_pond_flow(water_map, 'horizontal'),
        measure_pond_flow(water_map, 'vertical')
    )

    regions = calc_num_regions(water_map.astype(np.int8))

    IDEAL_WATER_RATIO = 0.4
    IDEAL_PURE_RATIO = 0.3
    IDEAL_SHORE_RATIO = 0.2
    MAX_FLOW_LENGTH = 5
    IDEAL_REGIONS = 1

    water_penalty = -abs(water_ratio - IDEAL_WATER_RATIO) * 100
    pure_penalty = -abs(pure_ratio - IDEAL_PURE_RATIO) * 100
    shore_penalty = -max(0, shore_ratio - IDEAL_SHORE_RATIO) * 100
    flow_penalty = -max(0, flow_length - MAX_FLOW_LENGTH) * 50
    region_penalty = -abs(regions - IDEAL_REGIONS) * 50

    total_reward = (water_penalty + pure_penalty + shore_penalty + flow_penalty + region_penalty)

    return total_reward, {}

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