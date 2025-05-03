import numpy as np

from .utils import calc_num_regions


def get_dominant_biome(grid: list[list[set[str]]]) -> str:
    """Enhanced biome detection with specific thresholds for ponds and rivers."""
    water_tiles = {
        "water",
        "water_tl",
        "water_tr",
        "water_t",
        "water_l",
        "water_r",
        "water_bl",
        "water_b",
        "water_br",
        "shore_tl",
        "shore_tr",
        "shore_bl",
        "shore_br",
        "shore_lr",
        "shore_rl",
    }

    # Count water cells and shore patterns
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

    # River detection - requires continuous flow and appropriate water ratio
    has_flow = check_continuous_flow(
        grid, water_tiles, "horizontal"
    ) or check_continuous_flow(grid, water_tiles, "vertical")

    if has_flow and 0.2 <= water_ratio <= 0.4:
        return "river"
    elif water_ratio >= 0.45 and shore_ratio <= 0.2:
        return "pond"
    return "unknown"


def check_continuous_flow(
    grid: list[list[set[str]]], water_tiles: set[str], direction: str
) -> bool:
    """Check if there's a continuous water path across the map in specified direction."""
    if direction == "horizontal":
        # Check from left to right
        for y in range(len(grid)):
            if has_water_path(grid, (0, y), (len(grid[0]) - 1, y), water_tiles):
                return True
    else:
        # Check from top to bottom
        for x in range(len(grid[0])):
            if has_water_path(grid, (x, 0), (x, len(grid) - 1), water_tiles):
                return True
    return False


def find_edge_water_cells(
    grid: list[list[set[str]]], water_tiles: set[str], edge: str
) -> list[tuple]:
    """Find water cells along a specific edge of the grid."""
    edge_cells = []
    if edge == "left":
        for y in range(len(grid)):
            if len(grid[y][0]) == 1 and next(iter(grid[y][0])).lower() in water_tiles:
                edge_cells.append((0, y))
    elif edge == "right":
        for y in range(len(grid)):
            if len(grid[y][-1]) == 1 and next(iter(grid[y][-1])).lower() in water_tiles:
                edge_cells.append((len(grid[0]) - 1, y))
    elif edge == "top":
        for x in range(len(grid[0])):
            if len(grid[0][x]) == 1 and next(iter(grid[0][x])).lower() in water_tiles:
                edge_cells.append((x, 0))
    elif edge == "bottom":
        for x in range(len(grid[0])):
            if len(grid[-1][x]) == 1 and next(iter(grid[-1][x])).lower() in water_tiles:
                edge_cells.append((x, len(grid) - 1))
    return edge_cells


def has_water_path(
    grid: list[list[set[str]]], start: tuple, end: tuple, water_tiles: set[str]
) -> bool:
    """Check if there's a continuous water path between two points."""
    from collections import deque

    # Convert grid to binary water map
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

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-way connectivity

    while queue:
        current = queue.popleft()
        if current == end:
            return True

        for dx, dy in directions:
            x, y = current[0] + dx, current[1] + dy
            if (
                0 <= x < len(grid[0])
                and 0 <= y < len(grid)
                and water_map[y, x]
                and (x, y) not in visited
            ):
                visited.add((x, y))
                queue.append((x, y))

    return False


def water_biome_reward(grid: list[list[set[str]]]) -> float:
    water_tiles = {
        "water",
        "water_tl",
        "water_tr",
        "water_t",
        "water_l",
        "water_r",
        "water_bl",
        "water_b",
        "water_br",
        "shore_tl",
        "shore_tr",
        "shore_bl",
        "shore_br",
        "shore_lr",
        "shore_rl",
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
        return 0.0

    water_ratio = water_cells / total_cells
    pure_water_ratio = pure_water_cells / total_cells
    shore_ratio = shore_cells / water_cells if water_cells > 0 else 0

    # Check biome type
    biome = get_dominant_biome(grid)

    if biome == "river":
        # River scoring
        has_flow = check_continuous_flow(
            grid, water_tiles, "horizontal"
        ) or check_continuous_flow(grid, water_tiles, "vertical")
        flow_score = 1.0 if has_flow else 0.0
        coverage_score = max(0.0, 1.0 - abs(water_ratio - 0.3) * 3.33)
        regions = calc_num_regions(water_map.astype(np.int8))
        connected_score = 1.0 / (regions**0.5)
        combined = 0.4 * flow_score + 0.3 * coverage_score + 0.3 * connected_score
        return float(combined * 100 * 1.5)  # Bonus for rivers

    elif biome == "pond":
        # Pond scoring - prioritize high water concentration
        coverage_score = max(
            0.0, 1.0 - abs(pure_water_ratio - 0.5) * 2.0
        )  # Target 50% pure water
        shore_penalty = max(0.0, 1.0 - shore_ratio * 5.0)  # Penalize shore tiles
        combined = 0.7 * coverage_score + 0.3 * shore_penalty
        return float(combined * 100 * 1.2)  # Smaller bonus for ponds

    # Default scoring for other cases
    coverage_score = max(0.0, 1.0 - abs(water_ratio - 0.35) * 2.86)
    return float(coverage_score * 100)
