from collections import deque
from typing import Callable

import numpy as np


def grid_to_binary_map(
    grid: list[list[set[str]]], is_empty: Callable[[str], bool]
) -> np.ndarray:
    """Converts the WFC grid into a binary map.
    Empty cells (0) are those whose single tile name starts with 'sand' or 'path',
    solid cells (1) are everything else.
    """
    height = len(grid)
    width = len(grid[0])
    binary_map = np.ones((height, width), dtype=np.int32)  # default solid (1)
    for y in range(height):
        for x in range(width):
            cell = grid[y][x]
            if len(cell) == 1:
                tile_name = next(iter(cell))
                if is_empty(tile_name):
                    binary_map[y, x] = 0  # empty
                else:
                    binary_map[y, x] = 1  # solid
            else:
                binary_map[y, x] = 1
    return binary_map

def percent_target_tiles_excluding_excluded_tiles(grid: list[list[set[str]]], is_target_tiles: Callable[[str], bool], is_excluded_tiles: Callable[[str], bool] | None = None) -> float:
    """Calculates the percentage of target tiles in the grid, excluding excluded tiles."""
    if is_excluded_tiles is None:
        is_excluded_tiles = lambda _: False

    total_target_tiles = 0
    total_excluded_tiles = 0
    total_tiles = len(grid) * len(grid[0])
    for row in grid:
        for cell in row:
            if len(cell) == 1:
                tile_name = next(iter(cell))
                if is_target_tiles(tile_name):
                    total_target_tiles += 1
                elif is_excluded_tiles(tile_name):
                    total_excluded_tiles += 1

    if total_target_tiles == 0:
        return 0.0

    return (total_target_tiles) / (total_tiles - total_excluded_tiles)

def calc_num_regions(binary_map: np.ndarray) -> int:
    """Counts connected regions of empty cells (value 0) using flood-fill."""
    h, w = binary_map.shape
    visited = np.zeros((h, w), dtype=bool)
    num_regions = 0

    def neighbors(y, x):
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx

    for y in range(h):
        for x in range(w):
            if binary_map[y, x] == 0 and not visited[y, x]:
                num_regions += 1
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if visited[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    for ny, nx in neighbors(cy, cx):
                        if binary_map[ny, nx] == 0 and not visited[ny, nx]:
                            stack.append((ny, nx))
    return num_regions


def calc_longest_path(binary_map: np.ndarray):
    def reconstruct_path(end, parent):
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def bfs_farthest(start_y, start_x, binary_map):
        h, w = binary_map.shape
        dist = -np.ones((h, w), dtype=int)
        parent = dict()  # maps (y,x) → (py,px)
        q = deque()

        dist[start_y, start_x] = 0
        parent[(start_y, start_x)] = None
        q.append((start_y, start_x))

        farthest = (start_y, start_x)
        while q:
            y, x = q.popleft()
            d = dist[y, x]

            # update farthest so far
            if d > dist[farthest]:
                farthest = (y, x)

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w:
                    if binary_map[ny, nx] == 0 and dist[ny, nx] == -1:
                        dist[ny, nx] = d + 1
                        parent[(ny, nx)] = (y, x)
                        q.append((ny, nx))

        return farthest, dist, parent

    h, w = binary_map.shape
    seen = np.zeros((h, w), dtype=bool)

    best_length = 0
    best_path = []

    for y in range(h):
        for x in range(w):
            if binary_map[y, x] == 0 and not seen[y, x]:
                # 3a) flood‐fill the component to mark it and collect one seed
                stack = [(y, x)]
                seen[y, x] = True
                component_cells = [(y, x)]

                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in (
                        (cy - 1, cx),
                        (cy + 1, cx),
                        (cy, cx - 1),
                        (cy, cx + 1),
                    ):
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and binary_map[ny, nx] == 0
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                            component_cells.append((ny, nx))

                # 3b) first sweep: from arbitrary seed → find A
                seed_y, seed_x = component_cells[0]
                A, _, _ = bfs_farthest(seed_y, seed_x, binary_map)

                # 3c) second sweep: from A → find B, get dist & parents
                B, dist, parent = bfs_farthest(A[0], A[1], binary_map)

                length = dist[B]
                if length > best_length:
                    best_length = length
                    best_path = reconstruct_path(B, parent)

    return best_length, best_path