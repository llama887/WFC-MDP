from collections import deque
from typing import Callable

import numpy as np


def grid_to_binary_map(
    grid: np.ndarray,  # 3D array [H, W, num_tiles]
    is_empty: np.ndarray  # 1D mask [num_tiles]
) -> np.ndarray:
    """Converts WFC grid to binary map using 3D mask operations"""
    return np.any(grid * is_empty[None, None, :], axis=2).astype(np.int32)

def percent_target_tiles_excluding_excluded_tiles(
    grid: np.ndarray,          # 3D array [H, W, num_tiles]
    target_mask: np.ndarray,   # 1D mask [num_tiles]
    exclude_mask: np.ndarray   # 1D mask [num_tiles]
) -> float:
    """Calculates percentage using 3D mask ops with broadcasting"""
    excluded_tiles = np.any(grid * exclude_mask[None, None, :], axis=2)
    target_tiles = np.any(grid * target_mask[None, None, :] * ~exclude_mask[None, None, :], axis=2)
    
    total_tiles = grid.shape[0] * grid.shape[1]
    excluded_count = np.sum(excluded_tiles)
    target_count = np.sum(target_tiles)
    
    valid_tiles = total_tiles - excluded_count
    return target_count / valid_tiles if valid_tiles > 0 else 0.0

def count_tiles(
    grid: np.ndarray,    # 3D array [H, W, num_tiles]
    mask: np.ndarray     # 1D mask [num_tiles]
) -> int:
    """Counts tiles using vectorized 3D mask operations"""
    return np.sum(grid * mask[None, None, :])

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

def count_tiles(grid: list[list[set[str]]], is_target_tile: Callable[[str], bool]) -> int:
    """Counts the number of tiles in the grid that match the given tile name."""
    count = 0
    for row in grid:
        for cell in row:
            if len(cell) == 1 and is_target_tile(next(iter(cell))):
                count += 1
    return count
