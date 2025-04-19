from collections import deque

import numpy as np


def grid_to_binary_map(grid: list[list[set[str]]]) -> np.ndarray:
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
                if tile_name.startswith("sand") or tile_name.startswith("path"):
                    binary_map[y, x] = 0  # empty
                else:
                    binary_map[y, x] = 1  # solid
            else:
                binary_map[y, x] = 1
    return binary_map


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


def calc_longest_path(binary_map: np.ndarray) -> int:
    """Computes the longest shortest path among all empty cells (value 0) using BFS."""
    h, w = binary_map.shape

    def bfs(start_y, start_x):
        visited = -np.ones((h, w), dtype=int)
        q = deque()
        visited[start_y, start_x] = 0
        q.append((start_y, start_x))
        max_dist = 0
        while q:
            y, x = q.popleft()
            d = visited[y, x]
            max_dist = max(max_dist, d)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w:
                    if binary_map[ny, nx] == 0 and visited[ny, nx] == -1:
                        visited[ny, nx] = d + 1
                        q.append((ny, nx))
        return max_dist

    overall_max = 0
    for y in range(h):
        for x in range(w):
            if binary_map[y, x] == 0:
                overall_max = max(overall_max, bfs(y, x))
    return overall_max
