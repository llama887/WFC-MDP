import sys

import numpy as np
from numba import boolean, float32, int32, njit


@njit(nogil=True)
def propagate(grid, rules, width, height, num_tiles, start_x, start_y):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    opposites = [2, 3, 0, 1]

    queue = [(start_x, start_y)]
    visited = np.zeros((height, width), dtype=boolean)

    idx = 0
    while idx < len(queue):
        x, y = queue[idx]
        idx += 1

        for d in range(4):
            dx, dy = directions[d]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue

            current_tiles = np.where(grid[y, x])[0]
            if len(current_tiles) == 0:
                continue

            od = opposites[d]
            allowed = np.zeros(num_tiles, dtype=boolean)
            for t in current_tiles:
                allowed |= rules[od, t]

            neighbor = grid[ny, nx].copy()
            new_possible = neighbor & allowed

            if not np.array_equal(neighbor, new_possible):
                grid[ny, nx] = new_possible
                if not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))

    return grid


@njit(nogil=True)
def wfc_core(width, height, weights, rules, num_tiles):
    grid = np.ones((height, width, num_tiles), dtype=boolean)
    output = np.full((height, width), -1, dtype=int32)
    total_weight = weights.sum()
    weight_map = (weights / total_weight).cumsum()

    while True:
        entropy = grid.sum(axis=2)
        entropy[entropy == 1] = sys.maxsize
        if np.all(entropy == sys.maxsize):
            break

        min_entropy = entropy.min()
        candidates = np.argwhere(entropy == min_entropy)
        y, x = candidates[np.random.randint(len(candidates))]

        possible = grid[y, x]
        possible_weights = weights.copy()
        possible_weights[~possible] = 0
        total = possible_weights.sum()
        if total == 0:
            break

        rand = np.random.random()
        selected = 0
        cumulative = possible_weights[0] / total
        while cumulative < rand and selected < num_tiles - 1:
            selected += 1
            cumulative += possible_weights[selected] / total

        grid[y, x] = False
        grid[y, x, selected] = True
        output[y, x] = selected

        grid = propagate(grid, rules, width, height, num_tiles, x, y)

    for y in range(height):
        for x in range(width):
            if output[y, x] == -1:
                possible = np.where(grid[y, x])[0]
                output[y, x] = possible[0] if len(possible) > 0 else -1

    return output


def wave_function_collapse(width, height, num_tiles, weights, adjacency_rules):
    rules = np.zeros((4, num_tiles, num_tiles), dtype=boolean)
    for d in range(4):
        for t in range(num_tiles):
            rules[d, t] = adjacency_rules[d][t]

    weights = weights.astype(float32)
    return wfc_core(width, height, weights, rules, num_tiles)


# Example configuration
width, height = 48, 48
num_tiles = 16
weights = np.random.rand(num_tiles)
adjacency_rules = [
    # For each direction (up, right, down, left), provide allowed tiles
    [
        np.random.choice([True, False], num_tiles)
        for _ in range(num_tiles)
        for _ in range(4)
    ]
]

result = wave_function_collapse(width, height, num_tiles, weights, adjacency_rules)
print(result)
