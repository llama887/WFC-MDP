import numpy as np
from numba import int32, njit


@njit(nogil=True)
def propagate_fast(grid, rules, width, height, num_tiles, start_x, start_y):
    # Preallocate a fixed‑size queue (maximum possible size = width * height)
    queue = np.empty((width * height, 2), dtype=np.int32)
    head = 0
    tail = 0
    queue[tail, 0] = start_x
    queue[tail, 1] = start_y
    tail += 1

    # Use a visited mask to avoid duplicate enqueues
    visited = np.zeros((height, width), dtype=np.bool_)
    visited[start_y, start_x] = True

    # Constant direction and opposites arrays
    directions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.int32)
    opposites = np.array([2, 3, 0, 1], dtype=np.int32)

    while head < tail:
        x = queue[head, 0]
        y = queue[head, 1]
        visited[y, x] = False  # Mark as processed
        head += 1

        for d in range(4):
            dx = directions[d, 0]
            dy = directions[d, 1]
            nx = x + dx
            ny = y + dy

            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            allowed = np.zeros(num_tiles, dtype=np.bool_)
            # Build the list of allowed tiles based on the current cell
            for t in range(num_tiles):
                if grid[y, x, t]:
                    for nt in range(num_tiles):
                        if rules[d, t, nt]:
                            allowed[nt] = True

            changed = False
            # Intersect neighbor's possibilities with allowed options
            for i in range(num_tiles):
                if grid[ny, nx, i] and not allowed[i]:
                    grid[ny, nx, i] = False
                    changed = True

            if changed and (not visited[ny, nx]):
                queue[tail, 0] = nx
                queue[tail, 1] = ny
                tail += 1
                visited[ny, nx] = True

    return grid


@njit(nogil=True)
def wfc_core(width, height, weights, rules, num_tiles):
    grid = np.ones((height, width, num_tiles), dtype=np.bool_)
    output = np.full((height, width), -1, dtype=int32)

    max_entropy = num_tiles + 1

    while True:
        entropy = grid.sum(axis=2)
        for i in range(entropy.shape[0]):
            for j in range(entropy.shape[1]):
                if entropy[i, j] == 1:
                    entropy[i, j] = max_entropy
        if np.all(entropy == max_entropy):
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

        grid = propagate_fast(grid, rules, width, height, num_tiles, x, y)

    for y in range(height):
        for x in range(width):
            if output[y, x] == -1:
                possible = np.where(grid[y, x])[0]
                output[y, x] = possible[0] if len(possible) > 0 else -1

    return output


def wave_function_collapse(width, height, num_tiles, weights, adjacency_rules):
    rules = np.zeros((4, num_tiles, num_tiles), dtype=np.bool_)
    for d in range(4):
        for t in range(num_tiles):
            rules[d, t] = adjacency_rules[d][t]

    weights = weights.astype(np.float32)
    return wfc_core(width, height, weights, rules, num_tiles)


# Example configuration
width, height = 48, 48
num_tiles = 16
weights = np.random.rand(num_tiles)
adjacency_rules = [
    [np.random.choice([True, False], num_tiles) for _ in range(num_tiles)]
    for _ in range(4)
]

result = wave_function_collapse(width, height, num_tiles, weights, adjacency_rules)
print(result)
