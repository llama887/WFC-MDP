import random

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------------------
# 1. TILE DEFINITIONS (using symbols, edges, and image filenames for reference)
# ---------------------------------------------------------------------------------------

PAC_TILES = {
    " ": {
        "image": "pac_floor.png",
        "edges": {"U": "OPEN", "R": "OPEN", "D": "OPEN", "L": "OPEN"},
    },
    "═": {
        "image": "pac_wall_h.png",
        "edges": {"U": "OPEN", "R": "LINE", "D": "OPEN", "L": "LINE"},
    },
    "║": {
        "image": "pac_wall_v.png",
        "edges": {"U": "LINE", "R": "OPEN", "D": "LINE", "L": "OPEN"},
    },
    "╔": {
        "image": "pac_corner_tl.png",
        "edges": {"U": "OPEN", "R": "LINE", "D": "LINE", "L": "OPEN"},
    },
    "╗": {
        "image": "pac_corner_tr.png",
        "edges": {"U": "OPEN", "R": "OPEN", "D": "LINE", "L": "LINE"},
    },
    "╚": {
        "image": "pac_corner_bl.png",
        "edges": {"U": "LINE", "R": "LINE", "D": "OPEN", "L": "OPEN"},
    },
    "╝": {
        "image": "pac_corner_br.png",
        "edges": {"U": "LINE", "R": "OPEN", "D": "OPEN", "L": "LINE"},
    },
}

# Opposite directions used for edge compatibility.
OPPOSITE_DIRECTION = {"U": "D", "D": "U", "L": "R", "R": "L"}
DIRECTIONS = ["U", "R", "D", "L"]

# Create a fixed order for tiles and a mapping from symbol to index.
tile_symbols = [" ", "═", "║", "╔", "╗", "╚", "╝"]
num_tiles = len(tile_symbols)
tile_to_index = {s: i for i, s in enumerate(tile_symbols)}

# ---------------------------------------------------------------------------------------
# 2. PRECOMPUTE ADJACENCY MATRIX (as a Boolean NumPy array)
# ---------------------------------------------------------------------------------------

# Build a boolean array of shape (num_tiles, 4, num_tiles). For each tile index i
# and direction d (0:U, 1:R, 2:D, 3:L), a True value for index j indicates that tile j
# is allowed as a neighbor.
adjacency_bool = np.zeros((num_tiles, 4, num_tiles), dtype=np.bool_)

for i, tile_a in enumerate(tile_symbols):
    for d, direction in enumerate(DIRECTIONS):
        for j, tile_b in enumerate(tile_symbols):
            edge_a = PAC_TILES[tile_a]["edges"][direction]
            edge_b = PAC_TILES[tile_b]["edges"][OPPOSITE_DIRECTION[direction]]
            if edge_a == edge_b:
                adjacency_bool[i, d, j] = True

# ---------------------------------------------------------------------------------------
# 3. FORCED BOUNDARIES
# ---------------------------------------------------------------------------------------


def get_forced_boundaries(width: int, height: int):
    """
    Returns a list of forced boundary cells as tuples (x, y, tile_index) based on the original rules.
    """
    boundaries = []
    # Corners.
    boundaries.append((0, 0, tile_to_index["╔"]))
    boundaries.append((width - 1, 0, tile_to_index["╗"]))
    boundaries.append((0, height - 1, tile_to_index["╚"]))
    boundaries.append((width - 1, height - 1, tile_to_index["╝"]))
    # Top and bottom borders (excluding corners).
    for x in range(1, width - 1):
        boundaries.append((x, 0, tile_to_index["═"]))
        boundaries.append((x, height - 1, tile_to_index["═"]))
    # Left and right borders (excluding corners).
    for y in range(1, height - 1):
        boundaries.append((0, y, tile_to_index["║"]))
        boundaries.append((width - 1, y, tile_to_index["║"]))
    return boundaries


# ---------------------------------------------------------------------------------------
# 4. NUMBA-ACCELERATED FUNCTIONS
# ---------------------------------------------------------------------------------------


@njit
def find_lowest_entropy_cell(grid, height, width, num_tiles):
    """
    Returns the non-collapsed cell (i.e. one with >1 possibility) with the fewest possibilities.

    Returns:
      (-2, -2) if a contradiction (cell with zero possibilities) is found.
      (-1, -1) if all cells are collapsed.
      Otherwise, returns (x, y) of the selected cell.
    """
    best_count = 10**9
    best_x = -1
    best_y = -1
    all_collapsed = True
    for y in range(height):
        for x in range(width):
            count = 0
            for t in range(num_tiles):
                if grid[y, x, t]:
                    count += 1
            if count == 0:
                return -2, -2  # Contradiction.
            if count > 1 and count < best_count:
                best_count = count
                best_x = x
                best_y = y
                all_collapsed = False
    if all_collapsed:
        return -1, -1
    return best_x, best_y


@njit
def choose_tile_with_action(grid, x, y, num_tiles, action, deterministic):
    """
    Chooses a tile index from cell (x, y) based on the given probabilities in 'action'.

    If deterministic is False:
      - Chooses randomly with weights given by action.
    If deterministic is True:
      - Chooses the tile with the highest weight among the possible ones.

    'action' is expected to be a 1D NumPy array of floats with length num_tiles.
    """
    if deterministic:
        max_weight = -1.0
        chosen = -1
        for t in range(num_tiles):
            if grid[y, x, t] and action[t] > max_weight:
                max_weight = action[t]
                chosen = t
        return chosen
    else:
        total_weight = 0.0
        for t in range(num_tiles):
            if grid[y, x, t]:
                total_weight += action[t]
        # If for some reason total_weight is zero, revert to uniform random selection.
        if total_weight <= 0.0:
            count = 0
            for t in range(num_tiles):
                if grid[y, x, t]:
                    count += 1
            target = np.random.randint(0, count)
            current = 0
            for t in range(num_tiles):
                if grid[y, x, t]:
                    if current == target:
                        return t
                    current += 1
            return -1  # Fallback
        else:
            # Weighted random selection.
            rand_val = np.random.random() * total_weight
            running_sum = 0.0
            for t in range(num_tiles):
                if grid[y, x, t]:
                    running_sum += action[t]
                    if running_sum >= rand_val:
                        return t
            return -1  # Fallback


@njit
def propagate_from_cell(
    grid, width, height, adjacency_bool, num_tiles, start_x, start_y
):
    """
    Propagates the constraints starting from cell (start_x, start_y) using a work queue.

    Returns True if propagation completes successfully or False if a contradiction is detected.
    """
    queue = np.empty((height * width, 2), dtype=np.int64)
    head = 0
    tail = 0
    queue[tail, 0] = start_x
    queue[tail, 1] = start_y
    tail += 1

    while head < tail:
        x = queue[head, 0]
        y = queue[head, 1]
        head += 1

        for d in range(4):
            if d == 0:
                nx = x
                ny = y - 1
            elif d == 1:
                nx = x + 1
                ny = y
            elif d == 2:
                nx = x
                ny = y + 1
            else:
                nx = x - 1
                ny = y
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            new_allowed = np.zeros(num_tiles, dtype=np.bool_)
            for p in range(num_tiles):
                if grid[y, x, p]:
                    for q in range(num_tiles):
                        if adjacency_bool[p, d, q]:
                            new_allowed[q] = True
            changed = False
            for q in range(num_tiles):
                new_val = grid[ny, nx, q] and new_allowed[q]
                if new_val != grid[ny, nx, q]:
                    grid[ny, nx, q] = new_val
                    changed = True
            if changed:
                has_possible = False
                for q in range(num_tiles):
                    if grid[ny, nx, q]:
                        has_possible = True
                        break
                if not has_possible:
                    return False
                queue[tail, 0] = nx
                queue[tail, 1] = ny
                tail += 1
    return True


def fast_wfc_collapse_step(grid, width, height, num_tiles, adjacency_bool, action, deterministic=False):
    """
    Performs a single collapse step on the given grid using the provided action vector.
    Returns (updated_grid, truncate) where truncate=True signals a contradiction.
    """
    x, y = find_lowest_entropy_cell(grid, height, width, num_tiles)
    if x == -2 and y == -2:
        # Contradiction detected.
        return grid, True
    if x == -1 and y == -1:
        # All cells are collapsed.
        return grid, False
    chosen = choose_tile_with_action(grid, x, y, num_tiles, action, deterministic)
    # Collapse the cell (set only the chosen possibility to True)
    for t in range(num_tiles):
        grid[y, x, t] = False
    grid[y, x, chosen] = True
    # Propagate constraints from the collapsed cell.
    if not propagate_from_cell(grid, width, height, adjacency_bool, num_tiles, x, y):
        return grid, True
    return grid, False


# ---------------------------------------------------------------------------------------
# 5. WAVE FUNCTION COLLAPSE (OPTIMIZED VERSION WITH ACTION)
# ---------------------------------------------------------------------------------------


def wave_function_collapse_optimized(
    width, height, adjacency_bool, num_tiles, forced_boundaries, deterministic=False
):
    """
    Performs the WFC algorithm using an optimized constraint propagation on a boolean grid.
    The collapse of each cell uses an action vector (a probability distribution) to
    influence which tile is chosen.

    Returns the collapsed grid if successful, or None if a contradiction occurs.
    """
    # Initialize grid: each cell starts with all possibilities (True).
    grid = np.ones((height, width, num_tiles), dtype=np.bool_)

    # Apply forced boundaries.
    for x, y, t in forced_boundaries:
        grid[y, x, :] = False
        grid[y, x, t] = True

    # Propagate constraints from forced boundary cells.
    for x, y, _ in forced_boundaries:
        if not propagate_from_cell(
            grid, width, height, adjacency_bool, num_tiles, x, y
        ):
            return None

    # Main collapse loop.
    while True:
        x, y = find_lowest_entropy_cell(grid, height, width, num_tiles)
        if x == -2 and y == -2:
            # A contradiction was detected.
            return None
        if x == -1 and y == -1:
            # All cells are collapsed.
            break

        # Create an action vector (weight for each tile). You can influence these values.
        action = np.empty(num_tiles, dtype=np.float64)
        for t in range(num_tiles):
            action[t] = random.random()  # Replace or modify as desired.

        chosen = choose_tile_with_action(grid, x, y, num_tiles, action, deterministic)

        # Collapse the cell at (x, y) to the chosen tile.
        for t in range(num_tiles):
            grid[y, x, t] = False
        grid[y, x, chosen] = True

        # Propagate constraints starting from the collapsed cell.
        if not propagate_from_cell(
            grid, width, height, adjacency_bool, num_tiles, x, y
        ):
            return None
    return grid


# ---------------------------------------------------------------------------------------
# 6. GENERATION WITH RETRIES
# ---------------------------------------------------------------------------------------


def generate_until_valid_optimized(
    width,
    height,
    adjacency_bool,
    num_tiles,
    forced_boundaries,
    max_attempts=50,
    deterministic=False,
):
    """
    Tries generating a valid layout up to max_attempts times.

    Returns the collapsed boolean grid if successful, or None otherwise.
    """
    for attempt in range(1, max_attempts + 1):
        result = wave_function_collapse_optimized(
            width, height, adjacency_bool, num_tiles, forced_boundaries, deterministic
        )
        if result is not None:
            print(f"Success on attempt {attempt}!")
            return result
    print("All attempts failed.")
    return None


# ---------------------------------------------------------------------------------------
# 7. RESULT CONVERSION AND MAIN ENTRYPOINT
# ---------------------------------------------------------------------------------------


def grid_to_layout(grid, tile_symbols):
    """
    Converts the boolean grid into a 2D list of tile symbols.
    """
    height, width, _ = grid.shape
    layout = []
    for y in range(height):
        row = []
        for x in range(width):
            tile_index = -1
            for t in range(len(tile_symbols)):
                if grid[y, x, t]:
                    tile_index = t
                    break
            row.append(tile_symbols[tile_index])
        layout.append(row)
    return layout


if __name__ == "__main__":
    # Set map dimensions.
    WIDTH, HEIGHT = 20, 12
    forced_boundaries = get_forced_boundaries(WIDTH, HEIGHT)

    # Try to generate a valid layout.
    # The 'deterministic' flag can be set to True to always choose the highest weight.
    result_grid = generate_until_valid_optimized(
        WIDTH,
        HEIGHT,
        adjacency_bool,
        num_tiles,
        forced_boundaries,
        max_attempts=50,
        deterministic=False,
    )

    if result_grid is not None:
        layout = grid_to_layout(result_grid, tile_symbols)
        print("Final Layout:")
        for row in layout:
            print(" ".join(row))
    else:
        print("No layout could be generated.")
        print("No layout could be generated.")
