import random

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------------------
# 3. FORCED BOUNDARIES
# ---------------------------------------------------------------------------------------


def get_forced_boundaries(
    width: int, height: int, tile_to_index: dict[str, int]
) -> list[tuple[int, int, int]]:
    """
    Returns a list of forced boundary cells as tuples (x, y, tile_index).

    Parameters:
      width (int): The width of the grid.
      height (int): The height of the grid.
      tile_to_index (dict[str, int]): Mapping from tile symbols to their corresponding indices.

    Returns:
      List[tuple[int, int, int]]: Each tuple contains the x-coordinate, y-coordinate, and forced tile index.
    """
    boundaries: list[tuple[int, int, int]] = []
    # Top-left corner.
    boundaries.append((0, 0, tile_to_index["╔"]))
    # Top-right corner.
    boundaries.append((width - 1, 0, tile_to_index["╗"]))
    # Bottom-left corner.
    boundaries.append((0, height - 1, tile_to_index["╚"]))
    # Bottom-right corner.
    boundaries.append((width - 1, height - 1, tile_to_index["╝"]))
    # Top and bottom borders (excluding corners).
    for x in range(1, width - 1):
        boundaries.append((x, 0, tile_to_index["═"]))  # Top border
        boundaries.append((x, height - 1, tile_to_index["═"]))  # Bottom border
    # Left and right borders (excluding corners).
    for y in range(1, height - 1):
        boundaries.append((0, y, tile_to_index["║"]))  # Left border
        boundaries.append((width - 1, y, tile_to_index["║"]))  # Right border
    return boundaries


# ---------------------------------------------------------------------------------------
# 4. NUMBA-ACCELERATED FUNCTIONS
# ---------------------------------------------------------------------------------------


@njit
def find_lowest_entropy_cell(
    grid: np.ndarray, height: int, width: int, num_tiles: int
) -> tuple[int, int]:
    """
    Returns the non-collapsed cell (i.e. one with >1 possibility)
    with the fewest possibilities.
    Returns:
      (-2, -2) on contradiction; (-1, -1) if all cells are collapsed;
      Otherwise, (x, y) coordinates of the candidate cell.
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
def choose_tile_with_action(
    grid: np.ndarray,
    x: int,
    y: int,
    num_tiles: int,
    action: np.ndarray,
    deterministic: bool,
) -> int:
    """
    Chooses a tile index from cell (x, y) based on the given action vector.

    Parameters:
      grid (np.ndarray): The current grid of possibilities.
      x (int): x-coordinate of the cell.
      y (int): y-coordinate of the cell.
      num_tiles (int): Total number of tile types.
      action (np.ndarray): A 1D array holding weights/probabilities for each tile.
      deterministic (bool): If True, selects the tile with the highest weight.

    Returns:
      int: The chosen tile index.
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
    grid: np.ndarray,
    width: int,
    height: int,
    adjacency_bool: np.ndarray,
    num_tiles: int,
    start_x: int,
    start_y: int,
) -> bool:
    """
    Propagates constraints from the starting cell (start_x, start_y).

    Parameters:
      grid (np.ndarray): The current grid.
      width (int): Width of the grid.
      height (int): Height of the grid.
      adjacency_bool (np.ndarray): Boolean array defining allowed adjacencies.
      num_tiles (int): Number of tile types.
      start_x (int): x-coordinate of initial cell.
      start_y (int): y-coordinate of initial cell.

    Returns:
      bool: True if propagation succeeds, False if a contradiction is found.
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


def fast_wfc_collapse_step(
    grid: np.ndarray,
    width: int,
    height: int,
    num_tiles: int,
    adjacency_bool: np.ndarray,
    action: np.ndarray,
    deterministic: bool = False,
) -> tuple[np.ndarray, bool, bool]:
    """
    Performs a single collapse step using the given action vector.

    Parameters:
      grid (np.ndarray): 3D boolean grid representing cell possibilities.
      width (int): Map width.
      height (int): Map height.
      num_tiles (int): Total tile types.
      adjacency_bool (np.ndarray): Boolean array (num_tiles x 4 x num_tiles) for adjacencies.
      action (np.ndarray): 1D array of tile selection weights.
      deterministic (bool): If True, chooses tile by max weight.

    Returns:
      tuple: (updated grid, terminate flag, contradiction (truncate) flag)
    """
    x, y = find_lowest_entropy_cell(grid, height, width, num_tiles)
    if x == -2 and y == -2:
        # Contradiction detected.
        return grid, False, True
    if x == -1 and y == -1:
        # All cells are collapsed.
        return grid, True, False
    chosen = choose_tile_with_action(grid, x, y, num_tiles, action, deterministic)
    # Collapse the cell (set only the chosen possibility to True)
    for t in range(num_tiles):
        grid[y, x, t] = False
    grid[y, x, chosen] = True
    # Propagate constraints from the collapsed cell.
    if not propagate_from_cell(grid, width, height, adjacency_bool, num_tiles, x, y):
        return grid, False, True
    return grid, False, False


# ---------------------------------------------------------------------------------------
# 5. WAVE FUNCTION COLLAPSE (OPTIMIZED VERSION WITH ACTION)
# ---------------------------------------------------------------------------------------


def fast_wave_function_collapse(
    width: int,
    height: int,
    adjacency_bool: np.ndarray,
    num_tiles: int,
    forced_boundaries: list[tuple[int, int, int]],
    deterministic: bool = False,
) -> np.ndarray | None:
    """
    Executes the entire WFC algorithm with forced boundaries, collapsing cells
    based on the action vector until a valid layout is achieved or a contradiction occurs.

    Parameters:
      width (int): Map width.
      height (int): Map height.
      adjacency_bool (np.ndarray): Allowed adjacencies as a boolean array.
      num_tiles (int): Total tile types.
      forced_boundaries (list[tuple[int, int, int]]): List of forced cell positions and tile indices.
      deterministic (bool): If True, uses deterministic collapse.

    Returns:
      np.ndarray or None: The collapsed grid if successful, or None if a contradiction is found.
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
    width: int,
    height: int,
    adjacency_bool: np.ndarray,
    num_tiles: int,
    forced_boundaries: list[tuple[int, int, int]],
    max_attempts: int = 50,
    deterministic: bool = False,
) -> np.ndarray | None:
    """
    Attempts to generate a valid collapsed grid up to max_attempts times.

    Parameters:
      width (int): Map width.
      height (int): Map height.
      adjacency_bool (np.ndarray): Precomputed Boolean adjacency matrix.
      num_tiles (int): Number of tile types.
      forced_boundaries (list[tuple[int, int, int]]): Forced boundary positions.
      max_attempts (int): Maximum generation attempts.
      deterministic (bool): If True, uses deterministic collapse.

    Returns:
      np.ndarray or None: Valid collapsed grid or None if all attempts fail.
    """
    for attempt in range(1, max_attempts + 1):
        result = fast_wave_function_collapse(
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


def grid_to_layout(grid: np.ndarray, tile_symbols: list[str]) -> list[list[str]]:
    """
    Converts the boolean grid into a 2D list of tile symbols.

    Parameters:
      grid (np.ndarray): The collapsed grid with shape (height, width, num_tiles).
      tile_symbols (list[str]): List of tile symbols ordered by their indices.

    Returns:
      list[list[str]]: 2D layout in tile symbols.
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
    # ---------------------------------------------------------------------------------------
    # 1. TILE DEFINITIONS (using symbolsand edges)
    # ---------------------------------------------------------------------------------------

    PAC_TILES = {
        " ": {
            "edges": {"U": "OPEN", "R": "OPEN", "D": "OPEN", "L": "OPEN"},
        },
        "X": {
            "edges": {"U": "OPEN", "R": "OPEN", "D": "OPEN", "L": "OPEN"},
        },
        "═": {
            "edges": {"U": "OPEN", "R": "LINE", "D": "OPEN", "L": "LINE"},
        },
        "║": {
            "edges": {"U": "LINE", "R": "OPEN", "D": "LINE", "L": "OPEN"},
        },
        "╔": {
            "edges": {"U": "OPEN", "R": "LINE", "D": "LINE", "L": "OPEN"},
        },
        "╗": {
            "edges": {"U": "OPEN", "R": "OPEN", "D": "LINE", "L": "LINE"},
        },
        "╚": {
            "edges": {"U": "LINE", "R": "LINE", "D": "OPEN", "L": "OPEN"},
        },
        "╝": {
            "edges": {"U": "LINE", "R": "OPEN", "D": "OPEN", "L": "LINE"},
        },
    }

    # Opposite directions used for edge compatibility.
    OPPOSITE_DIRECTION = {"U": "D", "D": "U", "L": "R", "R": "L"}
    DIRECTIONS = ["U", "R", "D", "L"]

    # Create a fixed order for tiles and a mapping from symbol to index.
    tile_symbols = list(PAC_TILES.keys())
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

    # Set map dimensions.
    WIDTH, HEIGHT = 20, 12
    forced_boundaries = get_forced_boundaries(WIDTH, HEIGHT, tile_to_index)

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
