import random
from typing import Generator, Optional

# ---------------------------------------------------------------------------------------
# 1) TILE DEFINITIONS
# ---------------------------------------------------------------------------------------

# Dictionary of all possible tile types and their connection rules
PAC_TILES: dict[str, dict[str, object]] = {
    # Floor tile, open on all sides
    " ": {
        "image": "pac_floor.png",
        "edges": {"U": "OPEN", "R": "OPEN", "D": "OPEN", "L": "OPEN"},
    },
    # Horizontal wall, open top/bottom, line left/right
    "═": {
        "image": "pac_wall_h.png",
        "edges": {"U": "OPEN", "R": "LINE", "D": "OPEN", "L": "LINE"},
    },
    # Vertical wall, open left/right, line top/bottom
    "║": {
        "image": "pac_wall_v.png",
        "edges": {"U": "LINE", "R": "OPEN", "D": "LINE", "L": "OPEN"},
    },
    # Corners: top-left, top-right, bottom-left, bottom-right
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

# ---------------------------------------------------------------------------------------
# 2) ADJACENCY RULES
# ---------------------------------------------------------------------------------------

# Direction opposites used for edge compatibility checks
OPPOSITE_DIRECTION: dict[str, str] = {"U": "D", "D": "U", "L": "R", "R": "L"}


def tiles_compatible(
    tile_a: str, tile_b: str, direction: str, tile_defs: dict[str, dict[str, object]]
) -> bool:
    """
    Returns True if tile_b can be placed in 'direction' from tile_a,
    meaning tile_a's 'direction' edge must match tile_b's opposite edge.
    """
    edge_a = tile_defs[tile_a]["edges"][direction]
    edge_b = tile_defs[tile_b]["edges"][OPPOSITE_DIRECTION[direction]]
    return edge_a == edge_b


def build_pacman_adjacency(
    tile_defs: dict[str, dict[str, object]],
) -> dict[str, dict[str, list[str]]]:
    """
    For each tile, compute valid neighbors in each direction by checking edge compatibility.
    """
    adjacency_rules: dict[str, dict[str, list[str]]] = {}
    all_tiles = list(tile_defs.keys())

    for tile_a in all_tiles:
        adjacency_rules[tile_a] = {d: [] for d in "URDL"}
        for direction in "URDL":
            valid_neighbors = []
            for tile_b in all_tiles:
                if tiles_compatible(tile_a, tile_b, direction, tile_defs):
                    valid_neighbors.append(tile_b)
            adjacency_rules[tile_a][direction] = valid_neighbors

    return adjacency_rules


# ---------------------------------------------------------------------------------------
# 3) WAVE FUNCTION COLLAPSE (WFC)
# ---------------------------------------------------------------------------------------


def wave_function_collapse(
    width: int,
    height: int,
    adjacency: dict[str, dict[str, list[str]]],
    tile_defs: dict[str, dict[str, object]],
) -> Optional[list[list[str]]]:
    """
    Executes the WFC algorithm to generate a map:
    - Starts with all possibilities in each cell
    - Collapses cells one-by-one and propagates constraints
    - Returns a valid layout or None if a contradiction occurs
    """
    all_tiles = list(tile_defs.keys())
    grid: list[list[set[str]]] = [
        [set(all_tiles) for _ in range(width)] for _ in range(height)
    ]

    def force_cell(x: int, y: int, tile: str) -> None:
        """Force a specific tile at (x, y), replacing all possibilities."""
        grid[y][x] = {tile}

    # Force boundary tiles
    force_cell(0, 0, "╔")
    force_cell(width - 1, 0, "╗")
    force_cell(0, height - 1, "╚")
    force_cell(width - 1, height - 1, "╝")
    for x in range(1, width - 1):
        force_cell(x, 0, "═")
        force_cell(x, height - 1, "═")
    for y in range(1, height - 1):
        force_cell(0, y, "║")
        force_cell(width - 1, y, "║")

    # Main WFC loop
    while True:
        action = {
            "╚": random.random(),
            "║": random.random(),
            "╝": random.random(),
            "╔": random.random(),
            "═": random.random(),
            " ": random.random(),
            "╗": random.random(),
        }
        best_cell, all_collapsed = wfc_next_collapse_position(grid)
        if all_collapsed:
            break
        grid, truncate = wfc_collapse(grid, best_cell, adjacency, action)
        if truncate:
            return None

    return [[next(iter(grid[y][x])) for x in range(width)] for y in range(height)]


def wfc_next_collapse_position(
    grid: list[list[set[str]]],
) -> tuple[tuple[int, int], bool]:
    all_collapsed = True
    min_options = float("inf")
    best_cell: Optional[tuple[int, int]] = None
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            num_options = len(grid[y][x])
            if num_options == 0:
                return None  # Contradiction
            elif num_options > 1:
                all_collapsed = False
                if num_options < min_options:
                    min_options = num_options
                    best_cell = (x, y)
    return best_cell, all_collapsed


def wfc_collapse(
    grid: list[list[set[str]]],
    best_cell: tuple[int, int],
    adjacency: dict[str, dict[str, list[str]]],
    action: dict[str, float],
    deterministic: bool = False,
) -> tuple[list[list[set[str]]], bool]:
    direction_offsets: dict[str, tuple[int, int]] = {
        "U": (0, -1),
        "R": (1, 0),
        "D": (0, 1),
        "L": (-1, 0),
    }

    def get_neighbors(x: int, y: int) -> Generator[tuple[int, int, str], None, None]:
        """Yield (neighbor_x, neighbor_y, direction) for all valid neighbors of (x, y)."""
        for direction, (dx, dy) in direction_offsets.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                yield nx, ny, direction

    def propagate_constraints() -> bool:
        """
        Applies constraint propagation to update valid tile options in all cells.
        Returns False if any cell becomes invalid (empty set).
        """
        changed = True
        while changed:
            changed = False
            for y in range(len(grid)):
                for x in range(len(grid[0])):
                    if len(grid[y][x]) == 1:
                        tile = next(iter(grid[y][x]))
                        for nx, ny, direction in get_neighbors(x, y):
                            allowed = {
                                t
                                for t in grid[ny][nx]
                                if t in adjacency[tile][direction]
                            }
                            if allowed != grid[ny][nx]:
                                grid[ny][nx] = allowed
                                changed = True
                                if not allowed:
                                    return False
        return True

    x, y = best_cell
    masked_action = [(k, v) for k, v in action.items() if k in grid[y][x]]
    if not deterministic:
        tiles, weights = zip(*masked_action)
        chosen = random.choices(tiles, weights=weights, k=1)[0]
    else:
        chosen = max(masked_action, key=lambda x: x[1])[0]
    grid[y][x] = {chosen}

    if not propagate_constraints():
        return grid, True

    return grid, False


# ---------------------------------------------------------------------------------------
# 4) GENERATION WITH RETRIES
# ---------------------------------------------------------------------------------------


def generate_until_valid(
    width: int,
    height: int,
    adjacency: dict[str, dict[str, list[str]]],
    tile_defs: dict[str, dict[str, object]],
    max_attempts: int = 50,
) -> Optional[list[list[str]]]:
    """
    Tries generating a valid layout up to 'max_attempts' times.
    Returns a completed layout if successful, or None if all attempts fail.
    """
    for attempt in range(1, max_attempts + 1):
        layout = wave_function_collapse(width, height, adjacency, tile_defs)
        if layout is not None:
            print(f"Success on attempt {attempt}!")
            return layout
    print("All attempts failed.")
    return None


# ---------------------------------------------------------------------------------------
# 5) DEMO ENTRYPOINT
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: Build adjacency rules based on tile edge matching
    adjacency_rules = build_pacman_adjacency(PAC_TILES)

    # Step 2: Set map dimensions
    WIDTH, HEIGHT = 20, 12

    # Step 3: Try to generate a valid layout with retries
    result = generate_until_valid(WIDTH, HEIGHT, adjacency_rules, PAC_TILES)

    # Step 4: Display result
    if result:
        print("Final Layout:")
        for row in result:
            print(" ".join(row))
    else:
        print("No layout could be generated.")
