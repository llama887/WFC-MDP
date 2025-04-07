import random

# ---------------------------------------------------------------------------------------
# TILE DEF
# ---------------------------------------------------------------------------------------

PAC_TILES = {
    # Floor tile, open on all sides
    " ": {
        "image": "pac_floor.png",
        "edges": {"U": "OPEN", "R": "OPEN", "D": "OPEN", "L": "OPEN"}
    },
    # Horizontal wall (line on left/right, open on top/bottom)
    "═": {
        "image": "pac_wall_h.png",
        "edges": {"U": "OPEN", "R": "LINE", "D": "OPEN", "L": "LINE"}
    },
    # Vertical wall (line on top/bottom, open on left/right)
    "║": {
        "image": "pac_wall_v.png",
        "edges": {"U": "LINE", "R": "OPEN", "D": "LINE", "L": "OPEN"}
    },
    # Top-left corner: open top/left, line right/bottom
    "╔": {
        "image": "pac_corner_tl.png",
        "edges": {"U": "OPEN", "R": "LINE", "D": "LINE", "L": "OPEN"}
    },
    # Top-right corner: open top/right, line left/bottom
    "╗": {
        "image": "pac_corner_tr.png",
        "edges": {"U": "OPEN", "R": "OPEN", "D": "LINE", "L": "LINE"}
    },
    # Bottom-left corner
    "╚": {
        "image": "pac_corner_bl.png",
        "edges": {"U": "LINE", "R": "LINE", "D": "OPEN", "L": "OPEN"}
    },
    # Bottom-right corner
    "╝": {
        "image": "pac_corner_br.png",
        "edges": {"U": "LINE", "R": "OPEN", "D": "OPEN", "L": "LINE"}
    },
}

# ---------------------------------------------------------------------------------------
# 2) ADJACENCY RULES
# ---------------------------------------------------------------------------------------

OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

def tiles_compatible(tile_a, tile_b, direction, tile_defs):
    """
    Returns True if tile_b can be placed in 'direction' from tile_a,
    meaning tile_a's 'direction' edge must match tile_b's opposite edge.
    """
    edge_a = tile_defs[tile_a]["edges"][direction]
    edge_b = tile_defs[tile_b]["edges"][OPPOSITE[direction]]
    return (edge_a == edge_b)

def build_pacman_adjacency(tile_defs):
    """
    For each tile, gather which tiles can appear in each direction
    by matching edge labels exactly.
    """
    adjacency = {}
    all_tiles = list(tile_defs.keys())

    for t_a in all_tiles:
        adjacency[t_a] = {"U": [], "R": [], "D": [], "L": []}
        for d in ["U", "R", "D", "L"]:
            valid_neighbors = []
            for t_b in all_tiles:
                if tiles_compatible(t_a, t_b, d, tile_defs):
                    valid_neighbors.append(t_b)
            adjacency[t_a][d] = valid_neighbors

    return adjacency

# ---------------------------------------------------------------------------------------
# 3) WFC
# ---------------------------------------------------------------------------------------

def wave_function_collapse(width, height, adjacency, tile_defs):
    """
    1) Each cell starts with all tiles.
    2) Force boundary cells:
       - corners get their exact corner tile
       - top/bottom edges (excluding corners) get WALL_H
       - left/right edges (excluding corners) get WALL_V
    3) Randomly collapse the cell with fewest candidates, propagate constraints.
    4) Return None if contradiction, or 2D layout if successful.
    """

    all_tiles = list(tile_defs.keys())
    grid = [[set(all_tiles) for _ in range(width)] for _ in range(height)]

    def force_cell(x, y, tile_name):
        grid[y][x] = {tile_name}

    # Force corners
    force_cell(0,          0,          "╔")  # top-left
    force_cell(width - 1,  0,          "╗")  # top-right
    force_cell(0,          height - 1, "╚")  # bottom-left
    force_cell(width - 1,  height - 1, "╝")  # bottom-right

    # Force top/bottom edges (excluding corners) to WALL_H
    for x in range(1, width - 1):
        force_cell(x, 0, "═")
        force_cell(x, height - 1, "═")

    # Force left/right edges (excluding corners) to WALL_V
    for y in range(1, height - 1):
        force_cell(0, y, "║")
        force_cell(width - 1, y, "║")

    dir_offsets = {"U": (0, -1), "R": (1, 0), "D": (0, 1), "L": (-1, 0)}

    def neighbors(cx, cy):
        for direction, (dx, dy) in dir_offsets.items():
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny, direction

    def propagate():
        changed = True
        while changed:
            changed = False
            for y_ in range(height):
                for x_ in range(width):
                    if len(grid[y_][x_]) == 1:
                        tile_a = next(iter(grid[y_][x_]))
                        for nx, ny, direction in neighbors(x_, y_):
                            valid_neighbors = set()
                            for tile_b in grid[ny][nx]:
                                if tile_b in adjacency[tile_a][direction]:
                                    valid_neighbors.add(tile_b)
                            if valid_neighbors != grid[ny][nx]:
                                grid[ny][nx] = valid_neighbors
                                changed = True
                                if not valid_neighbors:
                                    return False  # Contradiction
        return True

    # WFC loop
    while True:
        done = True
        best_len = float("inf")
        best_cell = None

        for y_ in range(height):
            for x_ in range(width):
                ccount = len(grid[y_][x_])
                if ccount == 0:
                    return None  # Contradiction
                elif ccount > 1:
                    done = False
                    if ccount < best_len:
                        best_len = ccount
                        best_cell = (x_, y_)

        if done:
            break  # all cells collapsed

        # Collapse the most constrained cell
        x_, y_ = best_cell
        candidates = list(grid[y_][x_])
        chosen_tile = random.choice(candidates)
        grid[y_][x_] = {chosen_tile}

        # Propagate constraints
        if not propagate():
            return None

    # Convert final sets to single tile names
    return [[next(iter(grid[row][col])) for col in range(width)] for row in range(height)]

# ---------------------------------------------------------------------------------------
# GENERATION UNTIL SUCCESS
# ---------------------------------------------------------------------------------------

def generate_until_valid(width, height, adjacency, tile_defs, max_attempts=50):
    """
    Attempts wave_function_collapse up to 'max_attempts' times.
    Returns the first successful layout (non-None).
    If all attempts fail (contradictions), returns None.
    """
    for attempt in range(1, max_attempts + 1):
        layout = wave_function_collapse(width, height, adjacency, tile_defs)
        if layout is not None:
            print(f"Success on attempt {attempt}!")
            return layout
    print("All attempts failed. No valid layout found.")
    return None

# ---------------------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Build adjacency rules
    adjacency_rules = build_pacman_adjacency(PAC_TILES)

    # Let’s pick a bigger map size
    WIDTH, HEIGHT = 20, 12

    # Keep trying until success or we exhaust attempts
    final_layout = generate_until_valid(WIDTH, HEIGHT, adjacency_rules, PAC_TILES, max_attempts=50)

    if final_layout is None:
        print("No layout could be generated after 50 attempts.")
    else:
        # Print the final layout
        print("Final Layout:")
        for row in final_layout:
            print(" ".join(row))
