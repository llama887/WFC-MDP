import random

# -----------------------------------------
# TILE DEF
# -----------------------------------------
SOKOBAN_TILES = {
    "W": {  # Wall
        "image": "wall.png"
    },
    " ": {  # Floor
        "image": "floor.png"
    },
    "G": {  # Goal
        "image": "goal.png"
    },
    "B": {  # Box
        "image": "box.png"
    }
}

def build_sokoban_adjacency():
    """
    Simple approach: any tile can border any other tile.
    """
    tiles = list(SOKOBAN_TILES.keys())
    adjacency = {}
    for t in tiles:
        adjacency[t] = {"U": [], "R": [], "D": [], "L": []}
        for direction in ["U","R","D","L"]:
            adjacency[t][direction] = tiles
    return adjacency

# -----------------------------------------
# WFC
# -----------------------------------------
def wave_function_collapse(width, height, adjacency, forced_positions):
    """
    forced_positions: dict of (x,y)-> tile (e.g. 'B','G') that must appear at (x,y).
    Boundary cells forced to 'W' automatically.
    """
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            if x in (0, width-1) or y in (0,height-1):
                # boundary => wall
                row.append({"W"})
            else:
                # interior => all four, until we remove extras or force
                row.append(set(["W"," ","B","G"]))
        grid.append(row)

    for (fx, fy), tile_name in forced_positions.items():
        if tile_name not in grid[fy][fx]:
            return None
        grid[fy][fx] = {tile_name}

    dir_offsets = {"U": (0,-1), "R":(1,0), "D":(0,1), "L":(-1,0)}
    def neighbors(cx,cy):
        for d,(dx,dy) in dir_offsets.items():
            nx, ny = cx+dx, cy+dy
            if 0<=nx<width and 0<=ny<height:
                yield nx, ny, d

    def propagate():
        changed = True
        while changed:
            changed = False
            for yy in range(height):
                for xx in range(width):
                    if len(grid[yy][xx])==1:
                        tile_a = next(iter(grid[yy][xx]))
                        for nx, ny, direction in neighbors(xx, yy):
                            valid_neighbors = set()
                            for tile_b in grid[ny][nx]:
                                if tile_b in adjacency[tile_a][direction]:
                                    valid_neighbors.add(tile_b)
                            if valid_neighbors != grid[ny][nx]:
                                grid[ny][nx] = valid_neighbors
                                changed = True
                                if not valid_neighbors:
                                    return False
        return True

    while True:
        done = True
        best_len = float("inf")
        best_cell = None
        for yy in range(height):
            for xx in range(width):
                ccount = len(grid[yy][xx])
                if ccount==0:
                    return None
                elif ccount>1:
                    done = False
                    if ccount<best_len:
                        best_len = ccount
                        best_cell = (xx,yy)
        if done:
            break

        x_, y_ = best_cell
        candidates = list(grid[y_][x_])
        chosen = random.choice(candidates)
        grid[y_][x_] = {chosen}

        if not propagate():
            return None

    return [[next(iter(grid[r][c])) for c in range(width)] for r in range(height)]

# -----------------------------------------
# LOCK CHECK
# -----------------------------------------
def is_box_fully_enclosed(layout, x, y):
    """
    Check if all four orthogonal neighbors are walls
    => box is fully enclosed on all sides
    """
    height = len(layout)
    width = len(layout[0]) if height>0 else 0

    def is_wall(xx,yy):
        return 0<=xx<width and 0<=yy<height and layout[yy][xx]=='W'

    up = is_wall(x, y-1)
    right = is_wall(x+1, y)
    down = is_wall(x, y+1)
    left = is_wall(x-1, y)
    return (up and right and down and left)

def is_box_corner_locked(layout, x, y):
    """
    Check if box at (x,y) has two perpendicular walls: up+left, up+right,
    down+left, or down+right.
    """
    height = len(layout)
    width = len(layout[0]) if height>0 else 0

    def is_wall(xx,yy):
        return 0<=xx<width and 0<=yy<height and layout[yy][xx]=='W'

    # up+left
    if is_wall(x, y-1) and is_wall(x-1, y):
        return True
    # up+right
    if is_wall(x, y-1) and is_wall(x+1, y):
        return True
    # down+left
    if is_wall(x, y+1) and is_wall(x-1, y):
        return True
    # down+right
    if is_wall(x, y+1) and is_wall(x+1, y):
        return True

    return False

def layout_has_locked_box(layout):
    """
    Return True if there's any box that is fully enclosed or corner-locked
    """
    height = len(layout)
    width = len(layout[0]) if height>0 else 0
    for y in range(height):
        for x in range(width):
            if layout[y][x] == 'B':
                if is_box_fully_enclosed(layout, x, y):
                    return True
                if is_box_corner_locked(layout, x, y):
                    return True
    return False

# -----------------------------------------
# EXACT N Boxes & N Goals, no locked boxes
# -----------------------------------------
def generate_sokoban_layout(width, height, n=2, max_tries=50):
    """
    We place exactly n boxes, exactly n goals. Then we check no box is locked.
    If locked => discard and retry.
    """
    adjacency = build_sokoban_adjacency()
    for attempt in range(1, max_tries+1):
        interior = [(x,y) for x in range(1,width-1) for y in range(1,height-1)]
        if 2*n>len(interior):
            print("Not enough interior for n boxes + n goals.")
            return None

        random.shuffle(interior)
        box_cells = interior[:n]
        goal_cells = interior[n:2*n]

        # Build forced dict
        forced = {}
        for (bx,by) in box_cells:
            forced[(bx,by)] = 'B'
        for (gx,gy) in goal_cells:
            forced[(gx,gy)] = 'G'

        layout = wave_function_collapse_exact_B_G_no_lock(width, height, adjacency, forced)
        if layout is not None:
            # check locking
            if layout_has_locked_box(layout):
                continue
            print(f"Success on attempt {attempt} with exactly {n} boxes & {n} goals, none locked.")
            return layout

    print("All attempts failed.")
    return None

def wave_function_collapse_exact_B_G_no_lock(width, height, adjacency, forced_positions):
    """
    Variation that ensures we have EXACT n boxes & n goals where forced,
    and we remove B/G from everywhere else, so no extras.
    """
    # Similar to previous approach
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            if x in (0,width-1) or y in (0,height-1):
                # boundary => W
                row.append({"W"})
            else:
                if (x,y) in forced_positions:
                    row.append({forced_positions[(x,y)]})
                else:
                    # remove 'B' & 'G' so no extras
                    row.append({"W"," "})
        grid.append(row)

    # Immediately collapse forced
    for (fx,fy), tile_name in forced_positions.items():
        if tile_name not in grid[fy][fx]:
            return None  # contradiction
        grid[fy][fx] = {tile_name}

    # Standard WFC logic
    dir_offsets = {"U": (0,-1),"R":(1,0),"D":(0,1),"L":(-1,0)}
    def neighbors(cx,cy):
        for d,(dx,dy) in dir_offsets.items():
            nx, ny = cx+dx, cy+dy
            if 0<=nx<width and 0<=ny<height:
                yield nx, ny, d

    def propagate():
        changed = True
        while changed:
            changed=False
            for yy in range(height):
                for xx in range(width):
                    if len(grid[yy][xx])==1:
                        tile_a = next(iter(grid[yy][xx]))
                        for nx, ny, direction in neighbors(xx,yy):
                            valid = set()
                            for tile_b in grid[ny][nx]:
                                if tile_b in adjacency[tile_a][direction]:
                                    valid.add(tile_b)
                            if valid != grid[ny][nx]:
                                grid[ny][nx] = valid
                                changed=True
                                if not valid:
                                    return False
        return True

    while True:
        done=True
        best_len=float("inf")
        best_cell=None
        for yy in range(height):
            for xx in range(width):
                ccount=len(grid[yy][xx])
                if ccount==0:
                    return None
                elif ccount>1:
                    done=False
                    if ccount<best_len:
                        best_len=ccount
                        best_cell=(xx,yy)
        if done:
            break

        x_,y_=best_cell
        candidates=list(grid[y_][x_])
        chosen=random.choice(candidates)
        grid[y_][x_]={chosen}
        if not propagate():
            return None

    return [[next(iter(grid[r][c])) for c in range(width)] for r in range(height)]

# -----------------------------------------
# Demo
# -----------------------------------------
if __name__=="__main__":
    W, H = 10, 8
    N = 2
    layout = generate_sokoban_layout(W, H, N, max_tries=100)
    if layout is None:
        print("No valid layout after 100 tries.")
    else:
        for row in layout:
            print(" ".join(row))
