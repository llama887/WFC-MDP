import numpy as np

TILES = {
    "grass": 0, "sand": 1, "water": 2, "cliff": 3, "dirt_path": 4,
    "vegetation": 5, "waterfall": 6, "reed": 7, "rocky_shore": 8,
    "sandy_shore": 9, "overgrown_path": 10, "tall_grass": 11,
    "dry_bush": 12, "rock_pile": 13
}

OPPOSITE_DIRECTION = {"U": "D", "D": "U", "L": "R", "R": "L"}
DIRECTIONS = ["U", "D", "L", "R"]
NUM_DIRECTIONS = len(DIRECTIONS)
NUM_TILES = len(TILES)

def generate_adjacency_rules(biome_type):
    adjacency_rules = {}

    if biome_type == "combined_biome":
        adjacency_rules = {
            "grass": [0, 1, 2, 4, 5, 7, 11, 12],
            "sand": [0, 1, 9, 12],
            "water": [0, 2, 3, 6, 7, 8, 9],
            "cliff": [0, 3, 6, 8, 13],
            "dirt_path": [0, 4, 10, 11, 12],
            "vegetation": [0, 3, 5, 10, 11, 12],
            "reed": [0, 2],
            "rocky_shore": [3, 2],
            "sandy_shore": [1, 2],
            "overgrown_path": [4, 5, 11],
            "tall_grass": [0, 11],
            "dry_bush": [0, 1, 4, 12],
            "rock_pile": [3, 12]
        }

    return adjacency_rules

# Initialize adjacency matrix
adjacency_bool = np.zeros((NUM_TILES, NUM_DIRECTIONS, NUM_TILES), dtype=bool)

# Get adjacency rules
adjacency_rules = generate_adjacency_rules("combined_biome")

# Convert adjacency rules to boolean matrix
tile_names = list(TILES.keys())
for i, tile_a in enumerate(tile_names):
    if tile_a in adjacency_rules:
        allowed_adjacent = adjacency_rules[tile_a]
        for direction in range(NUM_DIRECTIONS):
            for j, tile_b in enumerate(tile_names):
                if tile_b in allowed_adjacent:
                    adjacency_bool[i, direction, j] = True

def print_adjacency_compatibility(adjacency_rules):
    for tile, allowed_adjacent in adjacency_rules.items():
        print(f"{tile.capitalize()}: {allowed_adjacent}")

print_adjacency_compatibility(adjacency_rules)