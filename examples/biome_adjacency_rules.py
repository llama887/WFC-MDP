import numpy as np

tile_ids = {
    "grass": 0, "sand": 1, "water": 2, "cliff": 3, "dirt_path": 4,
    "vegetation": 5, "waterfall": 6, "reed": 7, "rocky_shore": 8,
    "sandy_shore": 9, "overgrown_path": 10, "tall_grass": 11,
    "dry_bush": 12, "rock_pile": 13
}

directions = ["up", "down", "left", "right"]
num_tiles, num_directions = len(tile_ids), len(directions)

adjacency_rules = [[[] for _ in range(num_directions)] for _ in range(num_tiles)]

def generate_adjacency_rules(biome_type):
    if biome_type == "combined_biome":
        # Grass - Symmetric adjacency with related biomes
        for d in range(4):
            adjacency_rules[tile_ids["grass"]][d] = [0, 1, 2, 4, 5, 7, 11, 12]
        
        # Sand - Symmetry with sandy shores and adjacent biomes
        for d in range(4):
            adjacency_rules[tile_ids["sand"]][d] = [0, 1, 9, 12]

        # Water - Including reeds, rocky shores, and sandy shores
        for d in range(4):
            adjacency_rules[tile_ids["water"]][d] = [0, 2, 3, 6, 7, 8, 9]

        # Cliff - Symmetric with rocky shores and waterfalls
        for d in range(4):
            adjacency_rules[tile_ids["cliff"]][d] = [0, 3, 6, 8, 13]

        # Dirt Path - Overgrown path and compatible ground types
        for d in range(4):
            adjacency_rules[tile_ids["dirt_path"]][d] = [0, 4, 10, 11, 12]

        # Vegetation - Blending with natural terrain
        for d in range(4):
            adjacency_rules[tile_ids["vegetation"]][d] = [0, 3, 5, 10, 11, 12]

        # Reed - Specifically between grass and water
        for d in range(4):
            adjacency_rules[tile_ids["reed"]][d] = [0, 2]

        # Rocky Shore - Where cliffs meet water
        for d in range(4):
            adjacency_rules[tile_ids["rocky_shore"]][d] = [3, 2]

        # Sandy Shore - Specifically between sand and water
        for d in range(4):
            adjacency_rules[tile_ids["sandy_shore"]][d] = [1, 2]

        # Overgrown Path - Transition between dirt path and vegetation
        for d in range(4):
            adjacency_rules[tile_ids["overgrown_path"]][d] = [4, 5, 11]

        # Tall Grass - Appears mainly in savannah regions
        for d in range(4):
            adjacency_rules[tile_ids["tall_grass"]][d] = [0, 11]

        # Dry Bush - Scrubland feature, compatible with dirt and grass
        for d in range(4):
            adjacency_rules[tile_ids["dry_bush"]][d] = [0, 1, 4, 12]

        # Rock Pile - Cliffs and dry ground
        for d in range(4):
            adjacency_rules[tile_ids["rock_pile"]][d] = [3, 12]

def get_tile_id(tile_name):
    return tile_ids.get(tile_name, -1)

def get_adjacent_tile_types(tile_id, direction):
    return adjacency_rules[tile_id][direction]

generate_adjacency_rules("combined_biome")
print("Grass compatible with Up:", get_adjacent_tile_types(get_tile_id("grass"), 0))
print("Water compatible with Right:", get_adjacent_tile_types(get_tile_id("water"), 3))
print("Cliff compatible with Left:", get_adjacent_tile_types(get_tile_id("cliff"), 2))
print("Dirt Path compatible with Down:", get_adjacent_tile_types(get_tile_id("dirt_path"), 1))
