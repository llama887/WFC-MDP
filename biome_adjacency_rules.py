import numpy as np

import pygame
import numpy as np
import os

TILES = {
    # grass tiles
    "grass": {
        "edges": {
            "U": ["grass", "tall_grass", "sand_bl", "sand_b", "sand_br", "water_bl", "water_b", "water_br", "flower", "grass_hill_b"], 
            "D": ["grass", "tall_grass", "sand_tl", "sand_t", "sand_tr", "water_tl", "water_t", "water_tr", "flower", "grass_hill_t"],
            "L": ["grass", "tall_grass", "sand_tr", "sand_r", "sand_br", "water_tr", "water_r", "water_br", "flower", "grass_hill_r", "grass_hill_tr", "grass_hill_br"], 
            "R": ["grass", "tall_grass", "sand_tl", "sand_l", "sand_bl", "water_tl", "water_l", "water_bl", "flower", "grass_hill_l", "grass_hill_tl", "grass_hill_bl"],
        },
        "image": "tiles_32x32_B/tile_0_1.png"
    },
    "tall_grass": {
        "edges": {
            "U": ["tall_grass", "grass", "flower"], 
            "D": ["tall_grass", "grass", "flower"],
            "L": ["tall_grass", "grass", "flower"], 
            "R": ["tall_grass", "grass", "flower"],
        },
        "image": "tiles_32x32_B/tile_0_2.png"
    },

    # flower tile
    "flower": {
        "edges": {
            "U": ["grass", "tall_grass", "grass_hill_t"], 
            "D": ["grass", "tall_grass", "grass_hill_b"],
            "L": ["grass", "tall_grass", "grass_hill_l"], 
            "R": ["grass", "tall_grass", "grass_hill_r"],
        },
        "image": "tiles_32x32_B/tile_13_1.png"
    },

    # hill tiles
    "grass_hill_tl": {  # top left
        "edges": {
            "U": ["grass", "grass_hill_br", "grass_hill_bl", "grass_hill_b"], 
            "D": ["grass_hill_l", "grass_hill_bl"],
            "L": ["grass", "grass_hill_br", "grass_hill_tr"], 
            "R": ["grass_hill_t", "grass_hill_tr"],
        },
        "image": "tiles_32x32_B/tile_12_0.png"
    },
    "grass_hill_t": {   # top
        "edges": {
            "U": ["grass", "grass_hill_b", "grass_hill_br", "grass_hill_bl"], 
            "D": ["grass", "flower", "grass_hill_b"],
            "L": ["grass_hill_tl", "grass_hill_t"], 
            "R": ["grass_hill_tr", "grass_hill_t"],
        },
        "image": "tiles_32x32_B/tile_12_1.png"
    },
    "grass_hill_tr": {   # top right
        "edges": {
            "U": ["grass", "grass_hill_br", "grass_hill_bl", "grass_hill_b"], 
            "D": ["grass_hill_r", "grass_hill_br"],
            "L": ["grass_hill_t", "grass_hill_tl"], 
            "R": ["grass", "grass_hill_bl", "grass_hill_tl"],
        },
        "image": "tiles_32x32_B/tile_12_2.png"
    },
    "grass_hill_l": {   # left
        "edges": {
            "U": ["grass_hill_l", "grass_hill_tl"], 
            "D": ["grass_hill_l", "grass_hill_bl"],
            "L": ["grass", "grass_hill_br", "grass_hill_tr"], 
            "R": ["grass", "flower", "grass_hill_r"],
        },
        "image": "tiles_32x32_B/tile_13_0.png"
    },
    "grass_hill_r": {   # right
        "edges": {
            "U": ["grass_hill_r", "grass_hill_tr"], 
            "D": ["grass_hill_r", "grass_hill_br"],
            "L": ["grass", "flower", "grass_hill_l"], 
            "R": ["grass", "grass_hill_bl", "grass_hill_tl"],
        },
        "image": "tiles_32x32_B/tile_13_2.png"
    },
    "grass_hill_bl": {   # bottom left
        "edges": {
            "U": ["grass_hill_l", "grass_hill_tl"], 
            "D": ["grass", "grass_hill_tr", "grass_hill_tl", "grass_hill_t"],
            "L": ["grass", "grass_hill_tr", "grass_hill_br"], 
            "R": ["grass_hill_b", "grass_hill_br"],
        },
        "image": "tiles_32x32_B/tile_14_0.png"
    },
    "grass_hill_b": {   # bottom
        "edges": {
            "U": ["grass", "flower", "grass_hill_t"], 
            "D": ["grass", "grass_hill_tr", "grass_hill_tl", "grass_hill_t"],
            "L": ["grass_hill_bl", "grass_hill_b"], 
            "R": ["grass_hill_br", "grass_hill_b"],
        },
        "image": "tiles_32x32_B/tile_14_1.png"
    },
    "grass_hill_br": {   # bottom right
        "edges": {
            "U": ["grass_hill_r", "grass_hill_tr"], 
            "D": ["grass", "grass_hill_tr", "grass_hill_tl", "grass_hill_t"],
            "L": ["grass_hill_b", "grass_hill_bl"], 
            "R": ["grass", "grass_hill_tl", "grass_hill_bl"],
        },
        "image": "tiles_32x32_B/tile_14_2.png"
    },

    # sand tiles
    "sand_1": {
        "edges": {
            "U": ["sand_1", "sand_2", "sand"],
            "D": ["sand_1", "sand_2", "sand"],
            "L": ["sand_1", "sand_2", "sand"],
            "R": ["sand_1", "sand_2", "sand"],
        },
        "image": "tiles_32x32_B/tile_0_3.png"
    },
    "sand_2": {
        "edges": {
            "U": ["sand_2", "sand_1", "sand"],
            "D": ["sand_2", "sand_1", "sand"],
            "L": ["sand_2", "sand_1", "sand"],
            "R": ["sand_2", "sand_1", "sand"],
        },
        "image": "tiles_32x32_B/tile_0_4.png"
    },
    "sand_tl": {    # top left
        "edges": {
            "U": ["grass"],
            "D": ["sand_l", "sand_bl"],
            "L": ["grass"],
            "R": ["sand_t", "sand_tr"],
        },
        "image": "tiles_32x32_B/tile_1_0.png"
    },
    "sand_t": {   # top
        "edges": {
            "U": ["grass"],
            "D": ["sand", "sand_1", "sand_2", "sand_b"],
            "L": ["sand_tl", "sand_t"],
            "R": ["sand_t", "sand_tr"],
        },
        "image": "tiles_32x32_B/tile_1_1.png"
    },
    "sand_tr": {   # top right
        "edges": {
            "U": ["grass"],
            "D": ["sand_r", "sand_br"],
            "L": ["sand_t", "sand_tl"],
            "R": ["grass"],
        },
        "image": "tiles_32x32_B/tile_1_2.png"
    },
    "sand_l": {   # left
        "edges": {
            "U": ["sand_tl", "sand_l"],
            "D": ["sand_l", "sand_bl"],
            "L": ["grass"],
            "R": ["sand", "sand_1", "sand_2", "sand_r"],
        },
        "image": "tiles_32x32_B/tile_2_0.png"
    },
    "sand": {
        "edges": {
            "U": ["sand", "sand_1", "sand_2", "sand_t"],
            "D": ["sand", "sand_1", "sand_2", "sand_b"],
            "L": ["sand_l", "sand", "sand_1", "sand_2"],
            "R": ["sand_r", "sand", "sand_1", "sand_2"],
        },
        "image": "tiles_32x32_B/tile_2_1.png"
    },
    "sand_r": {   # right
        "edges": {
            "U": ["sand_r", "sand_tr"],
            "D": ["sand_r", "sand_br"],
            "L": ["sand", "sand_1", "sand_2", "sand_l"],
            "R": ["grass"],
        },
        "image": "tiles_32x32_B/tile_2_2.png"
    },
    "sand_bl": {   # bottom left
        "edges": {
            "U": ["sand_l", "sand_tl"],
            "D": ["grass"],
            "L": ["grass"],
            "R": ["sand_b", "sand_br"],
        },
        "image": "tiles_32x32_B/tile_3_0.png"
    },
    "sand_b": {   # bottom
        "edges": {
            "U": ["sand_t", "sand", "sand_1", "sand_2"],
            "D": ["grass"],
            "L": ["sand_b", "sand_bl"],
            "R": ["sand_b", "sand_br"],
        },
        "image": "tiles_32x32_B/tile_3_1.png"
    },
    "sand_br": {   # bottom right
        "edges": {
            "U": ["sand_r", "sand_tr"],
            "D": ["grass"],
            "L": ["sand_b", "sand_bl"],
            "R": ["grass"],
        },
        "image": "tiles_32x32_B/tile_3_2.png"
    },

    # path tiles
    "path_tl": {    # top left
        "edges": {
            "U": ["sand_l", "sand_tl"],
            "D": ["sand", "sand_1", "sand_2", "sand_b", "path_bl", "path_br"],
            "L": ["sand_tl", "sand_t"],
            "R": ["sand", "sand_1", "sand_2", "sand_r", "path_tr", "path_br"],
        },
        "image": "tiles_32x32_B/tile_1_3.png"
    },
    "path_tr": {   # top right
        "edges": {
            "U": ["sand_r", "sand_tr"],
            "D": ["sand", "sand_1", "sand_2", "sand_b", "path_bl", "path_br"],
            "L": ["sand", "sand_1", "sand_2", "sand_l", "path_bl", "path_tl"],
            "R": ["sand_t", "sand_tr"],
        },
        "image": "tiles_32x32_B/tile_1_4.png"
    },
    "path_bl": {   # bottom left
        "edges": {
            "U": ["path_tl", "path_tr", "sand", "sand_1", "sand_2", "sand_t"],
            "D": ["sand_bl", "sand_l"],
            "L": ["sand_b", "sand_bl"],
            "R": ["sand", "sand_1", "sand_2", "path_tr", "path_br", "sand_r"],
        },
        "image": "tiles_32x32_B/tile_2_3.png"
    },
    "path_br": {   # bottom right
        "edges": {
            "U": ["path_tr", "path_tl", "sand", "sand_1", "sand_2", "sand_t"],
            "D": ["sand_r", "sand_br"],
            "L": ["sand", "sand_1", "sand_2", "path_bl", "path_tl", "sand_l"],
            "R": ["sand_b", "sand_br"],
        },
        "image": "tiles_32x32_B/tile_2_4.png"
    },
    "path_lr": {   # top left bottom right
        "edges": {
            "U": ["sand_l", "sand_tl"],
            "D": ["sand_r", "sand_br"],
            "L": ["sand_t", "sand_tl"],
            "R": ["sand_b", "sand_br"],
        },
        "image": "tiles_32x32_B/tile_3_3.png"
    },
    "path_rl": {   # top right bottom left
        "edges": {
            "U": ["sand_r", "sand_tr"],
            "D": ["sand_l", "sand_bl"],
            "L": ["sand_b", "sand_bl"],
            "R": ["sand_t", "sand_tr"],
        },
        "image": "tiles_32x32_B/tile_3_4.png"
    },

    # water tiles
    "water_tl": {   # top left
        "edges": {
            "U": ["grass", "water_br", "sand_br"],
            "D": ["water_l", "water_bl"],
            "L": ["grass"],
            "R": ["water_t", "water_tr"]
        },
        "image": "tiles_32x32_B/tile_4_0.png"
    },
    "water_t": {   # top
        "edges": {
            "U": ["grass"],
            "D": ["water", "water_b"],
            "L": ["water_tl", "water_t"],
            "R": ["water_tr", "water_t"]
        },
        "image": "tiles_32x32_B/tile_4_1.png"
    },
    "water_tr": {   # top right
        "edges": {
            "U": ["grass"],
            "D": ["water_r", "water_br"],
            "L": ["water_t", "water_tl"],
            "R": ["grass"]
        },
        "image": "tiles_32x32_B/tile_4_2.png"
    },
    "water_l": {   # left
        "edges": {
            "U": ["water_tl", "water_l"],
            "D": ["water_bl", "water_l"],
            "L": ["grass"],
            "R": ["water", "water_r"]
        },
        "image": "tiles_32x32_B/tile_5_0.png"
    },
    "water": {
        "edges": {
            "U": ["water_t", "water"],
            "D": ["water_b", "water"],
            "L": ["water_l", "water"],
            "R": ["water_r", "water"]
        },
        "image": "tiles_32x32_B/tile_5_1.png"
    },
    "water_r": {   # right
        "edges": {
            "U": ["water_tr", "water_r"],
            "D": ["water_br", "water_r"],
            "L": ["water", "water_l"],
            "R": ["grass"]
        },
        "image": "tiles_32x32_B/tile_5_2.png"
    },
    "water_bl": {   # bottom left
        "edges": {
            "U": ["water_l", "water_tl"],
            "D": ["grass"],
            "L": ["grass"],
            "R": ["water_b", "water_br"]
        },
        "image": "tiles_32x32_B/tile_6_0.png"
    },
    "water_b": {   # bottom
        "edges": {
            "U": ["water"],
            "D": ["grass"],
            "L": ["water_bl", "water_b"],
            "R": ["water_br", "water_b"]
        },
        "image": "tiles_32x32_B/tile_6_1.png"
    },
    "water_br": {   # bottom right
        "edges": {
            "U": ["water_r", "water_tr"],
            "D": ["grass"],
            "L": ["water_b", "water_bl"],
            "R": ["grass"]
        },
        "image": "tiles_32x32_B/tile_6_2.png"
    },

    # shore tiles
    "shore_tl": {   # top left
        "edges": {
            "U": ["water_l", "water_tl"],
            "D": ["water", "water_b", "shore_bl", "shore_ br"],
            "L": ["water_tl", "water_t"],
            "R": ["water_r", "shore_tr", "shore_br", "water"],
        },
        "image": "tiles_32x32_B/tile_4_3.png"
    },
    "shore_tr": {   # top right
        "edges": {
            "U": ["water_r", "water_tr"],
            "D": ["water_b", "water", "shore_bl", "shore_br"],
            "L": ["water_l", "shore_bl", "shore_tl", "water"],
            "R": ["water_t", "water_tr"],
        },
        "image": "tiles_32x32_B/tile_4_4.png"
    },
    "shore_bl": {   # bottom left
        "edges": {
            "U": ["shore_tl", "shore_tr", "water_t", "water"],
            "D": ["water_bl", "water_l"],
            "L": ["water_b", "water_bl"],
            "R": ["shore_tr", "shore_br", "water_r", "water"],
        },
        "image": "tiles_32x32_B/tile_5_3.png"
    },
    "shore_br": {   # bottom right
        "edges": {
            "U": ["shore_tr", "shore_tl", "water_t", "water"],
            "D": ["water_r", "water_br"],
            "L": ["shore_bl", "shore_tl", "water_l", "water"],
            "R": ["water_b", "water_br"],
        },
        "image": "tiles_32x32_B/tile_5_4.png"
    },
    "shore_lr": {   # top left bottom right
        "edges": {
            "U": ["water_l", "water_tl"],
            "D": ["water_r", "water_br"],
            "L": ["water_t", "water_tl"],
            "R": ["water_b", "water_br"],
        },
        "image": "tiles_32x32_B/tile_6_3.png"
    },
    "shore_rl": {   # top right bottom left
        "edges": {
            "U": ["water_r", "water_tr"],
            "D": ["water_l", "water_bl"],
            "L": ["water_b", "water_bl"],
            "R": ["water_t", "water_tr"],
        },
        "image": "tiles_32x32_B/tile_6_4.png"
    },
}

OPPOSITE_DIRECTION = {"U": "D", "D": "U", "L": "R", "R": "L"}
DIRECTIONS = ["U", "D", "L", "R"]

def create_adjacency_matrix():
    tile_to_index = {tile: idx for idx, tile in enumerate(TILES.keys())}
    tile_symbols = list(TILES.keys())
    num_tiles = len(tile_symbols)

    adjacency_bool = np.zeros((num_tiles, 4, num_tiles), dtype=bool)

    for tile_a, data in TILES.items():
        if isinstance(data, dict) and "edges" in data:
            tile_idx = tile_to_index[tile_a]
            for dir_idx, direction in enumerate(DIRECTIONS):
                allowed_tiles = data["edges"].get(direction, [])
                
                for tile_b in allowed_tiles:
                    if tile_b in tile_to_index:
                        neighbor_idx = tile_to_index[tile_b]
                        adjacency_bool[tile_idx, dir_idx, neighbor_idx] = True
                        
                        opposite_dir = OPPOSITE_DIRECTION[direction]
                        opp_dir_idx = DIRECTIONS.index(opposite_dir)
                        adjacency_bool[neighbor_idx, opp_dir_idx, tile_idx] = True

    return adjacency_bool, tile_symbols, tile_to_index

def print_adjacency_compatibility():
    for tile, data in TILES.items():
        if isinstance(data, dict) and "edges" in data:
            print(f"{tile}:")
            for direction in DIRECTIONS:
                allowed = data["edges"].get(direction, [])
                print(f"  {direction}: {allowed}")

# print_adjacency_compatibility()

# Initialize Pygame
pygame.init()
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
TILE_SIZE = 32

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tile Renderer")

# Load all tile images
def load_tile_images():
    tile_images = {}
    for tile_name, tile_data in TILES.items():
        image_path = tile_data["image"]
        try:
            # Load image and scale it to TILE_SIZE if needed
            image = pygame.image.load(image_path)
            if image.get_width() != TILE_SIZE or image.get_height() != TILE_SIZE:
                image = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
            tile_images[tile_name] = image
        except:
            print(f"Failed to load image: {image_path}")
            surf = pygame.Surface((TILE_SIZE, TILE_SIZE))
            surf.fill((200, 200, 200))
            font = pygame.font.SysFont(None, 20)
            text = font.render(tile_name, True, (0, 0, 0))
            surf.blit(text, (5, 5))
            tile_images[tile_name] = surf
    return tile_images

# Render tiles in a grid
def render_tiles(tile_images):
    screen.fill((255, 255, 255))
    tiles_per_row = SCREEN_WIDTH // TILE_SIZE
    x, y = 0, 0
    
    for idx, (tile_name, image) in enumerate(tile_images.items()):
        screen.blit(image, (x, y))
        x += TILE_SIZE
        if (idx + 1) % tiles_per_row == 0:
            x = 0
            y += TILE_SIZE
    pygame.display.flip()

def main():
    tile_images = load_tile_images()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        render_tiles(tile_images)
    pygame.quit()

if __name__ == "__main__":
    main()