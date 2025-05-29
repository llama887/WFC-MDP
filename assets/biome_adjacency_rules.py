import json

import numpy as np
import pygame

with open("assets/biome_adjacency_rules.json", "r") as f:
    TILES = json.load(f)
assert TILES is not None, "Tiles not loaded properly"

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
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
TILE_SIZE = 32


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
    # Initialize Pygame
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tile Renderer")
    main()
