import pygame
import numpy as np
import random
import os

from biome_tile import *

# Initialize Pygame
pygame.init()
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
TILE_SIZE = 32

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Evolving WFC")

# Load all tile images
def load_tile_images():
    tile_images = {}
    for tile_name, tile_data in TILES.items():
        image_path = tile_data["image"]
        try:
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

# Initialize WFC grid
def initialize_wfc_grid(width, height, tile_symbols):
    grid = []
    for _ in range(height):
        row = []
        for _ in range(width):
            row.append(set(tile_symbols))  # All tiles are possible initially
        grid.append(row)
    return grid

# Find the cell with the lowest entropy (fewest possibilities)
def find_lowest_entropy_cell(grid):
    min_entropy = float('inf')
    candidates = []
    
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if 1 < len(grid[y][x]) < min_entropy:
                min_entropy = len(grid[y][x])
                candidates = [(x, y)]
            elif len(grid[y][x]) == min_entropy:
                candidates.append((x, y))
    
    return random.choice(candidates) if candidates else None

# Collapse a cell to a single tile
def collapse_cell(grid, tile_symbols, x, y):
    possible_tiles = list(grid[y][x])
    chosen_tile = random.choice(possible_tiles)
    grid[y][x] = {chosen_tile}
    return chosen_tile

# Propagate constraints to neighbors
def propagate_constraints(grid, adjacency_bool, tile_to_index, x, y):
    stack = [(x, y)]
    
    while stack:
        x, y = stack.pop()
        current_possibilities = grid[y][x]
        
        for dir_idx, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):  # U, D, L, R
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                neighbor_possibilities = grid[ny][nx].copy()
                changed = False
                
                for neighbor_tile in list(grid[ny][nx]):
                    # Check if any current tile allows this neighbor in the given direction
                    compatible = False
                    for current_tile in current_possibilities:
                        current_idx = tile_to_index[current_tile]
                        neighbor_idx = tile_to_index[neighbor_tile]
                        if adjacency_bool[current_idx, dir_idx, neighbor_idx]:
                            compatible = True
                            break
                    
                    if not compatible:
                        neighbor_possibilities.discard(neighbor_tile)
                        changed = True
                
                if changed:
                    grid[ny][nx] = neighbor_possibilities
                    stack.append((nx, ny))

# Render the WFC grid
def render_wfc_grid(grid, tile_images):
    screen.fill((255, 255, 255))
    
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1:
                tile_name = next(iter(grid[y][x]))
                screen.blit(tile_images[tile_name], (x * TILE_SIZE, y * TILE_SIZE))
            else:
                pygame.draw.rect(screen, (255, 255, 255), (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
    
    pygame.display.flip()

# Main WFC Algorithm
def run_wfc(width, height, tile_images, adjacency_bool, tile_symbols, tile_to_index):
    grid = initialize_wfc_grid(width, height, tile_symbols)
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Find and collapse the next cell
        next_cell = find_lowest_entropy_cell(grid)
        if not next_cell:
            break  # All cells are resolved or in contradiction
        
        x, y = next_cell
        collapse_cell(grid, tile_symbols, x, y)
        propagate_constraints(grid, adjacency_bool, tile_to_index, x, y)
        
        render_wfc_grid(grid, tile_images)
        pygame.time.delay(100)
    
    return grid

if __name__ == "__main__":
    tile_images = load_tile_images()
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    
    # Define grid size (in tiles)
    GRID_WIDTH = SCREEN_WIDTH // TILE_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // TILE_SIZE
    
    final_grid = run_wfc(GRID_WIDTH, GRID_HEIGHT, tile_images, adjacency_bool, tile_symbols, tile_to_index)
    
    # Keep the window open until closed
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
    
    pygame.quit()