import random

import pygame

from biome_adjacency_rules import *

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
    min_entropy = float("inf")
    candidates = []

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if 1 < len(grid[y][x]) < min_entropy:
                min_entropy = len(grid[y][x])
                candidates = [(x, y)]
            elif len(grid[y][x]) == min_entropy:
                candidates.append((x, y))

    return random.choice(candidates) if candidates else None


# Collapse a cell to a single tile based on action probabilities
def collapse_cell(
    grid, tile_symbols, tile_to_index, x, y, action_probs, deterministic=False
):
    """
    Collapses the cell (x, y) by choosing a tile based on action_probs.

    Args:
        grid: The current WFC grid (list of lists of sets).
        tile_symbols: List of all tile names.
        tile_to_index: Mapping from tile name to index.
        x, y: Coordinates of the cell to collapse.
        action_probs: A list or numpy array of probabilities/weights for each tile index.
        deterministic: If True, choose the tile with the highest probability.

    Returns:
        The chosen tile name, or None if the cell had no possibilities.
    """
    possible_tiles = list(grid[y][x])
    if not possible_tiles:
        return None  # Should not happen if find_lowest_entropy found a valid cell

    chosen_tile = None  # Initialize chosen_tile

    if deterministic:
        best_tile = None
        max_prob = -1.0
        for tile in possible_tiles:
            idx = tile_to_index[tile]
            # Ensure index is within bounds of action_probs
            if 0 <= idx < len(action_probs):
                prob = action_probs[idx]
                if prob > max_prob:
                    max_prob = prob
                    best_tile = tile
        chosen_tile = best_tile
    else:
        # Weighted random selection
        weights = []
        valid_tiles = []
        total_weight = 0.0
        for tile in possible_tiles:
            idx = tile_to_index[tile]
            # Ensure index is within bounds and weight is non-negative
            if 0 <= idx < len(action_probs):
                weight = max(0.0, action_probs[idx])
                if (
                    weight > 1e-9
                ):  # Use a small epsilon to avoid floating point issues with zero weights
                    weights.append(weight)
                    valid_tiles.append(tile)
                    total_weight += weight

        if total_weight > 1e-9 and valid_tiles:  # Use weighted choice if possible
            rand_val = random.uniform(0, total_weight)
            current_sum = 0.0
            # Default to last valid tile in case of floating point issues
            chosen_tile = valid_tiles[-1]
            for i, tile in enumerate(valid_tiles):
                current_sum += weights[i]
                if rand_val <= current_sum:
                    chosen_tile = tile
                    break
        elif possible_tiles:  # Check if there were original possibilities
            # Fallback to uniform random choice among the original possibilities if all weights are zero or invalid
            chosen_tile = random.choice(possible_tiles)
        # If possible_tiles was empty initially, chosen_tile remains None

    if chosen_tile:
        grid[y][x] = {chosen_tile}

    return chosen_tile


# Propagate constraints to neighbors
def propagate_constraints(grid, adjacency_bool, tile_to_index, start_x, start_y):
    """
    Propagates constraints starting from cell (start_x, start_y).

    Returns:
        bool: True if propagation succeeded, False if a contradiction was found
              (a cell ended up with zero possibilities).
    """
    stack = [(start_x, start_y)]
    height = len(grid)
    width = len(grid[0])

    while stack:
        curr_x, curr_y = stack.pop()
        # If the cell that was popped was already empty (due to prior contradiction), skip
        if not grid[curr_y][curr_x]:
            continue

        current_possibilities = grid[curr_y][curr_x]

        # Directions: 0:U, 1:D, 2:L, 3:R (match adjacency_bool if it uses this order)
        # Check biome_adjacency_rules.py: DIRECTIONS = ["U", "D", "L", "R"] -> (0, -1), (0, 1), (-1, 0), (1, 0)
        for dir_idx, (dx, dy) in enumerate(
            [(0, -1), (0, 1), (-1, 0), (1, 0)]
        ):  # U, D, L, R
            nx, ny = curr_x + dx, curr_y + dy

            if 0 <= nx < width and 0 <= ny < height:
                original_neighbor_possibilities = grid[ny][nx]
                # If neighbor is already empty, no need to process further from it
                if not original_neighbor_possibilities:
                    continue

                removed_options = set()

                for neighbor_tile in original_neighbor_possibilities:
                    neighbor_idx = tile_to_index[neighbor_tile]
                    # Check if *any* tile currently possible in (curr_x, curr_y)
                    # allows 'neighbor_tile' in the direction 'dir_idx'.
                    is_compatible = False
                    for current_tile in current_possibilities:
                        current_idx = tile_to_index[current_tile]
                        # Check adjacency: current_idx -> neighbor_idx in direction dir_idx
                        # Ensure indices are valid before accessing adjacency_bool
                        if (
                            0 <= current_idx < adjacency_bool.shape[0]
                            and 0 <= dir_idx < adjacency_bool.shape[1]
                            and 0 <= neighbor_idx < adjacency_bool.shape[2]
                        ):
                            if adjacency_bool[current_idx, dir_idx, neighbor_idx]:
                                is_compatible = True
                                break  # Found a compatible tile, no need to check others
                        # else: Handle potential index out of bounds if necessary, though tile_to_index should be correct

                    if not is_compatible:
                        # This neighbor_tile is not supported by any possibility in the current cell.
                        removed_options.add(neighbor_tile)

                if removed_options:
                    new_neighbor_possibilities = (
                        original_neighbor_possibilities - removed_options
                    )
                    if not new_neighbor_possibilities:
                        # Contradiction detected!
                        grid[ny][nx] = set()  # Mark as empty
                        return False  # Signal failure

                    # Only update and add to stack if possibilities actually changed
                    if len(new_neighbor_possibilities) < len(
                        original_neighbor_possibilities
                    ):
                        grid[ny][nx] = new_neighbor_possibilities
                        # Add the neighbor to the stack to propagate constraints *from* it.
                        # Avoid adding if already in stack? Simple check:
                        if (nx, ny) not in stack:
                            stack.append((nx, ny))

    return True  # Propagation succeeded without contradiction


def biome_wfc_step(
    grid, adjacency_bool, tile_symbols, tile_to_index, action_probs, deterministic=False
):
    """
    Performs a single collapse and propagate step for the biome WFC.

    Args:
        grid: The current WFC grid state (list of lists of sets).
        adjacency_bool: The precomputed adjacency rules.
        tile_symbols: List of all tile names.
        tile_to_index: Mapping from tile name to index.
        action_probs: Probabilities/weights for tile selection from the RL agent.
        deterministic: Whether to use deterministic (max prob) or stochastic selection.

    Returns:
        tuple: (grid, terminated, truncated)
            - grid: The updated grid state.
            - terminated: True if all cells are collapsed.
            - truncated: True if a contradiction occurred.
    """
    # 1. Find the cell with the lowest entropy
    next_cell_coords = find_lowest_entropy_cell(grid)

    if next_cell_coords is None:
        # Check if it's because all cells are collapsed or because of a previous contradiction
        contradiction_found = False
        all_collapsed = True
        for r in grid:
            for cell in r:
                if not cell:  # Found an empty set -> contradiction occurred previously
                    contradiction_found = True
                    all_collapsed = False  # Ensure terminated is False if contradiction
                    break
                if len(cell) > 1:
                    all_collapsed = False
            if contradiction_found:
                break
        if contradiction_found:
            return grid, False, True  # Truncated (pre-existing contradiction)
        if all_collapsed:
            return grid, True, False  # Terminated (successfully completed)
        # If neither, it's an unexpected state, treat as truncated
        return grid, False, True

    x, y = next_cell_coords

    # 2. Collapse the chosen cell using the action probabilities
    chosen_tile = collapse_cell(
        grid, tile_symbols, tile_to_index, x, y, action_probs, deterministic
    )

    if chosen_tile is None:
        # This implies the cell chosen by find_lowest_entropy was already empty,
        # or collapse failed unexpectedly. Indicates a contradiction state.
        return grid, False, True  # Truncated (contradiction during collapse)

    # 3. Propagate constraints from the collapsed cell
    success = propagate_constraints(grid, adjacency_bool, tile_to_index, x, y)

    if not success:
        # Propagation itself detected a contradiction
        return grid, False, True  # Truncated (contradiction during propagation)

    # 4. Check if the process is finished *after* propagation
    # Re-calling find_lowest_entropy_cell tells us if any undecided cells remain.
    if find_lowest_entropy_cell(grid) is None:
        # Need to double-check if it's completion or contradiction again,
        # as propagation might have resolved everything or caused a new contradiction.
        contradiction_found = False
        all_collapsed = True
        for r in grid:
            for cell in r:
                if not cell:  # Check for contradiction created by propagation
                    contradiction_found = True
                    all_collapsed = False
                    break
                if len(cell) > 1:
                    all_collapsed = False
            if contradiction_found:
                break
        if contradiction_found:
            return grid, False, True  # Truncated (contradiction post-propagation)
        if all_collapsed:
            return grid, True, False  # Terminated (all collapsed post-propagation)
        # If neither, unexpected state
        return grid, False, True

    # Otherwise, the process continues, not terminated, not truncated
    return grid, False, False


# Render the WFC grid
def render_wfc_grid(grid, tile_images, save_filename=None):
    """
    Render the WFC grid and optionally save with reward info in filename.
    
    Args:
        grid: The WFC grid (list of lists of sets)
        tile_images: Dictionary mapping tile names to pygame surfaces
        save_filename: Optional base filename to save with reward info
    """
    screen.fill((255, 255, 255))
    
    # Count water tiles and total tiles
    water_count = 0
    total_tiles = len(grid) * len(grid[0])
    
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1:
                tile_name = next(iter(grid[y][x]))
                screen.blit(tile_images[tile_name], (x * TILE_SIZE, y * TILE_SIZE))
                
                # Count water tiles
                if "water" in tile_name.lower():
                    water_count += 1
            else:
                # Draw undecided cells in white
                pygame.draw.rect(
                    screen,
                    (255, 255, 255),
                    (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )
    
    # Calculate reward (percentage of water tiles)
    water_percentage = (water_count / total_tiles) * 100 if total_tiles > 0 else 0
    pygame.display.flip()
    
    # Save to file if requested
    if save_filename:
        # Generate filename with reward info
        reward_str = f"water_{water_count}_{water_percentage:.1f}%"
        filename = f"wfc_reward_img/{reward_str}.png"
        
        # Save the current screen surface
        pygame.image.save(screen, filename)
        print(f"Saved WFC output to: {filename}")
    
    return water_percentage


# Main WFC Algorithm (Standalone execution example)
def run_wfc(width, height, tile_images, adjacency_bool, tile_symbols, tile_to_index):
    grid = initialize_wfc_grid(width, height, tile_symbols)
    running = True
    terminated = False
    truncated = False
    num_tiles = len(tile_symbols)  # Get number of tiles for dummy action

    while running and not terminated and not truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:  # Exit if QUIT event was processed
            break

        # --- WFC Step ---
        # Create dummy action probabilities using random values for standalone run
        # In RL, this would come from the agent's policy
        dummy_action_probs = [random.random() for _ in range(num_tiles)]

        # Perform one step
        grid, terminated, truncated = biome_wfc_step(
            grid,
            adjacency_bool,
            tile_symbols,
            tile_to_index,
            dummy_action_probs,
            deterministic=False,  # Use stochastic collapse for variety
        )
        # --- End WFC Step ---

        # Render the current state
        render_wfc_grid(grid, tile_images)
        pygame.time.delay(1)  # Shorter delay for faster visualization

        if terminated:
            print("WFC Completed Successfully!")
        elif truncated:
            print("WFC Failed (Contradiction)!")
            # Optionally render the grid one last time to show the contradiction state
            render_wfc_grid(grid, tile_images)
            pygame.time.delay(1000)  # Pause to see the failure

    # Keep the final state visible until quit, unless already quit by event
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.time.delay(1)  # Prevent high CPU usage in final wait loop

    return grid


if __name__ == "__main__":
    tile_images = load_tile_images()
    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()

    # Define grid size (in tiles)
    GRID_WIDTH = SCREEN_WIDTH // TILE_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // TILE_SIZE

    final_grid = run_wfc(
        GRID_WIDTH,
        GRID_HEIGHT,
        tile_images,
        adjacency_bool,
        tile_symbols,
        tile_to_index,
    )

    # After the WFC completes, render one final time and save with reward
    water_score = render_wfc_grid(final_grid, tile_images, save_filename="wfc_output")
    
    # Keep the final state visible until quit
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.time.delay(1)

    pygame.quit()