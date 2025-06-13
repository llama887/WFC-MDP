import os
import random
import sys
from timeit import default_timer as timer

import numpy as np

import pygame

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from assets.biome_adjacency_rules import TILE_SIZE, TILES, create_adjacency_matrix


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
    # grid = []
    # for _ in range(height):
    #     row = []
    #     for _ in range(width):
    #         row.append(set(tile_symbols))  # All tiles are possible initially
    #     grid.append(row)
    grid = np.ones((height, width, len(tile_symbols)), dtype=bool)
    return grid


# Find the cell with the lowest entropy (fewest possibilities)
def find_lowest_entropy_cell(grid, deterministic: bool = False):
    # min_entropy = float("inf")
    # candidates = []

    # for y in range(len(grid)):
    #     for x in range(len(grid[0])):
    #         if 1 < len(grid[y][x]) < min_entropy:
    #             min_entropy = len(grid[y][x])
    #             candidates = [(x, y)]
    #         elif len(grid[y][x]) == min_entropy:
    #             candidates.append((x, y))
    # if deterministic:
    #     return candidates[0] if candidates else None
    # return random.choice(candidates) if candidates else None

    n_possibilities = np.sum(grid, axis=2)  # Sum across tile possibilities
    n_to_collapse = np.sum(n_possibilities >= 2)
    n_possibilities = np.where(n_possibilities < 2, np.inf, n_possibilities)  # Ignore cells with 0 or 1 possibility
    if n_to_collapse == 0:
        return None
    if deterministic:
        return np.unravel_index(np.argmin(n_possibilities), grid.shape[:2])

    min_entropy = np.min(n_possibilities)
    candidates = np.argwhere(n_possibilities == min_entropy)
    if len(candidates) == 0:
        return None
    else:
        idx = random.randint(0, len(candidates) - 1)
        return candidates[idx]


# Collapse a cell to a single tile based on action probabilities
def collapse_cell(
    grid, tile_symbols, tile_to_index, x, y, action_probs, deterministic: bool = False
):
    """
    Collapses the cell (x, y) by choosing a tile based on action_probs.

    If deterministic=True:
      1. Pick the possible tile (in tile_symbols order) with largest action_probs weight.
      2. If that max weight is <= 0, fall back to the first possible tile in tile_symbols.
    If deterministic=False:
      Do the weighted random selection as before, with uniform fallback if all weights zero.

    Returns:
        The chosen tile name, or None if the cell was already empty.
    """
    possible_tiles = grid[y, x]
    # if not possible_tiles:
    if not np.any(possible_tiles):
        return None  # Already empty / contradiction

    # chosen_tile = None

    # if deterministic:
    # 1) Scan in canonical order for the max-prob tile
    best_tile = None
    max_prob = -float("inf")
    # for tile in tile_symbols:
    #     if tile in possible_tiles:
    #         idx = tile_to_index.get(tile, None)
    #         prob = (
    #             action_probs[idx]
    #             if idx is not None and 0 <= idx < len(action_probs)
    #             else 0.0
    #         )
    #         if prob > max_prob:
    #             max_prob = prob
    #             best_tile = tile

    # TODO: Non-deterministic version of this.
    action_probs = np.where(possible_tiles, action_probs, 0.0)
    best_tile_idx = np.argmax(action_probs)
    # max_prob = action_probs[best_tile_idx]

    # 2) Use best_tile if it had positive weight; else fallback to first possible
    # if best_tile is not None and max_prob > 0.0:
    #     chosen_tile = best_tile
    # else:
    #     # If all action probabilities are zero or negative, return None to signal failure
    #     return None

    grid[y, x] = 0
    grid[y, x, best_tile_idx] = 1  # Collapse to the chosen tile
    return best_tile_idx

    # else:
    #     # — stochastic path (unchanged) —
    #     weights = []
    #     valid_tiles = []
    #     total = 0.0

    #     for tile in possible_tiles:
    #         idx = tile_to_index.get(tile, None)
    #         w = (
    #             action_probs[idx]
    #             if idx is not None and 0 <= idx < len(action_probs)
    #             else 0.0
    #         )
    #         w = max(0.0, w)
    #         if w > 1e-9:
    #             weights.append(w)
    #             valid_tiles.append(tile)
    #             total += w

    #     if total > 1e-9:
    #         r = random.uniform(0, total)
    #         cum = 0.0
    #         chosen_tile = valid_tiles[-1]  # fallback if rounding
    #         for w, tile in zip(weights, valid_tiles):
    #             cum += w
    #             if r <= cum:
    #                 chosen_tile = tile
    #                 break
    #     else:
    #         # all weights zero → uniform random among possibles
    #         chosen_tile = random.choice(list(possible_tiles))

    # # finally collapse
    # # grid[y][x] = {chosen_tile}
    # return chosen_tile

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # U, D, L, R

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
        # if not grid[curr_y][curr_x]:
        if not np.any(grid[curr_y, curr_x]):
            continue

        # Directions: 0:U, 1:D, 2:L, 3:R (match adjacency_bool if it uses this order)
        # Check biome_adjacency_rules.py: DIRECTIONS = ["U", "D", "L", "R"] -> (0, -1), (0, 1), (-1, 0), (1, 0)
        for dir_idx, (dx, dy) in enumerate(DIRS):
            nx, ny = curr_x + dx, curr_y + dy

            if 0 <= nx < width and 0 <= ny < height:
                original_neighbor_possibilities = grid[ny, nx]
                # If neighbor is already empty, no need to process further from it
                # if not original_neighbor_possibilities:
                if not np.any(original_neighbor_possibilities):
                    return False

                valid_neighb_tiles = (grid[curr_y, curr_x][..., None] * adjacency_bool[:, dir_idx, :]).any(axis=0)
                new_neighbor_possibilities = np.where(
                    valid_neighb_tiles, original_neighbor_possibilities, 0
                )
                removed_neighbor_possibilities = np.any(new_neighbor_possibilities != original_neighbor_possibilities)

                if not np.any(new_neighbor_possibilities):
                    # If the neighbor ended up with no possibilities, we have a contradiction
                    # print(f"Contradiction found at ({nx}, {ny}) relative to ({curr_x}, {curr_y})")
                    grid[ny, nx] = False
                    return False

                grid[ny, nx] = new_neighbor_possibilities

                if removed_neighbor_possibilities:
                    # If we removed any possibilities, add the neighbor to the stack
                    # to propagate constraints from it.
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
    next_cell_coords = find_lowest_entropy_cell(grid, deterministic)

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

    y, x = next_cell_coords

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
    # if find_lowest_entropy_cell(grid, deterministic) is None:
    if np.all(np.sum(grid, axis=2) <= 1):
        # Need to double-check if it's completion or contradiction again,
        # as propagation might have resolved everything or caused a new contradiction.
        contradiction_found = False
        all_collapsed = True
        for r in grid:
            for cell in r:
                # if not cell:  # Check for contradiction created by propagation
                if not np.any(cell):  # Check for contradiction created by propagation
                    contradiction_found = True
                    all_collapsed = False
                    break
                # if len(cell) > 1:
                if np.sum(cell) > 1:
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
def render_wfc_grid(grid, tile_images, save_filename=None, screen=None):
    """
    Render the WFC grid and optionally save with reward info in filename.

    Args:
        grid: The WFC grid (list of lists of sets)
        tile_images: Dictionary mapping tile names to pygame surfaces
        save_filename: Optional base filename to save with reward info
    """
    screen.fill((255, 255, 255))

    # Count river and pond tiles
    river_count = 0
    pond_count = 0
    total_tiles = len(grid) * len(grid[0])

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 1:
                tile_name = next(iter(grid[y][x]))
                screen.blit(tile_images[tile_name], (x * TILE_SIZE, y * TILE_SIZE))

                # Classify tiles based on their names
                if "shore" in tile_name.lower() or "water_" in tile_name.lower():
                    river_count += 1
                elif "water" in tile_name.lower():
                    pond_count += 1
            else:
                # Draw undecided cells in white
                pygame.draw.rect(
                    screen,
                    (255, 255, 255),
                    (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )

    # Calculate percentages
    river_percentage = (river_count / total_tiles) * 100 if total_tiles > 0 else 0
    pond_percentage = (pond_count / total_tiles) * 100 if total_tiles > 0 else 0

    # Determine which feature is dominant
    if river_count > pond_count:
        dominant_feature = "river"
        dominant_count = river_count
        dominant_percentage = river_percentage
    else:
        dominant_feature = "pond"
        dominant_count = pond_count
        dominant_percentage = pond_percentage

    pygame.display.flip()

    # Save to file if requested
    if save_filename:
        # Generate filename with feature info
        feature_str = f"{dominant_feature}_{dominant_count}_{dominant_percentage:.1f}%"
        filename = f"wfc_reward_img/{feature_str}.png"

        # Save the current screen surface
        pygame.image.save(screen, filename)
        print(f"Saved WFC output to: {filename}")

    return dominant_percentage


# Main WFC Algorithm (Standalone execution example)
def run_wfc(width, height, tile_images, adjacency_bool, tile_symbols, tile_to_index, screen=None):
    start_time = timer()
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
        render_wfc_grid(grid, tile_images, screen=screen)
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

    print(f"WFC run completed in {timer() - start_time:.2f} seconds")
    return grid


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 480
    TILE_SIZE = 32

    screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Evolving WFC")

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
        screen=screen,
    )

    # After the WFC completes, render one final time and save with reward
    river_score = render_wfc_grid(final_grid, tile_images, save_filename="wfc_output", screen=screen)
    pond_score = render_wfc_grid(final_grid, tile_images, save_filename="wfc_output", screen=screen)

    # Keep the final state visible until quit
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.time.delay(1)

    pygame.quit()