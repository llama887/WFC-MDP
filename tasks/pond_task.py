import numpy as np
from typing import Any
from .utils import calc_num_regions, percent_target_tiles_excluding_excluded_tiles, grid_to_binary_map, calc_longest_path

# Get tile definitions from biome rules
from assets.biome_adjacency_rules import create_adjacency_matrix
_, tile_symbols, _ = create_adjacency_matrix()
num_tiles = len(tile_symbols)

# Define masks for pond tasks
WATER_SHORE_MASK = np.zeros(num_tiles, dtype=bool)
PURE_WATER_MASK = np.zeros(num_tiles, dtype=bool)
SAND_PATH_MASK = np.zeros(num_tiles, dtype=bool)
HILL_MASK = np.zeros(num_tiles, dtype=bool)
GRASS_MASK = np.zeros(num_tiles, dtype=bool)

for idx, tile_name in enumerate(tile_symbols):
    if tile_name.startswith("water") or tile_name.startswith("shore") or tile_name == "pond":
        WATER_SHORE_MASK[idx] = True
    if tile_name in ["water", "pond"]:
        PURE_WATER_MASK[idx] = True
    if tile_name.startswith("sand") or tile_name.startswith("path"):
        SAND_PATH_MASK[idx] = True
    if "hill" in tile_name:
        HILL_MASK[idx] = True
    if "grass" in tile_name:
        GRASS_MASK[idx] = True

def pond_reward(grid: np.ndarray) -> tuple[float, dict[str, Any]]:
    water_percent = percent_target_tiles_excluding_excluded_tiles(
        grid, WATER_SHORE_MASK, SAND_PATH_MASK
    ) * 100

    pure_water_percent = percent_target_tiles_excluding_excluded_tiles(
        grid, PURE_WATER_MASK, SAND_PATH_MASK
    ) * 100

    grass_percent = percent_target_tiles_excluding_excluded_tiles(
        grid, GRASS_MASK, SAND_PATH_MASK
    ) * 100

    # Reward components
    water_penalty = water_percent - 25 if water_percent < 25 else 0
    water_center_penalty = pure_water_percent - 10 if pure_water_percent < 10 else 0
    grass_penalty = grass_percent - 15 if grass_percent < 15 else 0

    # Create binary maps
    water_binary_map = grid_to_binary_map(grid, WATER_SHORE_MASK)
    land_binary_map = grid_to_binary_map(grid, ~WATER_SHORE_MASK)

    # Calculate metrics
    water_regions = calc_num_regions(water_binary_map)
    land_regions = calc_num_regions(land_binary_map)
    water_path_length, _ = calc_longest_path(water_binary_map)

    # Apply penalties
    region_penalty = min(5 - water_regions, 0)
    path_penalty = min(25 - water_path_length, 0)
    land_region_penalty = min(2 - land_regions, 0)
    # hills_penalty = -int(np.sum(grid*HILL_MASK[None, None, :]))
    total_reward = (
        water_penalty +
        5 * water_center_penalty +
        region_penalty +
        land_region_penalty +
        # hills_penalty + 
        grass_penalty +
        path_penalty
    )
    diagnostic = {
        "percent_water": water_percent,
        "water_penalty": water_penalty,
        "grass_percent": grass_percent,
        "grass_penalty": grass_penalty,
        "percent_water_center": pure_water_percent,
        "water_center_penalty": water_center_penalty,
        "number_of_water_regions": water_regions,
        "region_penalty": region_penalty,
        "number_of_land_regions": land_regions,
        "land_region_penalty": land_region_penalty,
        "water_path_length": water_path_length,
        "path_penalty": path_penalty,
        # "hills_penalty": hills_penalty
    }
    assert total_reward <= 0, diagnostic

    total_reward = (
        water_penalty +
        3 * water_center_penalty +
        region_penalty +
        land_region_penalty +
        path_penalty
    )

    return total_reward, diagnostic

