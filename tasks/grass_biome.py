import numpy as np
from .utils import calc_num_regions


def classify_grass_biome(counts: dict, grass_cells: int) -> str:
    """Returns 'grassgrass', 'meadow', or 'unknown' — not used in scoring."""
    if grass_cells == 0:
        return "unknown"

    flower_ratio = counts["flower"] / grass_cells
    grass_ratio = grass_cells / sum(counts.values())

    if flower_ratio <= 0.15:
        return "grassgrass"
    elif flower_ratio > 0.15:
        return "meadow"
    return "unknown"


def grass_biome_reward(grid: list[list[set[str]]]) -> tuple[float, dict]:
    grass_tiles = {"grass", "tall_grass", "flower"}
    target_ratios = {"grass": 0.65, "tall_grass": 0.25, "flower": 0.10}

    total_cells = len(grid) * len(grid[0])
    grass_map = np.zeros((len(grid), len(grid[0])), dtype=bool)
    counts = {"grass": 0, "tall_grass": 0, "flower": 0}
    undecided = 0

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            cell = grid[y][x]
            if len(cell) != 1:
                undecided += 1
                continue
            tile = next(iter(cell)).lower()
            if tile in counts:
                counts[tile] += 1
                grass_map[y, x] = True

    grass_cells = np.sum(grass_map)
    if total_cells == 0 or grass_cells == 0:
        return 0.0, {"biome": "invalid"}

    grass_coverage = grass_cells / total_cells

    # Continuity
    num_regions = calc_num_regions(grass_map.astype(np.int8))
    continuity_score = 1.0 / (num_regions ** 0.5) if num_regions > 0 else 0.0

    # Distribution scoring
    actual_ratios = {t: counts[t] / grass_cells for t in counts}
    penalty = 0.0
    for tile in counts:
        actual = actual_ratios[tile]
        target = target_ratios[tile]
        if actual > target:
            penalty += (actual - target) ** 2 * 2.0
        else:
            penalty += (target - actual) * 0.5
    distribution_score = 1.0 - min(penalty, 1.0)

    # Final reward
    score = (
        0.5 * grass_coverage +
        0.3 * continuity_score +
        0.2 * distribution_score
    ) * 100

    # Biome label — used only for logging
    biome = classify_grass_biome(counts, grass_cells)

    return float(score), {
        "biome": biome,
        "coverage": grass_coverage,
        "continuity": continuity_score,
        "distribution": distribution_score,
        "ratios": actual_ratios,
        "num_regions": num_regions
    }
