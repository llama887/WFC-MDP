from .utils import calc_longest_path, calc_num_regions, grid_to_binary_map

MAX_BINARY_REWARD = 200


def binary_reward(
    grid: list[list[set[str]]], target_path_length: int
) -> tuple[float, int, int, list]:
    binary_map = grid_to_binary_map(
        grid,
        lambda tile_name: tile_name.startswith("sand") or tile_name.startswith("path"),
    )
    number_of_regions = calc_num_regions(binary_map)
    current_path_length, longest_path = calc_longest_path(binary_map)

    region_reward = 100.0 if number_of_regions == 1 else -100.0

    if current_path_length >= target_path_length:
        path_reward = 100.0
    else:
        path_reward = current_path_length - target_path_length
    return (
        region_reward + path_reward,
        number_of_regions,
        current_path_length,
        longest_path,
    )
