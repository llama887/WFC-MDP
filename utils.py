import json
import numpy as np

def extract_patterns(input_data):
    """
    Process the TILES dictionary to extract patterns, frequencies, and rules for WFC.
    
    Args:
        input_data: Dictionary of tile information including adjacency rules
        
    Returns:
        tuple: (patterns, frequencies, rules)
    """
    # If input_data is a file path, load it
    if isinstance(input_data, str):
        with open(input_data, 'r') as f:
            data = json.load(f)
    else:
        data = input_data
    
    tiles = data["tiles"]
    direction_mapping = data["direction_mapping"]
    
    num_patterns = len(tiles)
    
    # Create a mapping from tile names to their IDs
    tile_to_id = {name: info["id"] for name, info in tiles.items()}
    
    # Initialize uniform frequencies for all patterns
    frequencies = [1.0] * num_patterns
    
    # Initialize the propagator state for the WFC algorithm
    # propagator_state[pattern][direction] = list of compatible patterns
    propagator_state = []
    
    # For each pattern, compute its compatible neighbors in each direction
    for pattern_name, pattern_info in tiles.items():
        pattern_id = pattern_info["id"]
        
        # Ensure the propagator_state list is big enough
        while len(propagator_state) <= pattern_id:
            propagator_state.append([[] for _ in range(4)])
        
        # For each direction, find compatible patterns
        for direction_name, compatible_names in pattern_info["edges"].items():
            direction_id = direction_mapping[direction_name]
            
            # Convert tile names to IDs
            compatible_ids = [tile_to_id[name] for name in compatible_names]
            
            # Store in the propagator state
            propagator_state[pattern_id][direction_id] = compatible_ids
    
    return list(range(num_patterns)), frequencies, propagator_state

def load_tile_images(json_path, image_dir=None):
    """
    Load tile images from the JSON configuration.
    
    Args:
        json_path: Path to the JSON file with tile configuration
        image_dir: Optional directory prefix for image paths
        
    Returns:
        dict: Mapping from tile ID to image path
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tiles = data["tiles"]
    tile_images = {}
    
    for name, info in tiles.items():
        tile_id = info["id"]
        image_path = info["image"]
        
        if image_dir:
            image_path = f"{image_dir}/{image_path}"
        
        tile_images[tile_id] = image_path
    
    return tile_images