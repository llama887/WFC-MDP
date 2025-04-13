import numpy as np
import wfc_cpp  # The C++ extension

def fast_wfc_collapse_step(grid, width, height, num_patterns, adjacency, action_vector):
    """
    Perform a single collapse step using the C++ WFC implementation.
    
    Args:
        grid: 3D numpy boolean array of shape (height, width, num_patterns)
        width: Width of the grid
        height: Height of the grid
        num_patterns: Number of possible patterns
        adjacency: Adjacency rules (boolean array of shape (num_patterns, 4, num_patterns))
        action_vector: Action probabilities for each pattern
        
    Returns:
        tuple: (updated grid, terminate flag, truncate flag)
    """
    # Convert adjacency to the format expected by C++ WFC
    propagator_state = []
    for i in range(num_patterns):
        pattern_rules = []
        for d in range(4):  # Four directions: U, R, D, L
            compatible = []
            for j in range(num_patterns):
                if adjacency[i, d, j]:
                    compatible.append(j)
            pattern_rules.append(compatible)
        propagator_state.append(pattern_rules)
    
    # Create pattern frequencies (equal for now, can be customized)
    pattern_frequencies = np.ones(num_patterns) / num_patterns
    
    # Create the WFC instance
    seed = np.random.randint(0, 10000)  # Random seed
    periodic = False  # Non-periodic boundaries
    wfc = wfc_cpp.WFC(periodic, seed, pattern_frequencies.tolist(), 
                     propagator_state, height, width)
    
    # Set the current state of the grid
    wave_state = wfc.get_wave_state()
    for y in range(height):
        for x in range(width):
            for p in range(num_patterns):
                if not grid[y, x, p]:
                    wave_state.set(y, x, p, False)
    
    # Perform a single collapse step
    terminate, truncate = wfc.collapse_step(action_vector.tolist())
    
    # Get the updated state
    updated_grid = np.ones((height, width, num_patterns), dtype=bool)
    wave_state = wfc.get_wave_state()
    for y in range(height):
        for x in range(width):
            for p in range(num_patterns):
                updated_grid[y, x, p] = wave_state.get(y, x, p)
    
    return updated_grid, terminate, truncate