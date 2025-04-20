import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union

# Constants for directions: up, right, down, left
DIRECTIONS_X = np.array([0, 1, 0, -1])
DIRECTIONS_Y = np.array([-1, 0, 1, 0])

def get_opposite_direction(direction: int) -> int:
    """Get the opposite direction (0-3)"""
    return (direction + 2) % 4

class Wave:
    """
    Wave class to track possible patterns for each cell
    """
    def __init__(self, height: int, width: int, pattern_frequencies: List[float]) -> None:
        """
        Initialize the wave with all patterns possible in all cells
        
        Args:
            height: Grid height
            width: Grid width
            pattern_frequencies: List of pattern frequencies (normalized)
        """
        self.height = height
        self.width = width
        self.size = height * width
        
        # Pattern information
        self.pattern_frequencies = np.array(pattern_frequencies, dtype=np.float64)
        self.num_patterns = len(pattern_frequencies)
        
        # Wave data: data[y, x, pattern] = True if pattern is possible at (y, x)
        self.data = np.ones((height, width, self.num_patterns), dtype=bool)
        
        # Calculate p*log(p) for entropy calculation (precomputed)
        self.p_logp_pattern_frequencies = self.pattern_frequencies * np.log(self.pattern_frequencies)
        
        # Minimum absolute half p*log(p) for noise in entropy calculation
        nonzero_mask = self.p_logp_pattern_frequencies != 0
        if np.any(nonzero_mask):
            self.min_abs_half_p_logp = np.min(np.abs(self.p_logp_pattern_frequencies[nonzero_mask])) / 2.0
        else:
            self.min_abs_half_p_logp = 1e-6
        
        # Track if wave is in impossible state
        self.is_impossible = False
        
        # Initialize entropy memoization
        self._init_memoization()
    
    def _init_memoization(self) -> None:
        """Initialize entropy memoization values"""
        # Sum of p*log(p) for each cell
        base_entropy = np.sum(self.p_logp_pattern_frequencies)
        # Sum of p for each cell
        base_s = np.sum(self.pattern_frequencies)
        
        # Handle edge case where sum is zero or very small
        if base_s <= 1e-10:
            base_s = 1.0
            self.is_impossible = True
            
        # Log of sum for each cell
        log_base_s = np.log(base_s)
        # Base entropy value
        entropy_base = log_base_s - base_entropy / base_s
        
        # Create memoization arrays
        self.p_logp_sum = np.full((self.height, self.width), base_entropy, dtype=np.float64)
        self.p_sum = np.full((self.height, self.width), base_s, dtype=np.float64)
        self.log_sum = np.full((self.height, self.width), log_base_s, dtype=np.float64)
        self.num_patterns_array = np.full((self.height, self.width), self.num_patterns, dtype=np.int32)
        self.entropy = np.full((self.height, self.width), entropy_base, dtype=np.float64)
    
    def get(self, i: int, j: int, pattern: int) -> bool:
        """Check if pattern can be placed in cell at (i, j)"""
        return self.data[i, j, pattern]
    
    def set(self, i: int, j: int, pattern: int, value: bool) -> None:
        """Set pattern possibility in cell at (i, j)"""
        old_value = self.data[i, j, pattern]
        if old_value == value:
            return
        
        self.data[i, j, pattern] = value
        
        # Update entropy memoization if removing a pattern
        if value == False:
            # Remove pattern contribution from sum
            self.p_logp_sum[i, j] -= self.p_logp_pattern_frequencies[pattern]
            self.p_sum[i, j] -= self.pattern_frequencies[pattern]
            
            # Decrease pattern count
            self.num_patterns_array[i, j] -= 1
            
            # Check for contradiction or very small sum
            if self.num_patterns_array[i, j] == 0 or self.p_sum[i, j] <= 1e-10:
                self.is_impossible = True
                # Set entropy to infinity to avoid selecting this cell
                self.entropy[i, j] = float('inf')
                return
            
            # Update log sum (only if sum > 0)
            self.log_sum[i, j] = np.log(self.p_sum[i, j])
            
            # Recalculate entropy
            self.entropy[i, j] = self.log_sum[i, j] - self.p_logp_sum[i, j] / self.p_sum[i, j]
    
    def get_min_entropy(self, rng: np.random.Generator) -> int:
        """
        Find cell with minimum entropy
        
        Returns:
            int: -2 if contradiction, -1 if success (all cells decided), 
                 otherwise flattened index of cell with minimum entropy
        """
        if self.is_impossible:
            return -2
        
        # Get cells with more than one pattern (not yet collapsed)
        uncollapsed_mask = self.num_patterns_array > 1
        
        # If all cells collapsed, return success
        if not np.any(uncollapsed_mask):
            return -1
        
        # Get entropy values for uncollapsed cells
        entropy_values = self.entropy.copy()
        
        # Set entropy of collapsed cells to infinity
        entropy_values[~uncollapsed_mask] = np.inf
        
        # Add noise for tie-breaking
        noise = rng.uniform(0, self.min_abs_half_p_logp, size=entropy_values.shape)
        entropy_values += noise
        
        # Find minimum entropy cell (flattened index)
        min_idx = np.argmin(entropy_values)
        
        return min_idx


class Propagator:
    """
    Propagator class to propagate constraints
    """
    def __init__(self, wave_height: int, wave_width: int, periodic_output: bool, 
                 propagator_state: List[List[List[int]]]) -> None:
        """
        Initialize the propagator
        
        Args:
            wave_height: Height of the wave grid
            wave_width: Width of the wave grid
            periodic_output: Whether the output is toric/periodic
            propagator_state: Compatibility rules [pattern][direction] = compatible patterns
        """
        self.wave_height = wave_height
        self.wave_width = wave_width
        self.periodic_output = periodic_output
        self.propagator_state = propagator_state
        self.patterns_size = len(propagator_state)
        
        # Queue of (y, x, pattern) tuples to propagate
        self.propagating = []
        
        # Initialize compatible array
        # compatible[y, x, pattern, direction] = number of compatible patterns
        self.compatible = self._init_compatible()
    
    def _init_compatible(self) -> np.ndarray:
        """Initialize the compatible array"""
        compatible = np.zeros((self.wave_height, self.wave_width, self.patterns_size, 4), dtype=np.int32)
        
        for y in range(self.wave_height):
            for x in range(self.wave_width):
                for pattern in range(self.patterns_size):
                    for direction in range(4):
                        if pattern < len(self.propagator_state) and direction < len(self.propagator_state[pattern]):
                            compatible[y, x, pattern, direction] = len(
                                self.propagator_state[pattern][get_opposite_direction(direction)]
                            )
        
        return compatible
    
    def add_to_propagator(self, y: int, x: int, pattern: int) -> None:
        """Add a (y,x,pattern) to the propagation queue"""
        self.compatible[y, x, pattern, :] = 0
        self.propagating.append((y, x, pattern))
    
    def propagate(self, wave: Wave) -> None:
        """Propagate constraints through the wave"""
        while self.propagating:
            y1, x1, pattern = self.propagating.pop()
            
            # Propagate in all four directions
            for direction in range(4):
                # Skip if this pattern doesn't have this direction defined
                if pattern >= len(self.propagator_state) or direction >= len(self.propagator_state[pattern]):
                    continue
                
                dx = DIRECTIONS_X[direction]
                dy = DIRECTIONS_Y[direction]
                
                # Calculate the next cell coordinates, handling periodicity
                if self.periodic_output:
                    x2 = (x1 + dx) % self.wave_width
                    y2 = (y1 + dy) % self.wave_height
                else:
                    x2 = x1 + dx
                    y2 = y1 + dy
                    # Skip if outside grid boundaries
                    if x2 < 0 or x2 >= self.wave_width or y2 < 0 or y2 >= self.wave_height:
                        continue
                
                # Get compatible patterns in this direction
                compatible_patterns = self.propagator_state[pattern][direction]
                
                # Update compatible counts for affected patterns
                for compatible_pattern in compatible_patterns:
                    # Decrease the compatibility count
                    self.compatible[y2, x2, compatible_pattern, direction] -= 1
                    
                    # If no compatible patterns remain in this direction, remove pattern from wave
                    if self.compatible[y2, x2, compatible_pattern, direction] == 0:
                        self.add_to_propagator(y2, x2, compatible_pattern)
                        wave.set(y2, x2, compatible_pattern, False)


class WFC:
    """
    Wave Function Collapse algorithm implementation
    """
    def __init__(self, periodic_output: bool, seed: int,
                 pattern_frequencies: List[float],
                 propagator_state: List[List[List[int]]],
                 wave_height: int, wave_width: int) -> None:
        """
        Initialize the WFC algorithm
        
        Args:
            periodic_output: Whether the output is toric/periodic
            seed: Random seed
            pattern_frequencies: List of pattern frequencies
            propagator_state: Compatibility rules [pattern][direction] = compatible patterns
            wave_height: Height of the wave grid
            wave_width: Width of the wave grid
        """
        # Initialize random generator
        self.rng = np.random.RandomState(seed)
        
        # Normalize pattern frequencies
        self.pattern_frequencies = self._normalize(pattern_frequencies)
        self.num_patterns = len(pattern_frequencies)
        
        # Initialize wave and propagator
        self.wave = Wave(wave_height, wave_width, self.pattern_frequencies)
        self.propagator = Propagator(wave_height, wave_width, periodic_output, propagator_state)
    
    def _normalize(self, weights: List[float]) -> List[float]:
        """Normalize a vector of weights to sum to 1.0"""
        weights_array = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights_array)
        
        if sum_weights > 0:
            weights_array = weights_array / sum_weights
        
        return weights_array.tolist()
    
    def observe(self) -> int:
        """
        Observe the cell with minimum entropy
        
        Returns:
            int: Status (0=success, 1=failure, 2=continue)
        """
        # Enum for status (matching C++ implementation)
        SUCCESS = 0
        FAILURE = 1
        TO_CONTINUE = 2
        
        # Check if wave is already in impossible state
        if self.wave.is_impossible:
            return FAILURE
            
        # Get cell with minimum entropy
        argmin = self.wave.get_min_entropy(self.rng)
        
        if argmin == -2:
            return FAILURE  # Contradiction found
        
        if argmin == -1:
            return SUCCESS  # All cells decided
        
        # Convert flat index to 2D coordinates
        y = argmin // self.wave.width
        x = argmin % self.wave.width
        
        # Get possible patterns and their weights
        possible_patterns = np.where(self.wave.data[y, x])[0]
        
        # If no patterns are possible (contradiction), return failure
        if len(possible_patterns) == 0:
            self.wave.is_impossible = True
            return FAILURE
            
        weights = np.array([self.pattern_frequencies[p] for p in possible_patterns])
        
        # If no valid weights, return failure
        if np.sum(weights) <= 1e-10:
            self.wave.is_impossible = True
            return FAILURE
            
        # Normalize weights
        weights = weights / np.sum(weights)
        
        try:
            # Randomly choose a pattern according to weights
            chosen_idx = self.rng.choice(len(possible_patterns), p=weights)
            chosen_value = possible_patterns[chosen_idx]
            
            # Set the chosen pattern and remove others
            for pattern in range(self.num_patterns):
                if self.wave.get(y, x, pattern) != (pattern == chosen_value):
                    self.propagator.add_to_propagator(y, x, pattern)
                    self.wave.set(y, x, pattern, False)
            
            return TO_CONTINUE
        except Exception as e:
            print(f"Error in observe: {e}")
            print(f"weights: {weights}, sum: {np.sum(weights)}")
            self.wave.is_impossible = True
            return FAILURE
    
    def get_wave_state(self) -> np.ndarray:
        """
        Get full wave state (pattern probabilities for each cell)
        
        Returns:
            np.ndarray: 3D array of shape (height, width, num_patterns)
        """
        return self.wave.data
    
    def get_next_collapse_cell(self) -> Tuple[int, int, List[float]]:
        """
        Get next cell to collapse with probabilities
        
        Returns:
            Tuple[int, int, List[float]]: (x, y, probabilities)
        """
        # Get cell with minimum entropy
        argmin = self.wave.get_min_entropy(self.rng)
        
        # No more cells to collapse
        if argmin == -1 or argmin == -2:
            return -1, -1, []
        
        # Convert flat index to 2D coordinates
        y = argmin // self.wave.width
        x = argmin % self.wave.width
        
        # Calculate probabilities for each pattern
        probabilities = np.zeros(self.num_patterns, dtype=np.float64)
        total_weight = 0.0
        
        for p in range(self.num_patterns):
            if self.wave.get(y, x, p):
                probabilities[p] = self.pattern_frequencies[p]
                total_weight += probabilities[p]
        
        # Normalize probabilities
        if total_weight > 0:
            probabilities /= total_weight
        
        return x, y, probabilities.tolist()
    
    def collapse_step(self, action_vec: List[float]) -> Tuple[bool, bool]:
        """
        Single collapse step with action vector
        
        Args:
            action_vec: Action vector with weights for each pattern
            
        Returns:
            Tuple[bool, bool]: (terminate, truncate)
        """
        # Check if the wave is already in an impossible state
        if self.wave.is_impossible:
            return False, True  # terminate=False, truncate=True
            
        # Check if all cells are already decided
        argmin = self.wave.get_min_entropy(self.rng)
        if argmin == -1:
            return True, False  # terminate=True, truncate=False
        elif argmin == -2:
            return False, True  # terminate=False, truncate=True
            
        # Convert flat index to coordinates
        y = argmin // self.wave.width
        x = argmin % self.wave.width
        
        # Convert action to numpy array
        action_array = np.array(action_vec, dtype=np.float64)
        
        # Ensure action vector has the right size
        if len(action_array) != self.num_patterns:
            action_array = np.ones(self.num_patterns, dtype=np.float64)
        
        # Get possible patterns at this location
        possible_patterns = np.where(self.wave.data[y, x])[0]
        
        # If no patterns are possible (contradiction), return failure
        if len(possible_patterns) == 0:
            self.wave.is_impossible = True
            return False, True
            
        # Get weights for possible patterns
        possible_weights = action_array[possible_patterns]
        
        # Handle zero or negative weights
        if np.sum(possible_weights > 0) == 0:
            # If all weights are zero or negative, use uniform distribution
            possible_weights = np.ones_like(possible_weights)
        else:
            # Set negative weights to zero
            possible_weights = np.maximum(possible_weights, 0)
            
        # Normalize weights
        weights_sum = np.sum(possible_weights)
        if weights_sum > 0:
            possible_weights = possible_weights / weights_sum
        else:
            # Fallback to uniform distribution if sum is zero
            possible_weights = np.ones_like(possible_weights) / len(possible_weights)
        
        try:
            # Choose pattern
            chosen_idx = self.rng.choice(len(possible_patterns), p=possible_weights)
            chosen_pattern = possible_patterns[chosen_idx]
            
            # Collapse to chosen pattern
            for p in range(self.num_patterns):
                if p != chosen_pattern and self.wave.get(y, x, p):
                    self.propagator.add_to_propagator(y, x, p)
                    self.wave.set(y, x, p, False)
            
            # Propagate constraints
            self.propagator.propagate(self.wave)
            
            return False, False  # continue
        except Exception as e:
            print(f"Error in collapse_step: {e}")
            print(f"possible_weights: {possible_weights}, sum: {np.sum(possible_weights)}")
            self.wave.is_impossible = True
            return False, True  # terminate=False, truncate=True
    
    def propagate(self) -> None:
        """Propagate constraints in the wave"""
        self.propagator.propagate(self.wave)
    
    def remove_wave_pattern(self, i: int, j: int, pattern: int) -> None:
        """Remove a pattern from a cell"""
        if self.wave.get(i, j, pattern):
            self.wave.set(i, j, pattern, False)
            self.propagator.add_to_propagator(i, j, pattern)
    
    def run(self) -> Optional[np.ndarray]:
        """
        Run the full WFC algorithm
        
        Returns:
            Optional[np.ndarray]: 2D array of pattern indices or None if failed
        """
        # Status enum (matching C++ implementation)
        SUCCESS = 0
        FAILURE = 1
        TO_CONTINUE = 2
        
        while True:
            result = self.observe()
            
            if result == FAILURE:
                return None  # Algorithm failed
            elif result == SUCCESS:
                # Algorithm succeeded, convert wave to output
                output = np.zeros((self.wave.height, self.wave.width), dtype=np.int32)
                
                for y in range(self.wave.height):
                    for x in range(self.wave.width):
                        for p in range(self.num_patterns):
                            if self.wave.get(y, x, p):
                                output[y, x] = p
                                break
                
                return output
            
            self.propagator.propagate(self.wave)  # Propagate constraints