import numpy as np
import json
from utils import extract_patterns
from wfc import WFC

def test_extraction():
    """Test the extraction of patterns, frequencies, and rules."""
    # Load the JSON data
    with open('data/biome.json', 'r') as f:
        data = json.load(f)
    
    # Extract patterns
    patterns, frequencies, rules = extract_patterns(data)
    
    # Print info
    print(f"Number of patterns: {len(patterns)}")
    print(f"Number of frequencies: {len(frequencies)}")
    print(f"Number of rule sets: {len(rules)}")
    
    # Test some of the rules
    print("\nSample rules:")
    for i in range(3):
        pattern_id = i
        print(f"\nPattern {pattern_id} compatibility:")
        for direction, compatible in enumerate(rules[pattern_id]):
            direction_name = "URDL"[direction]
            print(f"  Direction {direction_name}: {compatible}")
    
    return patterns, frequencies, rules

def test_wfc_generation(patterns, frequencies, rules, width=20, height=15, seed=42):
    """Test WFC generation with the extracted patterns."""
    # Create a WFC instance
    wfc = WFC(
        False,  # periodic_output
        seed,   # random seed
        frequencies,
        rules,
        height,
        width
    )
    
    # Run the algorithm
    print("\nRunning WFC algorithm...")
    wave_state = wfc.get_wave_state()
    print(f"type of wave state: {type(wave_state)}")
    print(f"Initial wave state shape: {wave_state.shape}")
    
    # Get the next cell to collapse and its probabilities
    x, y, probs = wfc.get_next_collapse_cell()
    print(f"Next cell to collapse: ({x}, {y})")
    print(f"Probabilities: {probs[:5]}..." if len(probs) > 5 else f"Probabilities: {probs}")
    
    # Run a step with uniform probabilities
    terminated, truncated = wfc.collapse_step([1.0] * len(patterns))
    print(f"Step result: terminated={terminated}, truncated={truncated}")
    
    # Get the updated wave state
    wave_state = wfc.get_wave_state()
    print(f"Updated wave state shape: {wave_state.shape}")
    
    # Visualize a small part of the wave state
    print("\nWave state for the first 3x3 region and first 5 patterns:")
    for i in range(min(3, height)):
        for j in range(min(3, width)):
            print(f"({i},{j}): {wave_state[i,j,:5]}..." if wave_state.shape[2] > 5 else f"({i},{j}): {wave_state[i,j,:]}")
    
    return wfc

if __name__ == "__main__":
    # Test the extraction
    patterns, frequencies, rules = test_extraction()
    
    # Test WFC generation
    wfc = test_wfc_generation(patterns, frequencies, rules)