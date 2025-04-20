import os
import sys
import time
import pygame
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wfc import WFC
from render import WFCRenderer, create_renderer_for_env
from env import WFCEnv
from utils import extract_patterns


def test_standalone_renderer():
    """Test the standalone renderer with manual WFC running"""
    # Load the JSON data
    json_path = 'data/biome.json'
    
    # Extract patterns
    with open(json_path, 'r') as f:
        import json
        data = json.load(f)
    
    patterns, frequencies, rules = extract_patterns(data)
    
    # Create a WFC instance
    width, height = 20, 15
    seed = 42
    
    wfc = WFC(
        False,  # periodic_output
        seed,   # random seed
        frequencies,
        rules,
        height,
        width
    )
    
    # Create renderer
    renderer = WFCRenderer(
        json_path=json_path,
        screen_width=800,
        screen_height=600,
        tile_size=32
    )
    
    print("Testing standalone renderer...")
    print(f"Grid size: {width}x{height}")
    print(f"Number of patterns: {len(patterns)}")
    
    # Run and render steps
    running = True
    step_count = 0
    max_steps = 400  # Prevent infinite loops
    
    while running and step_count < max_steps:
        # Get current wave state
        wave_state = wfc.get_wave_state()
        
        # Render
        renderer.render(wave_state, show_entropy=False)
        
        # Handle events
        if renderer.handle_events():
            break
            
        # Get next cell to collapse
        x, y, probs = wfc.get_next_collapse_cell()
        
        if x == -1 or y == -1:  # No more cells to collapse
            print("WFC completed!")
            time.sleep(2)  # Pause to show final result
            break
        
        # Perform a step with expert (wfc) probabilities
        terminated, truncated = wfc.collapse_step(probs)
        
        if terminated or truncated:
            print(f"WFC {'completed successfully' if terminated else 'failed with contradiction'}")
            time.sleep(2)  # Pause to show final result
            break
            
        step_count += 1
        
        # Small delay to see the process
        time.sleep(0.05)
    
    print(f"Completed after {step_count} steps")
    renderer.close()


def test_wfc_env_renderer():
    """Test the renderer integrated with the WFCEnv class"""
    json_path = 'data/biome.json'
    width, height = 20, 15
    
    # Create environment with renderer
    env = WFCEnv(
        input_data=json_path,
        height=height,
        width=width,
        seed=42,
        render_mode="human",
        tile_size=32
    )
    
    print("Testing WFCEnv with integrated renderer...")
    print(f"Grid size: {width}x{height}")
    print(f"Number of patterns: {env.num_patterns}")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Run a fixed number of steps
    num_steps = 400
    step_count = 0
    
    while step_count < num_steps:
        # Create a random action (can be replaced with a trained agent)
        x, y, action = env.get_expert_action()
        formatted_action = [f"{a:.2f}" for a in action]
        print(f"Step {step_count}: collapsing cell ({x}, {y}) with action {formatted_action}\n") 
        # Take a step
        obs, reward, terminated, truncated, _ = env.step(action)
        
        step_count += 1
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return
        
        # Check if episode is done
        if terminated or truncated:
            print(f"Episode finished after {step_count} steps with reward {reward}")
            time.sleep(2)  # Pause to show final result
            break
            
        # Small delay to see the process
        time.sleep(0.05)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # Test standalone renderer
    try:
        test_standalone_renderer()
    except Exception as e:
        print(f"Error in standalone renderer test: {e}")
    
    # Test WFCEnv renderer
    try:
        test_wfc_env_renderer()
    except Exception as e:
        print(f"Error in WFCEnv renderer test: {e}")