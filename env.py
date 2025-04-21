import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from utils import extract_patterns
from wfc import WFC
from render import WFCRenderer

class WFCEnv(gym.Env):
    """
        Wave Function Collapse Environment for procedural generation.
        
        This environment allows for training agents to guide the WFC algorithm
        by deciding which patterns to collapse at each step.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, input_data, height, width, seed=None, render_mode=None, tile_size=32):
        """
        Initialize the WFC environment.
        
        Args:
            input_data: Path to JSON file or dictionary containing tile data
            height: Height of the output grid
            width: Width of the output grid
            seed: Random seed for reproducibility
        """
        self.patterns, self.frequencies, self.rules = extract_patterns(input_data)
        self.height = height
        self.width = width
        self.render_mode = render_mode
        self.tile_size = tile_size
        self.json_path = input_data if isinstance(input_data, str) else None

        # Renderer
        self.renderer = None
        self.window_width = width * tile_size
        self.window_height = height * tile_size


        # Define action and observation spaces
        self.num_patterns = len(self.patterns)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_patterns,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width, self.num_patterns))

        # Init env
        self.seed = seed
        _ = self.reset()

    def reset(self, seed=None):
        """
        Reset the environment to a fresh state.
        
        Args:
            seed: Optional seed for randomization
            
        Returns:
            Initial observation
            Info dictionary
        """
        if seed is not None:
            self.seed = seed
        elif self.seed is None:
            self.seed = np.random.randint(0, 1e4)
        
        self.wfc = WFC(
            False, 
            self.seed, 
            self.frequencies, 
            self.rules, 
            self.height, 
            self.width
        )
        # Initial observation
        obs =  self.get_obs()
        info = {}

        # Initialize renderer
        if self.render_mode == "human" and self.renderer is None and self.json_path is not None:
            self._init_renderer()
        
        # Render if human mode
        if self.render_mode == "human":
            self.render()

        return obs, info

    def get_obs(self):
        """
        Get current observation from WFC state.
        
        Returns:
            Numpy array representing the wave state
        """
        obs = self.wfc.get_wave_state()
        return obs.astype(np.float32)
    
    def step(self, action):
        """
        Apply action to collapse a cell in the wave.
        
        Args:
            action: Action vector with weights for each pattern
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """

        # Apply action to collapse next cell
        terminated, truncated = self.wfc.collapse_step(action)
        
        reward = 1.0 if terminated else 0.0 # Success
        reward = -1.0 if truncated else reward # Contradiction

        # Get observation
        obs = self.get_obs()

        # Info
        info = {}

         # Render if human mode
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    def get_expert_action(self):
        """
        Get expert action for the next cell to collapse.
        
        Returns:
            Tuple of (x, y, probabilities)
        """
        x, y, probabilities = self.wfc.get_next_collapse_cell()
        return x, y, probabilities
    
    def _init_renderer(self):
        """Initialize the pygame renderer"""
        self.renderer = WFCRenderer(
            json_path=self.json_path,
            screen_width=self.window_width,
            screen_height=self.window_height,
            tile_size=self.tile_size
        )

    def render(self):
        """
        Render the current state of the environment.
        
        Returns:
            Depends on render_mode:
            - 'human': None
            - 'rgb_array': RGB array of the rendered frame
        """
        if self.render_mode is None:
            return
            
        if self.renderer is None and self.json_path is not None:
            self._init_renderer()
        
        if self.renderer is None:
            raise ValueError("Renderer could not be initialized. Make sure json_path is provided.")
            
        # Get observation
        wave_state = self.get_obs()
        
        # Render
        if self.render_mode == "human":
            self.renderer.render(wave_state, show_entropy=False)
            return None
        elif self.render_mode == "rgb_array":
            screen = self.renderer.render(wave_state, show_entropy=False)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
            )
            
    def close(self):
        """Close the environment and free resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

if __name__ == "__main__":
    import time
    json_path = 'data/biome.json'
    width, height = 20, 15
    # Create environment with renderer
    env = WFCEnv(
        input_data=json_path,
        height=height,
        width=width,
        seed=123,
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
                break
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                break
        
        # Check if episode is done
        if terminated or truncated:
            print(f"Episode finished after {step_count} steps with reward {reward}")
            time.sleep(2)  # Pause to show final result
            break
    
    # Close environment
    env.close()
