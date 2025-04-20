import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils import extract_patterns
from wfc import WFC

class WFCEnv(gym.Env):
    """
        Wave Function Collapse Environment for procedural generation.
        
        This environment allows for training agents to guide the WFC algorithm
        by deciding which patterns to collapse at each step.
    """
    
    def __init__(self, input_data, height, width, seed=None):
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

        # Define action and observation spaces
        self.num_patterns = len(self.patterns)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_patterns,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width, self.num_patterns))

        # Init env
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
        obs =  self._get_obs()
        info = {}
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

        return obs, reward, terminated, truncated, info