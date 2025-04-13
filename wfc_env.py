import gymnasium as gym
from gymnasium import spaces
import wfc_cpp
import numpy as np

class WFCEnv(gym.Env):
    def __init__(self, input_data, height, width):
        # TODO: Extract patterns from input data
        self.patterns, self.frequencies, self.rules = extract_patterns(input_data)
        self.height = height
        self.width = width

        self.num_patterns = len(self.patterns)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_patterns,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.height, self.width, self.num_patterns))

        # Init env
        _ = self.reset()

    def reset(self):
        self.wfc = wfc_cpp.WFC(False, np.random.randint(0, 1e4), self.frequencies, self.rules, self.height, self.width)
        return self._get_obs()
    
    def get_obs(self):
        obs = self.wfc.get_wave_state()
        return obs.astype(np.float32)
    
    def step(self, action):
        # Apply action to collapse next cell
        terminated, truncated = self.wfc.collapse_step(action)
        
        reward = 1.0 if terminated else 0.0 # Success
        reward = -1.0 if truncated else reward # Contradiction

        # Get observation
        obs = self.get_obs()

        # Info
        info = {}

        return obs, reward, terminated, truncated, info