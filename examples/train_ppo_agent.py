from stable_baselines3 import PPO
from wfc_env import WFCEnv

# TODO: Load and process input data
env = WFCEnv(input_data, height=20, width=20)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)