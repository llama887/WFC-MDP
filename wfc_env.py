import os
import argparse # Import argparse
import yaml # Import yaml
import numpy as np
import gymnasium as gym # Use Gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy # For final evaluation

from fast_wfc import fast_wfc_collapse_step


def grid_to_array(
    grid: np.ndarray,
    all_tiles: list[str],
    map_length: int,
    map_width: int,
) -> np.ndarray:
    arr = np.empty((map_length, map_width), dtype=np.float32)
    for y in range(map_length):
        for x in range(map_width):
            possibilities = grid[y, x, :]
            if np.count_nonzero(possibilities) == 1:
                idx = int(np.argmax(possibilities))
                arr[y, x] = idx / (len(all_tiles) - 1)
            else:
                # Use a distinct value for undecided cells, e.g., -1, normalized later if needed by model
                # Or keep as 0.5 if model handles it. Let's use -1 for clarity.
                arr[y, x] = -1.0
    return arr.flatten()


def wfc_next_collapse_position(grid: np.ndarray) -> tuple[int, int]:
    """Finds the non-collapsed cell with the lowest entropy (fewest options)."""
    min_entropy = float("inf")
    best_cell = (-1, -1) # Initialize with invalid cell indicator
    map_length, map_width, num_tiles = grid.shape
    found_undecided = False

    # Precompute log factors if calculating true entropy, otherwise use count
    # weights = np.ones(num_tiles) # Example: assume uniform weight initially
    # log_weights_sum = np.log(np.sum(weights))

    for y in range(map_length):
        for x in range(map_width):
            possibilities = grid[y, x, :]
            options = np.count_nonzero(possibilities)

            if options > 1: # Only consider cells not yet collapsed
                found_undecided = True
                # Use number of options as a simple proxy for entropy (lower is better)
                entropy = options
                # Add small noise to break ties randomly (helps exploration)
                entropy += np.random.rand() * 1e-6

                if entropy < min_entropy:
                    min_entropy = entropy
                    best_cell = (x, y)

    # If no undecided cells found, return the indicator (-1, -1)
    if not found_undecided:
        return (-1, -1)

    # This fallback should ideally not be needed if the loop logic is correct,
    # but acts as a safeguard if all remaining cells have the same minimal entropy.
    if best_cell == (-1, -1) and found_undecided:
         for y in range(map_length):
            for x in range(map_width):
                 if np.count_nonzero(grid[y, x, :]) > 1:
                     return (x,y) # Return the first undecided cell found

    return best_cell


def fake_reward(grid: np.ndarray, num_tiles: int, tile_to_index: dict, terminated: bool, truncated: bool) -> float:
    """
    Calculates reward. Only gives non-zero reward at the end of an episode.
    Penalizes truncation (contradiction).
    Rewards successful termination based on proximity to target tile count.
    """
    if truncated:
        # print("Truncated, reward: -1000") # Debug print
        return -1000.0 # Heavy penalty for contradictions

    if not terminated:
        return 0.0 # No reward during the episode

    # --- Reward calculation only if terminated successfully ---
    target_tile = "X" # The tile we want to count
    desired_target_count = 20 # Example target count

    if target_tile not in tile_to_index:
        print(f"Warning: Target tile '{target_tile}' not found in tile_to_index.")
        return -500.0 # Penalize if setup is wrong

    target_idx = tile_to_index[target_tile]

    # Create a one-hot representation for the target tile
    target_one_hot = np.zeros(num_tiles, dtype=bool) # Use bool to match grid dtype
    target_one_hot[target_idx] = True

    # Count cells that have collapsed *exactly* to the target tile
    # Ensure grid is boolean before comparison
    matches = np.all(grid == target_one_hot, axis=-1)
    count = np.sum(matches)

    # Reward based on how close the count is to the desired count
    # Using squared error gives a stronger signal near the target.
    # Normalize the error? Max possible error is max(desired, width*height)
    max_possible_count = grid.shape[0] * grid.shape[1]
    # Max possible squared error: difference between desired and 0, or desired and max_possible
    max_error_sq = float(max((desired_target_count - 0)**2, (desired_target_count - max_possible_count)**2))
    error_sq = float((desired_target_count - count) ** 2)

    # Scale reward between 0 (max error) and +100 (perfect match)
    # Avoid division by zero if max_error_sq is 0 (e.g., 1x1 grid, desired=0)
    if max_error_sq > 1e-6:
        # Linear scaling: reward = 100 * (1 - sqrt(error_sq) / sqrt(max_error_sq))
        # Quadratic scaling (more penalty further away):
        normalized_reward = 100.0 * (1.0 - (error_sq / max_error_sq))
    else:
        normalized_reward = 100.0 if error_sq < 1e-6 else 0.0

    # Ensure reward is non-negative for successful termination
    final_reward = max(0.0, normalized_reward)
    # print(f"Terminated. Count={count}, Desired={desired_target_count}, Reward={final_reward}") # Debug print
    return final_reward


# # Target subarray to count
# target = np.array([0, 1, 0])

# # Count occurrences of the target subarray along the last dimension
# matches = np.all(array == target, axis=-1)
class WFCWrapper(gym.Env):
    """
    Gymnasium Environment for Wave Function Collapse controlled by an RL agent.

    Observation: Flattened grid state + normalized coordinates of the next cell to collapse.
                 Grid cells: Value is index/(num_tiles-1) if collapsed, -1.0 if undecided.
    Action: A vector of preferences (logits) for each tile type for the selected cell.
    Reward: Sparse reward given only at the end of the episode.
            + Scaled reward (0 to 100) based on proximity to target tile count for successful termination.
            - 1000 for truncation (contradiction or max steps).
            0 otherwise.
    Termination: Grid is fully collapsed (all cells have exactly one possibility).
    Truncation: A contradiction occurs during propagation OR max steps reached.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10} # Add metadata, adjust FPS

    def __init__(
        self,
        map_length: int,
        map_width: int,
        tile_symbols: list[str],
        adjacency_bool: np.ndarray,
        num_tiles: int,
        tile_to_index: dict[str, int], # Add tile_to_index
    ):
        super().__init__() # Call parent constructor
        self.all_tiles = tile_symbols
        self.adjacency = adjacency_bool
        self.num_tiles = num_tiles
        self.map_length: int = map_length
        self.map_width: int = map_width
        self.tile_to_index = tile_to_index # Store tile_to_index

        # Initial grid state (all possibilities True)
        self.initial_grid = np.ones(
            (self.map_length, self.map_width, self.num_tiles), dtype=bool
        )
        # self.grid will hold the current state during an episode
        self.grid = self.initial_grid.copy()

        # Action space: Agent outputs preferences (logits) for each tile type.
        # Needs to be float32 for SB3.
        self.action_space: spaces.Box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_tiles,), dtype=np.float32
        )

        # Observation space: Flattened map + normalized coordinates of the next cell to collapse
        # Map values range from -1 (undecided) to 1 (max index / max index).
        # Coordinates range from 0 to 1. Needs to be float32 for SB3.
        self.observation_space: spaces.Box = spaces.Box(
            low=-1.0, # Lower bound changed due to -1 for undecided cells
            high=1.0,
            shape=(self.map_length * self.map_width + 2,),
            dtype=np.float32,
        )
        self.current_step = 0
        # Set a maximum number of steps to prevent infinite loops if termination fails
        self.max_steps = self.map_length * self.map_width + 10 # Allow some buffer

    def get_observation(self) -> np.ndarray:
        """Constructs the observation array (needs to be float32)."""
        map_flat = grid_to_array(
            self.grid, self.all_tiles, self.map_length, self.map_width
        )
        pos = wfc_next_collapse_position(self.grid)

        # Handle case where grid is fully collapsed (pos is (-1, -1))
        if pos == (-1, -1):
             # If fully collapsed, the position doesn't matter much, use (0,0)
             # The episode should terminate based on this state anyway.
             pos_array = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # Normalize the collapse position (x, y) to be between 0 and 1
            # Avoid division by zero if width/length is 1
            norm_x = pos[0] / (self.map_width - 1) if self.map_width > 1 else 0.0
            norm_y = pos[1] / (self.map_length - 1) if self.map_length > 1 else 0.0
            pos_array = np.array([norm_x, norm_y], dtype=np.float32)

        # Ensure final observation is float32
        return np.concatenate([map_flat, pos_array]).astype(np.float32)

    def step(self, action: np.ndarray):
        """Performs one step of the WFC process based on the agent's action."""
        self.current_step += 1

        # Ensure action is float32 numpy array
        action = np.asarray(action, dtype=np.float32)

        # Convert action (potentially logits) to a probability distribution using softmax
        # Improve numerical stability by subtracting the max before exponentiating
        action_exp = np.exp(action - np.max(action))
        action_probs = action_exp / (np.sum(action_exp) + 1e-8) # Add epsilon for stability

        # Ensure probabilities are float64 for the Numba function if required
        action_probs_64 = action_probs.astype(np.float64)

        # Call the core WFC collapse and propagate step
        # deterministic=False during training allows stochastic choice based on probs
        self.grid, terminated, truncated = fast_wfc_collapse_step(
            self.grid,
            self.map_width,
            self.map_length,
            self.num_tiles,
            self.adjacency,
            action_probs_64, # Pass normalized probabilities as float64
            deterministic=False
        )

        # Check for truncation due to reaching max steps
        if not terminated and not truncated and self.current_step >= self.max_steps:
             # print(f"Max steps reached ({self.current_step}), truncating.") # Debug print
             truncated = True
             terminated = False # Cannot be both terminated and truncated

        # Calculate reward (only non-zero at the end)
        reward = fake_reward(self.grid, self.num_tiles, self.tile_to_index, terminated, truncated)

        # Get the next observation
        observation = self.get_observation()
        info = {} # Provide additional info if needed (e.g., current step count)
        info["steps"] = self.current_step
        if terminated:
            info["terminated_reason"] = "completed"
        if truncated:
            info["truncated_reason"] = "contradiction" if self.current_step < self.max_steps else "max_steps"


        # Ensure reward is float
        reward = float(reward)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed) # Handle seeding correctly via Gymnasium Env
        self.grid = self.initial_grid.copy()
        self.current_step = 0
        observation = self.get_observation()
        info = {} # Can provide initial info if needed
        # print("Environment Reset") # Debug print
        return observation, info

    def render(self, mode="human"):
        """Renders the current grid state to the console."""
        if mode == "human":
            print(f"--- Step: {self.current_step} ---")
            for y in range(self.map_length):
                row_str = ""
                for x in range(self.map_width):
                    possibilities = self.grid[y, x, :]
                    num_options = np.count_nonzero(possibilities)
                    if num_options == 1:
                        idx = np.argmax(possibilities)
                        row_str += self.all_tiles[idx] + " "
                    elif num_options == self.num_tiles:
                         row_str += "? " # Not touched yet (all possibilities)
                    elif num_options == 0:
                         row_str += "! " # Contradiction
                    else:
                        # Indicate superposition simply
                        row_str += ". "
                print(row_str.strip())
            print("-" * (self.map_width * 2))
        else:
             # Handle other modes or just pass as per gym interface
             # return super().render(mode=mode) # Use this if inheriting from gym.Env directly
             pass # No other render modes implemented


    def close(self):
        """Cleans up any resources used by the environment."""
        # No specific resources to clean up in this case
        pass
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train PPO for WFC Environment.")
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=None, # Default is None, meaning use defaults below
        help="Path to YAML file containing hyperparameters (e.g., best_wfc_hyperparams.yaml).",
    )
    parser.add_argument(
        "--load-best",
        action="store_true", # Make it a flag
        help="Load hyperparameters from the default 'best_wfc_hyperparams.yaml' file.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000, # Default training timesteps
        help="Total number of timesteps for training.",
    )
    parser.add_argument(
        "--map-length",
        type=int,
        default=12,
        help="Length (height) of the map.",
    )
    parser.add_argument(
        "--map-width",
        type=int,
        default=20,
        help="Width of the map.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./ppo_wfc_logs/",
        help="Directory to save logs and models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # --- Constants and Setup ---
    # Use map dimensions from args
    MAP_LENGTH = args.map_length
    MAP_WIDTH = args.map_width
    LOG_DIR = args.log_dir
    os.makedirs(LOG_DIR, exist_ok=True)
    DEFAULT_HYPERPARAMS_FILE = "best_wfc_hyperparams.yaml"

    # Define Tiles (ensure this matches the set used in tuning if loading params)
    PAC_TILES = {
        " ": {"edges": {"U": "OPEN", "R": "OPEN", "D": "OPEN", "L": "OPEN"}},
        "X": {"edges": {"U": "OPEN", "R": "OPEN", "D": "OPEN", "L": "OPEN"}}, # Example target tile
        "═": {"edges": {"U": "OPEN", "R": "LINE", "D": "OPEN", "L": "LINE"}},
        "║": {"edges": {"U": "LINE", "R": "OPEN", "D": "LINE", "L": "OPEN"}},
        "╔": {"edges": {"U": "OPEN", "R": "LINE", "D": "LINE", "L": "OPEN"}},
        "╗": {"edges": {"U": "OPEN", "R": "OPEN", "D": "LINE", "L": "LINE"}},
        "╚": {"edges": {"U": "LINE", "R": "LINE", "D": "OPEN", "L": "OPEN"}},
        "╝": {"edges": {"U": "LINE", "R": "OPEN", "D": "OPEN", "L": "LINE"}},
    }
    OPPOSITE_DIRECTION = {"U": "D", "D": "U", "L": "R", "R": "L"}
    DIRECTIONS = ["U", "R", "D", "L"]
    tile_symbols = list(PAC_TILES.keys())
    num_tiles = len(tile_symbols)
    tile_to_index = {s: i for i, s in enumerate(tile_symbols)}

    # Precompute Adjacency Matrix
    adjacency_bool = np.zeros((num_tiles, 4, num_tiles), dtype=np.bool_)
    for i, tile_a in enumerate(tile_symbols):
        for d, direction in enumerate(DIRECTIONS):
            for j, tile_b in enumerate(tile_symbols):
                # Ensure edges exist for both tiles before accessing
                if "edges" in PAC_TILES[tile_a] and direction in PAC_TILES[tile_a]["edges"] and \
                   "edges" in PAC_TILES[tile_b] and OPPOSITE_DIRECTION[direction] in PAC_TILES[tile_b]["edges"]:
                    edge_a = PAC_TILES[tile_a]["edges"][direction]
                    edge_b = PAC_TILES[tile_b]["edges"][OPPOSITE_DIRECTION[direction]]
                    if edge_a == edge_b:
                        adjacency_bool[i, d, j] = True
                # else: Handle tiles without defined edges if necessary (implicitly incompatible)

    # --- Hyperparameter Loading ---
    # Define default hyperparameters (SB3 PPO defaults + common adjustments)
    ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99, # Default gamma, not tuned but kept here
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": args.seed, # Pass seed from args
        "device": "cpu", # Default to CPU, can be changed
    }

    hyperparams_file_to_load = None
    if args.hyperparams:
        hyperparams_file_to_load = args.hyperparams
        print(f"Attempting to load hyperparameters from specified file: {hyperparams_file_to_load}")
    elif args.load_best:
        if os.path.exists(DEFAULT_HYPERPARAMS_FILE):
             hyperparams_file_to_load = DEFAULT_HYPERPARAMS_FILE
             print(f"Attempting to load hyperparameters from default best file: {hyperparams_file_to_load}")
        else:
             print(f"Warning: --load-best flag set, but default file '{DEFAULT_HYPERPARAMS_FILE}' not found. Using default parameters.")

    if hyperparams_file_to_load:
        try:
            with open(hyperparams_file_to_load, "r") as f:
                loaded_params_yaml = yaml.safe_load(f)
                # Expecting a structure like {'ppo': {'learning_rate': ...}}
                if loaded_params_yaml and isinstance(loaded_params_yaml, dict) and "ppo" in loaded_params_yaml:
                    loaded_ppo_params = loaded_params_yaml["ppo"]
                    if isinstance(loaded_ppo_params, dict):
                        # Update the defaults with the loaded params
                        ppo_params.update(loaded_ppo_params)
                        print("Successfully loaded and updated hyperparameters:")
                        # Print loaded params neatly
                        for key, value in loaded_ppo_params.items():
                             print(f"  - {key}: {value}")
                    else:
                         print(f"Warning: 'ppo' key in YAML file '{hyperparams_file_to_load}' does not contain a dictionary. Using default parameters.")
                else:
                     print(f"Warning: YAML file '{hyperparams_file_to_load}' has incorrect format or missing 'ppo' key. Using default parameters.")
        except FileNotFoundError:
            print(f"Error: Hyperparameter file not found: {hyperparams_file_to_load}. Using default parameters.")
        except yaml.YAMLError as e:
             print(f"Error parsing YAML file {hyperparams_file_to_load}: {e}. Using default parameters.")
        except Exception as e:
            print(f"Error loading hyperparameters from {hyperparams_file_to_load}: {e}. Using default parameters.")
    else:
         print("Using default hyperparameters:")
         # Print default params neatly
         for key, value in ppo_params.items():
              print(f"  - {key}: {value}")


    # --- Environment Creation ---
    # Use a lambda function to pass arguments easily, especially for VecEnv later if needed
    env_kwargs = dict(
        map_length=MAP_LENGTH,
        map_width=MAP_WIDTH,
        tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool,
        num_tiles=num_tiles,
        tile_to_index=tile_to_index,
    )
    # Wrap the environment with Monitor to log rewards and episode info
    env = Monitor(WFCWrapper(**env_kwargs), filename=os.path.join(LOG_DIR, "monitor.csv"))
    # Create a separate Monitor-wrapped instance for evaluation
    eval_env = Monitor(WFCWrapper(**env_kwargs))

    # Check if the environment follows the Gym interface.
    try:
        check_env(env, warn=True)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        print("Attempting to continue, but the environment might be incompatible.")
        # Consider exiting if check fails: exit(1)

    # --- Callbacks ---
    # Save checkpoints less frequently during longer training runs
    # Ensure save_freq is at least 1
    save_freq_checkpoints = max(args.total_timesteps // 20, 5000) # e.g., save 20 times or every 5k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_checkpoints,
        save_path=LOG_DIR,
        name_prefix="ppo_wfc_checkpoint"
    )

    # Evaluate less frequently but maybe more episodes for better estimate
    # Ensure eval_freq is at least 1
    eval_freq_callback = max(args.total_timesteps // 50, 2000) # e.g., eval 50 times or every 2k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"), # Save best model here
        log_path=LOG_DIR,
        eval_freq=eval_freq_callback,
        n_eval_episodes=10, # More episodes for robust evaluation
        deterministic=True,
        render=False,
        warn=False # Suppress callback warnings during training
    )

    # --- Model Training ---
    # Use the loaded/default ppo_params dictionary
    # Ensure seed is passed correctly if loaded from YAML or args
    if "seed" not in ppo_params:
         ppo_params["seed"] = args.seed # Ensure seed is set

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        **ppo_params # Unpack the hyperparameter dictionary
    )

    print("\n--- Training Configuration ---")
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Map Dimensions: {MAP_LENGTH}x{MAP_WIDTH}")
    print(f"Log Directory: {LOG_DIR}")
    print("PPO Hyperparameters:")
    # Print effective hyperparameters used by the model
    model_params = model.get_parameters()
    policy_params = model_params.get("policy", {}) # PPO stores most params directly
    for key, value in policy_params.items():
         # Filter out large objects like optimizer state for clarity
         if isinstance(value, (int, float, str, bool, list, tuple)) and key != 'optimizer_class' and key != 'optimizer_kwargs':
             print(f"  {key}: {value}")
    # Print others that might not be in policy dict directly
    print(f"  learning_rate: {model.learning_rate}")
    print(f"  n_steps: {model.n_steps}")
    print(f"  batch_size: {model.batch_size}")
    print(f"  n_epochs: {model.n_epochs}")
    print(f"  seed: {ppo_params.get('seed')}") # Get seed from original dict
    print("----------------------------\n")


    print(f"Starting training for {args.total_timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True # Show progress bar
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save the final model
    final_model_path = os.path.join(LOG_DIR, "ppo_wfc_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}.zip")

    # Optional: Load the best model found during training and evaluate it
    best_model_path = os.path.join(LOG_DIR, "best_model", "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final evaluation...")
        try:
            # Load the best model using the eval_env
            best_model = PPO.load(best_model_path, env=eval_env)
            # Evaluate the loaded best model
            mean_reward, std_reward = evaluate_policy(
                best_model,
                eval_env, # Use the monitored eval_env
                n_eval_episodes=20,
                deterministic=True,
                warn=False
            )
            print(f"Best model evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        except Exception as e:
            print(f"Could not load or evaluate best model: {e}")
    else:
        print("\nBest model checkpoint not found.")

    env.close()
    eval_env.close()
    print("\nTraining finished.")
