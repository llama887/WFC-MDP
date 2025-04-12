import os
import optuna
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym # Use Gymnasium instead of gym if using newer SB3/Gymnasium

# Assuming wfc_env.py is in the same directory or accessible
from wfc_env import WFCWrapper

# --- Constants (Copy relevant parts from wfc_env.py) ---
# Ensure these constants match exactly those used in wfc_env.py
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

MAP_LENGTH = 12 # Match wfc_env.py default
MAP_WIDTH = 20  # Match wfc_env.py default
N_EVAL_EPISODES = 5 # Number of episodes to run for evaluation
# Reduce EVAL_FREQ and total_timesteps_tuning for faster tuning trials
EVAL_FREQ = 2000 # Evaluate every N steps within a trial
TOTAL_TIMESTEPS_TUNING = 10000 # Timesteps per Optuna trial
N_TRIALS = 50 # Number of Optuna trials to run
STUDY_NAME = "ppo_wfc_study" # Optuna study name
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db" # Store results in a database
HYPERPARAMS_YAML_FILE = "best_wfc_hyperparams.yaml"
LOG_DIR_TUNE = "./optuna_wfc_logs/"
os.makedirs(LOG_DIR_TUNE, exist_ok=True)

# --- Objective Function for Optuna ---
def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    Trains a PPO agent and returns the mean reward during evaluation.
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    # Ensure batch_size is a divisor of n_steps (since num_envs=1)
    while n_steps % batch_size != 0:
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])
    # gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999]) # Removed as requested
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98, 1.0])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    vf_coef = trial.suggest_float("vf_coef", 0.4, 0.8)
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1, 5])

    # Create unique log dir for this trial
    trial_log_dir = os.path.join(LOG_DIR_TUNE, f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)

    # Create training environment
    # Pass tile_to_index to the environment wrapper
    env = Monitor(
        WFCWrapper(
            map_length=MAP_LENGTH,
            map_width=MAP_WIDTH,
            tile_symbols=tile_symbols,
            adjacency_bool=adjacency_bool,
            num_tiles=num_tiles,
            tile_to_index=tile_to_index,
        ),
        filename=os.path.join(trial_log_dir, "monitor.csv") # Log training rewards
    )

    # Create evaluation environment (important for unbiased evaluation)
    eval_env = Monitor(
        WFCWrapper(
            map_length=MAP_LENGTH,
            map_width=MAP_WIDTH,
            tile_symbols=tile_symbols,
            adjacency_bool=adjacency_bool,
            num_tiles=num_tiles,
            tile_to_index=tile_to_index,
        )
    )

    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(trial_log_dir, "best_model"),
        log_path=trial_log_dir,
        eval_freq=max(EVAL_FREQ // 1, 1), # Ensure eval_freq is at least 1
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        warn=False, # Suppress warnings during tuning
    )

    # Define PPO model with suggested hyperparameters
    # Note: gamma is NOT included here, will use SB3 default (0.99)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0, # Set to 1 to see training logs per trial
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        # gamma=gamma, # Removed
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=None, # Disable tensorboard logging during tuning for speed
        device="cpu",
        seed=trial.number # Use trial number for reproducibility if needed
    )

    mean_reward = -float("inf") # Default value if training fails
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS_TUNING, callback=eval_callback)
        # Retrieve the mean reward from the evaluation callback
        # Check if best_mean_reward is available (it might not be if eval_freq is too high)
        if hasattr(eval_callback, 'best_mean_reward') and eval_callback.best_mean_reward != -np.inf :
             mean_reward = eval_callback.best_mean_reward
        else:
             # Fallback: use last mean reward if best is not set or still -inf
             if hasattr(eval_callback, 'last_mean_reward'):
                 mean_reward = eval_callback.last_mean_reward
             else: # If no evaluation happened, we can't judge the trial well
                 print(f"Warning: No evaluation reward recorded for trial {trial.number}. Check eval_freq and total_timesteps_tuning.")
                 # Returning 0 might be better than -inf if it didn't crash
                 mean_reward = 0.0

        # Pruning based on intermediate values reported by the callback
        trial.report(mean_reward, step=TOTAL_TIMESTEPS_TUNING) # Report final value
        if trial.should_prune():
            raise optuna.TrialPruned()

    except optuna.TrialPruned:
         print(f"Trial {trial.number} pruned.")
         # Optuna handles pruned trials, return the current best reward reported
         return mean_reward
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Penalize failed trials (e.g., due to invalid hyperparameter combinations)
        mean_reward = -float("inf") # Optuna tries to maximize, so return a very low value

    finally:
        # Clean up environments
        env.close()
        eval_env.close()

    # Important: Optuna maximizes the returned value
    return mean_reward


# --- Main Optuna Study Execution ---
if __name__ == "__main__":
    # Set up the study, potentially resuming from the database
    # Added MedianPruner
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        load_if_exists=True, # Resume study if database exists
        direction="maximize", # We want to maximize the evaluation reward
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=EVAL_FREQ // 2), # Prune earlier based on intermediate reports
    )

    print(f"Starting Optuna study: {STUDY_NAME}")
    print(f"Using storage: {STORAGE_NAME}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Timesteps per trial: {TOTAL_TIMESTEPS_TUNING}")
    print(f"Evaluate every: {EVAL_FREQ} steps")

    try:
        # n_jobs=-1 can parallelize trials if objective function is safe, but requires care with file I/O and resources
        study.optimize(objective, n_trials=N_TRIALS, timeout=None) # Set timeout in seconds if needed
    except KeyboardInterrupt:
        print("Optimization stopped manually.")

    print("\nOptimization Finished!")
    print(f"Number of finished trials: {len(study.trials)}")

    # Find the best trial, considering only completed ones
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed_trials:
         print("No trials completed successfully.")
    else:
        # Sort completed trials by value (reward) in descending order
        completed_trials.sort(key=lambda t: t.value, reverse=True)
        best_trial = completed_trials[0] # The trial with the highest reward

        print(f"Best trial number: {best_trial.number}")
        print(f"Best value (mean reward): {best_trial.value}")

        print("Best hyperparameters:")
        best_params = best_trial.params
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        # Save the best hyperparameters to a YAML file
        # Wrap params in a dictionary for better structure in YAML
        hyperparams_to_save = {"ppo": best_params}
        try:
            with open(HYPERPARAMS_YAML_FILE, "w") as f:
                yaml.dump(hyperparams_to_save, f, default_flow_style=False)
            print(f"\nBest hyperparameters saved to {HYPERPARAMS_YAML_FILE}")
        except Exception as e:
            print(f"\nError saving hyperparameters to {HYPERPARAMS_YAML_FILE}: {e}")


    # You can visualize the study results if you have plotly installed
    if len(study.trials) > 0: # Check if there are any trials at all
        try:
            import plotly
            # Check if study has enough trials for plotting
            if len(study.trials) > 1:
                 fig_history = optuna.visualization.plot_optimization_history(study)
                 fig_history.show()
                 # Importance plot requires successful trials
                 if completed_trials:
                     fig_importance = optuna.visualization.plot_param_importances(study)
                     fig_importance.show()
                 else:
                     print("\nNot enough completed trials to generate parameter importances plot.")
            else:
                 print("\nNot enough trials to generate plots.")

        except ImportError:
            print("\nInstall plotly and kaleido to visualize and save optimization results: pip install plotly kaleido")
        except Exception as e:
            print(f"\nCould not plot results: {e}")
