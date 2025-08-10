from __future__ import annotations

import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pygame
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from assets.biome_adjacency_rules import create_adjacency_matrix
from core.wfc import load_tile_images
from core.wfc_env import CombinedReward, WFCWrapper
from tasks.binary_task import binary_reward
from tasks.grass_task import grass_reward
from tasks.hill_task import hill_reward
from tasks.pond_task import pond_reward
from tasks.river_task import river_reward


# ---------- Rendering helpers (unchanged API) ----------
def render_boolean_grid(grid_3d, tile_images, all_tiles, tile_size=32):
    height, width, _ = grid_3d.shape
    surface = pygame.Surface((width * tile_size, height * tile_size))
    surface.fill((255, 255, 255))
    for y in range(height):
        for x in range(width):
            cell = grid_3d[y, x]
            s = int(np.sum(cell))
            if s == 0:
                pygame.draw.rect(surface, (255, 0, 0), (x * tile_size, y * tile_size, tile_size, tile_size))
            elif s == 1:
                idx = int(np.argmax(cell))
                tile_name = all_tiles[idx]
                if tile_name in tile_images:
                    surface.blit(tile_images[tile_name], (x * tile_size, y * tile_size))
                else:
                    pygame.draw.rect(surface, (0, 255, 0), (x * tile_size, y * tile_size, tile_size, tile_size))
            else:
                pygame.draw.rect(surface, (200, 200, 200), (x * tile_size, y * tile_size, tile_size, tile_size))
    return surface

def deepcopy_env_state(env: WFCWrapper):
    return {
        'map_length': env.map_length,
        'map_width': env.map_width,
        'grid': copy.deepcopy(env.grid),
        'all_tiles': copy.copy(env.all_tiles),
    }


# ---------- PPO-specific bits ----------
@dataclass
class SavedRLAgent:
    model_path: str
    reward: float
    info: dict[str, Any]


class SaveBestRenderCallback(EvalCallback):
    """
    Extends EvalCallback:
      - Keeps eval early-stopping via patience, but counts ONLY actual evaluation events,
        not training steps.
      - Optionally skips the initial evaluation at step 0 (common SB3 behavior) to avoid
        triggering premature patience increments.
      - (Optionally) saves a rendered image whenever we get a new best.

    Parameters
    ----------
    eval_env : gym.Env
        Evaluation environment (same as EvalCallback).
    best_model_save_path : str
        Directory where the best model is saved.
    log_path : str
        Directory where evaluation logs are saved.
    task_info : str
        Label for filenames (e.g., task combination).
    tile_images : dict | None
        Mapping tile_name -> pygame.Surface used for rendering (optional).
    env_builder : Callable[[], WFCWrapper]
        Factory that returns a *fresh* environment instance for rendering.
    save_best_per_gen_dir : str | None
        If provided, save a PNG whenever a new best evaluation mean reward is found.
    eval_freq : int
        Evaluate every N calls (same as EvalCallback).
    patience_evals : int | None
        Number of evaluation events with no improvement before early-stopping.
        If None, never early-stop.
    deterministic : bool
        Deterministic evaluation actions (same as EvalCallback).
    skip_first_eval : bool
        If True (default), ignore the very first evaluation that happens at/near step 0.
    """

    def __init__(
        self,
        eval_env,
        best_model_save_path: str,
        log_path: str,
        task_info: str,
        tile_images: dict | None,
        env_builder: Callable[[], Any],
        save_best_per_gen_dir: str | None,
        eval_freq: int,
        patience_evals: int | None = None,
        deterministic: bool = True,
        skip_first_eval: bool = True,
    ):
        super().__init__(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            deterministic=deterministic,
            render=False,
        )
        self.task_info = task_info
        self.tile_images = tile_images
        self.env_builder = env_builder
        self.save_best_per_gen_dir = save_best_per_gen_dir
        self.patience_evals = patience_evals
        self.skip_first_eval = skip_first_eval

        self.no_improve_counter: int = 0
        self.best_mean_reward_seen: float | None = None

        # Track how many evaluation events we've *processed* to ensure
        # we increment patience only when a new evaluation happens.
        self._seen_eval_count: int = 0
        self._skipped_initial_eval: bool = False

    def _on_step(self) -> bool:
        # Let EvalCallback handle scheduling + metrics updates first
        result = super()._on_step()

        # Detect if an evaluation just ran by observing changes in evaluations_results
        just_evaluated: bool = (
            hasattr(self, "evaluations_results")
            and isinstance(self.evaluations_results, list)
            and len(self.evaluations_results) > self._seen_eval_count
        )

        # If no evaluation occurred this step, nothing to do here
        if not just_evaluated or self.last_mean_reward is None:
            return result

        # Update our local counter of seen evaluations
        self._seen_eval_count = len(self.evaluations_results)

        # Optionally skip the first eval (commonly at step 0)
        if self.skip_first_eval and not self._skipped_initial_eval:
            self._skipped_initial_eval = True
            # Do not touch patience, do not consider best, just ignore this eval
            return result

        # Evaluate improvement logic ONLY on evaluation events
        if self.best_mean_reward_seen is None or self.last_mean_reward > self.best_mean_reward_seen:
            self.best_mean_reward_seen = self.last_mean_reward
            self.no_improve_counter = 0

            # Save a render of the current best, if requested
            if self.save_best_per_gen_dir and self.tile_images is not None:
                try:
                    Path(self.save_best_per_gen_dir).mkdir(parents=True, exist_ok=True)

                    # Headless pygame
                    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
                    pygame.init()

                    env = self.env_builder()
                    env.tile_images = self.tile_images
                    env.render_mode = "human"
                    obs, _ = env.reset()

                    # rollout with the current best model
                    from stable_baselines3.common.base_class import BaseAlgorithm
                    assert isinstance(self.model, BaseAlgorithm)
                    done, truncated = False, False
                    while not (done or truncated):
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, info = env.step(action)

                    surface = env.render()
                    if surface:
                        timestamp = int(time.time())
                        filename = f"best_{self.task_info}_{self.last_mean_reward:.2f}_{timestamp}.png"
                        full_path = os.path.join(self.save_best_per_gen_dir, filename)
                        pygame.image.save(surface, full_path)
                        print(f"[DEBUG] Saved best render: {full_path}")
                    env.close()
                except Exception as e:
                    print(f"[WARN] Failed to render best: {e}")
                finally:
                    pygame.quit()
                    if "SDL_VIDEODRIVER" in os.environ:
                        del os.environ["SDL_VIDEODRIVER"]

        else:
            # No improvement in this evaluation
            self.no_improve_counter += 1
            if self.patience_evals is not None and self.no_improve_counter >= self.patience_evals:
                print(f"[DEBUG] Early stopping: no eval improvement for {self.no_improve_counter} evaluations.")
                return False

        return result


TASK_REWARDS: dict[str, Callable[..., tuple[float, dict[str, Any]]]] = {
        "binary_easy": partial(binary_reward, target_path_length=20),
        "binary_hard": partial(binary_reward, target_path_length=20, hard=True),
        "river": river_reward,
        "pond": pond_reward,
        "grass": grass_reward,
        "hill": hill_reward,
    }

def build_selected_reward(tasks: list[str]) -> Callable:
    if len(tasks) == 1:
        task = tasks[0]
        fn = TASK_REWARDS[task]
        def reward_fn(grid):
            out = fn(grid)
            if isinstance(out, tuple):
                return out
            return float(out), {}
        return reward_fn
    else:
        parts: list[Callable] = []
        for task in tasks:
            parts.append(lambda grid, t=task: TASK_REWARDS[t](grid))
        return CombinedReward(parts)


def make_env_factory(
    map_length: int,
    map_width: int,
    tile_symbols: list[str],
    adjacency_bool: np.ndarray,
    num_tiles: int,
    tile_to_index: dict[str, int],
    reward_fn: Callable,
) -> Callable[[], WFCWrapper]:
    def _make() -> WFCWrapper:
        return WFCWrapper(
            map_length=map_length,
            map_width=map_width,
            tile_symbols=tile_symbols,
            adjacency_bool=adjacency_bool,
            num_tiles=num_tiles,
            tile_to_index=tile_to_index,
            reward=reward_fn,
            deterministic=True,
            render_mode=None,
        )
    return _make


def train_with_ppo(
    env_builder: Callable[[], WFCWrapper],
    episodes: int,
    n_envs: int,
    eval_freq: int,
    patience_evals: int | None,
    output_dir: str,
    task_info: str,
    tile_images: dict | None,
    save_best_per_gen: bool,
    ppo_hparams: dict[str, Any],
    seed: int = 42,
) -> tuple[SavedRLAgent, list[float], list[float]]:
    """
    Train PPO. We map 'episodes' to total timesteps via:
        total_timesteps ≈ episodes * steps_per_episode
    using env.max_steps as steps_per_episode upper bound.
    Note: total_timesteps in SB3 counts total transitions across all vec envs.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    # for mapping episodes -> timesteps
    tmp_env = env_builder()
    steps_per_episode = int(getattr(tmp_env, "max_steps", tmp_env.map_width * tmp_env.map_length + 10))
    del tmp_env

    total_timesteps = int(episodes * steps_per_episode)

    vec_env = make_vec_env(env_builder, n_envs=max(1, n_envs), seed=seed)
    eval_env = Monitor(env_builder())

    best_dir = os.path.join(output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    best_gen_dir = None
    if save_best_per_gen:
        best_gen_dir = os.path.join(output_dir, "best_gen")
        os.makedirs(best_gen_dir, exist_ok=True)

    callback = SaveBestRenderCallback(
        eval_env=eval_env,
        best_model_save_path=best_dir,
        log_path=output_dir,
        task_info=task_info,
        tile_images=tile_images,
        env_builder=env_builder,
        save_best_per_gen_dir=best_gen_dir,
        eval_freq=eval_freq,
        patience_evals=patience_evals,
        deterministic=True,
    )

    default_hparams: dict[str, Any] = dict(
        policy="MlpPolicy",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=output_dir,
        seed=seed,
    )
    default_hparams.update(ppo_hparams or {})

    model = PPO(env=vec_env, **default_hparams)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    best_model_path = os.path.join(best_dir, "best_model.zip")
    if not os.path.exists(best_model_path):
        model.save(best_model_path)

    best_eval_reward = float(callback.best_mean_reward or 0.0)
    saved = SavedRLAgent(model_path=best_model_path, reward=best_eval_reward, info={})
    # histories (we keep the plotting API — single series)
    best_eval_history: list[float] = [best_eval_reward] if callback.best_mean_reward is not None else []
    mean_eval_history: list[float] = [best_eval_reward] if callback.best_mean_reward is not None else []
    return saved, best_eval_history, mean_eval_history


def objective(
    trial: optuna.Trial,
    episodes_per_trial: int,
    tasks_list: list[str],
    map_length: int,
    map_width: int,
    adjacency_bool: np.ndarray,
    tile_symbols: list[str],
    tile_to_index: dict[str, int],
) -> float:
    reward_fn = build_selected_reward(tasks_list)
    num_tiles = len(tile_symbols)
    env_builder = make_env_factory(
        map_length, map_width, tile_symbols, adjacency_bool, num_tiles, tile_to_index, reward_fn
    )

    # Suggest PPO hyperparams
    hparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs": trial.suggest_int("n_epochs", 5, 15),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.98),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
    }

    n_envs = trial.suggest_categorical("n_envs", [1, 4, 8])
    eval_freq = trial.suggest_categorical("eval_freq", [5000, 10000, 20000])

    out_dir = f"agents/ppo_optuna_trial_{trial.number}"
    os.makedirs(out_dir, exist_ok=True)

    saved, _, _ = train_with_ppo(
        env_builder=env_builder,
        episodes=episodes_per_trial,
        n_envs=n_envs,
        eval_freq=eval_freq,
        patience_evals=10,  # small patience during tuning
        output_dir=out_dir,
        task_info="_".join(tasks_list),
        tile_images=None,
        save_best_per_gen=False,
        ppo_hparams=hparams,
        seed=42 + trial.number,
    )
    return float(saved.reward)


def render_policy_once(
    model_path: str,
    env: WFCWrapper,
    tile_images: dict[str, Any],
    output_path: str,
    task_name: str = "",
) -> None:
    from stable_baselines3 import PPO

    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    pygame.init()
    SCREEN_WIDTH = env.map_width * 32
    SCREEN_HEIGHT = env.map_length * 32
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"PPO WFC Map - {task_name}")

    final_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    env.render_mode = "human"
    env.tile_images = tile_images
    obs, _ = env.reset()

    model = PPO.load(model_path)

    done, truncated = False, False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        screen.fill((0, 0, 0))
        final_surface.fill((0, 0, 0))

        for y in range(env.map_length):
            for x in range(env.map_width):
                cell_vec = env.grid[y][x]
                s = int(cell_vec.sum())
                if s == 1:
                    idx = int(np.argmax(cell_vec))
                    tile_name = env.all_tiles[idx]
                    if tile_name in tile_images:
                        screen.blit(tile_images[tile_name], (x * 32, y * 32))
                        final_surface.blit(tile_images[tile_name], (x * 32, y * 32))
                    else:
                        pygame.draw.rect(screen, (255, 0, 255), (x * 32, y * 32, 32, 32))
                        pygame.draw.rect(final_surface, (255, 0, 255), (x * 32, y * 32, 32, 32))
                elif s == 0:
                    pygame.draw.rect(screen, (255, 0, 0), (x * 32, y * 32, 32, 32))
                    pygame.draw.rect(final_surface, (255, 0, 0), (x * 32, y * 32, 32, 32))
                else:
                    pygame.draw.rect(screen, (100, 100, 100), (x * 32, y * 32, 32, 32))
                    pygame.draw.rect(final_surface, (100, 100, 100), (x * 32, y * 32, 32, 32))

        pygame.display.flip()

    pygame.image.save(final_surface, output_path)
    print(f"Saved final render to {output_path}")
    pygame.quit()
    if 'SDL_VIDEODRIVER' in os.environ:
        del os.environ['SDL_VIDEODRIVER']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WFC agents with PPO (RL-flavored flags).")
    parser.add_argument("--load-hyperparameters", type=str, default=None, help="YAML with PPO hparams.")
    parser.add_argument("--episodes", type=int, default=50, help="Approximate number of training episodes.")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Optuna trials for PPO hparams.")
    parser.add_argument("--episodes-per-trial", type=int, default=10, help="Episodes per Optuna trial.")
    parser.add_argument("--hyperparameter-dir", type=str, default="hyperparameters")
    parser.add_argument("--output-file", type=str, default="best_hyperparameters.yaml")
    parser.add_argument("--best-agent-pickle", type=str, help="Path to SavedRLAgent pickle OR SB3 .zip")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        choices=["binary_easy", "binary_hard", "river", "pond", "grass", "hill"],
    )
    parser.add_argument("--override-patience", type=int, default=None, help="Eval early-stopping patience (evals).")
    parser.add_argument("--save-best-per-gen", action="store_true", default=False, help="Save image on new eval best.")

    args = parser.parse_args()
    if not args.task:
        args.task = ["binary_easy"]

    MAP_LENGTH = 15
    MAP_WIDTH = 20

    adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
    num_tiles = len(tile_symbols)

    selected_reward = build_selected_reward(args.task)
    env = WFCWrapper(
        map_length=MAP_LENGTH,
        map_width=MAP_WIDTH,
        tile_symbols=tile_symbols,
        adjacency_bool=adjacency_bool,
        num_tiles=num_tiles,
        tile_to_index=tile_to_index,
        reward=selected_reward,
        deterministic=True,
    )
    tile_images = load_tile_images()

    env_builder = make_env_factory(
        MAP_LENGTH, MAP_WIDTH, tile_symbols, adjacency_bool, num_tiles, tile_to_index, selected_reward
    )

    os.makedirs("agents", exist_ok=True)

    best_agent: SavedRLAgent | None = None
    best_eval_history: list[float] = []
    mean_eval_history: list[float] = []

    if args.load_hyperparameters:
        print(f"Loading hyperparameters from: {args.load_hyperparameters}")
        try:
            with open(args.load_hyperparameters, "r") as f:
                hyperparams = yaml.safe_load(f) or {}
                hyperparams.pop("episodes", None)
                hyperparams.pop("patience_evals", None)
                hyperparams.pop("n_envs", None)
                hyperparams.pop("eval_freq", None)
        except FileNotFoundError:
            print(f"Error: Hyperparameter file not found: {args.load_hyperparameters}")
            sys.exit(1)

        patience_evals = args.override_patience if args.override_patience is not None else hyperparams.get("patience_evals", 50)
        episodes = int(hyperparams.get("episodes", args.episodes))
        n_envs = int(hyperparams.get("n_envs", 8))
        eval_freq = int(hyperparams.get("eval_freq", 10_000))

        out_dir = "ppo_loaded_hparams_run"
        task_info = "_".join(args.task)
        best_agent, best_eval_history, mean_eval_history = train_with_ppo(
            env_builder=env_builder,
            episodes=episodes,
            n_envs=n_envs,
            eval_freq=eval_freq,
            patience_evals=patience_evals,
            output_dir=out_dir,
            task_info=task_info,
            tile_images=tile_images,
            save_best_per_gen=args.save_best_per_gen,
            ppo_hparams=hyperparams,
            seed=42,
        )
        print(f"Training finished. Best eval reward: {best_agent.reward:.4f}")

        if best_eval_history:
            x_axis = np.arange(1, len(best_eval_history) + 1)
            plt.plot(x_axis, best_eval_history, label="Best Eval Mean Reward")
            plt.legend()
            plt.title(f"PPO Eval Performance: {task_info}")
            plt.xlabel("Eval checkpoints")
            plt.ylabel("Reward")
            plt.savefig(f"ppo_eval_performance_{task_info}.png")
            plt.close()

    elif not args.best_agent_pickle:
        print(f"Running Optuna hyperparameter search for {args.optuna_trials} trials...")
        study = optuna.create_study(direction="maximize")
        start_time = time.time()
        study.optimize(
            lambda trial: objective(
                trial,
                args.episodes_per_trial,
                tasks_list=args.task,
                map_length=MAP_LENGTH,
                map_width=MAP_WIDTH,
                adjacency_bool=adjacency_bool,
                tile_symbols=tile_symbols,
                tile_to_index=tile_to_index,
            ),
            n_trials=args.optuna_trials,
        )
        end_time = time.time()
        print(f"Optuna finished in {end_time - start_time:.2f} seconds.")
        print(f"Best trial reward: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        os.makedirs(args.hyperparameter_dir, exist_ok=True)
        output_path = os.path.join(args.hyperparameter_dir, args.output_file)
        payload = dict(study.best_params)
        payload.setdefault("episodes", args.episodes)
        payload.setdefault("n_envs", 8)
        payload.setdefault("eval_freq", 10_000)
        payload.setdefault("patience_evals", 50)

        with open(output_path, "w") as f:
            yaml.dump(payload, f, default_flow_style=False)
        print(f"Saved best hyperparameters to: {output_path}")

    elif args.best_agent_pickle:
        if args.best_agent_pickle.endswith(".zip") and os.path.exists(args.best_agent_pickle):
            best_agent = SavedRLAgent(model_path=args.best_agent_pickle, reward=0.0, info={})
            print(f"Loaded model path: {args.best_agent_pickle}")
        else:
            with open(args.best_agent_pickle, "rb") as f:
                best_agent = pickle.load(f)
            print(f"Loaded SavedRLAgent from pickle: {args.best_agent_pickle}")

    if best_agent:
        task_name = "_".join(args.task)
        env.render_mode = "human"
        env.tile_images = tile_images

        os.makedirs("rl_output", exist_ok=True)
        output_img = f"rl_output/{task_name}_reward_{best_agent.reward:.2f}.png"
        render_policy_once(best_agent.model_path, env, tile_images, output_img, task_name=task_name)

        agent_pkl_path = f"agents/best_rl_{task_name}_reward_{best_agent.reward:.2f}.pkl"
        with open(agent_pkl_path, "wb") as f:
            pickle.dump(best_agent, f)
        print(f"Saved best RL agent wrapper to {agent_pkl_path}")
    else:
        print("\nNo best agent produced (nothing to render).")

    print("Script finished.")
