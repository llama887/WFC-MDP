# WFC-MDP: A Markovian Framing of WaveFunctionCollapse for Procedurally Generating Aesthetically Complex Environments

[![Paper](https://img.shields.io/badge/Paper-Pending-lightgrey.svg)](https://example.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.1.1-green)](https://gymnasium.farama.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

Procedural content generation often requires satisfying both designer-specified objectives and adjacency constraints implicitly imposed by the underlying tile set. To address the challenges of jointly optimizing both constraints and objectives, we reformulate WaveFunctionCollapse (WFC) as a Markov Decision Process (MDP), enabling external optimization algorithms to focus exclusively on objective maximization while leveraging WFC's propagation mechanism to enforce constraint satisfaction. We empirically compare optimizing this MDP to traditional evolutionary approaches that jointly optimize global metrics and local tile placement. Across multiple domains with various difficulties, we find that joint optimization not only struggles as task complexity increases, but consistently underperforms relative to optimization over the WFC-MDP, underscoring the advantages of decoupling local constraint satisfaction from global objective optimization.

**Authors:** Franklin Yiu¹, Mohan Lu¹, Nina Li¹, Kevin Joseph¹, Tianxu Zhang¹, Julian Togelius¹, Timothy Merino¹, Sam Earle¹  
¹New York University, United States  


---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Key Innovation](#2-key-innovation-wfc-as-mdp)
3. [Results (MDP vs non‑MDP)](#3-results-mdp-vs-non-mdp)
4. [Methods](#4-methods)
5. [Setup and Usage](#5-setup-and-usage)
6. [Tasks and flags](#6-tasks-and-flags)
7. [Controllers and API](#7-controllers-and-api)

---

## 1. Introduction

Wave Function Collapse (WFC) is powerful at enforcing local adjacency constraints but offers limited leverage for optimizing global, gameplay‑relevant objectives. We reframe WFC as a Markov Decision Process (WFC‑MDP) so that constraint satisfaction is handled by propagation while external optimizers focus solely on objectives. Using a simple μ+λ Evolution controller, we evaluate this MDP formulation against non‑MDP baselines that operate directly on final maps (including FI‑2Pop). Across binary path‑length tasks, biome objectives, and their combinations, optimizing over the WFC‑MDP yields higher convergence rates and fewer generations than methods that must implicitly learn adjacency. This repo provides a lightweight Gymnasium environment, reward functions for several domains, and scripts to collect convergence data and produce comparison plots.

---

## 2. Key Innovation: WFC as MDP
- **State S**: boolean belief grid `G ∈ {0,1}^{H×W×n_t}`; a cell is collapsed iff its channel vector is one‑hot; otherwise it is in superposition. We also expose the next‑collapse index.
- **Action A**: length‑`n_t` preference logits over tiles for the next‑collapse cell.
- **Transition T**: collapse the feasible argmax tile and propagate adjacency constraints.
- **Reward R**: sparse terminal reward from a task‑specific objective; contradictions truncate with a large negative value.
- **Discount**: γ = 1.0 (episodic).

Action handling: We softmax logits, zero out infeasible tiles, select among the remaining, then propagate constraints. No explicit action‑mask argument is required.

---

## 3. Results (hard only)
Hard‑variant comparison plots from this repo (paper uses `figures/...`; this repo stores under `comparison_figures/...`):

- Binary (hard): `comparison_figures/binary_hard_comparison.png`
- River combo (hard): `comparison_figures/river_hard_comparison.png`
- Field/Grass combo (hard): `comparison_figures/grass_hard_comparison.png`

Across all, Evolution (WFC‑MDP) wins on fraction converged and generations to converge.

---

## 4. Methods

All optimization methods have various hyperparameters detailed in Section 7.1 of this README.

### 4.1 Direct Map Evolution
These methods operate directly on the final artifact and do not leverage WFC. Instead, the optimization process must learn to satisfy the adjacency rules. For a target map of length ℓ and width w, the genotype is represented as a 2D array of size ℓ × w, where each entry contains an integer corresponding to a tile index in the tileset.

**Baseline Evolution**: The baseline evolutionary algorithm treats each map genotype as an individual and applies standard genetic operators with a penalized fitness that subtracts adjacency violations from raw objective function. Given objective score o and v adjacency violations, the individuals will receive a fitness of o-v.

**FI-2Pop**: FI-2Pop attempts to leverage adjacency violations as an exploration medium by maintaining two equal–sized subpopulations, feasible (F) and infeasible (I), and applies tailored selection criteria to each: objective maximization in F and violation minimization in I.

### 4.2 MDP Representation
By formalizing WFC as a Markov Decision Process (MDP), we leverage its guarantees to offload the burden of learning adjacency constraints from the optimizer. This reformulation transforms the generation problem into a sequential decision process where every action results in a valid intermediate configuration.

### 4.3 Evolving an Action Sequence
We use a standard μ + λ evolutionary algorithm to optimize the full sequence of WFC collapse actions. Each individual in the population encodes a fixed-length sequence of collapse decisions, represented as logits over the tile set at each of the ℓ × w positions.

---

## 5. Setup and Usage

### 5.1 Repository Structure

```
WFC-MDP/
├── assets/
│   ├── biome_adjacency_rules.json         # Tile set definitions and adjacency graph
│   ├── biome_adjacency_rules.py           # Loads rules/images; builds boolean adjacency matrix
│   └── slice_tiles.py                     # Asset slicing/utilities (optional)
├── core/
│   ├── wfc_env.py                         # Gymnasium environment (WFC as MDP wrapper)
│   ├── wfc.py                             # WFC core: initialize, collapse, propagate, render helpers
│   ├── evolution.py                       # μ+λ evolution (1D/2D genotypes), QD hooks, Optuna objective
│   ├── fi2pop.py                          # FI-2Pop and baseline direct-map evolution pipelines
├── tasks/
│   ├── utils.py                           # Grid ops: masks, regions, longest path
│   ├── binary_task.py                     # Binary path-length reward
│   ├── river_task.py                      # River biome reward
│   ├── pond_task.py                       # Pond biome reward
│   ├── grass_task.py                      # Grass biome reward
│   └── hill_task.py                       # Hill biome reward
├── hyperparameters/
│   ├── binary_1d_hyperparameters.yaml     # Evolution params for 1D genotype
│   ├── binary_2d_hyperparameters.yaml     # Evolution params for 2D genotype
│   ├── combo_*_hyperparameters.yaml       # Binary+Biome combined runs
│   ├── fi2pop_*_hyperparameters.yaml      # FI-2Pop/baseline params
│   └── baseline_*_hyperparameters.yaml    # Baseline EA params
├── sbatch/
│   ├── evolution_plots/                   # SLURM scripts to launch plot sweeps
│   ├── fi2pop_plots/                      # SLURM scripts for FI-2Pop/baseline plots
│   ├── mcts_plots/                        # deprecated (MCTS not used)
│   └── baseline_plots/                    # SLURM scripts for baseline comparisons
├── plot.py                                # Data collection (CSV) and plotting/comparisons CLI
├── requirements.txt                       # Python dependencies
└── README.md                              # You are here
```

### 5.2 Installation

#### 5.2.1 Requirements
- Python 3.10+
- Windows/macOS/Linux
- 8GB RAM minimum recommended

#### 5.2.2 Setup

```bash
# From the repository root
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python -c "import gymnasium; print(gymnasium.__version__)"  # Expect 1.1.1
```

### 5.3 Quick Start

Minimal example (2-3 minutes on a typical laptop):

```bash
# Evolution (Binary easy)
python plot.py \
  --method evolution \
  --task binary_easy \
  --genotype-dimensions 1 \
  --load-hyperparameters hyperparameters/binary_1d_hyperparameters.yaml \
  --sample-size 5

# Outputs: CSVs under method-specific figure folders, e.g. `figures_evolution/1d/binary_easy_convergence.csv`.
# If you run evolution.py with --save-best-per-gen, best-per-generation images will save under `best_gen_maps/...`.
```

---

## 6. Tasks and flags (used in the paper)
Use these with `plot.py`. Reward implementations live in `tasks/*.py`.

| Task flag | Description | Reward function | Source |
| --- | --- | --- | --- |
| `--task binary_hard` | Binary path‑length (exact) | `binary_reward(target_path_length=P, hard=True)` | `tasks/binary_task.py` |
| `--task river` | River biome | `river_reward` | `tasks/river_task.py` |
| `--task grass` | Field (grass) biome | `grass_reward` | `tasks/grass_task.py` |
| `--task river --combo hard` | Binary + River (hard) | `CombinedReward([binary_reward(P, True), river_reward])` | `plot.py` |
| `--task grass --combo hard` | Binary + Field (hard) | `CombinedReward([binary_reward(P, True), grass_reward])` | `plot.py` |

Notes: results/plots are hard‑only; pond and hill are not used in the paper.

---

## 7. Controllers and API

### 7.1 Optimization Controllers

#### 7.1.1 Evolution (μ+λ)
```python
Parameters = {
    'population_size': 48,
    'survival_rate': 0.4151,  # Binary 1D
    'number_of_actions_mutated_mean': 97,
    'number_of_actions_mutated_standard_deviation': 120.1,
    'action_noise_standard_deviation': 0.1296,
    'cross_over_method': 'ONE_POINT',
    'cross_or_mutate': 0.7453,
    'random_offspring': 0.0,
}
```

Genotype modes:

```bash
# 1D (sequential playback of actions)
python plot.py --method evolution --task binary_easy \
  --genotype-dimensions 1 \
  --load-hyperparameters hyperparameters/binary_1d_hyperparameters.yaml --sample-size 5

# 2D (index by next-collapse cell)
python plot.py --method evolution --task binary_easy \
  --genotype-dimensions 2 \
  --load-hyperparameters hyperparameters/binary_2d_hyperparameters.yaml --sample-size 5
```

#### 7.1.2 FI-2Pop
Maintains two subpopulations of size N/2 each:
- Feasible: arg max f(x) subject to c(x) = 0
- Infeasible: arg min ||c(x)||

#### 7.1.3 MCTS
Removed. This repository no longer uses MCTS in experiments.

### 7.2 Plotting and aggregation with plot.py
```bash
# Evolution, Binary easy (P = 10..100), 1D genotype
python plot.py \
  --method evolution \
  --task binary_easy \
  --genotype-dimensions 1 \
  --load-hyperparameters hyperparameters/binary_1d_hyperparameters.yaml \
  --sample-size 20
# → figures_evolution/1d/binary_easy_convergence.csv

# Evolution, Binary hard (exact match)
python plot.py \
  --method evolution \
  --task binary_hard \
  --genotype-dimensions 1 \
  --load-hyperparameters hyperparameters/binary_1d_hyperparameters.yaml \
  --sample-size 20
# → figures_evolution/1d/binary_hard_convergence.csv

# FI-2Pop baseline, Binary easy
python plot.py \
  --method fi2pop \
  --task binary_easy \
  --load-hyperparameters hyperparameters/fi2pop_binary_hyperparameters.yaml \
  --sample-size 20
# → figures_fi2pop/binary_convergence.csv

# Biome averages with evolution (Pond, River)
python plot.py \
  --method evolution \
  --task biomes \
  --genotype-dimensions 1 \
  --load-hyperparameters hyperparameters/biomes_1d_hyperparameters.yaml \
  --sample-size 20
# → figures_evolution/1d/biome_average_convergence.csv

# Combo objective (Binary + River), easy
python plot.py \
  --method evolution \
  --task river \
  --combo easy \
  --genotype-dimensions 1 \
  --load-hyperparameters hyperparameters/combo_river_1d_hyperparameters.yaml \
  --sample-size 20
# → figures_evolution/1d/combo_river_convergence.csv

# Field/Grass combo (hard)
python plot.py --method evolution --task grass --combo hard \
  --genotype-dimensions 1 \
  --load-hyperparameters hyperparameters/combo_grass_1d_hyperparameters.yaml \
  --sample-size 20
```

Comparison plots from CSVs:
```bash
python plot.py --compare \
  --csv-files \
    figures_evolution/1d/binary_easy_convergence.csv \
    figures_fi2pop/binary_convergence.csv \
  --labels Evolution FI-2Pop \
  --title "Binary Easy: Evolution vs FI-2Pop" \
  --output comparison_figures/binary_easy_comparison.png
```

Tips:
- For 2D genotypes, set `--genotype-dimensions 2` and load the corresponding `binary_2d_hyperparameters.yaml`.
- To remove random offspring in evolution sweeps, use `--no-random-offspring`.
- Debug per-run reward curves can be enabled with `--debug` (PNG saved under `debug_plots/`).

### 7.3 Gymnasium Environment API
```python
import numpy as np
from functools import partial
from core.wfc_env import WFCWrapper
from assets.biome_adjacency_rules import create_adjacency_matrix
from tasks.binary_task import binary_reward

adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()
reward_fn = partial(binary_reward, target_path_length=40, hard=True)

env = WFCWrapper(
    map_length=15,
    map_width=20,
    tile_symbols=tile_symbols,
    adjacency_bool=adjacency_bool,
    num_tiles=len(tile_symbols),
    tile_to_index=tile_to_index,
    reward=reward_fn,
    deterministic=True,
)

obs, info = env.reset(seed=42)
for _ in range(env.map_length * env.map_width):
    action = np.random.rand(env.action_space.shape[0])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print({"terminated": terminated, "truncated": truncated, **info})
```

### 7.4 Combined objectives (CombinedReward)
```python
from functools import partial
from core.wfc_env import CombinedReward
from tasks.binary_task import binary_reward
from tasks.river_task import river_reward

reward = CombinedReward([
    partial(binary_reward, target_path_length=40, hard=True),
    river_reward,
])
```
