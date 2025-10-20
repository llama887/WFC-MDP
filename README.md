# WFC-MDP: A Markovian Framing of WaveFunctionCollapse for Procedurally Generating Aesthetically Complex Environments

[![Paper](https://img.shields.io/badge/Paper-Pending-lightgrey.svg)](https://example.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.1.1-green)](https://gymnasium.farama.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

Procedural content generation often requires satisfying both designer-specified objectives and adjacency constraints implicitly imposed by the underlying tile set. We reformulate Wave Function Collapse (WFC) as a Markov Decision Process (MDP), enabling external optimization algorithms to focus exclusively on objective maximization while leveraging WFC's propagation mechanism to enforce constraint satisfaction. Across multiple domains with varying difficulties, joint optimization not only struggles as task complexity increases but consistently underperforms relative to optimization over the WFC-MDP, underscoring the advantages of decoupling local constraint satisfaction from global objective optimization.

**Authors:** Franklin Yiu¹, Mohan Lu¹, Nina Li¹, Kevin Joseph¹, Tianxu Zhang¹, Julian Togelius¹, Timothy Merino¹, Sam Earle¹  
¹New York University, Brooklyn, NY 11201, USA  

**Correspondence:** fyy2003@nyu.edu  
**Paper:** Link pending  
**Version:** 1.0.0

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Key Innovation](#3-key-innovation-wfc-as-mdp)
4. [Experimental Results](#4-experimental-results)
5. [Repository Structure](#5-repository-structure)
6. [Installation](#6-installation)
7. [Quick Start](#7-quick-start)
8. [Domains and Objectives](#8-domains-and-objectives)
9. [Controllers](#9-controllers)
10. [Gymnasium Environment API](#10-gymnasium-environment-api)
11. [Running Experiments](#11-running-experiments)
12. [Hyperparameter Configuration](#12-hyperparameter-configuration)
13. [Reproducibility](#13-reproducibility)
14. [Extending the Framework](#14-extending-the-framework)
15. [Computational Requirements](#15-computational-requirements)
16. [Known Limitations](#16-known-limitations)
17. [Testing](#17-testing)
18. [Citation](#18-citation)
19. [License and Ethics](#19-license-and-ethics)
20. [Acknowledgments](#20-acknowledgments)

---

## 1. Introduction

Wave Function Collapse (WFC) excels at maintaining local adjacency constraints but struggles with global functional properties critical for gameplay. This repository presents a novel MDP formulation of WFC that decouples constraint satisfaction from objective optimization, enabling the application of standard optimization algorithms while maintaining structural validity.

### 1.1 Problem Statement

Traditional PCG methods must simultaneously learn:
- Local adjacency constraints (aesthetics)
- Global objective optimization (functionality)

This joint optimization becomes harder as aesthetic complexity increases. Our code focuses on decoupling constraints from objectives; specific performance numbers depend on configuration and are not claimed here.

### 1.2 Our Solution

By reformulating WFC as an MDP, we:
- Offload constraint satisfaction to WFC's propagation mechanism
- Enable optimizers to focus solely on objective maximization

---

## 2. Related Work

### 2.1 Wave Function Collapse Extensions
- **Karth & Smith (2017)**: WFC as constraint satisfaction problem
- **Nie et al. (2023)**: Nested WFC for scalability  
- **Our contribution**: First MDP formulation enabling RL/search methods

### 2.2 Procedural Content Generation via Learning
- **Khalifa et al. (2020)**: PCGRL - RL for level generation
- **Summerville et al. (2018)**: PCGML survey
- **Our distinction**: Explicit constraint/objective decoupling

### 2.3 Constrained Optimization in PCG
- **Kimbrough et al. (2008)**: FI-2Pop for constrained optimization

---

## 3. Key Innovation: WFC as MDP

### 3.1 Formal Definition

- **State S**: Grid of shape HxW with values in [-1, n_t-1]; -1 means uncollapsed. Implementation: `np.ndarray[H,W]` + next-cell position.
- **Action A**: Length-n_t vector in [0,1] (tile preference logits). Implementation: `Box(low=0, high=1, shape=(num_tiles,))`.
- **Transition T**: Deterministic propagation from state and action to next state. Implementation: `propagate_constraints()`.
- **Reward R**: Sparse terminal reward computed from the final grid. Implementation: `tasks/*.reward()`.
- **Discount**: gamma = 1.0 (episodic task).

### 3.2 State Representation
```python
State = {
    'grid': np.ndarray,      # Shape: (H, W), dtype: int8
                             # Values: -1 (uncollapsed) or tile_index
    'next_pos': (int, int),  # Next cell to collapse (lowest entropy)
    'valid_tiles': np.ndarray # Shape: (H, W, nₜ), dtype: bool
}
```

### 3.3 Action Handling
The environment consumes a length-`num_tiles` vector of preferences and converts logits to probabilities internally. Constraint propagation enforces feasibility; there is no separate action mask.

---

## 4. Experimental Results

This repo provides tooling to collect and plot convergence data; run the commands in Section 11 to generate your own CSVs and figures. No fixed numbers are claimed here.

---

## 5. Repository Structure

```
WFC-MDP/
├── assets/
│   ├── biome_adjacency_rules.json
│   ├── biome_adjacency_rules.py
│   └── slice_tiles.py
├── core/
│   ├── wfc_env.py
│   ├── wfc.py
│   ├── evolution.py
│   ├── fi2pop.py
│   └── mcts.py
├── tasks/
│   ├── utils.py
│   ├── binary_task.py
│   ├── river_task.py
│   ├── pond_task.py
│   ├── grass_task.py
│   └── hill_task.py
├── hyperparameters/
│   ├── binary_1d_hyperparameters.yaml
│   ├── binary_2d_hyperparameters.yaml
│   ├── combo_*_hyperparameters.yaml
│   ├── fi2pop_*_hyperparameters.yaml
│   └── baseline_*_hyperparameters.yaml
├── sbatch/
│   ├── evolution_plots/
│   ├── mcts_plots/
│   ├── fi2pop_plots/
│   └── baseline_plots/
├── plot.py
├── requirements.txt
└── README.md
```

---

## 6. Installation

### 6.1 Requirements
- Python 3.10+
- Windows/macOS/Linux
- 8GB RAM minimum recommended

### 6.2 Setup

```bash
# Clone repository
git clone https://github.com/your-org/WFC-MDP.git
cd WFC-MDP

# Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import gymnasium; print(gymnasium.__version__)"  # Should print 1.1.1
```

### 6.3 Dependencies
See `requirements.txt` for the authoritative list. Current pins:
```
gymnasium==1.1.1
optuna==4.2.1
scipy==1.15.2
matplotlib==3.10.1
pygame==2.6.1
pillow==11.2.1
pandas==2.2.3
pydantic==2.11.7
jinja2==3.1.6
```

---

## 7. Quick Start

Minimal example (2-3 minutes on Intel i7-9700K):

```bash
# Evolution (Binary easy)
python plot.py \
  --method evolution \
  --task binary_easy \
  --genotype-dimensions 1 \
  --load-hyperparameters hyperparameters/binary_1d_hyperparameters.yaml \
  --sample-size 5

# MCTS (no hyperparameters file required)
python plot.py \
  --method mcts \
  --task binary_easy \
  --sample-size 5 \
  --mcts-iterations 1000

# Outputs: CSVs under method-specific figure folders, e.g. `figures_evolution/1d/binary_easy_convergence.csv`.
```

---

## 8. Domains and Objectives

### 8.1 Binary Domain
**Objective Function:**
```
f(p) = -|p - P|
```
Where p = longest shortest path, P = target path length

### 8.2 River/Binary Hybrid
**River Objective (qualitative):** Encourages a single connected, winding river of at least length 35, discourages fully surrounded “lake” tiles, and penalizes excessive fragmentation of land regions. This mirrors the implementation in `tasks/river_task.py` (connected water regions close to 1, river length below threshold is penalized, pure water tiles discouraged, land regions capped).

**Combined:** f(p, oᵣ) = -|p - P| + oᵣ

### 8.3 Pond/Grass/Hill Hybrids
The repo includes `pond`, `grass`, and `hill` reward functions. You can also combine them with the binary path-length reward using `CombinedReward` (see Section 11 for commands).

---

## 9. Controllers

### 9.1 Evolution (μ+λ)
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

### 9.2 FI-2Pop
Maintains two subpopulations of size N/2 each:
- Feasible: arg max f(x) subject to c(x) = 0
- Infeasible: arg min ||c(x)||

### 9.3 MCTS
UCT formula: Q(s,a) + C√(ln N(s) / N(s,a))
- Exploration constant C = √2
- Default iterations in our scripts: 1000 (configurable via `--mcts-iterations`)

---

## 10. Gymnasium Environment API

```python
import numpy as np
from functools import partial
from core.wfc_env import WFCWrapper
from assets.biome_adjacency_rules import create_adjacency_matrix
from tasks.binary_task import binary_reward

# Build adjacency and tiles
adjacency_bool, tile_symbols, tile_to_index = create_adjacency_matrix()

# Use a reward callable (e.g., binary path length with target 40)
reward_fn = partial(binary_reward, target_path_length=40, hard=False)

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
```

---

## 11. Running Experiments

### 11.1 CLI Recipes (that match `plot.py`)

```bash
# Evolution (easy binary path length across P=10..100)
python plot.py \
  --method evolution \
  --task binary_easy \
  --genotype-dimensions 1 \
  --load-hyperparameters hyperparameters/binary_1d_hyperparameters.yaml \
  --sample-size 20

# FI-2Pop baseline (easy binary)
python plot.py \
  --method fi2pop \
  --task binary_easy \
  --load-hyperparameters hyperparameters/fi2pop_binary_hyperparameters.yaml \
  --sample-size 20

# MCTS (easy binary)
python plot.py --method mcts --task binary_easy --sample-size 20 --mcts-iterations 1000

# Biome-only averages
python plot.py --method evolution --task biomes --load-hyperparameters hyperparameters/biomes_1d_hyperparameters.yaml --sample-size 20

# Combo objectives (binary + river/pond/grass/hill)
python plot.py \
  --method evolution \
  --task river \
  --combo easy \
  --load-hyperparameters hyperparameters/combo_river_1d_hyperparameters.yaml \
  --sample-size 20
```

### 11.2 Viewing Included Comparison Figures

The repo includes several pre-rendered comparison plots under `comparison_figures/`. To generate your own, run the commands above and then use the comparison mode in `plot.py` to overlay CSVs.

---

## 12. Hyperparameter Configuration

### 12.1 Tuning Notes
- Example YAMLs in `hyperparameters/` were tuned offline.
- If you tune yourself, we recommend Optuna, but no tuner is bundled.

### 12.2 Example Configuration (Binary 1D Evolution)
```yaml
# hyperparameters/binary_1d_hyperparameters.yaml
method: evolution
genotype_dimensions: 1
number_of_actions_mutated_mean: 97
number_of_actions_mutated_standard_deviation: 120.0876
action_noise_standard_deviation: 0.1296
survival_rate: 0.4151
cross_over_method: ONE_POINT
cross_or_mutate: 0.7453
random_offspring_fraction: 0.0
```

---

## 13. Reproducibility

### 13.1 Random Seeds
- Training seeds: {0, 42, 123, 2025, 7777}
- NumPy: `np.random.seed(seed)`
- Gymnasium: `env.reset(seed=seed)`
- Python random: `random.seed(seed)`

### 13.2 Notes on Reproducibility
- Set seeds in Python, NumPy, and via `env.reset(seed=...)`.
- Results may vary by platform and package versions.

---

## 14. Extending the Framework

### 14.1 Adding a New Task

```python
# tasks/custom_task.py
import numpy as np
from tasks.utils import calc_num_regions

def reward(grid_3d: np.ndarray) -> float:
    """
    Args:
        grid_3d: Shape (H, W, num_tiles) boolean array
    Returns:
        Scalar reward (0 is optimal)
    """
    # Implementation
    return reward_value
```

### 14.2 Adding a New Controller

Implement the interface:
```python
class CustomController:
    def __init__(self, config: dict):
        pass
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Return action logits of shape (num_tiles,)"""
        pass
```

---

## 15. Computational Requirements

Runtime depends on hardware, problem size, and method. Start with small sample sizes (e.g., 5–10 runs) and scale up as needed.

---

## 16. Known Limitations

1. **Population Size:** Fixed at 48 due to hardware constraints
2. **Observation Utilization:** Evolution doesn't leverage intermediate observations
3. **Exploration:** Standard μ+λ exhibits inadequate exploration (consider Quality Diversity)
4. **Scalability:** O(H×W×nₜ) memory complexity limits map size
5. **Tile Set:** Currently limited to Biome Pack B tiles

---

## 17. Testing

This repository does not include a formal test suite yet. If you add tests, place them under a new `tests/` directory and run with `pytest`.

---

## 18. Citation

```bibtex
@inproceedings{yiu2025markovian,
  title     = {A Markovian Framing of WaveFunctionCollapse for 
               Procedurally Generating Aesthetically Complex Environments},
  author    = {Yiu, Franklin and Lu, Mohan and Li, Nina and Joseph, Kevin 
               and Zhang, Tianxu and Togelius, Julian and Merino, Timothy 
               and Earle, Sam},
  booktitle = {Proceedings of the AIIDE Workshop on Experimental AI in Games},
  pages     = {1--15},
  year      = {2025},
  publisher = {AAAI Press}
}
```

---

## 19. License and Ethics

### License
MIT License - see [LICENSE](LICENSE)

### Ethical Considerations
- Computational resources: ~200 GPU-hours total
- Carbon footprint: Estimated 18kg CO₂ (NYU carbon-neutral by 2040)
- No human subjects involved

---

## 20. Acknowledgments

Thanks to contributors and to the maintainers of Gymnasium and matplotlib. Tile assets are adapted from VectoRaith's Biome Tileset Pack B.
