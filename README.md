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
2. [Related Work](#2-related-work)
3. [Key Innovation](#3-key-innovation-wfc-as-mdp)
4. [Experimental Results](#4-experimental-results)
5. [Repository Structure](#5-repository-structure)
6. [Installation](#6-installation)
7. [Quick Start](#7-quick-start)
8. [Problem Domain](#8-problem-domain)
9. [Domains and Objectives](#9-domains-and-objectives)
10. [Controllers](#10-controllers)
11. [Gymnasium Environment API](#11-gymnasium-environment-api)
12. [Running Experiments](#12-running-experiments)

---

## 1. Introduction

Wave Function Collapse (WFC) excels at maintaining local adjacency constraints but struggles with global functional properties critical for gameplay. This repository presents a novel MDP formulation of WFC that decouples constraint satisfaction from objective optimization, enabling the application of standard optimization algorithms while maintaining structural validity.

### 1.1 Problem Statement

Traditional PCG methods must simultaneously learn:
- Local adjacency constraints (aesthetics)
- Global objective optimization (functionality)

This joint optimization becomes harder as aesthetic complexity increases. Our code focuses on decoupling constraints from objectives; specific performance numbers depend on configuration and are not claimed here.

### 1.2 Our Solution

We offer the following contributions:
- We demonstrate that forcing learning algorithms to learn local adjacency constraints leads to degraded performance in highly constrained domains.
- We present a novel formulation of WFC as an MDP, along with a corresponding Gymnasium environment to facilitate the evaluation of alternative optimization algorithms.

---

## 2. Related Work

### 2.1 Wave Function Collapse Algorithm and Modifications

**WaveFunctionCollapse**: WFC is an algorithm for procedural content generation, particularly for tile-based environments. It can create coherent and visually consistent outputs based on simple input examples or constraints. The algorithm's core functionality resembles constraint satisfaction with a quantum mechanics-inspired approach: maintaining a superposition of possible states for each tile that gradually "collapses" to definite states through observation and propagation steps.

The algorithm operates in two main modes: the simple tiled model, which uses explicit adjacency rules, and the overlapping model, which automatically extracts patterns from example inputs. In both models, WFC can be seen as a form of self-supervised learning that learns a distribution from minimal examples—often just one—and generates similar content by maintaining the learned adjacency patterns. While WFC excels at maintaining local adjacency constraints, it struggles with global optimization objectives that are critical for gameplay, such as ensuring level solvability or balanced resource distribution.

**Enhancements to WaveFunctionCollapse**: Researchers have developed numerous modifications to improve WFC's capabilities beyond its original formulation. To improve scalability to large environments, Nested Wave Function Collapse (N-WFC) was proposed, which decomposes the generation process into nested subproblems. This approach significantly reduces time complexity from exponential to polynomial while maintaining consistency across the generated content. For non-grid environments, researchers have developed graph-based extensions of WFC that enable content generation for arbitrary topologies, enabling applications to 3D worlds and non-uniform structures.

### 2.2 Evolutionary Approaches to PCG

**Evolutionary Algorithms**: Evolutionary algorithms, widely applied to PCG problems, are a flexible approach to searching the space of possible game content while optimizing for designer-specified objectives. These approaches typically encode content as genomes that evolve through operations such as crossover and mutation, with fitness functions guiding the search toward desired properties.

**FI-2Pop for Constrained Optimization**: The Feasible-Infeasible Two-Population (FI-2Pop) genetic algorithm is a solution to constrained optimization problems that maintains two separate populations: one of feasible solutions that optimize the objective function, and another of infeasible solutions that minimize constraint violations. This approach has proven particularly effective for PCG applications where content must simultaneously satisfy hard constraints (e.g., playability) while optimizing soft objectives (e.g., challenge or balance).

### 2.3 Procedural Content Generation via Reinforcement Learning (PCGRL)

PCGRL was introduced as a novel approach to procedural content generation that frames level design as a sequential decision-making process optimized through reinforcement learning. By formulating level generation as a Markov Decision Process (MDP), PCGRL enables an agent to learn a policy that maximizes expected level quality through incremental modifications. This approach can operate without human-authored examples and generates content extremely quickly once trained.

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

We evaluate each optimization method across desired path lengths 10-100 in intervals of 10 for both binary and hybrid domains. Convergence robustness is defined as the proportion of runs that achieve the maximal reward of 0, while sample efficiency is measured by the number of generations it takes to evolve at least one population member with reward of 0. All methods were run with a fixed sample budget (population size = 48) to enable fair comparisons.

### 4.1 Key Findings

**MDP Encapsulation of Constraints is Crucial**: Across all desired path lengths, methods that offload constraint enforcement to WFC consistently outperform those that must learn it implicitly. This discrepancy is especially pronounced given more difficult objectives (i.e. higher target path lengths, hybrid biome/binary domains), where the feasibility space is severely constrained.

**Feasible Region Shrinkage Limits Optimization**: At high path lengths, even MDP methods fail to converge reliably. This likely stems from the exponentially shrinking volume of the feasible space and limited tendency toward exploration in vanilla μ+λ evolution—as compared to e.g. Quality Diversity. Despite valid intermediate states, the reward landscape remains highly sparse and multi-modal.

These findings highlight a fundamental insight: procedural generation under complex constraints benefits most when constraint satisfaction is externalized and search is guided through structurally aligned representations. The clear failure of joint optimization approaches, particularly in aesthetically constrained domains, emphasizes the importance of architectural modularity in generative design systems.

### 4.2 Performance Data

This repo provides tooling to collect and plot convergence data; run the commands in Section 12 to generate your own CSVs and figures. The paper includes detailed performance tables showing convergence rates and generation counts across different path lengths and domains.

---

## 5. Methods

All optimization methods have various hyperparameters detailed in Section 13 of this README.

### 5.1 Direct Map Evolution
These methods operate directly on the final artifact and do not leverage WFC. Instead, the optimization process must learn to satisfy the adjacency rules. For a target map of length ℓ and width w, the genotype is represented as a 2D array of size ℓ × w, where each entry contains an integer corresponding to a tile index in the tileset.

**Baseline Evolution**: The baseline evolutionary algorithm treats each map genotype as an individual and applies standard genetic operators with a penalized fitness that subtracts adjacency violations from raw objective function. Given objective score o and v adjacency violations, the individuals will receive a fitness of o-v.

**FI-2Pop**: FI-2Pop attempts to leverage adjacency violations as an exploration medium by maintaining two equal–sized subpopulations, feasible (F) and infeasible (I), and applies tailored selection criteria to each: objective maximization in F and violation minimization in I.

### 5.2 MDP Representation
By formalizing WFC as a Markov Decision Process (MDP), we leverage its guarantees to offload the burden of learning adjacency constraints from the optimizer. This reformulation transforms the generation problem into a sequential decision process where every action results in a valid intermediate configuration.

### 5.3 Evolving an Action Sequence
We use a standard μ + λ evolutionary algorithm to optimize the full sequence of WFC collapse actions. Each individual in the population encodes a fixed-length sequence of collapse decisions, represented as logits over the tile set at each of the ℓ × w positions.

---

## 6. Repository Structure

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

## 8. Problem Domain

All maps and thus objective functions were constructed based on a small subset of *Biome Tileset Pack B - Grassland, Savannah, and Scrubland* which offers combinations of different grass, path, and water tiles. We used a subset of the tiles from the tileset. We generate the adjacency rules via manual human labeling, though they could also be extracted from an input image.

## 9. Domains and Objectives

### 9.1 Binary Domain
**Objective Function:**
```
f(p) = -|p - P|
```
Where p = longest shortest path, P = target path length

### 9.2 River/Binary Hybrid
**River Objective:** Let rᵣ = number of connected river regions, ℓ = length of current river path, nᶜ = number of water "center" tiles, rₗ = number of connected land regions. Then:

oᵣ = (1 - rᵣ) + min(0, ℓ - 35) - nᶜ + min(0, 3 - rₗ)

The objective attains its maximum value of 0 when: rᵣ = 1 (exactly one contiguous river region), ℓ ≥ 35 (river path length of at least 35 tiles), nᶜ = 0 (no fully surrounded water tiles), rₗ ≤ 3 (no more than three separate land regions).

**Combined:** f(p, oᵣ) = -|p - P| + oᵣ

### 9.3 Grass/Binary Hybrid
**Grass Objective:** Let nᵥ = number of water tiles, nₕ = number of hill tiles, g = percent of grass tiles, f = percent of flower tiles. Then:

oᵍ = -nᵥ - nₕ + min(0, g - 20) + min(0, f - 20)

The objective attains its maximum value of 0 when: nᵥ = 0 (no water or shore tiles), nₕ = 0 (no hill tiles), g ≥ 20 (at least 20% of tiles are grass), f ≥ 20 (at least 20% of tiles are flowers).

**Combined:** f(p, oᵍ) = -|p - P| + oᵍ

### 9.4 Other Biomes
The repo includes `pond` and `hill` reward functions. You can also combine them with the binary path-length reward using `CombinedReward` (see Section 12 for commands).

---

## 10. Controllers

### 10.1 Evolution (μ+λ)
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

### 10.2 FI-2Pop
Maintains two subpopulations of size N/2 each:
- Feasible: arg max f(x) subject to c(x) = 0
- Infeasible: arg min ||c(x)||

### 10.3 MCTS
UCT formula: Q(s,a) + C√(ln N(s) / N(s,a))
- Exploration constant C = √2
- Default iterations in our scripts: 1000 (configurable via `--mcts-iterations`)

---

## 11. Gymnasium Environment API

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

## 12. Running Experiments

### 12.1 CLI Recipes (that match `plot.py`)

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
