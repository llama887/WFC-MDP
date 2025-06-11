#!/bin/bash

sbatch binary_1d.s
sbatch binary_2d.s
sbatch combo_grass_1d.s
sbatch combo_grass_2d.s
sbatch combo_pond_1d.s
sbatch combo_pond_2d.s
sbatch combo_river_1d.s
sbatch combo_river_2d.s

sbatch binary_fi2pop.s
sbatch combo_grass_fi2pop.s
sbatch combo_pond_fi2pop.s
sbatch combo_river_fi2pop.s

sbatch binary_baseline.s
sbatch combo_grass_baseline.s
sbatch combo_pond_baseline.s
sbatch combo_river_baseline.s

sbatch binary_mcts.s
sbatch combo_grass_mcts.s
sbatch combo_pond_mcts.s
sbatch combo_river_mcts.s