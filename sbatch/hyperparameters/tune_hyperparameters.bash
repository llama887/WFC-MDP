#!/bin/bash

sbatch sbatch/hyperparameters/binary_1d.s
sbatch sbatch/hyperparameters/binary_2d.s
sbatch sbatch/hyperparameters/combo_grass_1d.s
sbatch sbatch/hyperparameters/combo_grass_2d.s
sbatch sbatch/hyperparameters/combo_pond_1d.s
sbatch sbatch/hyperparameters/combo_pond_2d.s
sbatch sbatch/hyperparameters/combo_river_1d.s
sbatch sbatch/hyperparameters/combo_river_2d.s

sbatch sbatch/hyperparameters/binary_fi2pop.s
sbatch sbatch/hyperparameters/combo_grass_fi2pop.s
sbatch sbatch/hyperparameters/combo_pond_fi2pop.s
sbatch sbatch/hyperparameters/combo_river_fi2pop.s

sbatch sbatch/hyperparameters/binary_baseline.s
sbatch sbatch/hyperparameters/combo_grass_baseline.s
sbatch sbatch/hyperparameters/combo_pond_baseline.s
sbatch sbatch/hyperparameters/combo_river_baseline.s

sbatch sbatch/hyperparameters/binary_mcts.s
sbatch sbatch/hyperparameters/combo_grass_mcts.s
sbatch sbatch/hyperparameters/combo_pond_mcts.s
sbatch sbatch/hyperparameters/combo_river_mcts.s