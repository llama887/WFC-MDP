#!/bin/bash

sbatch sbatch/hyperparameters/binary_1d.s
sbatch sbatch/hyperparameters/binary_2d.s
sbatch sbatch/hyperparameters/combo_grass_1d.s
sbatch sbatch/hyperparameters/combo_grass_2d.s
sbatch sbatch/hyperparameters/combo_pond_1d.s
sbatch sbatch/hyperparameters/combo_pond_2d.s
sbatch sbatch/hyperparameters/combo_river_1d.s
sbatch sbatch/hyperparameters/combo_river_2d.s

sbatch sbatch/hyperparameters/binary_1d_fi2pop.s
sbatch sbatch/hyperparameters/binary_2d_fi2pop.s
sbatch sbatch/hyperparameters/combo_grass_1d_fi2pop.s
sbatch sbatch/hyperparameters/combo_grass_2d_fi2pop.s
sbatch sbatch/hyperparameters/combo_pond_1d_fi2pop.s
sbatch sbatch/hyperparameters/combo_pond_2d_fi2pop.s
sbatch sbatch/hyperparameters/combo_river_1d_fi2pop.s
sbatch sbatch/hyperparameters/combo_river_2d_fi2pop.s

sbatch sbatch/hyperparameters/binary_1d_mcts.s
sbatch sbatch/hyperparameters/binary_2d_mcts.s
sbatch sbatch/hyperparameters/combo_grass_1d_mcts.s
sbatch sbatch/hyperparameters/combo_grass_2d_mcts.s
sbatch sbatch/hyperparameters/combo_pond_1d_mcts.s
sbatch sbatch/hyperparameters/combo_pond_2d_mcts.s
sbatch sbatch/hyperparameters/combo_river_1d_mcts.s
sbatch sbatch/hyperparameters/combo_river_2d_mcts.s