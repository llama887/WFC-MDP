#!/bin/bash

sbatch --requeue sbatch/mcts_plots/plot_mcts_river_easy.s 
sbatch --requeue sbatch/mcts_plots/plot_mcts_river_hard.s 
sbatch --requeue sbatch/mcts_plots/plot_mcts_pond_hard.s 
sbatch --requeue sbatch/mcts_plots/plot_mcts_pond_easy.s 
sbatch --requeue sbatch/mcts_plots/plot_mcts_grass_easy.s 
sbatch --requeue sbatch/mcts_plots/plot_mcts_grass_hard.s 
sbatch --requeue sbatch/mcts_plots/plot_mcts_binary_hard.s 
sbatch --requeue sbatch/mcts_plots/plot_mcts_binary_easy.s 
sbatch --requeue sbatch/mcts_plots/plot_mcts_biomes.s