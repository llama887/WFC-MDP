#!/bin/bash

mkdir comparison_figures
echo "Plotting..."
source venv/bin/activate
python plot.py --compare --csv-files figures_baseline/binary_convergence.csv \
                                    figures_evolution/1d/binary_easy_convergence.csv \
                                    figures_evolution/2d/binary_easy_convergence.csv \
                                    figures_fi2pop/binary_convergence.csv \
                                    --labels baseline evolution_1d evolution_2d fi2pop \
                                    --title "Binary Easy Convergence Behavior" \
                                    --output comparison_figures/binary_easy_comparison.png &

python plot.py --compare --csv-files figures_baseline/binary_hard_convergence.csv \
                                    figures_evolution/1d/binary_hard_convergence.csv \
                                    figures_evolution/2d/binary_hard_convergence.csv \
                                    figures_fi2pop/binary_hard_convergence.csv \
                                    --labels baseline evolution_1d evolution_2d fi2pop \
                                    --title "Binary Hard Convergence Behavior" \
                                    --output comparison_figures/binary_hard_comparison.png &

wait
echo "Plotting grass..."

python plot.py --compare --csv-files figures_baseline/combo_grass_convergence.csv \
                                    figures_evolution/1d/combo_grass_convergence.csv \
                                    figures_evolution/2d/combo_grass_convergence.csv \
                                    figures_fi2pop/combo_grass_convergence.csv \
                                    --labels baseline evolution_1d evolution_2d fi2pop \
                                    --title "Grass Combo Convergence Behavior" \
                                    --output comparison_figures/grass_comparison.png &

python plot.py --compare --csv-files figures_baseline/combo_grass_hard_convergence.csv \
                                    figures_evolution/1d/combo_grass_hard_convergence.csv \
                                    figures_evolution/2d/combo_grass_hard_convergence.csv \
                                    figures_fi2pop/combo_grass_hard_convergence.csv \
                                    --labels baseline evolution_1d evolution_2d fi2pop \
                                    --title "Hard Grass Combo Convergence Behavior" \
                                    --output comparison_figures/grass_hard_comparison.png &

wait
echo "Plotting River..."

python plot.py --compare --csv-files figures_baseline/combo_river_convergence.csv \
                                    figures_evolution/1d/combo_river_convergence.csv \
                                    figures_evolution/2d/combo_river_convergence.csv \
                                    figures_fi2pop/combo_river_convergence.csv \
                                    --labels baseline evolution_1d evolution_2d fi2pop \
                                    --title "River Combo Convergence Behavior" \
                                    --output comparison_figures/river_comparison.png &

python plot.py --compare --csv-files figures_baseline/combo_river_hard_convergence.csv \
                                    figures_evolution/1d/combo_river_hard_convergence.csv \
                                    figures_evolution/2d/combo_river_hard_convergence.csv \
                                    figures_fi2pop/combo_river_hard_convergence.csv \
                                    --labels baseline evolution_1d evolution_2d fi2pop \
                                    --title "River Hard Combo Convergence Behavior" \
                                    --output comparison_figures/river_hard_comparison.png & 

wait
echo "Plotting Pond..."

python plot.py --compare --csv-files figures_baseline/combo_pond_convergence.csv \
                                    figures_evolution/1d/combo_pond_convergence.csv \
                                    figures_evolution/2d/combo_pond_convergence.csv \
                                    figures_fi2pop/combo_pond_convergence.csv \
                                    --labels baseline evolution_1d evolution_2d fi2pop \
                                    --title "Pond Combo Convergence Behavior" \
                                    --output comparison_figures/pond_comparison.png &

python plot.py --compare --csv-files figures_baseline/combo_pond_hard_convergence.csv \
                                    figures_evolution/1d/combo_pond_hard_convergence.csv \
                                    figures_evolution/2d/combo_pond_hard_convergence.csv \
                                    figures_fi2pop/combo_pond_hard_convergence.csv \
                                    --labels baseline evolution_1d evolution_2d fi2pop \
                                    --title "Pond Hard Combo Convergence Behavior" \
                                    --output comparison_figures/pond_hard_comparison.png &

wait
# echo "Plotting ablations..."

# python plot.py --compare --csv-files figures_baseline/binary_convergence.csv \
#                                     figures_evolution/1d/0_random_offspring/binary_easy_convergence.csv \
#                                     figures_evolution/2d/0_random_offspring/binary_easy_convergence.csv \
#                                     figures_fi2pop/binary_convergence.csv \
#                                     --labels baseline evolution_1d_no_random_agents evolution_2d_no_random_agents fi2pop \
#                                     --title "Binary Easy Convergence Behavior With No Random Agents" \
#                                     --output comparison_figures/0_random_offspring_binary_easy_comparison.png &

# python plot.py --compare --csv-files figures_baseline/binary_hard_convergence.csv \
#                                     figures_evolution/1d/0_random_offspring/binary_hard_convergence.csv \
#                                     figures_evolution/2d/0_random_offspring/binary_hard_convergence.csv \
#                                     figures_fi2pop/binary_hard_convergence.csv \
#                                     --labels baseline evolution_1d_no_random_agents evolution_2d_no_random_agents fi2pop \
#                                     --title "Binary Hard Convergence Behavior With No Random Agents" \
#                                     --output comparison_figures/0_random_offspring_binary_hard_comparison.png &

# python plot.py --compare --csv-files figures_baseline/combo_grass_convergence.csv \
#                                     figures_evolution/1d/0_random_offspring/combo_grass_convergence.csv \
#                                     figures_evolution/2d/0_random_offspring/combo_grass_convergence.csv \
#                                     figures_fi2pop/combo_grass_convergence.csv \
#                                     --labels baseline evolution_1d_no_random_agents evolution_2d_no_random_agents fi2pop \
#                                     --title "Grass Combo Convergence Behavior With No Random Agents" \
#                                     --output comparison_figures/0_random_offspring_grass_comparison.png &

# python plot.py --compare --csv-files figures_baseline/combo_grass_hard_convergence.csv \
#                                     figures_evolution/1d/0_random_offspring/combo_grass_hard_convergence.csv \
#                                     figures_evolution/2d/0_random_offspring/combo_grass_hard_convergence.csv \
#                                     figures_fi2pop/combo_grass_hard_convergence.csv \
#                                     --labels baseline evolution_1d_no_random_agents evolution_2d_no_random_agents fi2pop \
#                                     --title "Hard Grass Combo Convergence Behavior With No Random Agents" \
#                                     --output comparison_figures/0_random_offspring_grass_hard_comparison.png &

# python plot.py --compare --csv-files figures_baseline/combo_river_convergence.csv \
#                                     figures_evolution/1d/0_random_offspring/combo_river_convergence.csv \
#                                     figures_evolution/2d/0_random_offspring/combo_river_convergence.csv \
#                                     figures_fi2pop/combo_river_convergence.csv \
#                                     --labels baseline evolution_1d_no_random_agents evolution_2d_no_random_agents fi2pop \
#                                     --title "River Combo Convergence Behavior With No Random Agents" \
#                                     --output comparison_figures/0_random_offspring_river_comparison.png &

# python plot.py --compare --csv-files figures_baseline/combo_river_hard_convergence.csv \
#                                     figures_evolution/1d/0_random_offspring/combo_river_hard_convergence.csv \
#                                     figures_evolution/2d/0_random_offspring/combo_river_hard_convergence.csv \
#                                     figures_fi2pop/combo_river_hard_convergence.csv \
#                                     --labels baseline evolution_1d_no_random_agents evolution_2d_no_random_agents fi2pop \
#                                     --title "River Hard Combo Convergence Behavior With No Random Agents" \
#                                     --output comparison_figures/0_random_offspring_river_hard_comparison.png & 

# python plot.py --compare --csv-files figures_baseline/combo_pond_convergence.csv \
#                                     figures_evolution/1d/0_random_offspring/combo_pond_convergence.csv \
#                                     figures_evolution/2d/0_random_offspring/combo_pond_convergence.csv \
#                                     figures_fi2pop/combo_pond_convergence.csv \
#                                     --labels baseline evolution_1d_no_random_agents evolution_2d_no_random_agents fi2pop \
#                                     --title "Pond Combo Convergence Behavior With No Random Agents" \
#                                     --output comparison_figures/0_random_offspring_pond_comparison.png &

# python plot.py --compare --csv-files figures_baseline/combo_pond_hard_convergence.csv \
#                                     figures_evolution/1d/0_random_offspring/combo_pond_hard_convergence.csv \
#                                     figures_evolution/2d/0_random_offspring/combo_pond_hard_convergence.csv \
#                                     figures_fi2pop/combo_pond_hard_convergence.csv \
#                                     --labels baseline evolution_1d_no_random_agents evolution_2d_no_random_agents fi2pop \
#                                     --title "Pond Hard Combo Convergence Behavior With No Random Agents" \
#                                     --output comparison_figures/0_random_offspring_pond_hard_comparison.png &

# wait
