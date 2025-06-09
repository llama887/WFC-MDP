#!/bin/bash
#SBATCH --job-name=hyperparams_binary_mcts
#SBATCH --output=hyperparams_binary_mcts_%j.out
#SBATCH --error=hyperparams_binary_mcts_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --time=48:00:00 
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1           
#SBATCH --cpus-per-task=48           
#SBATCH --mem=180G                             
#SBATCH --account=pr_100_tandon_priority

### -------------------- Logging Setup -------------------- ###
LOG_DIR="/scratch/$USER/optimizing_WFC/output"
GPU_LOG_FILE="$LOG_DIR/gpu_used_${SLURM_JOB_ID}.txt"
mkdir -p $LOG_DIR
log_and_email() {
    MESSAGE="$1"
    echo "$MESSAGE" | tee -a "$GPU_LOG_FILE"
    echo -e "Subject:[Slurm Job: $SLURM_JOB_ID] Status Update\n\n$MESSAGE" | sendmail fyy2003@nyu.edu
}
log_and_email "Starting job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"

module purge
cd /scratch/fyy2003/optimizing_WFC
source venv/bin/activate
python core/fi2pop.py --generations-per-trial 500 --hyperparameter-dir hyperparameters --output-file mcts_binary_hyperparameters.yaml --task binary_hard --optuna-trials 5



