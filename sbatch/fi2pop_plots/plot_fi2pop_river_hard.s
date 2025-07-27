#!/bin/bash
#SBATCH --job-name=plot_river_hard_fi2pop
#SBATCH --output=plot_river_hard_fi2pop_%j.out
#SBATCH --error=plot_river_hard_fi2pop_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --time=48:00:00
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1           
#SBATCH --cpus-per-task=48           
#SBATCH --mem=64G                             
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
# python plot.py --load-hyperparameters hyperparameters/qd_binary_hyperparameters.yaml --qd
python plot.py --load-hyperparameters hyperparameters/fi2pop_combo_river_hyperparameters.yaml --task river --combo hard --debug --method fi2pop




