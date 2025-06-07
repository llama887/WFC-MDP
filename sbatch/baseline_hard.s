#!/bin/bash
#SBATCH --job-name=baseline_plots_hard
#SBATCH --output=baseline_plots_hard%j.out
#SBATCH --error=baseline_plots_hard%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --time=10:00:00 
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1           
#SBATCH --cpus-per-task=48           
#SBATCH --mem=128G                             
#SBATCH --account=pr_100_tandon_priority

### -------------------- Logging Setup -------------------- ###
 LOG_DIR="${OUTPUT_DIR:-/scratch/$USER/optimizing_WFC/output}"
 LOG_FILE="$LOG_DIR/baseline_plots_hard_${SLURM_JOB_ID}.log"
...
 log_and_email() {
     MESSAGE="$1"
    echo "$MESSAGE" | tee -a "$LOG_FILE"
     echo -e "Subject:[Slurm Job: $SLURM_JOB_ID] Status Update\n\n$MESSAGE" \
          | sendmail "${MAIL_USER:-fyy2003@nyu.edu}"
 }
 
# Notify on failure or normal exit
 trap 'log_and_email "Job FAILED: $SLURM_JOB_NAME ($SLURM_JOB_ID) at $(date)"' ERR
 trap 'log_and_email "Job COMPLETED: $SLURM_JOB_NAME ($SLURM_JOB_ID) at $(date)"' EXIT

module purge
cd /scratch/fyy2003/optimizing_WFC
source venv/bin/activate
python fi2pop.py   -l hyperparameters/binary_hyperparameters.yaml   --hard   --binary --bar-graph

