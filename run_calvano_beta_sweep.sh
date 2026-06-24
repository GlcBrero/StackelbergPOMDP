#!/bin/bash
#SBATCH --job-name=betaswp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/calvano_beta_sweep/slurm_%A_%a.out
#SBATCH --error=results/calvano_beta_sweep/slurm_%A_%a.err
#SBATCH --array=1-50

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# 5 betas x 10 seeds = 50 jobs
TASK=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$(( (TASK % 10) + 1 ))
BETA_IDX=$((TASK / 10))

BETAS=(4e-6 1e-5 4e-5 1e-4 4e-4)
BETA=${BETAS[$BETA_IDX]}

echo "Beta=$BETA, Seed=$SEED"

python -u calvano_replication.py --m 5 --beta $BETA --n_sessions 1 --seed_offset $((SEED - 1))
