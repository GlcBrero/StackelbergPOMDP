#!/bin/bash
#SBATCH --job-name=m4grid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=results/calvano_m4/slurm_%A_%a.out
#SBATCH --error=results/calvano_m4/slurm_%A_%a.err
#SBATCH --array=1-75

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# 3 alphas x 5 betas x 5 seeds = 75 jobs
TASK=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$(( (TASK % 5) + 1 ))
TASK=$((TASK / 5))
BETA_IDX=$((TASK % 5))
ALPHA_IDX=$((TASK / 5))

ALPHAS=(0.05 0.15 0.25)
BETAS=(4e-5 1e-4 4e-4 1e-3 4e-3)
ALPHA=${ALPHAS[$ALPHA_IDX]}
BETA=${BETAS[$BETA_IDX]}

echo "Alpha=$ALPHA, Beta=$BETA, Seed=$SEED, m=4"

python -u calvano_replication.py --m 4 --alpha $ALPHA --beta $BETA --n_sessions 1 --seed_offset $((SEED - 1))
