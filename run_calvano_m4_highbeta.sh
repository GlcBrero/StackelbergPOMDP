#!/bin/bash
#SBATCH --job-name=m4hb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=results/calvano_m4_highbeta/slurm_%A_%a.out
#SBATCH --error=results/calvano_m4_highbeta/slurm_%A_%a.err
#SBATCH --array=1-50

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# 2 alphas x 5 betas x 5 seeds = 50 jobs
TASK=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$(( (TASK % 5) + 1 ))
TASK=$((TASK / 5))
BETA_IDX=$((TASK % 5))
ALPHA_IDX=$((TASK / 5))

ALPHAS=(0.15 0.25)
BETAS=(1e-2 4e-2 1e-1 4e-1 1)
ALPHA=${ALPHAS[$ALPHA_IDX]}
BETA=${BETAS[$BETA_IDX]}

echo "Alpha=$ALPHA, Beta=$BETA, Seed=$SEED, m=4"

python -u calvano_replication.py --m 4 --alpha $ALPHA --beta $BETA --n_sessions 1 --seed_offset $((SEED - 1))
