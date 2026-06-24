#!/bin/bash
#SBATCH --job-name=base_m4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/baselines_m4/slurm_%A_%a.out
#SBATCH --error=results/baselines_m4/slurm_%A_%a.err
#SBATCH --array=1-75

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# 3 interventions x 25 seeds = 75 jobs
TASK=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$(( (TASK % 25) + 1 ))
INT_IDX=$((TASK / 25))

INTERVENTIONS=(no_intervene pdp dpdp)
INT=${INTERVENTIONS[$INT_IDX]}

echo "Baseline: $INT, Seed=$SEED, m=4"

python -u calvano_replication.py --m 4 --alpha 0.25 --beta 1e-4 --platform_intervention $INT --n_sessions 1 --seed_offset $((SEED - 1))
