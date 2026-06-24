#!/bin/bash
#SBATCH --job-name=k1pol
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/k1_policies/slurm_%A_%a.out
#SBATCH --error=results/k1_policies/slurm_%A_%a.err
#SBATCH --array=0-149

source /etc/profile
module load conda/latest
module load gurobi/12.0.0
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP

# 15 policies x 10 seeds = 150 jobs
N_POLICIES=15
N_SEEDS=10

IDX=$SLURM_ARRAY_TASK_ID
POLICY_IDX=$((IDX / N_SEEDS))
SEED=$((IDX % N_SEEDS + 1))

echo "Job $IDX: policy=$POLICY_IDX, seed=$SEED"
python experiments/test_k1_policies.py $POLICY_IDX $SEED
