#!/bin/bash
#SBATCH --job-name=rr_bin
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/rr_binary/slurm_%A_%a.out
#SBATCH --error=results/rr_binary/slurm_%A_%a.err
#SBATCH --array=0-14

source /etc/profile
module load conda/latest
module load gurobi/12.0.0
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP

# 3 conditions x 5 seeds = 15 jobs
SEEDS=(1 2 3 4 5)

IDX=$SLURM_ARRAY_TASK_ID
COND=$((IDX / 5))
SEED_IDX=$((IDX % 5))
SEED=${SEEDS[$SEED_IDX]}

if [ $COND -eq 0 ]; then
    OBS="price_profile"
    SORT="unsorted"
elif [ $COND -eq 1 ]; then
    OBS="price_profile"
    SORT="sorted"
else
    OBS="no_observation"
    SORT="unsorted"
fi

echo "Job $IDX: seed=$SEED, obs=$OBS, sort=$SORT, RoundRobin, binary_threshold, lambda=0.5"
python experiments/run_rr_binary.py $SEED $OBS $SORT
