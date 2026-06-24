#!/bin/bash
#SBATCH --job-name=ev_lam
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/eval_lambda/slurm_%A_%a.out
#SBATCH --error=results/eval_lambda/slurm_%A_%a.err
#SBATCH --array=1-50

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

# 5 lambdas x 2 obs x 5 seeds = 50 jobs
TASK=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$(( (TASK % 5) + 1 ))
TASK=$((TASK / 5))
OBS_IDX=$((TASK % 2))
LAMBDA_IDX=$((TASK / 2))

LAMBDAS=(0 0.01 0.05 0.1 0.5)
LAMBDA=${LAMBDAS[$LAMBDA_IDX]}

OBS_TYPES=(price_profile no_observation)
OBS=${OBS_TYPES[$OBS_IDX]}

# Find the log directory for this config
LOG_DIR=$(ls -d logs/exp.RL:Standard.bertrand.50000000.A2C.${SEED}.${SEED}.30.50000.full.True.Qlearning.100.${OBS}.nocp.lam${LAMBDA}.nows 2>/dev/null | head -1)

if [ -z "$LOG_DIR" ]; then
    echo "No log dir found for Lambda=$LAMBDA, Obs=$OBS, Seed=$SEED"
    exit 1
fi

echo "Eval: Lambda=$LAMBDA, Obs=$OBS, Seed=$SEED, Dir=$LOG_DIR"

python -u eval_intervention.py --log_dir "$LOG_DIR" --obs_type $OBS --seed $SEED
