#!/bin/bash
#SBATCH --job-name=ql_p13
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/ql_pmin13/slurm_%A_%a.out
#SBATCH --error=results/ql_pmin13/slurm_%A_%a.err
#SBATCH --array=0-39
#SBATCH --exclude=uri-cpu001,uri-cpu002,uri-cpu003,uri-cpu004,uri-cpu005,uri-cpu006,uri-cpu007,uri-cpu008,uri-cpu009,uri-cpu010,uri-cpu011,uri-cpu012,uri-cpu013,uri-cpu014,uri-cpu015,uri-cpu016,uri-cpu017,uri-cpu018,uri-cpu019,uri-cpu020,uri-cpu021,uri-cpu022,uri-cpu023,uri-cpu024

source /etc/profile
module load conda/latest
module load gurobi/12.0.0
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP

# 4 lambdas x 2 obs x 5 seeds = 40 jobs
LAMBDAS=(0.01 0.05 0.1 0.5)
OBS_TYPES=("price_profile" "no_observation")
SEEDS=(1 2 3 4 5)

IDX=$SLURM_ARRAY_TASK_ID
LAM_IDX=$((IDX / 10))
REMAIN=$((IDX % 10))
OBS_IDX=$((REMAIN / 5))
SEED_IDX=$((REMAIN % 5))

LAM=${LAMBDAS[$LAM_IDX]}
OBS=${OBS_TYPES[$OBS_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Job $IDX: seed=$SEED, lambda=$LAM, obs=$OBS, price_min=1.3"
python experiments/run_ql_pmin13.py $SEED $LAM $OBS
