#!/bin/bash
#SBATCH --job-name=calv5
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/calvano/slurm_m5_%A_%a.out
#SBATCH --error=results/calvano/slurm_m5_%A_%a.err
#SBATCH --array=1-25

source /etc/profile
module load conda/latest
conda activate elec_collusion

cd ~/StackelbergPOMDP/Code/StackelbergPOMDP/stackelberg_pomdp
export PYTHONPATH=..

python -u calvano_replication.py --m 5 --n_sessions 1 --seed_offset $((SLURM_ARRAY_TASK_ID - 1)) --output_dir ../results/calvano_m5
