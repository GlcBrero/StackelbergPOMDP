#!/bin/bash
#SBATCH --job-name=matrix-rllib-es
#SBATCH --partition=cpu
#SBATCH --array=0-24
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=replication/matrix_ablations/native_es_%A_%a.out
#SBATCH --error=replication/matrix_ablations/native_es_%A_%a.err

set -euo pipefail

source /etc/profile
module load conda/latest
conda activate stackelberg-pomdp-ray-es

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matrix_es_mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit this array from the repository root}"
RESULT_ROOT="${PROJECT_ROOT}/replication/matrix_ablations/results/hidden_queries_rllib_es_s1to25"
RESPONSE_DIR="${PROJECT_ROOT}/replication/matrix_ablations/checkpoints/certified_response_seed2"
RESPONSE_CHECKPOINT="${RESPONSE_DIR}/model.zip"
RESPONSE_CONTRACT="${RESPONSE_DIR}/response_contract.json"
SEED="$((SLURM_ARRAY_TASK_ID + 1))"

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_ROOT}" "${MPLCONFIGDIR}"

for CONDITION in observed hidden; do
  python -m stackelberg_pomdp.experiments.matrix_ablations leader \
    --experiment hidden_queries \
    --condition "${CONDITION}" \
    --algorithm ES \
    --profile-id paper_joint_v1 \
    --response-checkpoint "${RESPONSE_CHECKPOINT}" \
    --response-contract-path "${RESPONSE_CONTRACT}" \
    --response-algorithm REINFORCE \
    --seed "${SEED}" \
    --es-iterations 300 \
    --es-num-workers 1 \
    --es-episodes-per-batch 1000 \
    --es-train-batch-size 1000 \
    --es-sigma 0.02 \
    --es-stepsize 0.01 \
    --es-l2-coeff 0.005 \
    --es-eval-prob 0.03 \
    --es-report-length 10 \
    --es-noise-size 250000000 \
    --es-eval-every 10 \
    --eval-episodes 20 \
    --final-eval-episodes 100 \
    --wandb \
    --wandb-project StackPOMDP \
    --wandb-group native-rllib-es-paper-s1to25 \
    --wandb-name "native-rllib-es-seed${SEED}-${CONDITION}" \
    --output-root "${RESULT_ROOT}"
done
