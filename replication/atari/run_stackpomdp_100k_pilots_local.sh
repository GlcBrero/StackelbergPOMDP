#!/usr/bin/env bash

# Run the matched seed-1 Atari StackPOMDP diagnostics on one local machine.
#
# The stages are deliberately serialized to avoid oversubscribing the laptop:
#   1. seller meta-response to five queried buyer thresholds;
#   2. seller leader against the independently audited buyer meta-response;
#   3. buyer leader against the selected seller meta-response.
#
# Each scientific stage has its own checkpoint, log, W&B job type, and failure
# sentinel.  A failed meta-seller blocks only the dependent buyer-leader stage;
# the seller-leader diagnostic can still run against the audited meta-buyer.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${CODE_ROOT}/../.." && pwd)"
PYTHON="/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python"
RUN_TAG="${STACKPOMDP_RUN_TAG:-stackpomdp_20260727_100k_v1}"
LOG_DIR="${WORKSPACE_ROOT}/Research Artifacts/experiment_logs/StackelbergPOMDP/atari/${RUN_TAG}"

E0="${CODE_ROOT}/replication/atari/checkpoints/sb3/space_invaders_e0_ppo_seed1_10m_best.zip"
META_BUYER="${CODE_ROOT}/replication/atari/checkpoints/stackpomdp/meta_buyer_response_ppo_seed1_100k_audited.zip"
META_SELLER="${CODE_ROOT}/replication/atari/checkpoints/stackpomdp/meta_seller_response_ppo_seed1_100k_v1.zip"
META_SELLER_BEST="${CODE_ROOT}/replication/atari/checkpoints/stackpomdp/meta_seller_response_ppo_seed1_100k_v1_best.zip"
META_SELLER_EVAL="${CODE_ROOT}/replication/atari/checkpoints/stackpomdp/meta_seller_response_ppo_seed1_100k_v1.evaluation.json"
META_SELLER_AUDIT="${CODE_ROOT}/replication/atari/checkpoints/stackpomdp/meta_seller_response_ppo_seed1_100k_v1_best.full_evaluation_seed200103.json"

SELLER_LEADER="${CODE_ROOT}/replication/atari/checkpoints/stackpomdp/space_invaders_seller_leader_ppo_seed1_100k_v1.zip"
SELLER_LEADER_EVAL="${CODE_ROOT}/replication/atari/checkpoints/stackpomdp/space_invaders_seller_leader_ppo_seed1_100k_v1.evaluation.json"
BUYER_LEADER="${CODE_ROOT}/replication/atari/checkpoints/stackpomdp/space_invaders_buyer_leader_ppo_seed1_100k_v1.zip"
BUYER_LEADER_EVAL="${CODE_ROOT}/replication/atari/checkpoints/stackpomdp/space_invaders_buyer_leader_ppo_seed1_100k_v1.evaluation.json"

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/private/tmp/stackpomdp-matplotlib
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_DIR}"
cd "${CODE_ROOT}"

if [[ -e "${LOG_DIR}/pipeline.started" ]]; then
    echo "Refusing to reuse existing run directory: ${LOG_DIR}" >&2
    exit 2
fi
touch "${LOG_DIR}/pipeline.started"

pipeline_ok=0
trap 'rc=$?; if [[ ${pipeline_ok} -eq 1 && ${rc} -eq 0 ]]; then touch "${LOG_DIR}/pipeline.ok"; else touch "${LOG_DIR}/pipeline.failed"; fi' EXIT

run_stage() {
    local stage="$1"
    shift
    local log="${LOG_DIR}/${stage}.log"
    local started="${LOG_DIR}/${stage}.started"
    local ok="${LOG_DIR}/${stage}.ok"
    local failed="${LOG_DIR}/${stage}.failed"

    if [[ -e "${started}" || -e "${ok}" || -e "${failed}" ]]; then
        echo "Refusing to overwrite stage state for ${stage}" >&2
        return 2
    fi
    touch "${started}"
    {
        echo "stage=${stage}"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'command='
        printf '%q ' "$@"
        printf '\n'
    } | tee -a "${log}"

    set +e
    "$@" 2>&1 | tee -a "${log}"
    local rc=${PIPESTATUS[0]}
    set -e

    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc}" | tee -a "${log}"
    if [[ ${rc} -eq 0 ]]; then
        touch "${ok}"
    else
        touch "${failed}"
    fi
    return "${rc}"
}

for required in "${PYTHON}" "${E0}" "${META_BUYER}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing required file: ${required}" >&2
        exit 3
    fi
done

run_stage preflight \
    "${PYTHON}" -c \
    'import sys; sys.modules["readline"] = None; import pytest; raise SystemExit(pytest.main(["-q", "tests/test_atari_meta_response_trainer.py", "tests/test_atari_query_trace.py", "tests/test_atari_stackpomdp_env.py", "tests/test_atari_stackpomdp_full_leader_env.py", "tests/test_atari_stackpomdp_leader_trainer.py", "tests/test_atari_stackpomdp_policy.py", "tests/test_atari_stochastic_timing.py", "tests/test_atari_sb3_pipeline.py"]))'

meta_seller_ok=0
if run_stage meta_seller \
    "${PYTHON}" -u -m replication.atari.train_atari_meta_response_sb3 \
    --role seller \
    --seed 1 \
    --timesteps 100000 \
    --game-checkpoint "${E0}" \
    --checkpoint "${META_SELLER}" \
    --output "${META_SELLER_EVAL}" \
    --num-envs 4 \
    --start-method spawn \
    --n-steps 100 \
    --batch-size 100 \
    --n-epochs 4 \
    --learning-rate 0.0003 \
    --entropy-coeff 0.01 \
    --gameplay-horizon 200 \
    --event-tail-steps 50 \
    --eval-seed 200001 \
    --eval-episodes-per-context 3 \
    --random-eval-episodes 20 \
    --eval-every 25000 \
    --checkpoint-every 25000 \
    --log-every 5000 \
    --device cpu \
    --wandb \
    --wandb-project StackPOMDP \
    --wandb-entity glcbrero \
    --wandb-group atari_stackpomdp \
    --wandb-job-type meta_seller_response \
    --wandb-name atari_meta_seller_response_ppo_seed1_100k_local_v1
then
    if run_stage meta_seller_full_audit \
        "${PYTHON}" -u -m replication.atari.train_atari_meta_response_sb3 \
        --role seller \
        --resume "${META_SELLER_BEST}" \
        --eval-only \
        --game-checkpoint "${E0}" \
        --gameplay-horizon 200 \
        --event-tail-steps 50 \
        --eval-seed 200103 \
        --eval-episodes-per-context 20 \
        --random-eval-episodes 100 \
        --output "${META_SELLER_AUDIT}" \
        --device cpu \
        --no-wandb
    then
        meta_seller_ok=1
    fi
fi

seller_leader_ok=0
if run_stage seller_leader \
    "${PYTHON}" -u -m replication.atari.train_atari_stackpomdp_leader_sb3 \
    --leader-role seller \
    --response-checkpoint "${META_BUYER}" \
    --game-checkpoint "${E0}" \
    --seed 1 \
    --timesteps 100000 \
    --checkpoint "${SELLER_LEADER}" \
    --output "${SELLER_LEADER_EVAL}" \
    --num-envs 4 \
    --start-method spawn \
    --n-steps 210 \
    --batch-size 210 \
    --n-epochs 10 \
    --learning-rate 0.0003 \
    --entropy-coeff 0.01 \
    --gameplay-horizon 200 \
    --event-tail-steps 50 \
    --eval-episodes 20 \
    --eval-seed 200003 \
    --eval-every 25000 \
    --checkpoint-every 25000 \
    --log-every 1000 \
    --device cpu \
    --wandb \
    --wandb-project StackPOMDP \
    --wandb-entity glcbrero \
    --wandb-group atari_stackpomdp \
    --wandb-job-type seller_leader \
    --wandb-name atari_stackpomdp_seller_leader_ppo_seed1_100k_local_v1
then
    seller_leader_ok=1
fi

buyer_leader_ok=0
if [[ ${meta_seller_ok} -eq 1 ]]; then
    if run_stage buyer_leader \
        "${PYTHON}" -u -m replication.atari.train_atari_stackpomdp_leader_sb3 \
        --leader-role buyer \
        --response-checkpoint "${META_SELLER_BEST}" \
        --game-checkpoint "${E0}" \
        --seed 1 \
        --timesteps 100000 \
        --checkpoint "${BUYER_LEADER}" \
        --output "${BUYER_LEADER_EVAL}" \
        --num-envs 4 \
        --start-method spawn \
        --n-steps 210 \
        --batch-size 210 \
        --n-epochs 10 \
        --learning-rate 0.0003 \
        --entropy-coeff 0.01 \
        --gameplay-horizon 200 \
        --event-tail-steps 50 \
        --eval-episodes 20 \
        --eval-seed 200003 \
        --eval-every 25000 \
        --checkpoint-every 25000 \
        --log-every 1000 \
        --device cpu \
        --wandb \
        --wandb-project StackPOMDP \
        --wandb-entity glcbrero \
        --wandb-group atari_stackpomdp \
        --wandb-job-type buyer_leader \
        --wandb-name atari_stackpomdp_buyer_leader_ppo_seed1_100k_local_v1
    then
        buyer_leader_ok=1
    fi
else
    echo "Skipping buyer leader because the selected meta-seller failed validation." | tee -a "${LOG_DIR}/buyer_leader.skipped"
fi

{
    echo "meta_seller_ok=${meta_seller_ok}"
    echo "seller_leader_ok=${seller_leader_ok}"
    echo "buyer_leader_ok=${buyer_leader_ok}"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${LOG_DIR}/pipeline.summary"

if [[ ${meta_seller_ok} -ne 1 || ${seller_leader_ok} -ne 1 || ${buyer_leader_ok} -ne 1 ]]; then
    exit 1
fi

pipeline_ok=1
