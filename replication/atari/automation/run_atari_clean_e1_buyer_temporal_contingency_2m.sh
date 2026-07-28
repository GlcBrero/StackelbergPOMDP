#!/bin/zsh
set -euo pipefail

source "${0:A:h}/atari_e1_temporal_contingency_common.zsh"

set +e
e1_activate
activation_status=$?
set -e
if (( activation_status == 3 )); then
  print "temporal contingency remains dormant: both final uniform all-six failures are required"
  exit 3
elif (( activation_status != 0 )); then
  exit "$activation_status"
fi

e1_prepare_runtime
e1_run_preflight

if [[ -f "$E1_FAMILY" ]]; then
  e1_validate_existing_family
  "$E1_CODE_ROOT/replication/atari/automation/run_e1_buyer_temporal_contingency_final_selector.sh"
  exit $?
fi

if ! mkdir "$E1_TRAINING_LOCK" 2>/dev/null; then
  e1_die "another temporal-contingency training process is running"
fi
trap 'rmdir "$E1_TRAINING_LOCK" 2>/dev/null || true' EXIT INT TERM

resume=$(jq -r '.resume_source.path' "$E1_ACTIVATION")
e0b=$(jq -r '.e0b_source.path' "$E1_ACTIVATION")
e1_require_regular_sha "$resume" "$(jq -r '.resume_source.sha256' "$E1_ACTIVATION")"
e1_require_regular_sha "$e0b" "$(jq -r '.e0b_source.sha256' "$E1_ACTIVATION")"

e1_refuse_path "$E1_BASE"
e1_refuse_path "${E1_BASE%.zip}.evaluation.json"
e1_refuse_path "${E1_BASE%.zip}.fixed_contexts.csv"
e1_refuse_path "${E1_BASE%.zip}.training.jsonl"
e1_refuse_path "$E1_TRAIN_LOG"
while IFS= read -r candidate; do
  [[ "$candidate" == "$E1_BASE" ]] || e1_refuse_path "$candidate"
done < <(e1_candidates)

mkdir -p "$E1_CHECKPOINT_ROOT" "$E1_LOG_ROOT" "$E1_WANDB_ROOT"
export WANDB_DIR="$E1_WANDB_ROOT"
export WANDB_MODE=online
export MPLCONFIGDIR=/private/tmp/mpl-stackpomdp-e1-temporal-training

set +e
(
  cd "$E1_CODE_ROOT"
  "$E1_PYTHON" -u -m replication.atari.train_atari_meta_response_sb3 \
    --role buyer \
    --seed 1 \
    --timesteps 2000800 \
    --gameplay-horizon 200 \
    --event-tail-steps 0 \
    --e1-sampler-mode temporal-marginal-v1 \
    --num-envs 4 \
    --start-method spawn \
    --n-steps 205 \
    --batch-size 820 \
    --n-epochs 4 \
    --learning-rate 0.0001 \
    --pretrained-lr-scale 0.1 \
    --actor-loss-mode balanced \
    --entropy-coeff 0.01 \
    --clip-range 0.1 \
    --value-coefficient 0.5 \
    --max-grad-norm 0.5 \
    --noop-max 30 \
    --max-frames 100000 \
    --rom-path "$E1_ROM" \
    --e0b-checkpoint "$e0b" \
    --resume "$resume" \
    --checkpoint "$E1_BASE" \
    --checkpoint-every 400160 \
    --eval-episodes 100 \
    --fixed-eval-episodes 20 \
    --device cpu \
    --wandb \
    --wandb-project StackPOMDP \
    --wandb-group atari_clean_curriculum \
    --wandb-name "$E1_RUN_NAME"
) 2>&1 | tee "$E1_TRAIN_LOG"
training_status=$pipestatus[1]
set -e
if (( training_status != 0 )); then
  e1_die "temporal E1 training failed with exit status $training_status"
fi

while IFS= read -r candidate; do
  e1_wait_for_stable_zip "$candidate"
done < <(e1_candidates)
e1_validate_family

rmdir "$E1_TRAINING_LOCK"
trap - EXIT INT TERM
"$E1_CODE_ROOT/replication/atari/automation/run_e1_buyer_temporal_contingency_final_selector.sh"
