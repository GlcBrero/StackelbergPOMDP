#!/bin/zsh
set -euo pipefail

# Ordered v5 seller path: execution-only smoke, learned-conditioning preflight,
# then (and only then) a fresh formal online W&B run.  Selection is separate.

source "${0:A:h}/atari_e1_seller_shared_context_v5_common.zsh"

typeset -gx STACKPOMDP_E1V5_LOCK_TOKEN="${STACKPOMDP_E1V5_LOCK_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"
stackpomdp_claim_owned_lock \
  "$E1V5_LOCK" "$STACKPOMDP_E1V5_LOCK_TOKEN" \
  "v5 shared-context seller pipeline" || exit $?
typeset -g E1V5_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"

function release_e1v5_lock() {
  local cleanup_status=0
  e2_release_transient_lock || cleanup_status=$?
  stackpomdp_release_owned_lock \
    "$E1V5_LOCK" "$STACKPOMDP_E1V5_LOCK_TOKEN" "$E1V5_LOCK_OWNED" \
    "v5 shared-context seller pipeline" || cleanup_status=$?
  typeset -g E1V5_LOCK_OWNED=0
  return "$cleanup_status"
}

function interrupt_e1v5() {
  local exit_status="$1"
  trap - EXIT HUP INT TERM
  e1v5_cancel_active_job
  release_e1v5_lock || :
  exit "$exit_status"
}

trap 'release_e1v5_lock' EXIT
trap 'interrupt_e1v5 129' HUP
trap 'interrupt_e1v5 130' INT
trap 'interrupt_e1v5 143' TERM

function e1v5_run_logged() {
  local log_path="$1"
  shift
  set +e
  (
    set +e
    ("$@") 2>&1 | tee "$log_path"
    pipeline_status=(${pipestatus[@]})
    (( pipeline_status[2] == 0 )) || exit "${pipeline_status[2]}"
    exit "${pipeline_status[1]}"
  ) &
  E1V5_ACTIVE_JOB_PID=$!
  wait "$E1V5_ACTIVE_JOB_PID"
  local job_status=$?
  E1V5_ACTIVE_JOB_PID=0
  set -e
  return "$job_status"
}

function e1v5_train_command() {
  local code_root="$1"
  local timesteps="$2"
  local checkpoint_every="$3"
  local eval_episodes="$4"
  local fixed_eval_episodes="$5"
  local checkpoint="$6"
  shift 6
  (
    cd "$code_root"
    "$E1V5_PYTHON" -u -m replication.atari.train_atari_meta_response_sb3 \
      --role seller \
      --e0b-checkpoint "$E1V5_E0B" \
      --economic-architecture "$E1V5_ARCHITECTURE" \
      --e1-sampler-mode uniform \
      --actor-loss-mode balanced \
      --seed 1 \
      --timesteps "$timesteps" \
      --gameplay-horizon 200 \
      --event-tail-steps 0 \
      --num-envs 4 \
      --start-method spawn \
      --n-steps 205 \
      --batch-size 820 \
      --n-epochs 4 \
      --learning-rate 0.0005 \
      --pretrained-lr-scale 0.1 \
      --entropy-coeff 0.01 \
      --clip-range 0.1 \
      --value-coefficient 0.5 \
      --max-grad-norm 0.5 \
      --noop-max 30 \
      --max-frames 100000 \
      --rom-path "$ROM" \
      --checkpoint-every "$checkpoint_every" \
      --eval-episodes "$eval_episodes" \
      --fixed-eval-episodes "$fixed_eval_episodes" \
      --checkpoint "$checkpoint" \
      "$@"
  )
}

e1v5_require_scoped_clean
typeset -gr E1V5_REVISION=$(git -C "$E1V5_SOURCE_ROOT" rev-parse HEAD)
[[ ${#E1V5_REVISION} -eq 40 && "$E1V5_REVISION" != *[!0-9a-f]* ]] || \
  e1v5_die "v5 source revision is not a full lowercase SHA"
typeset -gr E1V5_DIAGNOSTIC_REVISION=${E1V5_EVIDENCE_REVISION:-$E1V5_REVISION}
[[ ${#E1V5_DIAGNOSTIC_REVISION} -eq 40 \
    && "$E1V5_DIAGNOSTIC_REVISION" != *[!0-9a-f]* ]] || \
  e1v5_die "v5 diagnostic revision is not a full lowercase SHA"
export STACKPOMDP_SPACE_INVADERS_ROM="$ROM"
export PYTHONPATH="$E1V5_SOURCE_ROOT"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$CHECKPOINT_ROOT" "$WANDB_ROOT" "$LOG_ROOT" "$E1_OUTPUT"

if [[ ! -f "$E1V5_SMOKE" || -L "$E1V5_SMOKE" ]]; then
  for artifact in \
      "$E1V5_SMOKE" \
      "${E1V5_SMOKE%.zip}_step${E1V5_SMOKE_TIMESTEPS}.zip" \
      "$E1V5_SMOKE_TRACE" \
      "$E1V5_SMOKE_EVALUATION" \
      "${E1V5_SMOKE%.zip}.fixed_contexts.csv" \
      "$E1V5_SMOKE_LOG"; do
    e1v5_refuse_path "$artifact"
  done
  print "starting v5 ${E1V5_PROTOCOL} no-W&B ${E1V5_SMOKE_TIMESTEPS}-step Uniform(0,1)^5 mechanics smoke"
  e1v5_run_logged "$E1V5_SMOKE_LOG" \
    e1v5_train_command \
      "$E1V5_SOURCE_ROOT" "$E1V5_SMOKE_TIMESTEPS" \
      "$E1V5_SMOKE_TIMESTEPS" 20 20 "$E1V5_SMOKE" \
      --no-wandb
fi
e1v5_wait_for_stable_zip "$E1V5_SMOKE"
(
  cd "$E1V5_SOURCE_ROOT"
  "$E1V5_PYTHON" "$E1V5_VALIDATOR" \
    "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" validate-smoke \
    --checkpoint "$E1V5_SMOKE" \
    --training-log "$E1V5_SMOKE_TRACE" \
    --evaluation "$E1V5_SMOKE_EVALUATION" \
    --e0b "$E1V5_E0B" \
    --code-revision "$E1V5_DIAGNOSTIC_REVISION"
)

if [[ ! -f "$E1V5_PREFLIGHT" || -L "$E1V5_PREFLIGHT" ]]; then
  for artifact in \
      "$E1V5_PREFLIGHT" \
      "${E1V5_PREFLIGHT%.zip}_step${E1V5_PREFLIGHT_TIMESTEPS}.zip" \
      "$E1V5_PREFLIGHT_TRACE" \
      "$E1V5_PREFLIGHT_EVALUATION" \
      "${E1V5_PREFLIGHT%.zip}.fixed_contexts.csv" \
      "$E1V5_PREFLIGHT_PROBE" \
      "$E1V5_PREFLIGHT_BEHAVIOR" \
      "$E1V5_PREFLIGHT_LOG"; do
    e1v5_refuse_path "$artifact"
  done
  print "starting v5 ${E1V5_PROTOCOL} no-W&B ${E1V5_PREFLIGHT_TIMESTEPS}-step Uniform(0,1)^5 conditioning preflight"
  e1v5_run_logged "$E1V5_PREFLIGHT_LOG" \
    e1v5_train_command \
      "$E1V5_SOURCE_ROOT" "$E1V5_PREFLIGHT_TIMESTEPS" \
      "$E1V5_PREFLIGHT_TIMESTEPS" 20 20 "$E1V5_PREFLIGHT" \
      --no-wandb
fi
e1v5_wait_for_stable_zip "$E1V5_PREFLIGHT"
if [[ ! -e "$E1V5_PREFLIGHT_PROBE" && ! -L "$E1V5_PREFLIGHT_PROBE" ]]; then
  (
    cd "$E1V5_SOURCE_ROOT"
    "$E1V5_PYTHON" -u -m \
      replication.atari.probe_atari_e1_seller_shared_context \
      --checkpoint "$E1V5_PREFLIGHT" \
      --e0b-checkpoint "$E1V5_E0B" \
      --device cpu \
      --output "$E1V5_PREFLIGHT_PROBE" \
      --require-pass
  )
fi
if [[ ! -e "$E1V5_PREFLIGHT_BEHAVIOR" \
    && ! -L "$E1V5_PREFLIGHT_BEHAVIOR" ]]; then
  (
    cd "$E1V5_SOURCE_ROOT"
    "$E1V5_PYTHON" -u "$E1V5_VALIDATOR" \
      "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" behavioral-preflight \
      --checkpoint "$E1V5_PREFLIGHT" \
      --e0b "$E1V5_E0B" \
      --rom "$ROM" \
      --output "$E1V5_PREFLIGHT_BEHAVIOR" \
      --code-revision "$E1V5_DIAGNOSTIC_REVISION"
  )
fi
(
  cd "$E1V5_SOURCE_ROOT"
  "$E1V5_PYTHON" "$E1V5_VALIDATOR" \
    "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" validate-preflight \
    --checkpoint "$E1V5_PREFLIGHT" \
    --training-log "$E1V5_PREFLIGHT_TRACE" \
    --evaluation "$E1V5_PREFLIGHT_EVALUATION" \
    --probe "$E1V5_PREFLIGHT_PROBE" \
    --behavior-report "$E1V5_PREFLIGHT_BEHAVIOR" \
    --e0b "$E1V5_E0B" \
    --rom "$ROM" \
    --code-revision "$E1V5_DIAGNOSTIC_REVISION"
)

if [[ ! -f "$E1V5_GATE" || -L "$E1V5_GATE" ]]; then
  for artifact in \
      "$E1V5_GATE" \
      "$E1V5_FORMAL" \
      "$E1V5_FORMAL_TRACE" \
      "$E1V5_FORMAL_EVALUATION" \
      "${E1V5_FORMAL%.zip}.fixed_contexts.csv" \
      "$E1V5_FORMAL_LOG"; do
    e1v5_refuse_path "$artifact"
  done
  for step in $E1V5_FORMAL_STEPS; do
    e1v5_refuse_path "${E1V5_FORMAL%.zip}_step${step}.zip"
  done
fi
(
  cd "$E1V5_SOURCE_ROOT"
  "$E1V5_PYTHON" "$E1V5_VALIDATOR" \
    "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" gate \
    --seller-release "$E1V5_SELLER_RELEASE" \
    --e0b "$E1V5_E0B" \
    --rom "$ROM" \
    --smoke-checkpoint "$E1V5_SMOKE" \
    --smoke-training-log "$E1V5_SMOKE_TRACE" \
    --smoke-evaluation "$E1V5_SMOKE_EVALUATION" \
    --preflight-checkpoint "$E1V5_PREFLIGHT" \
    --preflight-training-log "$E1V5_PREFLIGHT_TRACE" \
    --preflight-evaluation "$E1V5_PREFLIGHT_EVALUATION" \
    --preflight-probe "$E1V5_PREFLIGHT_PROBE" \
    --preflight-behavior "$E1V5_PREFLIGHT_BEHAVIOR" \
    --formal-checkpoint "$E1V5_FORMAL" \
    --code-root "$E1V5_SOURCE_ROOT" \
    --evidence-code-revision "$E1V5_DIAGNOSTIC_REVISION" \
    --output "$E1V5_GATE"
)

e1v5_prepare_runtime

if [[ -e "$E1V5_FORMAL" || -L "$E1V5_FORMAL" ]]; then
  e1v5_die \
    "v5 formal checkpoint already exists; refusing overwrite before the separate selector/family validator"
fi
for artifact in \
    "$E1V5_FORMAL" \
    "$E1V5_FORMAL_TRACE" \
    "$E1V5_FORMAL_EVALUATION" \
    "${E1V5_FORMAL%.zip}.fixed_contexts.csv" \
    "$E1V5_FORMAL_LOG"; do
  e1v5_refuse_path "$artifact"
done
for step in $E1V5_FORMAL_STEPS; do
  e1v5_refuse_path "${E1V5_FORMAL%.zip}_step${step}.zip"
done

print "v5 diagnostics passed; starting fresh 2000800-step online W&B formal run"
e1v5_run_logged "$E1V5_FORMAL_LOG" \
  e1v5_train_command \
    "$E1V5_CODE_ROOT" 2000800 400160 100 20 "$E1V5_FORMAL" \
    --wandb \
    --wandb-project StackPOMDP \
    --wandb-group atari_clean_curriculum \
    --wandb-job-type "$E1V5_WANDB_JOB_TYPE" \
    --wandb-name "$E1V5_WANDB_NAME"
e1v5_wait_for_stable_zip "$E1V5_FORMAL"
[[ -f "$E1V5_FORMAL_TRACE" && -f "$E1V5_FORMAL_EVALUATION" ]] || \
  e1v5_die "v5 formal run lacks its trace or evaluation sidecar"

print "v5 formal training complete; selection/family validation remains a separate gated step"
release_e1v5_lock
trap - EXIT HUP INT TERM
