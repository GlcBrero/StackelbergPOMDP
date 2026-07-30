#!/bin/zsh
set -euo pipefail

# Ordered v4 seller path: execution-only smoke, learned-conditioning preflight,
# then (and only then) a fresh formal online W&B run.  Selection is separate.

source "${0:A:h}/atari_e1_seller_two_branch_v4_common.zsh"

typeset -gx STACKPOMDP_E1V4_LOCK_TOKEN="${STACKPOMDP_E1V4_LOCK_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"
stackpomdp_claim_owned_lock \
  "$E1V4_LOCK" "$STACKPOMDP_E1V4_LOCK_TOKEN" \
  "v4 two-branch seller pipeline" || exit $?
typeset -g E1V4_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"

function release_e1v4_lock() {
  local cleanup_status=0
  e2_release_transient_lock || cleanup_status=$?
  stackpomdp_release_owned_lock \
    "$E1V4_LOCK" "$STACKPOMDP_E1V4_LOCK_TOKEN" "$E1V4_LOCK_OWNED" \
    "v4 two-branch seller pipeline" || cleanup_status=$?
  typeset -g E1V4_LOCK_OWNED=0
  return "$cleanup_status"
}

function interrupt_e1v4() {
  local exit_status="$1"
  trap - EXIT HUP INT TERM
  e1v4_cancel_active_job
  release_e1v4_lock || :
  exit "$exit_status"
}

trap 'release_e1v4_lock' EXIT
trap 'interrupt_e1v4 129' HUP
trap 'interrupt_e1v4 130' INT
trap 'interrupt_e1v4 143' TERM

function e1v4_run_logged() {
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
  E1V4_ACTIVE_JOB_PID=$!
  wait "$E1V4_ACTIVE_JOB_PID"
  local job_status=$?
  E1V4_ACTIVE_JOB_PID=0
  set -e
  return "$job_status"
}

function e1v4_train_command() {
  local code_root="$1"
  local timesteps="$2"
  local checkpoint_every="$3"
  local eval_episodes="$4"
  local fixed_eval_episodes="$5"
  local checkpoint="$6"
  shift 6
  (
    cd "$code_root"
    "$E1V4_PYTHON" -u -m replication.atari.train_atari_meta_response_sb3 \
      --role seller \
      --e0b-checkpoint "$E1V4_E0B" \
      --economic-architecture "$E1V4_ARCHITECTURE" \
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

e1v4_require_scoped_clean
typeset -gr E1V4_REVISION=$(git -C "$E1V4_SOURCE_ROOT" rev-parse HEAD)
[[ ${#E1V4_REVISION} -eq 40 && "$E1V4_REVISION" != *[!0-9a-f]* ]] || \
  e1v4_die "v4 source revision is not a full lowercase SHA"
export STACKPOMDP_SPACE_INVADERS_ROM="$ROM"
export PYTHONPATH="$E1V4_SOURCE_ROOT"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$CHECKPOINT_ROOT" "$WANDB_ROOT" "$LOG_ROOT" "$E1_OUTPUT"

if [[ ! -f "$E1V4_SMOKE" || -L "$E1V4_SMOKE" ]]; then
  for artifact in \
      "$E1V4_SMOKE" \
      "${E1V4_SMOKE%.zip}_step20500.zip" \
      "$E1V4_SMOKE_TRACE" \
      "$E1V4_SMOKE_EVALUATION" \
      "${E1V4_SMOKE%.zip}.fixed_contexts.csv" \
      "$E1V4_SMOKE_LOG"; do
    e1v4_refuse_path "$artifact"
  done
  print "starting v4 no-W&B 20500-step Uniform(0,1)^5 mechanics smoke"
  e1v4_run_logged "$E1V4_SMOKE_LOG" \
    e1v4_train_command \
      "$E1V4_SOURCE_ROOT" 20500 20500 20 20 "$E1V4_SMOKE" \
      --no-wandb
fi
e1v4_wait_for_stable_zip "$E1V4_SMOKE"
(
  cd "$E1V4_SOURCE_ROOT"
  "$E1V4_PYTHON" "$E1V4_VALIDATOR" validate-smoke \
    --checkpoint "$E1V4_SMOKE" \
    --training-log "$E1V4_SMOKE_TRACE" \
    --evaluation "$E1V4_SMOKE_EVALUATION" \
    --e0b "$E1V4_E0B" \
    --code-revision "$E1V4_REVISION"
)

if [[ ! -f "$E1V4_PREFLIGHT" || -L "$E1V4_PREFLIGHT" ]]; then
  for artifact in \
      "$E1V4_PREFLIGHT" \
      "${E1V4_PREFLIGHT%.zip}_step82000.zip" \
      "$E1V4_PREFLIGHT_TRACE" \
      "$E1V4_PREFLIGHT_EVALUATION" \
      "${E1V4_PREFLIGHT%.zip}.fixed_contexts.csv" \
      "$E1V4_PREFLIGHT_PROBE" \
      "$E1V4_PREFLIGHT_BEHAVIOR" \
      "$E1V4_PREFLIGHT_LOG"; do
    e1v4_refuse_path "$artifact"
  done
  print "starting v4 no-W&B 82000-step Uniform(0,1)^5 conditioning preflight"
  e1v4_run_logged "$E1V4_PREFLIGHT_LOG" \
    e1v4_train_command \
      "$E1V4_SOURCE_ROOT" 82000 82000 20 20 "$E1V4_PREFLIGHT" \
      --no-wandb
fi
e1v4_wait_for_stable_zip "$E1V4_PREFLIGHT"
if [[ ! -e "$E1V4_PREFLIGHT_PROBE" && ! -L "$E1V4_PREFLIGHT_PROBE" ]]; then
  (
    cd "$E1V4_SOURCE_ROOT"
    "$E1V4_PYTHON" -u -m \
      replication.atari.probe_atari_e1_seller_two_branch \
      --checkpoint "$E1V4_PREFLIGHT" \
      --e0b-checkpoint "$E1V4_E0B" \
      --device cpu \
      --output "$E1V4_PREFLIGHT_PROBE" \
      --require-pass
  )
fi
if [[ ! -e "$E1V4_PREFLIGHT_BEHAVIOR" \
    && ! -L "$E1V4_PREFLIGHT_BEHAVIOR" ]]; then
  (
    cd "$E1V4_SOURCE_ROOT"
    "$E1V4_PYTHON" -u "$E1V4_VALIDATOR" behavioral-preflight \
      --checkpoint "$E1V4_PREFLIGHT" \
      --e0b "$E1V4_E0B" \
      --rom "$ROM" \
      --output "$E1V4_PREFLIGHT_BEHAVIOR" \
      --code-revision "$E1V4_REVISION"
  )
fi
(
  cd "$E1V4_SOURCE_ROOT"
  "$E1V4_PYTHON" "$E1V4_VALIDATOR" validate-preflight \
    --checkpoint "$E1V4_PREFLIGHT" \
    --training-log "$E1V4_PREFLIGHT_TRACE" \
    --evaluation "$E1V4_PREFLIGHT_EVALUATION" \
    --probe "$E1V4_PREFLIGHT_PROBE" \
    --behavior-report "$E1V4_PREFLIGHT_BEHAVIOR" \
    --e0b "$E1V4_E0B" \
    --rom "$ROM" \
    --code-revision "$E1V4_REVISION"
)

if [[ ! -f "$E1V4_GATE" || -L "$E1V4_GATE" ]]; then
  for artifact in \
      "$E1V4_GATE" \
      "$E1V4_FORMAL" \
      "$E1V4_FORMAL_TRACE" \
      "$E1V4_FORMAL_EVALUATION" \
      "${E1V4_FORMAL%.zip}.fixed_contexts.csv" \
      "$E1V4_FORMAL_LOG"; do
    e1v4_refuse_path "$artifact"
  done
  for step in $E1V4_FORMAL_STEPS; do
    e1v4_refuse_path "${E1V4_FORMAL%.zip}_step${step}.zip"
  done
fi
(
  cd "$E1V4_SOURCE_ROOT"
  "$E1V4_PYTHON" "$E1V4_VALIDATOR" gate \
    --seller-release "$E1V4_SELLER_RELEASE" \
    --e0b "$E1V4_E0B" \
    --rom "$ROM" \
    --smoke-checkpoint "$E1V4_SMOKE" \
    --smoke-training-log "$E1V4_SMOKE_TRACE" \
    --smoke-evaluation "$E1V4_SMOKE_EVALUATION" \
    --preflight-checkpoint "$E1V4_PREFLIGHT" \
    --preflight-training-log "$E1V4_PREFLIGHT_TRACE" \
    --preflight-evaluation "$E1V4_PREFLIGHT_EVALUATION" \
    --preflight-probe "$E1V4_PREFLIGHT_PROBE" \
    --preflight-behavior "$E1V4_PREFLIGHT_BEHAVIOR" \
    --formal-checkpoint "$E1V4_FORMAL" \
    --code-root "$E1V4_SOURCE_ROOT" \
    --output "$E1V4_GATE"
)

e1v4_prepare_runtime

if [[ -e "$E1V4_FORMAL" || -L "$E1V4_FORMAL" ]]; then
  e1v4_die \
    "v4 formal checkpoint already exists; refusing overwrite before the separate selector/family validator"
fi
for artifact in \
    "$E1V4_FORMAL" \
    "$E1V4_FORMAL_TRACE" \
    "$E1V4_FORMAL_EVALUATION" \
    "${E1V4_FORMAL%.zip}.fixed_contexts.csv" \
    "$E1V4_FORMAL_LOG"; do
  e1v4_refuse_path "$artifact"
done
for step in $E1V4_FORMAL_STEPS; do
  e1v4_refuse_path "${E1V4_FORMAL%.zip}_step${step}.zip"
done

print "v4 diagnostics passed; starting fresh 2000800-step online W&B formal run"
e1v4_run_logged "$E1V4_FORMAL_LOG" \
  e1v4_train_command \
    "$E1V4_CODE_ROOT" 2000800 400160 100 20 "$E1V4_FORMAL" \
    --wandb \
    --wandb-project StackPOMDP \
    --wandb-group atari_clean_curriculum \
    --wandb-job-type "$E1V4_WANDB_JOB_TYPE" \
    --wandb-name "$E1V4_WANDB_NAME"
e1v4_wait_for_stable_zip "$E1V4_FORMAL"
[[ -f "$E1V4_FORMAL_TRACE" && -f "$E1V4_FORMAL_EVALUATION" ]] || \
  e1v4_die "v4 formal run lacks its trace or evaluation sidecar"

print "v4 formal training complete; selection/family validation remains a separate gated step"
release_e1v4_lock
trap - EXIT HUP INT TERM
