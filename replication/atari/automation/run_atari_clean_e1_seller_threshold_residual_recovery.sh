#!/bin/zsh
set -euo pipefail

# Separately versioned residual seller recovery.  Its activation recomputes the
# exact immutable v1 warm-up failure.  The unchanged all-equal gate must pass
# before independent-uniform training or selection can begin.

source "${0:A:h}/atari_e1_seller_threshold_residual_recovery_common.zsh"

typeset -gx STACKPOMDP_E1R2_GUARD_TOKEN="${STACKPOMDP_E1R2_GUARD_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"
stackpomdp_claim_owned_lock "$E1R2_V1_GUARD_LOCK" \
  "$STACKPOMDP_E1R2_GUARD_TOKEN" "superseded v1 seller recovery guard" || exit $?
typeset -g E1R2_GUARD_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"

typeset -gx STACKPOMDP_E1R2_LOCK_TOKEN="${STACKPOMDP_E1R2_LOCK_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"
stackpomdp_claim_owned_lock "$E1R2_LOCK" \
    "$STACKPOMDP_E1R2_LOCK_TOKEN" "$E1R2_PROFILE_LABEL recovery" || {
  lock_status=$?
  stackpomdp_release_owned_lock "$E1R2_V1_GUARD_LOCK" \
    "$STACKPOMDP_E1R2_GUARD_TOKEN" "$E1R2_GUARD_OWNED" \
    "superseded v1 seller recovery guard" || :
  exit "$lock_status"
}
typeset -g E1R2_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"

function release_e1r2_locks() {
  local cleanup_status=0
  e2_release_transient_lock || cleanup_status=$?
  stackpomdp_release_owned_lock "$E1R2_LOCK" \
    "$STACKPOMDP_E1R2_LOCK_TOKEN" "$E1R2_LOCK_OWNED" \
    "$E1R2_PROFILE_LABEL recovery" || cleanup_status=$?
  typeset -g E1R2_LOCK_OWNED=0
  stackpomdp_release_owned_lock "$E1R2_V1_GUARD_LOCK" \
    "$STACKPOMDP_E1R2_GUARD_TOKEN" "$E1R2_GUARD_OWNED" \
    "superseded v1 seller recovery guard" || cleanup_status=$?
  typeset -g E1R2_GUARD_OWNED=0
  return "$cleanup_status"
}

function interrupt_e1r2() {
  local exit_status="$1"
  trap - EXIT HUP INT TERM
  e1r2_cancel_active_job
  release_e1r2_locks || :
  exit "$exit_status"
}

trap 'release_e1r2_locks' EXIT
trap 'interrupt_e1r2 129' HUP
trap 'interrupt_e1r2 130' INT
trap 'interrupt_e1r2 143' TERM

function run_v2_warmup_probe() {
  print "running predeclared no-ALE $E1R2_PROFILE_LABEL conditioning probe"
  (
    cd "$E1R2_CODE_ROOT"
    "$E1R2_PYTHON" -u -m \
      "$E1R2_PROBE_MODULE" \
      --checkpoint "$E1R2_WARMUP_BASE" \
      --e0b-checkpoint "$E1R2_E0B" \
      --device cpu \
      --output "$E1R2_WARMUP_PROBE" \
      --require-pass
  )
}

function run_v2_pure64_preflight() {
  e1r2_require_scoped_clean
  export STACKPOMDP_SPACE_INVADERS_ROM="$ROM"
  export PYTHONPATH="$E1R2_SOURCE_ROOT"
  export PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT" "$E1_OUTPUT"
  if [[ ! -f "$E1R2_PREFLIGHT_BASE" || -L "$E1R2_PREFLIGHT_BASE" ]]; then
    for artifact in \
        "$E1R2_PREFLIGHT_BASE" \
        "${E1R2_PREFLIGHT_BASE%.zip}_step20500.zip" \
        "$E1R2_PREFLIGHT_EVALUATION" \
        "${E1R2_PREFLIGHT_BASE%.zip}.fixed_contexts.csv" \
        "${E1R2_PREFLIGHT_BASE%.zip}.training.jsonl" \
        "$E1R2_PREFLIGHT_PROBE" \
        "$E1R2_PREFLIGHT_LOG"; do
      e1r2_refuse_path "$artifact"
    done
    print "starting no-W&B 20500-step $E1R2_PROFILE_LABEL real-ALE preflight"
    set +e
    (
      set +e
      (
        cd "$E1R2_SOURCE_ROOT"
        "$E1R2_PYTHON" -u -m replication.atari.train_atari_meta_response_sb3 \
          --role seller \
          --e0b-checkpoint "$E1R2_E0B" \
          "${E1R2_ARCHITECTURE_FLAGS[@]}" \
          --e1-sampler-mode all-equal-v1 \
          --actor-loss-mode balanced \
          --seed 1 \
          --timesteps 20500 \
          --gameplay-horizon 200 \
          --event-tail-steps 0 \
          --num-envs 4 \
          --start-method spawn \
          --n-steps 205 \
          --batch-size 820 \
          --n-epochs 4 \
          --learning-rate "$E1R2_LEARNING_RATE" \
          --pretrained-lr-scale 0.1 \
          --entropy-coeff 0.01 \
          --clip-range 0.1 \
          --value-coefficient 0.5 \
          --max-grad-norm 0.5 \
          --noop-max 30 \
          --max-frames 100000 \
          --rom-path "$ROM" \
          --checkpoint-every 20500 \
          --eval-episodes 20 \
          --fixed-eval-episodes 20 \
          --checkpoint "$E1R2_PREFLIGHT_BASE" \
          --no-wandb
      ) 2>&1 | tee "$E1R2_PREFLIGHT_LOG"
      pipeline_status=(${pipestatus[@]})
      (( pipeline_status[2] == 0 )) || exit "${pipeline_status[2]}"
      exit "${pipeline_status[1]}"
    ) &
    E1R2_ACTIVE_JOB_PID=$!
    wait "$E1R2_ACTIVE_JOB_PID"
    preflight_status=$?
    E1R2_ACTIVE_JOB_PID=0
    set -e
    (( preflight_status == 0 )) || exit "$preflight_status"
  fi
  e1r2_wait_for_stable_zip "$E1R2_PREFLIGHT_BASE"
  [[ -f "$E1R2_PREFLIGHT_EVALUATION" ]] || \
    e1r2_die "$E1R2_PROFILE_LABEL preflight lacks real-ALE evaluation"
  if [[ ! -e "$E1R2_PREFLIGHT_PROBE" && ! -L "$E1R2_PREFLIGHT_PROBE" ]]; then
    (
      cd "$E1R2_SOURCE_ROOT"
      "$E1R2_PYTHON" -u -m \
        "$E1R2_PROBE_MODULE" \
        --checkpoint "$E1R2_PREFLIGHT_BASE" \
        --e0b-checkpoint "$E1R2_E0B" \
        --device cpu \
        --output "$E1R2_PREFLIGHT_PROBE" \
        --require-pass
    )
  fi
}

run_v2_pure64_preflight
e1r2_activate
e1r2_prepare_runtime

if [[ -f "$E1R2_FAMILY" && ! -L "$E1R2_FAMILY" ]]; then
  e1r2_validate_existing_family
  release_e1r2_locks
  trap - EXIT HUP INT TERM
  exec "${0:A:h}/$E1R2_SELECTOR_SCRIPT"
fi

if [[ -f "$E1R2_WARMUP_BASE" && ! -L "$E1R2_WARMUP_BASE" ]]; then
  e1r2_wait_for_stable_zip "$E1R2_WARMUP_BASE"
  [[ -f "${E1R2_WARMUP_BASE%.zip}.evaluation.json" \
      && -f "${E1R2_WARMUP_BASE%.zip}.training.jsonl" ]] || \
    e1r2_die "existing v2 warm-up lacks immutable evaluation or trace"
  if [[ ! -e "$E1R2_WARMUP_PROBE" && ! -L "$E1R2_WARMUP_PROBE" ]]; then
    run_v2_warmup_probe
  fi
  # This command exits 2 for a fully validated scientific gate failure.
  e1r2_validate_warmup_probe
  e1r2_validate_warmup_stage
else
  for artifact in \
      "$E1R2_WARMUP_BASE" \
      "${E1R2_WARMUP_BASE%.zip}_step400160.zip" \
      "${E1R2_WARMUP_BASE%.zip}.evaluation.json" \
      "${E1R2_WARMUP_BASE%.zip}.fixed_contexts.csv" \
      "${E1R2_WARMUP_BASE%.zip}.training.jsonl" \
      "$E1R2_WARMUP_PROBE" \
      "$E1R2_WARMUP_LOG"; do
    e1r2_refuse_path "$artifact"
  done
  print "starting 400160-step all-equal-v1 v2 seller warm-up"
  set +e
  (
    set +e
    (
      cd "$E1R2_CODE_ROOT"
      "$E1R2_PYTHON" -u -m replication.atari.train_atari_meta_response_sb3 \
        --role seller \
        --e0b-checkpoint "$E1R2_E0B" \
        "${E1R2_ARCHITECTURE_FLAGS[@]}" \
        --e1-sampler-mode all-equal-v1 \
        --actor-loss-mode balanced \
        --seed 1 \
        --timesteps 400160 \
        --gameplay-horizon 200 \
        --event-tail-steps 0 \
        --num-envs 4 \
        --start-method spawn \
        --n-steps 205 \
        --batch-size 820 \
        --n-epochs 4 \
        --learning-rate "$E1R2_LEARNING_RATE" \
        --pretrained-lr-scale 0.1 \
        --entropy-coeff 0.01 \
        --clip-range 0.1 \
        --value-coefficient 0.5 \
        --max-grad-norm 0.5 \
        --noop-max 30 \
        --max-frames 100000 \
        --rom-path "$ROM" \
        --checkpoint-every 400160 \
        --eval-episodes 20 \
        --fixed-eval-episodes 20 \
        --checkpoint "$E1R2_WARMUP_BASE" \
        --wandb \
        --wandb-project StackPOMDP \
        --wandb-group atari_clean_curriculum \
        --wandb-job-type "$E1R2_WARMUP_JOB_TYPE" \
        --wandb-name "$E1R2_WARMUP_RUN"
    ) 2>&1 | tee "$E1R2_WARMUP_LOG"
    pipeline_status=(${pipestatus[@]})
    (( pipeline_status[2] == 0 )) || exit "${pipeline_status[2]}"
    exit "${pipeline_status[1]}"
  ) &
  E1R2_ACTIVE_JOB_PID=$!
  wait "$E1R2_ACTIVE_JOB_PID"
  warmup_status=$?
  E1R2_ACTIVE_JOB_PID=0
  set -e
  (( warmup_status == 0 )) || exit "$warmup_status"
  e1r2_wait_for_stable_zip "$E1R2_WARMUP_BASE"
  run_v2_warmup_probe
  e1r2_validate_warmup_stage
fi

# No target command is reachable unless the exact warm-up ZIP, trace,
# evaluation, architecture, and freshly recomputed no-ALE gate all pass.
if [[ ! -f "$E1R2_TARGET_BASE" ]]; then
  for artifact in \
      "$E1R2_TARGET_BASE" \
      "${E1R2_TARGET_BASE%.zip}.evaluation.json" \
      "${E1R2_TARGET_BASE%.zip}.fixed_contexts.csv" \
      "${E1R2_TARGET_BASE%.zip}.training.jsonl" \
      "$E1R2_TARGET_LOG" \
      "$E1R2_FAMILY"; do
    e1r2_refuse_path "$artifact"
  done
  for step in $E1R2_TARGET_STEPS; do
    e1r2_refuse_path "${E1R2_TARGET_BASE%.zip}_step${step}.zip"
  done
  print "starting v2 independent-uniform target from full warm-up state"
  set +e
  (
    set +e
    (
      cd "$E1R2_CODE_ROOT"
      "$E1R2_PYTHON" -u -m replication.atari.train_atari_meta_response_sb3 \
        --role seller \
        --e0b-checkpoint "$E1R2_E0B" \
        --resume "$E1R2_WARMUP_BASE" \
        "${E1R2_ARCHITECTURE_FLAGS[@]}" \
        --e1-sampler-mode uniform \
        --actor-loss-mode balanced \
        --seed 1 \
        --timesteps 2000800 \
        --gameplay-horizon 200 \
        --event-tail-steps 0 \
        --num-envs 4 \
        --start-method spawn \
        --n-steps 205 \
        --batch-size 820 \
        --n-epochs 4 \
        --learning-rate "$E1R2_LEARNING_RATE" \
        --pretrained-lr-scale 0.1 \
        --entropy-coeff 0.01 \
        --clip-range 0.1 \
        --value-coefficient 0.5 \
        --max-grad-norm 0.5 \
        --noop-max 30 \
        --max-frames 100000 \
        --rom-path "$ROM" \
        --checkpoint-every 400160 \
        --eval-episodes 100 \
        --fixed-eval-episodes 20 \
        --checkpoint "$E1R2_TARGET_BASE" \
        --wandb \
        --wandb-project StackPOMDP \
        --wandb-group atari_clean_curriculum \
        --wandb-job-type "$E1R2_TARGET_JOB_TYPE" \
        --wandb-name "$E1R2_TARGET_RUN"
    ) 2>&1 | tee "$E1R2_TARGET_LOG"
    pipeline_status=(${pipestatus[@]})
    (( pipeline_status[2] == 0 )) || exit "${pipeline_status[2]}"
    exit "${pipeline_status[1]}"
  ) &
  E1R2_ACTIVE_JOB_PID=$!
  wait "$E1R2_ACTIVE_JOB_PID"
  target_status=$?
  E1R2_ACTIVE_JOB_PID=0
  set -e
  (( target_status == 0 )) || exit "$target_status"
fi

e1r2_validate_family
e1r2_validate_existing_family
release_e1r2_locks
trap - EXIT HUP INT TERM
exec "${0:A:h}/$E1R2_SELECTOR_SCRIPT"
