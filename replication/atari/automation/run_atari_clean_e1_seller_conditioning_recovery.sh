#!/bin/zsh
set -euo pipefail

# Two-stage local E1 seller recovery.  The all-equal warm-up must pass its
# immutable no-ALE conditioning probe before the independent-uniform target
# stage can start.  Only the target-stage family reaches the selector.

source "${0:A:h}/atari_e1_seller_conditioning_recovery_common.zsh"

typeset -gx STACKPOMDP_E1R_LOCK_TOKEN="${STACKPOMDP_E1R_LOCK_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"
stackpomdp_claim_owned_lock \
  "$E1R_LOCK" "$STACKPOMDP_E1R_LOCK_TOKEN" \
  "E1 seller conditioning recovery" || exit $?
typeset -g E1R_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"
function release_e1r_lock() {
  e2_release_transient_lock || return $?
  stackpomdp_release_owned_lock \
    "$E1R_LOCK" "$STACKPOMDP_E1R_LOCK_TOKEN" \
    "$E1R_LOCK_OWNED" "E1 seller conditioning recovery" || return $?
  typeset -g E1R_LOCK_OWNED=0
}
function interrupt_e1r() {
  local exit_status="$1"
  trap - EXIT HUP INT TERM
  e1r_cancel_active_job
  release_e1r_lock || :
  exit "$exit_status"
}
trap 'release_e1r_lock' EXIT
trap 'interrupt_e1r 129' HUP
trap 'interrupt_e1r 130' INT
trap 'interrupt_e1r 143' TERM

function run_warmup_conditioning_probe() {
  print "running predeclared no-ALE warm-up conditioning probe"
  (
    cd "$E1R_CODE_ROOT"
    "$E1R_PYTHON" -u -m \
      replication.atari.probe_atari_e1_seller_conditioning \
      --checkpoint "$E1R_WARMUP_BASE" \
      --e0b-checkpoint "$E1R_E0B" \
      --device cpu \
      --output "$E1R_WARMUP_PROBE" \
      --require-pass
  )
}

e1r_activate
e1r_prepare_runtime

if [[ -f "$E1R_FAMILY" && ! -L "$E1R_FAMILY" ]]; then
  e1r_validate_existing_family
  release_e1r_lock
  trap - EXIT HUP INT TERM
  exec "${0:A:h}/run_e1_seller_conditioning_recovery_selector.sh"
fi

if [[ -f "$E1R_WARMUP_BASE" && ! -L "$E1R_WARMUP_BASE" ]]; then
  e1r_wait_for_stable_zip "$E1R_WARMUP_BASE"
  [[ -f "${E1R_WARMUP_BASE%.zip}.evaluation.json" \
      && -f "${E1R_WARMUP_BASE%.zip}.training.jsonl" ]] || \
    e1r_die "existing warm-up lacks its immutable evaluation or training trace; eval-only recovery would alter resume provenance, so this family remains fail-closed"
  e1r_validate_warmup_checkpoint
  if [[ ! -e "$E1R_WARMUP_PROBE" && ! -L "$E1R_WARMUP_PROBE" ]]; then
    # The probe is a deterministic, read-only inference artifact.  It is safe
    # to resume this narrow post-training publication window.
    run_warmup_conditioning_probe
  fi
  e1r_validate_warmup_probe
else
  for artifact in \
      "$E1R_WARMUP_BASE" \
      "${E1R_WARMUP_BASE%.zip}_step400160.zip" \
      "${E1R_WARMUP_BASE%.zip}.evaluation.json" \
      "${E1R_WARMUP_BASE%.zip}.fixed_contexts.csv" \
      "${E1R_WARMUP_BASE%.zip}.training.jsonl" \
      "$E1R_WARMUP_PROBE" \
      "$E1R_WARMUP_LOG"; do
    e1r_refuse_path "$artifact"
  done
  print "starting 400160-step all-equal-v1 seller warm-up"
  set +e
  (
    set +e
    (
      cd "$E1R_CODE_ROOT"
      "$E1R_PYTHON" -u -m replication.atari.train_atari_meta_response_sb3 \
        --role seller \
        --e0b-checkpoint "$E1R_E0B" \
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
        --learning-rate 0.0001 \
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
        --fixed-eval-episodes 5 \
        --checkpoint "$E1R_WARMUP_BASE" \
        --wandb \
        --wandb-project StackPOMDP \
        --wandb-group atari_clean_curriculum \
        --wandb-name "$E1R_WARMUP_RUN"
    ) 2>&1 | tee "$E1R_WARMUP_LOG"
    pipeline_status=(${pipestatus[@]})
    (( pipeline_status[2] == 0 )) || exit "${pipeline_status[2]}"
    exit "${pipeline_status[1]}"
  ) &
  E1R_ACTIVE_JOB_PID=$!
  wait "$E1R_ACTIVE_JOB_PID"
  warmup_status=$?
  E1R_ACTIVE_JOB_PID=0
  set -e
  (( warmup_status == 0 )) || {
    print -u2 "seller all-equal warm-up failed with exit status $warmup_status"
    exit "$warmup_status"
  }
  e1r_wait_for_stable_zip "$E1R_WARMUP_BASE"
  run_warmup_conditioning_probe
  # This independently reloads the ZIP and requires a fresh one-stage
  # all-equal lineage, the exact 400160 boundary, and the probe's SHA binding.
  e1r_validate_warmup_probe
fi

# A failed/malformed probe exits above.  It is impossible to reach target
# training merely because the warm-up checkpoint exists.
if [[ ! -f "$E1R_TARGET_BASE" ]]; then
  for artifact in \
      "$E1R_TARGET_BASE" \
      "${E1R_TARGET_BASE%.zip}.evaluation.json" \
      "${E1R_TARGET_BASE%.zip}.fixed_contexts.csv" \
      "${E1R_TARGET_BASE%.zip}.training.jsonl" \
      "$E1R_TARGET_LOG" \
      "$E1R_FAMILY"; do
    e1r_refuse_path "$artifact"
  done
  for step in $E1R_TARGET_STEPS; do
    e1r_refuse_path "${E1R_TARGET_BASE%.zip}_step${step}.zip"
  done
  print "starting independent-uniform target stage from full warm-up state"
  set +e
  (
    set +e
    (
      cd "$E1R_CODE_ROOT"
      "$E1R_PYTHON" -u -m replication.atari.train_atari_meta_response_sb3 \
        --role seller \
        --e0b-checkpoint "$E1R_E0B" \
        --resume "$E1R_WARMUP_BASE" \
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
        --learning-rate 0.0001 \
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
        --checkpoint "$E1R_TARGET_BASE" \
        --wandb \
        --wandb-project StackPOMDP \
        --wandb-group atari_clean_curriculum \
        --wandb-name "$E1R_TARGET_RUN"
    ) 2>&1 | tee "$E1R_TARGET_LOG"
    pipeline_status=(${pipestatus[@]})
    (( pipeline_status[2] == 0 )) || exit "${pipeline_status[2]}"
    exit "${pipeline_status[1]}"
  ) &
  E1R_ACTIVE_JOB_PID=$!
  wait "$E1R_ACTIVE_JOB_PID"
  target_status=$?
  E1R_ACTIVE_JOB_PID=0
  set -e
  (( target_status == 0 )) || {
    print -u2 "seller independent-uniform target failed with exit status $target_status"
    exit "$target_status"
  }
else
  print "found an existing target endpoint; validating the complete family"
fi

e1r_validate_family
e1r_validate_existing_family
release_e1r_lock
trap - EXIT HUP INT TERM
exec "${0:A:h}/run_e1_seller_conditioning_recovery_selector.sh"
