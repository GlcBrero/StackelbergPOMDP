#!/bin/zsh

# Shared fail-closed paths/helpers for the separately versioned v2 seller
# threshold-residual recovery.  Sourcing this file never starts work.

source "${${(%):-%N}:A:h}/atari_e2_pipeline_common.zsh"

typeset -gr E1R2_SOURCE_ROOT="$AUTOMATION_ROOT"
typeset -gr E1R2_ACTIVE_ROOT="$ROOT"
typeset -gr E1R2_PYTHON="$PYTHON"
typeset -gr E1R2_E0B="$CHECKPOINT_ROOT/space_invaders_e0b_ppo_seed1_firefix_retrain_selected.zip"
typeset -gr E1R2_RELEASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_seed1_firefix_retrain.buyer_gate.json"

typeset -gr E1R2_V1_ACTIVATION="$E1_OUTPUT/e1_seller_conditioning_recovery_activation_v1.json"
typeset -gr E1R2_V1_WARMUP="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_conditioning_recovery_v1_seed1_all_equal_warmup.zip"
typeset -gr E1R2_V1_PROBE="$E1_OUTPUT/e1_seller_conditioning_recovery_warmup_probe_v1.json"
typeset -gr E1R2_V1_TRACE="${E1R2_V1_WARMUP%.zip}.training.jsonl"
typeset -gr E1R2_V1_EVALUATION="${E1R2_V1_WARMUP%.zip}.evaluation.json"

case "${STACKPOMDP_E1_SELLER_RESIDUAL_PROFILE:-v2}" in
v3-direct)
  typeset -gr E1R2_TOKEN=conditioning_recovery_v3_direct_threshold_residual_v1
  typeset -gr E1R2_PROFILE_LABEL="v3 direct-threshold residual"
  typeset -gr E1R2_PROBE_MODULE=replication.atari.probe_atari_e1_seller_direct_threshold_residual
  typeset -gra E1R2_ARCHITECTURE_FLAGS=(
    --economic-threshold-residual
    --economic-threshold-residual-direct-input
  )
  typeset -gr E1R2_PREFLIGHT_SUFFIX=direct65_preflight
  typeset -gr E1R2_WARMUP_RUN=atari_clean_e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_all_equal_warmup_seed1_400160_local
  typeset -gr E1R2_TARGET_RUN=atari_clean_e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_uniform_target_seed1_2000800_local
  typeset -gr E1R2_WARMUP_JOB_TYPE=atari_e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_warmup
  typeset -gr E1R2_TARGET_JOB_TYPE=atari_e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_target
  typeset -gr E1R2_SELECTOR_NAME=e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_all6_selector_v3
  typeset -gr E1R2_LOCK=/private/tmp/stackpomdp-atari-e1-seller-conditioning-recovery-v3-direct-threshold-residual-v1.lock
  typeset -gr E1R2_VALIDATOR_RELATIVE=replication/atari/automation/validate_atari_e1_seller_direct_threshold_residual_recovery.py
  typeset -gr E1R2_SELECTOR_SCRIPT=run_e1_seller_direct_threshold_residual_recovery_selector.sh
  ;;
v2)
  typeset -gr E1R2_TOKEN=conditioning_recovery_v2_threshold_residual_v1
  typeset -gr E1R2_PROFILE_LABEL="v2 threshold-residual"
  typeset -gr E1R2_PROBE_MODULE=replication.atari.probe_atari_e1_seller_threshold_residual
  typeset -gra E1R2_ARCHITECTURE_FLAGS=(--economic-threshold-residual)
  typeset -gr E1R2_PREFLIGHT_SUFFIX=pure64_preflight
  typeset -gr E1R2_WARMUP_RUN=atari_clean_e1_seller_conditioning_recovery_v2_threshold_residual_v1_all_equal_warmup_seed1_400160_local
  typeset -gr E1R2_TARGET_RUN=atari_clean_e1_seller_conditioning_recovery_v2_threshold_residual_v1_uniform_target_seed1_2000800_local
  typeset -gr E1R2_WARMUP_JOB_TYPE=atari_e1_seller_conditioning_recovery_v2_threshold_residual_v1_warmup
  typeset -gr E1R2_TARGET_JOB_TYPE=atari_e1_seller_conditioning_recovery_v2_threshold_residual_v1_target
  typeset -gr E1R2_SELECTOR_NAME=e1_seller_conditioning_recovery_v2_threshold_residual_v1_all6_selector_v2
  typeset -gr E1R2_LOCK=/private/tmp/stackpomdp-atari-e1-seller-conditioning-recovery-v2-threshold-residual-v1.lock
  typeset -gr E1R2_VALIDATOR_RELATIVE=replication/atari/automation/validate_atari_e1_seller_threshold_residual_recovery.py
  typeset -gr E1R2_SELECTOR_SCRIPT=run_e1_seller_threshold_residual_recovery_selector.sh
  ;;
*)
  print -u2 -- "invalid seller residual profile: ${STACKPOMDP_E1_SELLER_RESIDUAL_PROFILE}"
  return 1
  ;;
esac
typeset -gr E1R2_ACTIVATION="$E1_OUTPUT/e1_seller_${E1R2_TOKEN}_activation.json"
typeset -gr E1R2_FAMILY="$E1_OUTPUT/e1_seller_${E1R2_TOKEN}_family.json"
typeset -gr E1R2_WARMUP_BASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1R2_TOKEN}_seed1_all_equal_warmup.zip"
typeset -gr E1R2_WARMUP_PROBE="$E1_OUTPUT/e1_seller_${E1R2_TOKEN}_warmup_probe.json"
typeset -gr E1R2_TARGET_BASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1R2_TOKEN}_seed1_uniform_target.zip"
typeset -gr E1R2_PREFLIGHT_BASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1R2_TOKEN}_seed1_${E1R2_PREFLIGHT_SUFFIX}.zip"
typeset -gr E1R2_PREFLIGHT_PROBE="$E1_OUTPUT/e1_seller_${E1R2_TOKEN}_${E1R2_PREFLIGHT_SUFFIX}_probe.json"
typeset -gr E1R2_PREFLIGHT_EVALUATION="${E1R2_PREFLIGHT_BASE%.zip}.evaluation.json"
typeset -gr E1R2_PREFLIGHT_LOG="$LOG_ROOT/atari_clean_e1_seller_${E1R2_TOKEN}_${E1R2_PREFLIGHT_SUFFIX}_seed1_20500_local.log"
typeset -gr E1R2_WARMUP_LOG="$LOG_ROOT/${E1R2_WARMUP_RUN}.log"
typeset -gr E1R2_TARGET_LOG="$LOG_ROOT/${E1R2_TARGET_RUN}.log"
typeset -gr E1R2_REPORT="$E1_OUTPUT/${E1R2_SELECTOR_NAME}.json"
typeset -gr E1R2_GATE="$E1_OUTPUT/${E1R2_SELECTOR_NAME}.gate.json"
typeset -gr E1R2_SELECTED="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1R2_TOKEN}_seed1_selected.zip"
typeset -gr E1R2_SELECTOR_LOG="$LOG_ROOT/${E1R2_SELECTOR_NAME}.log"
typeset -gr E1R2_V1_GUARD_LOCK=/private/tmp/stackpomdp-atari-e1-seller-conditioning-recovery.lock
typeset -gra E1R2_TARGET_STEPS=(800320 1200480 1600640 2000800 2400960)
typeset -gr E1R2_LEARNING_RATE=0.0001

typeset -g E1R2_CODE_ROOT="$E1R2_SOURCE_ROOT"
typeset -g E1R2_VALIDATOR="$E1R2_SOURCE_ROOT/$E1R2_VALIDATOR_RELATIVE"
typeset -gi E1R2_ACTIVE_JOB_PID=0

function e1r2_die() {
  print -u2 -- "$*"
  return 1
}

function e1r2_terminate_process_tree() {
  local parent_pid="$1"
  local signal_name="${2:-TERM}"
  local children child
  [[ "$parent_pid" == <-> ]] || return 0
  children=$(pgrep -P "$parent_pid" 2>/dev/null) || children=""
  for child in ${(f)children}; do
    [[ "$child" == <-> ]] || continue
    e1r2_terminate_process_tree "$child" "$signal_name"
  done
  kill -s "$signal_name" "$parent_pid" 2>/dev/null || :
}

function e1r2_cancel_active_job() {
  local active_pid="$E1R2_ACTIVE_JOB_PID"
  (( active_pid > 0 )) || return 0
  e1r2_terminate_process_tree "$active_pid" TERM
  wait "$active_pid" 2>/dev/null || :
  typeset -g E1R2_ACTIVE_JOB_PID=0
}

function e1r2_require_scoped_clean() {
  local dirty
  dirty=$(git -C "$E1R2_SOURCE_ROOT" status --porcelain -- \
    replication/atari/train_atari_meta_response_sb3.py \
    replication/atari/evaluate_atari_meta_response_sb3.py \
    replication/atari/probe_atari_e1_seller_conditioning.py \
    replication/atari/probe_atari_e1_seller_threshold_residual.py \
    replication/atari/sb3_common.py \
    replication/atari/automation \
    stackelberg_pomdp/atari)
  [[ -z "$dirty" ]] || e1r2_die "refusing uncommitted v2 seller-recovery code:\n$dirty"
}

function e1r2_refuse_path() {
  [[ ! -e "$1" && ! -L "$1" ]] || \
    e1r2_die "refusing to overwrite v2 seller-recovery artifact: $1"
}

function e1r2_activate() {
  if [[ -f "$E1R2_ACTIVATION" && ! -L "$E1R2_ACTIVATION" ]]; then
    (
      cd "$E1R2_SOURCE_ROOT"
      PYTHONPATH="$E1R2_SOURCE_ROOT" PYTHONNOUSERSITE=1 \
        "$E1R2_PYTHON" "$E1R2_VALIDATOR" validate-activation \
          --activation "$E1R2_ACTIVATION"
    ) || return $?
    [[ "$(jq -r '.protocol.training_config.learning_rate' "$E1R2_ACTIVATION")" == "$E1R2_LEARNING_RATE" ]] || \
      e1r2_die "v2 activation learning rate is not the fixed 0.0001"
    return 0
  fi
  e1r2_require_scoped_clean
  (
    cd "$E1R2_SOURCE_ROOT"
    PYTHONPATH="$E1R2_SOURCE_ROOT" PYTHONNOUSERSITE=1 \
      "$E1R2_PYTHON" "$E1R2_VALIDATOR" activation \
        --v1-activation "$E1R2_V1_ACTIVATION" \
        --v1-warmup-checkpoint "$E1R2_V1_WARMUP" \
        --v1-warmup-probe "$E1R2_V1_PROBE" \
        --v1-warmup-training-log "$E1R2_V1_TRACE" \
        --v1-warmup-evaluation "$E1R2_V1_EVALUATION" \
        --seller-release "$E1R2_RELEASE" \
        --e0b "$E1R2_E0B" \
        --rom "$ROM" \
        --warmup-checkpoint "$E1R2_WARMUP_BASE" \
        --warmup-probe "$E1R2_WARMUP_PROBE" \
        --target-checkpoint "$E1R2_TARGET_BASE" \
        --preflight-checkpoint "$E1R2_PREFLIGHT_BASE" \
        --preflight-probe "$E1R2_PREFLIGHT_PROBE" \
        --preflight-evaluation "$E1R2_PREFLIGHT_EVALUATION" \
        --code-root "$E1R2_SOURCE_ROOT" \
        --output "$E1R2_ACTIVATION"
  )
}

function e1r2_prepare_runtime() {
  local revision runtime head dirty init_token
  revision=$(jq -r '.code_revision' "$E1R2_ACTIVATION") || return $?
  [[ ${#revision} -eq 40 && "$revision" != *[!0-9a-f]* ]] || \
    e1r2_die "invalid v2 seller-recovery activation revision"
  runtime="/private/tmp/stackpomdp-e1-seller-threshold-residual-code-${revision[1,12]}"
  if [[ -d "$runtime/.git" || -f "$runtime/.git" ]]; then
    head=$(git -C "$runtime" rev-parse HEAD) || return $?
    [[ "$head" == "$revision" ]] || e1r2_die "existing v2 runtime has another revision: $runtime"
  elif [[ -e "$runtime" ]]; then
    e1r2_die "v2 runtime path is not a git worktree: $runtime"
  else
    init_token="$(hostname)-$$-$(date +%s)-${RANDOM}"
    e2_claim_transient_lock "${runtime}.init.lock" "$init_token" \
      "v2 seller-recovery worktree initialization" || return $?
    local init_status=0
    if [[ ! -e "$runtime" ]]; then
      git -C "$E1R2_SOURCE_ROOT" worktree add --detach "$runtime" "$revision" || init_status=$?
    fi
    e2_release_transient_lock || return $?
    (( init_status == 0 )) || return "$init_status"
  fi
  dirty=$(git -C "$runtime" status --porcelain) || return $?
  [[ -z "$dirty" ]] || e1r2_die "v2 seller-recovery runtime is dirty: $runtime"
  E1R2_CODE_ROOT="$runtime"
  E1R2_VALIDATOR="$runtime/$E1R2_VALIDATOR_RELATIVE"
  local activated_learning_rate
  activated_learning_rate=$(jq -r '.protocol.training_config.learning_rate' "$E1R2_ACTIVATION") || return $?
  [[ "$activated_learning_rate" == "$E1R2_LEARNING_RATE" ]] || \
    e1r2_die "configured v2 learning rate differs from immutable activation: $E1R2_LEARNING_RATE != $activated_learning_rate"
  export STACKPOMDP_SPACE_INVADERS_ROM="$ROM"
  export PYTHONPATH="$runtime"
  export PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export WANDB_MODE=online
  export WANDB_START_METHOD=thread
  export WANDB_DIR="$WANDB_ROOT"
  mkdir -p "$CHECKPOINT_ROOT" "$WANDB_ROOT" "$LOG_ROOT" "$E1_OUTPUT"
  (
    cd "$E1R2_CODE_ROOT"
    "$E1R2_PYTHON" "$E1R2_VALIDATOR" validate-activation \
      --activation "$E1R2_ACTIVATION" \
      --code-root "$E1R2_CODE_ROOT"
  )
}

function e1r2_wait_for_stable_zip() {
  local file_path="$1"
  local first second
  [[ -f "$file_path" && ! -L "$file_path" ]] || e1r2_die "missing v2 checkpoint: $file_path"
  first=$(stat -f '%z:%m' "$file_path") || return $?
  sleep 4
  second=$(stat -f '%z:%m' "$file_path") || return $?
  [[ "$first" == "$second" ]] || e1r2_die "v2 checkpoint is still changing: $file_path"
  unzip -tq "$file_path"
}

function e1r2_candidate_arguments() {
  local step
  for step in $E1R2_TARGET_STEPS; do
    print -r -- "${E1R2_TARGET_BASE%.zip}_step${step}.zip"
  done
  print -r -- "$E1R2_TARGET_BASE"
}

function e1r2_validate_warmup_probe() {
  (
    cd "$E1R2_CODE_ROOT"
    "$E1R2_PYTHON" "$E1R2_VALIDATOR" validate-warmup-probe \
      --activation "$E1R2_ACTIVATION" \
      --probe "$E1R2_WARMUP_PROBE" \
      --checkpoint "$E1R2_WARMUP_BASE"
  )
}

function e1r2_validate_warmup_stage() {
  (
    cd "$E1R2_CODE_ROOT"
    "$E1R2_PYTHON" "$E1R2_VALIDATOR" validate-warmup-stage \
      --activation "$E1R2_ACTIVATION" \
      --probe "$E1R2_WARMUP_PROBE" \
      --checkpoint "$E1R2_WARMUP_BASE" \
      --training-log "${E1R2_WARMUP_BASE%.zip}.training.jsonl" \
      --evaluation "${E1R2_WARMUP_BASE%.zip}.evaluation.json" \
      --device cpu
  )
}

function e1r2_validate_family() {
  local -a arguments
  local candidate
  arguments=()
  while IFS= read -r candidate; do
    e1r2_wait_for_stable_zip "$candidate" || return $?
    arguments+=(--candidate "$candidate")
  done < <(e1r2_candidate_arguments)
  (
    cd "$E1R2_CODE_ROOT"
    "$E1R2_PYTHON" "$E1R2_VALIDATOR" training-family \
      --activation "$E1R2_ACTIVATION" \
      --code-root "$E1R2_CODE_ROOT" \
      --warmup-checkpoint "$E1R2_WARMUP_BASE" \
      --warmup-probe "$E1R2_WARMUP_PROBE" \
      --target-checkpoint "$E1R2_TARGET_BASE" \
      "${arguments[@]}" \
      --warmup-training-log "${E1R2_WARMUP_BASE%.zip}.training.jsonl" \
      --target-training-log "${E1R2_TARGET_BASE%.zip}.training.jsonl" \
      --warmup-evaluation "${E1R2_WARMUP_BASE%.zip}.evaluation.json" \
      --target-evaluation "${E1R2_TARGET_BASE%.zip}.evaluation.json" \
      --output "$E1R2_FAMILY" \
      --device cpu
  )
}

function e1r2_validate_existing_family() {
  (
    cd "$E1R2_CODE_ROOT"
    "$E1R2_PYTHON" "$E1R2_VALIDATOR" validate-training-family --family "$E1R2_FAMILY"
  )
}

function e1r2_validate_existing_gate() {
  (
    cd "$E1R2_CODE_ROOT"
    "$E1R2_PYTHON" "$E1R2_VALIDATOR" validate-selection-gate \
      --gate "$E1R2_GATE" \
      --family "$E1R2_FAMILY" \
      --report "$E1R2_REPORT" \
      --selected "$E1R2_SELECTED"
  )
}
