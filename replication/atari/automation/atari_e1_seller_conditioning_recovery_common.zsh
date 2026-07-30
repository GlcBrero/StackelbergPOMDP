#!/bin/zsh

# Shared paths and fail-closed helpers for the two-stage E1 seller recovery.
# This file is sourced by the training and selector launchers; it never starts
# work on its own.

source "${${(%):-%N}:A:h}/atari_e2_pipeline_common.zsh"

typeset -gr E1R_SOURCE_ROOT="$AUTOMATION_ROOT"
typeset -gr E1R_ACTIVE_ROOT="$ROOT"
typeset -gr E1R_PYTHON="$PYTHON"
typeset -gr E1R_E0B="$CHECKPOINT_ROOT/space_invaders_e0b_ppo_seed1_firefix_retrain_selected.zip"
typeset -gr E1R_FAILED_REPORT="$E1_OUTPUT/e1_seller_balanced_all6_selector_v2.json"
typeset -gr E1R_RELEASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_seed1_firefix_retrain.buyer_gate.json"
typeset -gr E1R_ACTIVATION="$E1_OUTPUT/e1_seller_conditioning_recovery_activation_v1.json"
typeset -gr E1R_FAMILY="$E1_OUTPUT/e1_seller_conditioning_recovery_family_v1.json"
typeset -gr E1R_WARMUP_BASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_conditioning_recovery_v1_seed1_all_equal_warmup.zip"
typeset -gr E1R_WARMUP_PROBE="$E1_OUTPUT/e1_seller_conditioning_recovery_warmup_probe_v1.json"
typeset -gr E1R_TARGET_BASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_conditioning_recovery_v1_seed1_uniform_target.zip"
typeset -gr E1R_WARMUP_RUN=atari_clean_e1_seller_conditioning_recovery_v1_all_equal_warmup_seed1_400160_local
typeset -gr E1R_TARGET_RUN=atari_clean_e1_seller_conditioning_recovery_v1_uniform_target_seed1_2000800_local
typeset -gr E1R_WARMUP_LOG="$LOG_ROOT/${E1R_WARMUP_RUN}.log"
typeset -gr E1R_TARGET_LOG="$LOG_ROOT/${E1R_TARGET_RUN}.log"
typeset -gr E1R_SELECTOR_NAME=e1_seller_conditioning_recovery_all6_selector_v2
typeset -gr E1R_REPORT="$E1_OUTPUT/${E1R_SELECTOR_NAME}.json"
typeset -gr E1R_GATE="$E1_OUTPUT/${E1R_SELECTOR_NAME}.gate.json"
typeset -gr E1R_SELECTED="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_conditioning_recovery_v1_seed1_selected.zip"
typeset -gr E1R_SELECTOR_LOG="$LOG_ROOT/${E1R_SELECTOR_NAME}.log"
typeset -gr E1R_LOCK=/private/tmp/stackpomdp-atari-e1-seller-conditioning-recovery.lock
typeset -gra E1R_TARGET_STEPS=(800320 1200480 1600640 2000800 2400960)

typeset -g E1R_CODE_ROOT="$E1R_SOURCE_ROOT"
typeset -g E1R_VALIDATOR="$E1R_SOURCE_ROOT/replication/atari/automation/validate_atari_e1_seller_conditioning_recovery.py"
typeset -gi E1R_ACTIVE_JOB_PID=0

function e1r_die() {
  print -u2 -- "$*"
  return 1
}

function e1r_terminate_process_tree() {
  local parent_pid="$1"
  local signal_name="${2:-TERM}"
  local children child
  [[ "$parent_pid" == <-> ]] || return 0
  children=$(pgrep -P "$parent_pid" 2>/dev/null) || children=""
  for child in ${(f)children}; do
    [[ "$child" == <-> ]] || continue
    e1r_terminate_process_tree "$child" "$signal_name"
  done
  kill -s "$signal_name" "$parent_pid" 2>/dev/null || :
}

function e1r_cancel_active_job() {
  local active_pid="$E1R_ACTIVE_JOB_PID"
  (( active_pid > 0 )) || return 0
  e1r_terminate_process_tree "$active_pid" TERM
  wait "$active_pid" 2>/dev/null || :
  typeset -g E1R_ACTIVE_JOB_PID=0
}

function e1r_require_scoped_clean() {
  local dirty
  dirty=$(git -C "$E1R_SOURCE_ROOT" status --porcelain -- \
    replication/atari/train_atari_meta_response_sb3.py \
    replication/atari/evaluate_atari_meta_response_sb3.py \
    replication/atari/probe_atari_e1_seller_conditioning.py \
    replication/atari/sb3_common.py \
    replication/atari/automation \
    stackelberg_pomdp/atari)
  [[ -z "$dirty" ]] || e1r_die "refusing uncommitted seller-recovery code:\n$dirty"
}

function e1r_refuse_path() {
  [[ ! -e "$1" && ! -L "$1" ]] || \
    e1r_die "refusing to overwrite seller-recovery artifact: $1"
}

function e1r_activate() {
  if [[ -f "$E1R_ACTIVATION" && ! -L "$E1R_ACTIVATION" ]]; then
    (
      cd "$E1R_SOURCE_ROOT"
      PYTHONPATH="$E1R_SOURCE_ROOT" PYTHONNOUSERSITE=1 \
        "$E1R_PYTHON" "$E1R_VALIDATOR" validate-activation \
          --activation "$E1R_ACTIVATION"
    )
    return $?
  fi
  e1r_require_scoped_clean
  (
    cd "$E1R_SOURCE_ROOT"
    PYTHONPATH="$E1R_SOURCE_ROOT" PYTHONNOUSERSITE=1 \
      "$E1R_PYTHON" "$E1R_VALIDATOR" activation \
        --failed-report "$E1R_FAILED_REPORT" \
        --seller-release "$E1R_RELEASE" \
        --e0b "$E1R_E0B" \
        --rom "$ROM" \
        --warmup-checkpoint "$E1R_WARMUP_BASE" \
        --warmup-probe "$E1R_WARMUP_PROBE" \
        --target-checkpoint "$E1R_TARGET_BASE" \
        --code-root "$E1R_SOURCE_ROOT" \
        --output "$E1R_ACTIVATION"
  )
}

function e1r_prepare_runtime() {
  local revision runtime head dirty init_token
  revision=$(jq -r '.code_revision' "$E1R_ACTIVATION") || return $?
  [[ ${#revision} -eq 40 && "$revision" != *[!0-9a-f]* ]] || \
    e1r_die "invalid seller-recovery activation revision"
  runtime="/private/tmp/stackpomdp-e1-seller-recovery-code-${revision[1,12]}"
  if [[ -d "$runtime/.git" || -f "$runtime/.git" ]]; then
    head=$(git -C "$runtime" rev-parse HEAD) || return $?
    [[ "$head" == "$revision" ]] || \
      e1r_die "existing seller-recovery runtime has another revision: $runtime"
  elif [[ -e "$runtime" ]]; then
    e1r_die "seller-recovery runtime path is not a git worktree: $runtime"
  else
    init_token="$(hostname)-$$-$(date +%s)-${RANDOM}"
    e2_claim_transient_lock \
      "${runtime}.init.lock" "$init_token" \
      "seller-recovery code-worktree initialization" || return $?
    local init_status=0
    if [[ ! -e "$runtime" ]]; then
      git -C "$E1R_SOURCE_ROOT" worktree add --detach "$runtime" "$revision" || \
        init_status=$?
    fi
    e2_release_transient_lock || return $?
    (( init_status == 0 )) || return "$init_status"
  fi
  dirty=$(git -C "$runtime" status --porcelain) || return $?
  [[ -z "$dirty" ]] || e1r_die "seller-recovery runtime is dirty: $runtime"
  E1R_CODE_ROOT="$runtime"
  E1R_VALIDATOR="$runtime/replication/atari/automation/validate_atari_e1_seller_conditioning_recovery.py"
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
    cd "$E1R_CODE_ROOT"
    "$E1R_PYTHON" "$E1R_VALIDATOR" validate-activation \
      --activation "$E1R_ACTIVATION" \
      --code-root "$E1R_CODE_ROOT"
  )
}

function e1r_wait_for_stable_zip() {
  local file_path="$1"
  local first second
  [[ -f "$file_path" && ! -L "$file_path" ]] || \
    e1r_die "missing seller-recovery checkpoint: $file_path"
  first=$(stat -f '%z:%m' "$file_path") || return $?
  sleep 4
  second=$(stat -f '%z:%m' "$file_path") || return $?
  [[ "$first" == "$second" ]] || \
    e1r_die "seller-recovery checkpoint is still changing: $file_path"
  unzip -tq "$file_path"
}

function e1r_candidate_arguments() {
  local step
  for step in $E1R_TARGET_STEPS; do
    print -r -- "${E1R_TARGET_BASE%.zip}_step${step}.zip"
  done
  print -r -- "$E1R_TARGET_BASE"
}

function e1r_validate_warmup_probe() {
  (
    cd "$E1R_CODE_ROOT"
    "$E1R_PYTHON" "$E1R_VALIDATOR" validate-warmup-stage \
      --activation "$E1R_ACTIVATION" \
      --probe "$E1R_WARMUP_PROBE" \
      --checkpoint "$E1R_WARMUP_BASE" \
      --training-log "${E1R_WARMUP_BASE%.zip}.training.jsonl" \
      --evaluation "${E1R_WARMUP_BASE%.zip}.evaluation.json" \
      --device cpu
  )
}

function e1r_validate_warmup_checkpoint() {
  (
    cd "$E1R_CODE_ROOT"
    "$E1R_PYTHON" "$E1R_VALIDATOR" validate-warmup-checkpoint \
      --activation "$E1R_ACTIVATION" \
      --checkpoint "$E1R_WARMUP_BASE" \
      --device cpu
  )
}

function e1r_validate_family() {
  local -a arguments
  local candidate
  arguments=()
  while IFS= read -r candidate; do
    e1r_wait_for_stable_zip "$candidate" || return $?
    arguments+=(--candidate "$candidate")
  done < <(e1r_candidate_arguments)
  (
    cd "$E1R_CODE_ROOT"
    "$E1R_PYTHON" "$E1R_VALIDATOR" training-family \
      --activation "$E1R_ACTIVATION" \
      --code-root "$E1R_CODE_ROOT" \
      --warmup-checkpoint "$E1R_WARMUP_BASE" \
      --warmup-probe "$E1R_WARMUP_PROBE" \
      --target-checkpoint "$E1R_TARGET_BASE" \
      "${arguments[@]}" \
      --warmup-training-log "${E1R_WARMUP_BASE%.zip}.training.jsonl" \
      --target-training-log "${E1R_TARGET_BASE%.zip}.training.jsonl" \
      --warmup-evaluation "${E1R_WARMUP_BASE%.zip}.evaluation.json" \
      --target-evaluation "${E1R_TARGET_BASE%.zip}.evaluation.json" \
      --output "$E1R_FAMILY" \
      --device cpu
  )
}

function e1r_validate_existing_family() {
  (
    cd "$E1R_CODE_ROOT"
    "$E1R_PYTHON" "$E1R_VALIDATOR" validate-training-family \
      --family "$E1R_FAMILY"
  )
}

function e1r_validate_existing_gate() {
  (
    cd "$E1R_CODE_ROOT"
    "$E1R_PYTHON" "$E1R_VALIDATOR" validate-selection-gate \
      --gate "$E1R_GATE" \
      --family "$E1R_FAMILY" \
      --report "$E1R_REPORT" \
      --selected "$E1R_SELECTED"
  )
}
