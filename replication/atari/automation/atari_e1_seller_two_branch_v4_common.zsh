#!/bin/zsh

# Shared, side-effect-free paths and helpers for the opt-in v4 seller actor.
# This namespace never admits v1-v3 recovery checkpoints or artifacts.

source "${${(%):-%N}:A:h}/atari_e2_pipeline_common.zsh"

typeset -gr E1V4_SOURCE_ROOT="$AUTOMATION_ROOT"
typeset -gr E1V4_PYTHON="$PYTHON"
typeset -gr E1V4_TOKEN=conditioning_recovery_v4_two_branch_v1
typeset -gr E1V4_SOURCE_KIND=seller_conditioning_recovery_v4_two_branch_v1
typeset -gr E1V4_ARCHITECTURE=seller_two_branch_beta_v4
typeset -gr E1V4_E0B="$CHECKPOINT_ROOT/space_invaders_e0b_ppo_seed1_firefix_retrain_selected.zip"
typeset -gr E1V4_SELLER_RELEASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_seed1_firefix_retrain.buyer_gate.json"
typeset -gr E1V4_VALIDATOR_RELATIVE=replication/atari/automation/validate_atari_e1_seller_two_branch_v4.py

typeset -gr E1V4_SMOKE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1V4_TOKEN}_seed1_uniform_mechanics_smoke.zip"
typeset -gr E1V4_SMOKE_TRACE="${E1V4_SMOKE%.zip}.training.jsonl"
typeset -gr E1V4_SMOKE_EVALUATION="${E1V4_SMOKE%.zip}.evaluation.json"
typeset -gr E1V4_SMOKE_LOG="$LOG_ROOT/atari_clean_e1_seller_${E1V4_TOKEN}_uniform_mechanics_smoke_seed1_20500_local.log"

typeset -gr E1V4_PREFLIGHT="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1V4_TOKEN}_seed1_uniform_conditioning_preflight.zip"
typeset -gr E1V4_PREFLIGHT_TRACE="${E1V4_PREFLIGHT%.zip}.training.jsonl"
typeset -gr E1V4_PREFLIGHT_EVALUATION="${E1V4_PREFLIGHT%.zip}.evaluation.json"
typeset -gr E1V4_PREFLIGHT_PROBE="$E1_OUTPUT/e1_seller_${E1V4_TOKEN}_uniform_conditioning_preflight_probe.json"
typeset -gr E1V4_PREFLIGHT_BEHAVIOR="$E1_OUTPUT/e1_seller_${E1V4_TOKEN}_uniform_conditioning_preflight_behavior.json"
typeset -gr E1V4_PREFLIGHT_LOG="$LOG_ROOT/atari_clean_e1_seller_${E1V4_TOKEN}_uniform_conditioning_preflight_seed1_82000_local.log"

typeset -gr E1V4_GATE="$E1_OUTPUT/e1_seller_${E1V4_TOKEN}_diagnostics_gate.json"
typeset -gr E1V4_FORMAL="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1V4_TOKEN}_seed1_uniform_formal.zip"
typeset -gr E1V4_FORMAL_TRACE="${E1V4_FORMAL%.zip}.training.jsonl"
typeset -gr E1V4_FORMAL_EVALUATION="${E1V4_FORMAL%.zip}.evaluation.json"
typeset -gr E1V4_FORMAL_LOG="$LOG_ROOT/atari_clean_e1_seller_${E1V4_TOKEN}_uniform_formal_seed1_2000800_local.log"
typeset -gr E1V4_WANDB_NAME=atari_clean_e1_seller_conditioning_recovery_v4_two_branch_v1_uniform_seed1_2000800_local
typeset -gr E1V4_WANDB_JOB_TYPE=atari_e1_seller_conditioning_recovery_v4_two_branch_v1_uniform_formal
typeset -gr E1V4_LOCK=/private/tmp/stackpomdp-atari-e1-seller-conditioning-recovery-v4-two-branch-v1.lock
typeset -gra E1V4_FORMAL_STEPS=(400160 800320 1200480 1600640 2000800)

typeset -g E1V4_CODE_ROOT="$E1V4_SOURCE_ROOT"
typeset -g E1V4_VALIDATOR="$E1V4_SOURCE_ROOT/$E1V4_VALIDATOR_RELATIVE"
typeset -gi E1V4_ACTIVE_JOB_PID=0

function e1v4_die() {
  print -u2 -- "$*"
  return 1
}

function e1v4_terminate_process_tree() {
  local parent_pid="$1"
  local signal_name="${2:-TERM}"
  local children child
  [[ "$parent_pid" == <-> ]] || return 0
  children=$(pgrep -P "$parent_pid" 2>/dev/null) || children=""
  for child in ${(f)children}; do
    [[ "$child" == <-> ]] || continue
    e1v4_terminate_process_tree "$child" "$signal_name"
  done
  kill -s "$signal_name" "$parent_pid" 2>/dev/null || :
}

function e1v4_cancel_active_job() {
  local active_pid="$E1V4_ACTIVE_JOB_PID"
  (( active_pid > 0 )) || return 0
  e1v4_terminate_process_tree "$active_pid" TERM
  wait "$active_pid" 2>/dev/null || :
  typeset -g E1V4_ACTIVE_JOB_PID=0
}

function e1v4_require_scoped_clean() {
  local dirty
  dirty=$(git -C "$E1V4_SOURCE_ROOT" status --porcelain -- \
    replication/atari/train_atari_meta_response_sb3.py \
    replication/atari/evaluate_atari_meta_response_sb3.py \
    replication/atari/probe_atari_e1_seller_two_branch.py \
    replication/atari/sb3_common.py \
    replication/atari/automation/atari_e1_seller_two_branch_v4_common.zsh \
    replication/atari/automation/run_atari_clean_e1_seller_two_branch_v4.sh \
    replication/atari/automation/validate_atari_e1_seller_two_branch_v4.py \
    replication/atari/automation/atari_e2_pipeline_common.zsh \
    replication/atari/automation/validate_atari_e1_seller_conditioning_recovery.py \
    stackelberg_pomdp/atari)
  [[ -z "$dirty" ]] || \
    e1v4_die "refusing uncommitted v4 seller code:\n$dirty"
}

function e1v4_refuse_path() {
  [[ ! -e "$1" && ! -L "$1" ]] || \
    e1v4_die "refusing to overwrite v4 artifact: $1"
}

function e1v4_wait_for_stable_zip() {
  local file_path="$1"
  local first second
  [[ -f "$file_path" && ! -L "$file_path" ]] || \
    e1v4_die "missing v4 checkpoint: $file_path"
  first=$(stat -f '%z:%m' "$file_path") || return $?
  sleep 4
  second=$(stat -f '%z:%m' "$file_path") || return $?
  [[ "$first" == "$second" ]] || \
    e1v4_die "v4 checkpoint is still changing: $file_path"
  unzip -tq "$file_path"
}

function e1v4_prepare_runtime() {
  local revision runtime head dirty init_token init_status=0
  revision=$(jq -r '.code_revision' "$E1V4_GATE") || return $?
  [[ ${#revision} -eq 40 && "$revision" != *[!0-9a-f]* ]] || \
    e1v4_die "invalid v4 diagnostics-gate revision"
  runtime="/private/tmp/stackpomdp-e1-seller-two-branch-v4-code-${revision[1,12]}"
  if [[ -d "$runtime/.git" || -f "$runtime/.git" ]]; then
    head=$(git -C "$runtime" rev-parse HEAD) || return $?
    [[ "$head" == "$revision" ]] || \
      e1v4_die "existing v4 runtime has another revision: $runtime"
  elif [[ -e "$runtime" ]]; then
    e1v4_die "v4 runtime path is not a git worktree: $runtime"
  else
    init_token="$(hostname)-$$-$(date +%s)-${RANDOM}"
    e2_claim_transient_lock "${runtime}.init.lock" "$init_token" \
      "v4 seller worktree initialization" || return $?
    if [[ ! -e "$runtime" ]]; then
      git -C "$E1V4_SOURCE_ROOT" worktree add --detach \
        "$runtime" "$revision" || init_status=$?
    fi
    e2_release_transient_lock || return $?
    (( init_status == 0 )) || return "$init_status"
  fi
  dirty=$(git -C "$runtime" status --porcelain) || return $?
  [[ -z "$dirty" ]] || e1v4_die "v4 runtime is dirty: $runtime"
  E1V4_CODE_ROOT="$runtime"
  E1V4_VALIDATOR="$runtime/$E1V4_VALIDATOR_RELATIVE"
  export STACKPOMDP_SPACE_INVADERS_ROM="$ROM"
  export PYTHONPATH="$runtime"
  export PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export WANDB_MODE=online
  export WANDB_START_METHOD=thread
  export WANDB_DIR="$WANDB_ROOT"
  (
    cd "$runtime"
    "$E1V4_PYTHON" "$E1V4_VALIDATOR" validate-gate \
      --gate "$E1V4_GATE" --code-root "$runtime"
  )
}
