#!/bin/zsh

# Shared, side-effect-free paths and helpers for the opt-in v5 seller actor.
# This namespace never admits v1-v3 recovery checkpoints or artifacts.

source "${${(%):-%N}:A:h}/atari_e2_pipeline_common.zsh"

typeset -gr E1V5_SOURCE_ROOT="$AUTOMATION_ROOT"
typeset -gr E1V5_PYTHON="$PYTHON"
typeset -gr E1V5_PROTOCOL="${STACKPOMDP_E1V5_PROTOCOL:-standard_v1}"
case "$E1V5_PROTOCOL" in
  standard_v1)
    typeset -gr E1V5_TOKEN=conditioning_recovery_v5_shared_context_v1
    typeset -gr E1V5_SOURCE_KIND=seller_conditioning_recovery_v5_shared_context_v1
    typeset -gr E1V5_SMOKE_TIMESTEPS=20500
    typeset -gr E1V5_PREFLIGHT_TIMESTEPS=82000
    typeset -gra E1V5_VALIDATOR_PROTOCOL_ARGS=()
    typeset -gr E1V5_EVIDENCE_REVISION=""
    ;;
  exposure_v2)
    typeset -gr E1V5_TOKEN=conditioning_recovery_v5_shared_context_exposure_v2
    typeset -gr E1V5_SOURCE_KIND=seller_conditioning_recovery_v5_shared_context_exposure_v2
    typeset -gr E1V5_SMOKE_TIMESTEPS=20500
    typeset -gr E1V5_PREFLIGHT_TIMESTEPS=400160
    typeset -gra E1V5_VALIDATOR_PROTOCOL_ARGS=(--protocol exposure_v2)
    typeset -gr E1V5_EVIDENCE_REVISION=c4a7dcd92b621c0884f3dcef0b170961e1ec625b
    ;;
  *)
    print -u2 -- "unknown v5 seller exposure protocol: $E1V5_PROTOCOL"
    return 1
    ;;
esac
typeset -gr E1V5_ARCHITECTURE=seller_shared_context_beta_v5
typeset -gr E1V5_E0B="$CHECKPOINT_ROOT/space_invaders_e0b_ppo_seed1_firefix_retrain_selected.zip"
typeset -gr E1V5_SELLER_RELEASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_seed1_firefix_retrain.buyer_gate.json"
typeset -gr E1V5_VALIDATOR_RELATIVE=replication/atari/automation/validate_atari_e1_seller_shared_context_v5.py

typeset -gr E1V5_SMOKE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1V5_TOKEN}_seed1_uniform_mechanics_smoke.zip"
typeset -gr E1V5_SMOKE_TRACE="${E1V5_SMOKE%.zip}.training.jsonl"
typeset -gr E1V5_SMOKE_EVALUATION="${E1V5_SMOKE%.zip}.evaluation.json"
typeset -gr E1V5_SMOKE_LOG="$LOG_ROOT/atari_clean_e1_seller_${E1V5_TOKEN}_uniform_mechanics_smoke_seed1_${E1V5_SMOKE_TIMESTEPS}_local.log"

typeset -gr E1V5_PREFLIGHT="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1V5_TOKEN}_seed1_uniform_conditioning_preflight.zip"
typeset -gr E1V5_PREFLIGHT_TRACE="${E1V5_PREFLIGHT%.zip}.training.jsonl"
typeset -gr E1V5_PREFLIGHT_EVALUATION="${E1V5_PREFLIGHT%.zip}.evaluation.json"
typeset -gr E1V5_PREFLIGHT_PROBE="$E1_OUTPUT/e1_seller_${E1V5_TOKEN}_uniform_conditioning_preflight_probe.json"
typeset -gr E1V5_PREFLIGHT_BEHAVIOR="$E1_OUTPUT/e1_seller_${E1V5_TOKEN}_uniform_conditioning_preflight_behavior.json"
typeset -gr E1V5_PREFLIGHT_LOG="$LOG_ROOT/atari_clean_e1_seller_${E1V5_TOKEN}_uniform_conditioning_preflight_seed1_${E1V5_PREFLIGHT_TIMESTEPS}_local.log"

typeset -gr E1V5_GATE="$E1_OUTPUT/e1_seller_${E1V5_TOKEN}_diagnostics_gate.json"
typeset -gr E1V5_FORMAL="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1V5_TOKEN}_seed1_uniform_formal.zip"
typeset -gr E1V5_FORMAL_TRACE="${E1V5_FORMAL%.zip}.training.jsonl"
typeset -gr E1V5_FORMAL_EVALUATION="${E1V5_FORMAL%.zip}.evaluation.json"
typeset -gr E1V5_FORMAL_LOG="$LOG_ROOT/atari_clean_e1_seller_${E1V5_TOKEN}_uniform_formal_seed1_2000800_local.log"
typeset -gr E1V5_WANDB_NAME=atari_clean_e1_seller_${E1V5_TOKEN}_uniform_seed1_2000800_local
typeset -gr E1V5_WANDB_JOB_TYPE=atari_e1_seller_${E1V5_TOKEN}_uniform_formal
typeset -gr E1V5_LOCK=/private/tmp/stackpomdp-atari-e1-seller-${E1V5_TOKEN//_/-}.lock
typeset -gra E1V5_FORMAL_STEPS=(400160 800320 1200480 1600640 2000800)

typeset -g E1V5_CODE_ROOT="$E1V5_SOURCE_ROOT"
typeset -g E1V5_VALIDATOR="$E1V5_SOURCE_ROOT/$E1V5_VALIDATOR_RELATIVE"
typeset -gi E1V5_ACTIVE_JOB_PID=0

function e1v5_die() {
  print -u2 -- "$*"
  return 1
}

function e1v5_terminate_process_tree() {
  local parent_pid="$1"
  local signal_name="${2:-TERM}"
  local children child
  [[ "$parent_pid" == <-> ]] || return 0
  children=$(pgrep -P "$parent_pid" 2>/dev/null) || children=""
  for child in ${(f)children}; do
    [[ "$child" == <-> ]] || continue
    e1v5_terminate_process_tree "$child" "$signal_name"
  done
  kill -s "$signal_name" "$parent_pid" 2>/dev/null || :
}

function e1v5_cancel_active_job() {
  local active_pid="$E1V5_ACTIVE_JOB_PID"
  (( active_pid > 0 )) || return 0
  e1v5_terminate_process_tree "$active_pid" TERM
  wait "$active_pid" 2>/dev/null || :
  typeset -g E1V5_ACTIVE_JOB_PID=0
}

function e1v5_require_scoped_clean() {
  local dirty
  dirty=$(git -C "$E1V5_SOURCE_ROOT" status --porcelain -- \
    replication/atari/train_atari_meta_response_sb3.py \
    replication/atari/evaluate_atari_meta_response_sb3.py \
    replication/atari/probe_atari_e1_seller_conditioning.py \
    replication/atari/probe_atari_e1_seller_shared_context.py \
    replication/atari/sb3_common.py \
    replication/atari/automation/atari_e1_seller_shared_context_v5_common.zsh \
    replication/atari/automation/run_atari_clean_e1_seller_shared_context_v5.sh \
    replication/atari/automation/run_atari_clean_e1_seller_shared_context_v5_exposure_v2.sh \
    replication/atari/automation/validate_atari_e1_seller_shared_context_v5.py \
    replication/atari/automation/atari_e2_pipeline_common.zsh \
    replication/atari/automation/validate_atari_e1_seller_conditioning_recovery.py \
    stackelberg_pomdp/atari)
  [[ -z "$dirty" ]] || \
    e1v5_die "refusing uncommitted v5 seller code:\n$dirty"
}

function e1v5_refuse_path() {
  [[ ! -e "$1" && ! -L "$1" ]] || \
    e1v5_die "refusing to overwrite v5 artifact: $1"
}

function e1v5_wait_for_stable_zip() {
  local file_path="$1"
  local first second
  [[ -f "$file_path" && ! -L "$file_path" ]] || \
    e1v5_die "missing v5 checkpoint: $file_path"
  first=$(stat -f '%z:%m' "$file_path") || return $?
  sleep 4
  second=$(stat -f '%z:%m' "$file_path") || return $?
  [[ "$first" == "$second" ]] || \
    e1v5_die "v5 checkpoint is still changing: $file_path"
  unzip -tq "$file_path"
}

function e1v5_prepare_runtime() {
  local revision runtime head dirty init_token init_status=0
  revision=$(jq -r '.code_revision' "$E1V5_GATE") || return $?
  [[ ${#revision} -eq 40 && "$revision" != *[!0-9a-f]* ]] || \
    e1v5_die "invalid v5 diagnostics-gate revision"
  runtime="/private/tmp/stackpomdp-e1-seller-shared-context-v5-code-${revision[1,12]}"
  if [[ -d "$runtime/.git" || -f "$runtime/.git" ]]; then
    head=$(git -C "$runtime" rev-parse HEAD) || return $?
    [[ "$head" == "$revision" ]] || \
      e1v5_die "existing v5 runtime has another revision: $runtime"
  elif [[ -e "$runtime" ]]; then
    e1v5_die "v5 runtime path is not a git worktree: $runtime"
  else
    init_token="$(hostname)-$$-$(date +%s)-${RANDOM}"
    e2_claim_transient_lock "${runtime}.init.lock" "$init_token" \
      "v5 seller worktree initialization" || return $?
    if [[ ! -e "$runtime" ]]; then
      git -C "$E1V5_SOURCE_ROOT" worktree add --detach \
        "$runtime" "$revision" || init_status=$?
    fi
    e2_release_transient_lock || return $?
    (( init_status == 0 )) || return "$init_status"
  fi
  dirty=$(git -C "$runtime" status --porcelain) || return $?
  [[ -z "$dirty" ]] || e1v5_die "v5 runtime is dirty: $runtime"
  E1V5_CODE_ROOT="$runtime"
  E1V5_VALIDATOR="$runtime/$E1V5_VALIDATOR_RELATIVE"
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
    "$E1V5_PYTHON" "$E1V5_VALIDATOR" \
      "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" validate-gate \
      --gate "$E1V5_GATE" --code-root "$runtime"
  )
}
