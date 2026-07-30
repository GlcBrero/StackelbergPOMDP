#!/bin/zsh

# Side-effect-free paths/helpers for the seller-v5 formal family selector.

source "${${(%):-%N}:A:h}/atari_e1_seller_shared_context_v5_common.zsh"

typeset -gr E1V5S_VALIDATOR_RELATIVE=replication/atari/automation/validate_atari_e1_seller_shared_context_v5_selection.py
typeset -gr E1V5S_FAMILY="$E1_OUTPUT/e1_seller_${E1V5_TOKEN}_formal_family.json"
typeset -gr E1V5S_REPORT="$E1_OUTPUT/e1_seller_${E1V5_TOKEN}_all6_selector_v1.json"
typeset -gr E1V5S_GATE="$E1_OUTPUT/e1_seller_${E1V5_TOKEN}_all6_selector_v1.gate.json"
typeset -gr E1V5S_SELECTED="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_${E1V5_TOKEN}_seed1_selected.zip"
typeset -gr E1V5S_RUN_NAME="e1_seller_${E1V5_TOKEN}_all6_selector_v1"
typeset -gr E1V5S_LOG="$LOG_ROOT/${E1V5S_RUN_NAME}.log"

case "$E1V5_PROTOCOL" in
  standard_v1)
    typeset -gr E1V5S_SCREEN_SEED_START=12000001
    typeset -gr E1V5S_CONFIRMATION_SEED_START=12100001
    typeset -gr E1V5S_FIXED_SEED_START=12200001
    typeset -gr E1V5S_TIMING_SEED_START=12300001
    ;;
  exposure_v2)
    typeset -gr E1V5S_SCREEN_SEED_START=13000001
    typeset -gr E1V5S_CONFIRMATION_SEED_START=13100001
    typeset -gr E1V5S_FIXED_SEED_START=13200001
    typeset -gr E1V5S_TIMING_SEED_START=13300001
    ;;
  *) e1v5_die "unsupported seller-v5 selection protocol: $E1V5_PROTOCOL" ;;
esac

typeset -g E1V5S_CODE_ROOT="$E1V5_SOURCE_ROOT"
typeset -g E1V5S_VALIDATOR="$E1V5_SOURCE_ROOT/$E1V5S_VALIDATOR_RELATIVE"

function e1v5s_candidate_arguments() {
  local step
  for step in $E1V5_FORMAL_STEPS; do
    print -r -- "${E1V5_FORMAL%.zip}_step${step}.zip"
  done
  print -r -- "$E1V5_FORMAL"
}

function e1v5s_prepare_runtime() {
  local revision runtime head dirty init_token init_status=0
  e1v5_require_scoped_clean || return $?
  revision=$(git -C "$E1V5_SOURCE_ROOT" rev-parse HEAD) || return $?
  [[ ${#revision} -eq 40 && "$revision" != *[!0-9a-f]* ]] || \
    e1v5_die "invalid seller-v5 selector revision"
  runtime="/private/tmp/stackpomdp-e1-seller-shared-context-v5-selector-code-${revision[1,12]}"
  if [[ -d "$runtime/.git" || -f "$runtime/.git" ]]; then
    head=$(git -C "$runtime" rev-parse HEAD) || return $?
    [[ "$head" == "$revision" ]] || \
      e1v5_die "existing seller-v5 selector runtime has another revision"
  elif [[ -e "$runtime" ]]; then
    e1v5_die "seller-v5 selector runtime path is not a git worktree: $runtime"
  else
    init_token="$(hostname)-$$-$(date +%s)-${RANDOM}"
    e2_claim_transient_lock "${runtime}.init.lock" "$init_token" \
      "seller-v5 selector worktree initialization" || return $?
    if [[ ! -e "$runtime" ]]; then
      git -C "$E1V5_SOURCE_ROOT" worktree add --detach \
        "$runtime" "$revision" || init_status=$?
    fi
    e2_release_transient_lock || return $?
    (( init_status == 0 )) || return "$init_status"
  fi
  dirty=$(git -C "$runtime" status --porcelain) || return $?
  [[ -z "$dirty" ]] || \
    e1v5_die "seller-v5 selector runtime is dirty: $runtime"
  E1V5S_CODE_ROOT="$runtime"
  E1V5S_VALIDATOR="$runtime/$E1V5S_VALIDATOR_RELATIVE"
  export STACKPOMDP_SPACE_INVADERS_ROM="$ROM"
  export PYTHONPATH="$runtime"
  export PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT" "$E1_OUTPUT"
}

function e1v5s_validate_family() {
  (
    cd "$E1V5S_CODE_ROOT"
    "$E1V5_PYTHON" "$E1V5S_VALIDATOR" \
      "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" validate-formal-family \
      --family "$E1V5S_FAMILY" --code-root "$E1V5S_CODE_ROOT"
  )
}

function e1v5s_validate_gate() {
  (
    cd "$E1V5S_CODE_ROOT"
    "$E1V5_PYTHON" "$E1V5S_VALIDATOR" \
      "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" validate-selection-gate \
      --gate "$E1V5S_GATE" \
      --family "$E1V5S_FAMILY" \
      --report "$E1V5S_REPORT" \
      --selected "$E1V5S_SELECTED"
  )
}
