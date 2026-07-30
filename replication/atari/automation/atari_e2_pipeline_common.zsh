#!/bin/zsh

# Shared, collision-safe helpers for clean Atari E2 launchers.
# This file is sourced by the role-specific scripts; it does not launch work.

# tmux may retain a restricted launch-time PATH across long unattended runs.
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin

typeset -gr AUTOMATION_DIR="${${(%):-%N}:A:h}"
typeset -gr AUTOMATION_ROOT="${AUTOMATION_DIR:h:h:h}"
typeset -gr ROOT=/Users/gbrero/active-research/StackelbergPOMDP/code/StackelbergPOMDP
typeset -gr CODE_ROOT=/private/tmp/stackpomdp-e2-code-87fc165
typeset -gr PYTHON=/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python
typeset -gr EXPECTED_HEAD=87fc165000517e874881cac63850133b9982de7f
typeset -gr VALIDATOR="$AUTOMATION_DIR/validate_atari_e2_pipeline_artifact.py"
typeset -gr CHECKPOINT_ROOT="$ROOT/replication/atari/checkpoints/clean"
typeset -gr WANDB_ROOT="$CHECKPOINT_ROOT/wandb_runs"
typeset -gr LOG_ROOT="$ROOT/replication/atari/results/run_logs/clean_20260728"
typeset -gr E1_OUTPUT="$ROOT/replication/atari/results/e1_selections"
typeset -gr E2_OUTPUT="$ROOT/replication/atari/results/e2_selections"
typeset -gr ROM="$ROOT/stackelberg_pomdp/atari/roms/space_invaders.bin"
typeset -gr ROM_SHA256=7224b17462b992d67f4e06a3c85f269c9822b06df6015bf038b55f384ced0301

# The legacy v3 profile remains the default.  The v5 launchers opt into a
# disjoint namespace, cohort, code pin, lock, and W&B job-type family.
typeset -gr E2_PROFILE="${STACKPOMDP_ATARI_E2_PROFILE:-v3-direct-threshold-residual}"
typeset -gr E2_V3_EXPECTED_HEAD=87fc165000517e874881cac63850133b9982de7f
typeset -gr E2_V5_EXPECTED_HEAD=c4a7dcd92b621c0884f3dcef0b170961e1ec625b
case "$E2_PROFILE" in
  v3-direct-threshold-residual)
    typeset -gr E2_ACTIVE_CODE_ROOT="$CODE_ROOT"
    typeset -gr E2_ACTIVE_EXPECTED_HEAD="$EXPECTED_HEAD"
    typeset -gr E2_NAMESPACE=e1seller_direct_threshold_residual_v3
    typeset -gr E1_REQUIRED_SELLER_SOURCE_KIND=seller_conditioning_recovery_v3_direct_threshold_residual_v1
    typeset -gr E2_PIPELINE_LOCK=/private/tmp/stackpomdp-atari-e2-direct-threshold-residual-v3-sequential.lock
    typeset -gr E2_WANDB_BUYER_JOB_TYPE=atari_e2_direct_threshold_residual_v3_buyer_leader
    typeset -gr E2_WANDB_SELLER_JOB_TYPE=atari_e2_direct_threshold_residual_v3_seller_leader
    ;;
  v5-shared-context-exposure-v2)
    typeset -gr E2_ACTIVE_CODE_ROOT=/private/tmp/stackpomdp-e2-v5-shared-context-code-c4a7dcd
    typeset -gr E2_ACTIVE_EXPECTED_HEAD="$E2_V5_EXPECTED_HEAD"
    typeset -gr E2_NAMESPACE=e1seller_shared_context_v5_exposure_v2
    typeset -gr E1_REQUIRED_SELLER_SOURCE_KIND=seller_conditioning_recovery_v5_shared_context_exposure_v2
    typeset -gr E2_PIPELINE_LOCK=/private/tmp/stackpomdp-atari-e2-shared-context-v5-exposure-v2-sequential.lock
    typeset -gr E2_WANDB_BUYER_JOB_TYPE=atari_e2_shared_context_v5_exposure_v2_buyer_leader
    typeset -gr E2_WANDB_SELLER_JOB_TYPE=atari_e2_shared_context_v5_exposure_v2_seller_leader
    ;;
  *)
    print -u2 -- "unknown Atari E2 profile: $E2_PROFILE"
    return 1
    ;;
esac
typeset -gr E1_COHORT="$CHECKPOINT_ROOT/e2_e1_gate_cohort_${E2_NAMESPACE}.json"

typeset -gra E2_STEPS=(400680 800520 1200360 1600200 2000040)
typeset -gr E2_TIMESTEPS=2000040
typeset -gr E2_N_STEPS=210
typeset -gr E2_NUM_ENVS=4
typeset -gr E2_BATCH_SIZE=840

function stackpomdp_claim_owned_lock() {
  local lock="$1"
  local token="$2"
  local label="$3"
  local owner_record owner owner_pid owner_host before after current_host
  local lock_mtime lock_age now
  local -a entries
  current_host=$(hostname)
  while true; do
    if mkdir "$lock" 2>/dev/null; then
      if ! print -r -- "${token}"$'\t'"$$"$'\t'"${current_host}"$'\t'"$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          > "$lock/owner.tsv"; then
        rm -f "$lock/owner.tsv" 2>/dev/null || true
        rmdir "$lock" 2>/dev/null || true
        print -u2 "failed to publish $label owner record: $lock/owner.tsv"
        return 1
      fi
      typeset -g STACKPOMDP_LOCK_RESULT_OWNED=1
      return 0
    fi
    [[ -d "$lock" && ! -L "$lock" ]] || {
      print -u2 "$label lock path is not a regular directory: $lock"
      return 1
    }
    if [[ ! -e "$lock/owner.tsv" ]]; then
      entries=("$lock"/*(DN))
      if (( ${#entries} != 0 )); then
        print -u2 "$label lock has no owner record but is not empty: $lock"
        return 1
      fi
      lock_mtime=$(stat -f '%m' "$lock") || return 1
      now=$(date +%s) || return 1
      lock_age=$(( now - lock_mtime ))
      if (( lock_age < 60 )); then
        print -u2 "$label ownerless lock is younger than 60 seconds: $lock"
        return 1
      fi
      before=$(stat -f '%d:%i:%m:%c' "$lock") || return 1
      sleep 1
      entries=("$lock"/*(DN))
      after=$(stat -f '%d:%i:%m:%c' "$lock") || return 1
      if (( ${#entries} == 0 )) && [[ "$before" == "$after" ]]; then
        if rmdir "$lock" 2>/dev/null; then
          print "reclaimed stable empty $label lock: $lock"
          continue
        fi
      fi
      print -u2 "$label lock has no stable owner record: $lock"
      return 1
    fi
    [[ -f "$lock/owner.tsv" && ! -L "$lock/owner.tsv" ]] || {
      print -u2 "$label owner record is not a regular file: $lock/owner.tsv"
      return 1
    }
    owner_record=$(<"$lock/owner.tsv")
    IFS=$'\t' read -r owner owner_pid owner_host _ <<< "$owner_record"
    [[ "$owner_pid" == <-> && "$owner_pid" -gt 0 && -n "$owner_host" ]] || {
      print -u2 "$label lock has malformed owner metadata: $lock"
      return 1
    }
    if [[ "$owner_host" != "$current_host" ]]; then
      print -u2 "$label lock belongs to foreign host $owner_host: $lock"
      return 1
    fi
    if [[ "$owner" == "$token" ]]; then
      if ! kill -0 "$owner_pid" 2>/dev/null; then
        print -u2 "$label reentrant token names absent PID $owner_pid; refusing automatic reclaim: $lock"
        return 1
      fi
      typeset -g STACKPOMDP_LOCK_RESULT_OWNED=0
      return 0
    fi
    if kill -0 "$owner_pid" 2>/dev/null; then
      print -u2 "$label lock is live on PID $owner_pid: $lock"
      return 1
    fi
    print -u2 "$label owner PID $owner_pid is absent, but descendants may still be active; refusing automatic reclaim: $lock"
    return 1
  done
}

function stackpomdp_release_owned_lock() {
  local lock="$1"
  local token="$2"
  local owned="$3"
  local label="$4"
  local owner
  [[ "$owned" == 1 ]] || return 0
  [[ -f "$lock/owner.tsv" && ! -L "$lock/owner.tsv" ]] || return 1
  IFS=$'\t' read -r owner _ < "$lock/owner.tsv"
  [[ "$owner" == "$token" ]] || {
    print -u2 "refusing to release a $label lock owned by another token"
    return 1
  }
  rm "$lock/owner.tsv"
  rmdir "$lock"
}

function e2_claim_pipeline_lock() {
  local token
  token="${STACKPOMDP_E2_LOCK_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"
  stackpomdp_claim_owned_lock \
    "$E2_PIPELINE_LOCK" "$token" "E2 pipeline" || return $?
  typeset -gx STACKPOMDP_E2_LOCK_TOKEN="$token"
  typeset -g E2_LOCK_OWNED_BY_CALLER="$STACKPOMDP_LOCK_RESULT_OWNED"
}

function e2_release_pipeline_lock() {
  stackpomdp_release_owned_lock \
    "$E2_PIPELINE_LOCK" "${STACKPOMDP_E2_LOCK_TOKEN:-}" \
    "${E2_LOCK_OWNED_BY_CALLER:-0}" "E2 pipeline" || return $?
  typeset -g E2_LOCK_OWNED_BY_CALLER=0
}

function e2_clear_transient_lock_record() {
  typeset -g E2_TRANSIENT_LOCK_PATH=
  typeset -g E2_TRANSIENT_LOCK_TOKEN=
  typeset -g E2_TRANSIENT_LOCK_OWNED=0
  typeset -g E2_TRANSIENT_LOCK_LABEL=
}

function e2_claim_transient_lock() {
  local lock="$1"
  local token="$2"
  local label="$3"
  local claim_status
  [[ -z "${E2_TRANSIENT_LOCK_PATH:-}" ]] || {
    print -u2 "cannot nest transient locks: ${E2_TRANSIENT_LOCK_PATH}"
    return 1
  }
  # Pre-register ownership so a signal arriving immediately after mkdir can
  # still remove this unique-token lock. A mismatched pre-existing owner is
  # protected by stackpomdp_release_owned_lock's token check.
  typeset -g E2_TRANSIENT_LOCK_PATH="$lock"
  typeset -g E2_TRANSIENT_LOCK_TOKEN="$token"
  typeset -g E2_TRANSIENT_LOCK_OWNED=1
  typeset -g E2_TRANSIENT_LOCK_LABEL="$label"
  stackpomdp_claim_owned_lock "$lock" "$token" "$label" || {
    claim_status=$?
    e2_clear_transient_lock_record
    return "$claim_status"
  }
  typeset -g E2_TRANSIENT_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"
}

function e2_release_transient_lock() {
  [[ -n "${E2_TRANSIENT_LOCK_PATH:-}" ]] || return 0
  stackpomdp_release_owned_lock \
    "$E2_TRANSIENT_LOCK_PATH" "$E2_TRANSIENT_LOCK_TOKEN" \
    "${E2_TRANSIENT_LOCK_OWNED:-0}" "$E2_TRANSIENT_LOCK_LABEL" || return $?
  e2_clear_transient_lock_record
}

function e2_release_active_locks() {
  local aggregate_status=0 release_status
  e2_release_transient_lock || {
    release_status=$?
    aggregate_status=$release_status
  }
  e2_release_pipeline_lock || {
    release_status=$?
    (( aggregate_status == 0 )) && aggregate_status=$release_status
  }
  return "$aggregate_status"
}

function e2_prepare_runtime() {
  local head worktree_status automation_status lock
  typeset -g E2_AUTOMATION_REVISION=$(git -C "$AUTOMATION_ROOT" rev-parse HEAD)
  [[ ${#E2_AUTOMATION_REVISION} -eq 40 \
      && "$E2_AUTOMATION_REVISION" != *[!0-9a-f]* ]] || {
    print -u2 "Atari automation has no full git revision: $AUTOMATION_ROOT"
    return 1
  }
  automation_status=$(git -C "$AUTOMATION_ROOT" status --porcelain -- \
    replication/atari/automation \
    replication/atari/evaluate_atari_meta_response_sb3.py \
    replication/atari/evaluate_atari_stackpomdp_leader_sb3.py \
    replication/atari/probe_atari_e1_seller_conditioning.py \
    replication/atari/probe_atari_e1_seller_threshold_residual.py \
    replication/atari/probe_atari_e1_seller_direct_threshold_residual.py \
    replication/atari/probe_atari_e1_seller_shared_context.py \
    replication/atari/train_atari_meta_response_sb3.py \
    replication/atari/train_atari_stackpomdp_leader_sb3.py \
    replication/atari/sb3_common.py \
    stackelberg_pomdp/atari)
  if [[ -n "$automation_status" ]]; then
    print -u2 "refusing automation from uncommitted Atari code: $AUTOMATION_ROOT"
    print -u2 "$automation_status"
    return 1
  fi
  if [[ ! -e "$E2_ACTIVE_CODE_ROOT" ]]; then
    # Isolate E2 from later changes on the active branch while all generated
    # checkpoints, reports, W&B files, and logs still go to the active repo.
    lock="${E2_ACTIVE_CODE_ROOT}.init.lock"
    local init_token="$(hostname)-$$-$(date +%s)-${RANDOM}"
    local init_status=0
    e2_claim_transient_lock \
      "$lock" "$init_token" "E2 code-worktree initialization" || return $?
    if [[ ! -e "$E2_ACTIVE_CODE_ROOT" ]]; then
      set +e
      git -C "$ROOT" worktree add --detach "$E2_ACTIVE_CODE_ROOT" "$E2_ACTIVE_EXPECTED_HEAD"
      init_status=$?
      set -e
    fi
    e2_release_transient_lock || return $?
    (( init_status == 0 )) || return "$init_status"
  elif [[ ! -d "$E2_ACTIVE_CODE_ROOT/.git" && ! -f "$E2_ACTIVE_CODE_ROOT/.git" ]]; then
    print -u2 "reserved E2 code path exists but is not a git worktree: $E2_ACTIVE_CODE_ROOT"
    return 1
  fi
  head=$(git -C "$E2_ACTIVE_CODE_ROOT" rev-parse HEAD)
  if [[ "$head" != "$E2_ACTIVE_EXPECTED_HEAD" ]]; then
    print -u2 "refusing E2 launch from unreviewed code: expected $E2_ACTIVE_EXPECTED_HEAD, observed $head"
    return 1
  fi
  worktree_status=$(git -C "$E2_ACTIVE_CODE_ROOT" status --short --untracked-files=no)
  if [[ -n "$worktree_status" ]]; then
    print -u2 "refusing E2 launch from a modified detached worktree: $E2_ACTIVE_CODE_ROOT"
    print -u2 "$worktree_status"
    return 1
  fi
  if [[ ! -x "$PYTHON" ]]; then
    print -u2 "Python environment is unavailable: $PYTHON"
    return 1
  fi
  if [[ ! -f "$VALIDATOR" ]]; then
    print -u2 "E2 integrity validator is unavailable: $VALIDATOR"
    return 1
  fi
  mkdir -p "$CHECKPOINT_ROOT" "$WANDB_ROOT" "$LOG_ROOT" "$E2_OUTPUT"
  cd "$E2_ACTIVE_CODE_ROOT"
  export STACKPOMDP_CODE_ROOT="$E2_ACTIVE_CODE_ROOT"
  export STACKPOMDP_SPACE_INVADERS_ROM="$ROM"
  export PYTHONPATH=.
  export PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export WANDB_MODE=online
  export WANDB_START_METHOD=thread
  export WANDB_DIR="$WANDB_ROOT"
  "$PYTHON" "$VALIDATOR" validate-rom --rom "$ROM" --sha256 "$ROM_SHA256"
}

function e2_wait_for_file() {
  # `path` is a special zsh array tied to PATH; never shadow it locally.
  local file_path="$1"
  local label="${2:-file}"
  while [[ ! -f "$file_path" ]]; do
    print "waiting for $label: $file_path"
    sleep 20
  done
}

function e2_wait_for_stable_zip() {
  local file_path="$1"
  local first second
  e2_wait_for_file "$file_path" "checkpoint ZIP"
  while true; do
    first=$(stat -f '%z:%m' "$file_path")
    sleep 10
    [[ -f "$file_path" ]] || continue
    second=$(stat -f '%z:%m' "$file_path")
    if [[ "$first" == "$second" ]] && unzip -tq "$file_path"; then
      return 0
    fi
    print "checkpoint is not yet stable; waiting: $file_path"
  done
}

function e2_refuse_path() {
  local file_path="$1"
  if [[ -e "$file_path" || -L "$file_path" ]]; then
    print -u2 "refusing to overwrite existing pipeline artifact: $file_path"
    return 1
  fi
}

function e2_refuse_role_pipeline_outputs() {
  local role="$1"
  local stem run_name selected selector_log step file_path
  local -a result_artifacts
  e2_role_paths "$role"
  stem="${E2_BASE%.zip}"
  for file_path in \
      "$E2_BASE" \
      "${stem}.evaluation.json" \
      "${stem}.provenance.json" \
      "${stem}.training.jsonl" \
      "$E2_INPUT_MANIFEST" \
      "$E2_TRAIN_LOG"; do
    e2_refuse_path "$file_path"
  done
  for step in $E2_STEPS; do
    e2_refuse_path "${stem}_step${step}.zip"
  done
  run_name="e2_${role}_balanced_all6_selector_v2_${E2_NAMESPACE}"
  selected="$CHECKPOINT_ROOT/leader_${role}_e2_ppo_balanced_seed1_firefix_retrain_${E2_NAMESPACE}_selected.zip"
  selector_log="$LOG_ROOT/${run_name}.log"
  e2_refuse_path "$selected"
  e2_refuse_path "$selector_log"
  result_artifacts=("$E2_OUTPUT/${run_name}".*(N))
  for file_path in $result_artifacts; do
    e2_refuse_path "$file_path"
  done
}

function e2_resolve_e1_gate() {
  local role="$1"
  local require_mode override choice exit_code found source_kind
  case "$role" in
    buyer)
      require_mode=""
      override="${E2_BUYER_GATE_REPORT:-}"
      ;;
    seller)
      require_mode=balanced
      override="${E2_SELLER_GATE_REPORT:-}"
      ;;
    *)
      print -u2 "unknown E1 role: $role"
      return 1
      ;;
  esac
  while true; do
    local -a command
    command=(
      "$PYTHON" "$VALIDATOR" discover-e1-gate
      --role "$role"
      --output-dir "$E1_OUTPUT"
    )
    [[ -n "$require_mode" ]] && command+=(--require-mode "$require_mode")
    [[ -n "$override" ]] && command+=(--override-report "$override")
    set +e
    choice=$("${command[@]}")
    exit_code=$?
    set -e
    if (( exit_code != 0 )); then
      print -u2 "strict E1 $role gate discovery failed"
      return "$exit_code"
    fi
    found=$(print -r -- "$choice" | jq -r '.found')
    if [[ "$found" == true ]]; then
      source_kind=$(print -r -- "$choice" | jq -r '.source_kind')
      if [[ "$role" == seller \
          && "$source_kind" != "$E1_REQUIRED_SELLER_SOURCE_KIND" ]]; then
        print "waiting for authoritative E1 seller source $E1_REQUIRED_SELLER_SOURCE_KIND"
        sleep 20
        continue
      fi
      case "$role" in
        buyer)
          typeset -g E1_BUYER_REPORT=$(print -r -- "$choice" | jq -r '.report')
          typeset -g E1_BUYER=$(print -r -- "$choice" | jq -r '.checkpoint')
          typeset -g E1_BUYER_MODE=$(print -r -- "$choice" | jq -r '.actor_loss_mode')
          typeset -g E1_BUYER_SOURCE_KIND=$(print -r -- "$choice" | jq -r '.source_kind')
          ;;
        seller)
          typeset -g E1_SELLER_REPORT=$(print -r -- "$choice" | jq -r '.report')
          typeset -g E1_SELLER=$(print -r -- "$choice" | jq -r '.checkpoint')
          typeset -g E1_SELLER_MODE=$(print -r -- "$choice" | jq -r '.actor_loss_mode')
          typeset -g E1_SELLER_SOURCE_KIND=$(print -r -- "$choice" | jq -r '.source_kind')
          ;;
      esac
      print "selected strict E1 $role gate: $choice"
      return 0
    fi
    print "waiting for a passing final E1 $role selector (failed reports remain immutable)"
    sleep 20
  done
}

function e2_load_e1_cohort() {
  local choice
  choice=$("$PYTHON" "$VALIDATOR" read-e1-gate-cohort \
    --cohort-manifest "$E1_COHORT")
  typeset -g E1_BUYER_REPORT=$(print -r -- "$choice" | jq -r '.e1_gates.buyer.report')
  typeset -g E1_BUYER=$(print -r -- "$choice" | jq -r '.e1_gates.buyer.checkpoint')
  typeset -g E1_BUYER_MODE=$(print -r -- "$choice" | jq -r '.e1_gates.buyer.actor_loss_mode')
  typeset -g E1_BUYER_SOURCE_KIND=$(print -r -- "$choice" | jq -r '.e1_gates.buyer.source_kind')
  typeset -g E1_SELLER_REPORT=$(print -r -- "$choice" | jq -r '.e1_gates.seller.report')
  typeset -g E1_SELLER=$(print -r -- "$choice" | jq -r '.e1_gates.seller.checkpoint')
  typeset -g E1_SELLER_MODE=$(print -r -- "$choice" | jq -r '.e1_gates.seller.actor_loss_mode')
  typeset -g E1_SELLER_SOURCE_KIND=$(print -r -- "$choice" | jq -r '.e1_gates.seller.source_kind')
  if [[ "$E1_SELLER_SOURCE_KIND" != "$E1_REQUIRED_SELLER_SOURCE_KIND" ]]; then
    print -u2 \
      "E2 cohort does not contain required seller source $E1_REQUIRED_SELLER_SOURCE_KIND"
    return 1
  fi
}

function e2_wait_for_both_e1_gates() {
  # One collision-safe cohort fixes both role launchers to identical E1 bytes.
  # The buyer must be the authoritative primary-economic release, and it must
  # be byte/report-identical to the buyer that authorized seller training.
  # The seller is balanced-only. Intermediate-step reports are never gates.
  if [[ -f "$E1_COHORT" ]]; then
    e2_load_e1_cohort
    return 0
  fi
  e2_resolve_e1_gate seller
  e2_resolve_e1_gate buyer

  local lock="${E1_COHORT}.init.lock"
  local token="$(hostname)-$$-$(date +%s)-${RANDOM}"
  e2_claim_transient_lock \
    "$lock" "$token" "E1 cohort initialization" || return $?
  if [[ ! -f "$E1_COHORT" ]]; then
    # Refresh under the cohort lock. The validator proves that these buyer
    # bytes are exactly those in the seller's immutable release manifest.
    set +e
    (
      set -e
      e2_resolve_e1_gate seller
      e2_resolve_e1_gate buyer
      "$PYTHON" "$VALIDATOR" write-e1-gate-cohort \
        --output "$E1_COHORT" \
        --buyer-report "$E1_BUYER_REPORT" \
        --buyer-checkpoint "$E1_BUYER" \
        --buyer-actor-loss-mode "$E1_BUYER_MODE" \
        --seller-report "$E1_SELLER_REPORT" \
        --seller-checkpoint "$E1_SELLER" \
        --seller-actor-loss-mode "$E1_SELLER_MODE"
    )
    local exit_code=$?
    set -e
    if ! e2_release_transient_lock; then
      print -u2 "failed to release E1 cohort initialization lock: $lock"
      return 1
    fi
    (( exit_code == 0 )) || return "$exit_code"
  else
    if ! e2_release_transient_lock; then
      print -u2 "failed to release E1 cohort initialization lock: $lock"
      return 1
    fi
  fi
  e2_load_e1_cohort
}

function e2_role_paths() {
  local role="$1"
  case "$role" in
    buyer)
      E2_RESPONSE="$E1_SELLER"
      E2_LEADER_E1="$E1_BUYER"
      ;;
    seller)
      E2_RESPONSE="$E1_BUYER"
      E2_LEADER_E1="$E1_SELLER"
      ;;
    *)
      print -u2 "unknown E2 role: $role"
      return 1
      ;;
  esac
  E2_BASE="$CHECKPOINT_ROOT/leader_${role}_e2_ppo_balanced_seed1_firefix_retrain_${E2_NAMESPACE}.zip"
  E2_INPUT_MANIFEST="${E2_BASE%.zip}.pipeline_inputs.json"
  E2_RUN_NAME="atari_clean_e2_${role}_balanced_seed1_firefix_retrain_2m_local_e1buyer${E1_BUYER_MODE}_${E1_BUYER_SOURCE_KIND}_e1seller${E1_SELLER_MODE}_${E1_SELLER_SOURCE_KIND}"
  E2_TRAIN_LOG="$LOG_ROOT/${E2_RUN_NAME}.log"
}

function e2_write_input_manifest() {
  local role="$1"
  e2_role_paths "$role"
  e2_refuse_path "$E2_INPUT_MANIFEST"
  "$PYTHON" "$VALIDATOR" write-e2-input-manifest \
    --role "$role" \
    --output "$E2_INPUT_MANIFEST" \
    --cohort-manifest "$E1_COHORT"
}

function e2_load_input_manifest() {
  local role="$1"
  local manifest choice
  manifest="$CHECKPOINT_ROOT/leader_${role}_e2_ppo_balanced_seed1_firefix_retrain_${E2_NAMESPACE}.pipeline_inputs.json"
  e2_wait_for_file "$manifest" "immutable E2 $role pipeline-input manifest"
  choice=$("$PYTHON" "$VALIDATOR" read-e2-input-manifest \
    --role "$role" --input-manifest "$manifest")
  typeset -g E1_BUYER_REPORT=$(print -r -- "$choice" | jq -r '.e1_gates.buyer.report')
  typeset -g E1_BUYER=$(print -r -- "$choice" | jq -r '.e1_gates.buyer.checkpoint')
  typeset -g E1_BUYER_MODE=$(print -r -- "$choice" | jq -r '.e1_gates.buyer.actor_loss_mode')
  typeset -g E1_BUYER_SOURCE_KIND=$(print -r -- "$choice" | jq -r '.e1_gates.buyer.source_kind')
  typeset -g E1_SELLER_REPORT=$(print -r -- "$choice" | jq -r '.e1_gates.seller.report')
  typeset -g E1_SELLER=$(print -r -- "$choice" | jq -r '.e1_gates.seller.checkpoint')
  typeset -g E1_SELLER_MODE=$(print -r -- "$choice" | jq -r '.e1_gates.seller.actor_loss_mode')
  typeset -g E1_SELLER_SOURCE_KIND=$(print -r -- "$choice" | jq -r '.e1_gates.seller.source_kind')
  e2_role_paths "$role"
  if [[ "$E2_INPUT_MANIFEST" != "$manifest" ]]; then
    print -u2 "E2 pipeline-input manifest path resolution is inconsistent"
    return 1
  fi
}

function e2_validate_checkpoint() {
  local role="$1"
  local checkpoint="$2"
  local timesteps="$3"
  e2_role_paths "$role"
  "$PYTHON" "$VALIDATOR" e2-checkpoint \
    --role "$role" \
    --checkpoint "$checkpoint" \
    --response "$E2_RESPONSE" \
    --leader-e1 "$E2_LEADER_E1" \
    --timesteps "$timesteps" \
    --input-manifest "$E2_INPUT_MANIFEST"
}

function e2_validate_report() {
  local role="$1"
  local report="$2"
  local selected="$3"
  local expected="$4"
  local stem step
  local -a candidate_args
  e2_role_paths "$role"
  stem="${E2_BASE%.zip}"
  candidate_args=()
  for step in $E2_STEPS; do
    candidate_args+=(--candidate "${stem}_step${step}.zip")
  done
  candidate_args+=(--candidate "$E2_BASE")
  "$PYTHON" "$VALIDATOR" e2-report \
    --role "$role" \
    --report "$report" \
    --selected "$selected" \
    --response "$E2_RESPONSE" \
    --input-manifest "$E2_INPUT_MANIFEST" \
    "${candidate_args[@]}" \
    --expect "$expected"
}

function e2_validate_family() {
  local role="$1"
  local stem step checkpoint
  local -a family_args
  e2_role_paths "$role"
  stem="${E2_BASE%.zip}"
  family_args=()
  for step in $E2_STEPS; do
    checkpoint="${stem}_step${step}.zip"
    family_args+=(--step-checkpoint "$checkpoint")
  done
  "$PYTHON" "$VALIDATOR" e2-family \
    --role "$role" \
    --response "$E2_RESPONSE" \
    --leader-e1 "$E2_LEADER_E1" \
    --input-manifest "$E2_INPUT_MANIFEST" \
    "${family_args[@]}" \
    --base-checkpoint "$E2_BASE"
}

function e2_existing_report_outcome() {
  local report="$1"
  local outcome
  outcome=$(jq -r 'if .passed == true then "passed" elif .passed == false then "failed" else "invalid" end' "$report")
  if [[ "$outcome" == invalid ]]; then
    print -u2 "existing E2 report has no Boolean outcome: $report"
    return 1
  fi
  print "$outcome"
}

function e2_run_final_selector() {
  local role="$1"
  local stem checkpoint run_name selected report log outcome exit_code step
  local -a checkpoint_args
  # Selectors use the immutable gate choices made before this training run;
  # they never rediscover a newer E1 artifact after seeing evaluation data.
  e2_load_input_manifest "$role"
  stem="${E2_BASE%.zip}"

  e2_wait_for_stable_zip "$E2_BASE"
  e2_wait_for_file "${stem}.evaluation.json" "completed E2 $role evaluation"
  "$PYTHON" "$VALIDATOR" e2-output \
    --role "$role" \
    --checkpoint "$E2_BASE" \
    --response "$E2_RESPONSE" \
    --leader-e1 "$E2_LEADER_E1" \
    --timesteps "$E2_TIMESTEPS" \
    --input-manifest "$E2_INPUT_MANIFEST"

  checkpoint_args=()
  for step in $E2_STEPS; do
    checkpoint="${stem}_step${step}.zip"
    e2_wait_for_stable_zip "$checkpoint"
    checkpoint_args+=(--checkpoint "$checkpoint")
  done
  e2_validate_family "$role"
  checkpoint_args+=(--checkpoint "$E2_BASE")

  run_name="e2_${role}_balanced_all6_selector_v2_${E2_NAMESPACE}"
  selected="$CHECKPOINT_ROOT/leader_${role}_e2_ppo_balanced_seed1_firefix_retrain_${E2_NAMESPACE}_selected.zip"
  report="$E2_OUTPUT/${run_name}.json"
  log="$LOG_ROOT/${run_name}.log"
  if [[ -f "$report" ]]; then
    outcome=$(e2_existing_report_outcome "$report")
    e2_validate_report "$role" "$report" "$selected" "$outcome"
    print "retaining validated existing final E2 $role report ($outcome): $report"
    [[ "$outcome" == passed ]] && return 0
    return 2
  fi
  e2_refuse_path "$selected"
  e2_refuse_path "$log"

  set +e
  "$PYTHON" -u -m replication.atari.evaluate_atari_stackpomdp_leader_sb3 \
    --leader-role "$role" \
    --response-checkpoint "$E2_RESPONSE" \
    "${checkpoint_args[@]}" \
    --selected-checkpoint "$selected" \
    --screen-episodes 20 \
    --screen-seed-start 4000001 \
    --confirmation-episodes 100 \
    --confirmation-seed-start 5000001 \
    --gameplay-horizon 200 \
    --event-tail-steps 0 \
    --noop-max 30 \
    --max-frames 100000 \
    --rom-path "$ROM" \
    --device cpu \
    --output-dir "$E2_OUTPUT" \
    --run-name "$run_name" \
    2>&1 | tee "$log"
  exit_code=$pipestatus[1]
  set -e
  case "$exit_code" in
    0)
      e2_validate_report "$role" "$report" "$selected" passed
      ;;
    2)
      e2_validate_report "$role" "$report" "$selected" failed
      print -u2 "final E2 $role family failed confirmation; no selected alias and no fallback"
      return 2
      ;;
    *)
      print -u2 "final E2 $role selector crashed with exit status $exit_code"
      return "$exit_code"
      ;;
  esac
}
