#!/bin/zsh
set -euo pipefail

# Bind the six formal v5 checkpoints, screen all six, and confirm only the
# screen winner on fresh held-out seeds.  No fallback is permitted.

source "${0:A:h}/atari_e1_seller_shared_context_v5_selection_common.zsh"

typeset -gx STACKPOMDP_E1V5_SELECTOR_TOKEN="${STACKPOMDP_E1V5_SELECTOR_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"
stackpomdp_claim_owned_lock \
  "$E1V5_LOCK" "$STACKPOMDP_E1V5_SELECTOR_TOKEN" \
  "seller-v5 formal selector" || exit $?
typeset -g E1V5_SELECTOR_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"

function release_e1v5_selector_lock() {
  local cleanup_status=0
  e2_release_transient_lock || cleanup_status=$?
  stackpomdp_release_owned_lock \
    "$E1V5_LOCK" "$STACKPOMDP_E1V5_SELECTOR_TOKEN" \
    "$E1V5_SELECTOR_LOCK_OWNED" "seller-v5 formal selector" || \
    cleanup_status=$?
  typeset -g E1V5_SELECTOR_LOCK_OWNED=0
  return "$cleanup_status"
}

function interrupt_e1v5_selector() {
  local exit_status="$1"
  trap - EXIT HUP INT TERM
  e1v5_cancel_active_job
  release_e1v5_selector_lock || :
  exit "$exit_status"
}

trap 'release_e1v5_selector_lock' EXIT
trap 'interrupt_e1v5_selector 129' HUP
trap 'interrupt_e1v5_selector 130' INT
trap 'interrupt_e1v5_selector 143' TERM

# Revalidate the immutable diagnostics gate in the exact formal-training
# runtime before moving to the newer selector runtime.  For exposure-v2 this
# proves the preregistered c4a7dcd evidence -> 180c84f validator-only bridge,
# including the shortened W&B job-type contract, without reinterpreting the
# diagnostic evidence under later selector code.
e1v5_prepare_runtime
e1v5s_prepare_runtime
(
  cd "$E1V5S_CODE_ROOT"
  "$E1V5_PYTHON" "$E1V5S_DIAGNOSTICS_VALIDATOR" \
    "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" validate-gate \
    --gate "$E1V5_GATE"
)

typeset -a candidate_args
candidate_args=()
while IFS= read -r candidate; do
  e1v5_wait_for_stable_zip "$candidate"
  candidate_args+=(--candidate "$candidate")
done < <(e1v5s_candidate_arguments)

if [[ ! -f "$E1V5S_FAMILY" || -L "$E1V5S_FAMILY" ]]; then
  e1v5_refuse_path "$E1V5S_FAMILY"
  (
    cd "$E1V5S_CODE_ROOT"
    "$E1V5_PYTHON" "$E1V5S_VALIDATOR" \
      "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" formal-family \
      --diagnostics-gate "$E1V5_GATE" \
      "${candidate_args[@]}" \
      --training-log "$E1V5_FORMAL_TRACE" \
      --evaluation "$E1V5_FORMAL_EVALUATION" \
      --console-log "$E1V5_FORMAL_LOG" \
      --code-root "$E1V5S_CODE_ROOT" \
      --output "$E1V5S_FAMILY"
  )
fi
e1v5s_validate_family

if [[ -f "$E1V5S_REPORT" && ! -L "$E1V5S_REPORT" ]]; then
  outcome=$(jq -r \
    'if .passed == true then "passed" elif .passed == false then "failed" else "invalid" end' \
    "$E1V5S_REPORT")
  case "$outcome" in
    passed)
      [[ -f "$E1V5S_SELECTED" && ! -L "$E1V5S_SELECTED" ]] || \
        e1v5_die "passing seller-v5 report lacks its selected alias"
      if [[ ! -e "$E1V5S_GATE" && ! -L "$E1V5S_GATE" ]]; then
        (
          cd "$E1V5S_CODE_ROOT"
          "$E1V5_PYTHON" "$E1V5S_VALIDATOR" \
            "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" selection \
            --family "$E1V5S_FAMILY" \
            --report "$E1V5S_REPORT" \
            --selected "$E1V5S_SELECTED" \
            --gate-output "$E1V5S_GATE"
        )
      fi
      e1v5s_validate_gate
      exit 0
      ;;
    failed)
      [[ ! -e "$E1V5S_SELECTED" && ! -L "$E1V5S_SELECTED" \
          && ! -e "$E1V5S_GATE" && ! -L "$E1V5S_GATE" ]] || \
        e1v5_die "failed seller-v5 selector retained alias or gate"
      exit 2
      ;;
    *) e1v5_die "existing seller-v5 report has no Boolean outcome" ;;
  esac
fi

e1v5_refuse_path "$E1V5S_SELECTED"
e1v5_refuse_path "$E1V5S_GATE"
e1v5_refuse_path "$E1V5S_LOG"

revision=$(git -C "$E1V5S_CODE_ROOT" rev-parse HEAD)
export MPLCONFIGDIR="/private/tmp/mpl-stackpomdp-${E1V5S_RUN_NAME}"
set +e
(
  set +e
  (
    cd "$E1V5S_CODE_ROOT"
    "$E1V5_PYTHON" -u -m \
      replication.atari.evaluate_atari_meta_response_sb3 \
      --role seller \
      --e0b-checkpoint "$E1V5_E0B" \
      "${candidate_args[@]/--candidate/--checkpoint}" \
      --selected-checkpoint "$E1V5S_SELECTED" \
      --screen-seed-start "$E1V5S_SCREEN_SEED_START" \
      --confirmation-seed-start "$E1V5S_CONFIRMATION_SEED_START" \
      --fixed-seed-start "$E1V5S_FIXED_SEED_START" \
      --timing-seed-start "$E1V5S_TIMING_SEED_START" \
      --selector-code-revision "$revision" \
      --rom-path "$ROM" \
      --output-dir "$E1_OUTPUT" \
      --run-name "$E1V5S_RUN_NAME" \
      --device cpu
  ) 2>&1 | tee "$E1V5S_LOG"
  pipeline_status=(${pipestatus[@]})
  (( pipeline_status[2] == 0 )) || exit "${pipeline_status[2]}"
  exit "${pipeline_status[1]}"
) &
E1V5_ACTIVE_JOB_PID=$!
wait "$E1V5_ACTIVE_JOB_PID"
selector_status=$?
E1V5_ACTIVE_JOB_PID=0
set -e

case "$selector_status" in
  0)
    (
      cd "$E1V5S_CODE_ROOT"
      "$E1V5_PYTHON" "$E1V5S_VALIDATOR" \
        "${E1V5_VALIDATOR_PROTOCOL_ARGS[@]}" selection \
        --family "$E1V5S_FAMILY" \
        --report "$E1V5S_REPORT" \
        --selected "$E1V5S_SELECTED" \
        --gate-output "$E1V5S_GATE"
    )
    e1v5s_validate_gate
    ;;
  2)
    [[ ! -e "$E1V5S_SELECTED" && ! -L "$E1V5S_SELECTED" \
        && ! -e "$E1V5S_GATE" && ! -L "$E1V5S_GATE" ]] || \
      e1v5_die "failed seller-v5 selector retained alias or gate"
    print -u2 "seller-v5 screen winner failed fresh confirmation; E2 remains closed"
    ;;
  *)
    print -u2 "seller-v5 selector crashed with status $selector_status"
    exit "$selector_status"
    ;;
esac
exit "$selector_status"
