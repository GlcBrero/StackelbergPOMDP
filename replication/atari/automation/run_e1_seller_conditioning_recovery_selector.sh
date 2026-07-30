#!/bin/zsh
set -euo pipefail

# One common-screen, winner-only confirmation over the six preregistered
# target-stage checkpoints.  The all-equal warm-up is not passed to the
# evaluator and can never become a selected fallback.

source "${0:A:h}/atari_e1_seller_conditioning_recovery_common.zsh"

typeset -gx STACKPOMDP_E1R_SELECTOR_LOCK_TOKEN="${STACKPOMDP_E1R_SELECTOR_LOCK_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"
stackpomdp_claim_owned_lock \
  "$E1R_LOCK" "$STACKPOMDP_E1R_SELECTOR_LOCK_TOKEN" \
  "E1 seller recovery selector" || exit $?
typeset -g E1R_SELECTOR_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"
function release_e1r_selector_lock() {
  e2_release_transient_lock || return $?
  stackpomdp_release_owned_lock \
    "$E1R_LOCK" "$STACKPOMDP_E1R_SELECTOR_LOCK_TOKEN" \
    "$E1R_SELECTOR_LOCK_OWNED" "E1 seller recovery selector" || return $?
  typeset -g E1R_SELECTOR_LOCK_OWNED=0
}
function interrupt_e1r_selector() {
  local exit_status="$1"
  trap - EXIT HUP INT TERM
  e1r_cancel_active_job
  release_e1r_selector_lock || :
  exit "$exit_status"
}
trap 'release_e1r_selector_lock' EXIT
trap 'interrupt_e1r_selector 129' HUP
trap 'interrupt_e1r_selector 130' INT
trap 'interrupt_e1r_selector 143' TERM

e1r_activate
e1r_prepare_runtime
e1r_validate_existing_family

if [[ -f "$E1R_REPORT" && ! -L "$E1R_REPORT" ]]; then
  outcome=$(jq -r 'if .passed == true then "passed" elif .passed == false then "failed" else "invalid" end' "$E1R_REPORT")
  case "$outcome" in
    passed)
      [[ -f "$E1R_SELECTED" && ! -L "$E1R_SELECTED" ]] || \
        e1r_die "passing recovery report has no selected checkpoint alias"
      if [[ ! -e "$E1R_GATE" && ! -L "$E1R_GATE" ]]; then
        # The evaluator atomically publishes the passing report and alias
        # before the deterministic validator publishes the gate.  Resume only
        # this byte-validating publication step after an interruption.
        (
          cd "$E1R_CODE_ROOT"
          "$E1R_PYTHON" "$E1R_VALIDATOR" selection \
            --family "$E1R_FAMILY" \
            --report "$E1R_REPORT" \
            --selected "$E1R_SELECTED" \
            --gate-output "$E1R_GATE"
        )
      fi
      e1r_validate_existing_gate
      exit 0
      ;;
    failed)
      [[ ! -e "$E1R_SELECTED" && ! -L "$E1R_SELECTED" \
          && ! -e "$E1R_GATE" && ! -L "$E1R_GATE" ]] || \
        e1r_die "failed recovery selector retained an alias or gate"
      exit 2
      ;;
    *)
      e1r_die "existing recovery report has no Boolean outcome"
      ;;
  esac
fi

e1r_refuse_path "$E1R_SELECTED"
e1r_refuse_path "$E1R_GATE"
e1r_refuse_path "$E1R_SELECTOR_LOG"

typeset -a checkpoint_arguments
checkpoint_arguments=()
while IFS= read -r candidate; do
  e1r_wait_for_stable_zip "$candidate"
  checkpoint_arguments+=(--checkpoint "$candidate")
done < <(e1r_candidate_arguments)

revision=$(jq -r '.code_revision' "$E1R_ACTIVATION")
export MPLCONFIGDIR=/private/tmp/mpl-stackpomdp-e1-seller-conditioning-recovery
set +e
(
  set +e
  (
    cd "$E1R_CODE_ROOT"
    "$E1R_PYTHON" -u -m replication.atari.evaluate_atari_meta_response_sb3 \
      --role seller \
      --e0b-checkpoint "$E1R_E0B" \
      "${checkpoint_arguments[@]}" \
      --selected-checkpoint "$E1R_SELECTED" \
      --screen-seed-start 9000001 \
      --confirmation-seed-start 9100001 \
      --fixed-seed-start 9200001 \
      --timing-seed-start 9300001 \
      --selector-code-revision "$revision" \
      --rom-path "$ROM" \
      --output-dir "$E1_OUTPUT" \
      --run-name "$E1R_SELECTOR_NAME" \
      --device cpu
  ) 2>&1 | tee "$E1R_SELECTOR_LOG"
  pipeline_status=(${pipestatus[@]})
  (( pipeline_status[2] == 0 )) || exit "${pipeline_status[2]}"
  exit "${pipeline_status[1]}"
) &
E1R_ACTIVE_JOB_PID=$!
wait "$E1R_ACTIVE_JOB_PID"
selector_status=$?
E1R_ACTIVE_JOB_PID=0
set -e

case "$selector_status" in
  0)
    (
      cd "$E1R_CODE_ROOT"
      "$E1R_PYTHON" "$E1R_VALIDATOR" selection \
        --family "$E1R_FAMILY" \
        --report "$E1R_REPORT" \
        --selected "$E1R_SELECTED" \
        --gate-output "$E1R_GATE"
    )
    e1r_validate_existing_gate
    ;;
  2)
    (
      cd "$E1R_CODE_ROOT"
      "$E1R_PYTHON" "$E1R_VALIDATOR" selection \
        --family "$E1R_FAMILY" \
        --report "$E1R_REPORT" \
        --selected "$E1R_SELECTED"
    )
    [[ ! -e "$E1R_SELECTED" && ! -L "$E1R_SELECTED" \
        && ! -e "$E1R_GATE" && ! -L "$E1R_GATE" ]] || \
      e1r_die "failed recovery selector retained an alias or gate"
    ;;
  *)
    print -u2 "seller recovery selector crashed with exit status $selector_status"
    exit "$selector_status"
    ;;
esac
exit "$selector_status"
