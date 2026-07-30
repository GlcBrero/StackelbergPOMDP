#!/bin/zsh
set -euo pipefail

# Screen only the six v2 uniform-target checkpoints.  The v2 warm-up is never
# selectable, and only the screen winner receives fresh-seed confirmation.

source "${0:A:h}/atari_e1_seller_threshold_residual_recovery_common.zsh"

typeset -gx STACKPOMDP_E1R2_SELECTOR_GUARD_TOKEN="${STACKPOMDP_E1R2_SELECTOR_GUARD_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"
stackpomdp_claim_owned_lock "$E1R2_V1_GUARD_LOCK" \
  "$STACKPOMDP_E1R2_SELECTOR_GUARD_TOKEN" \
  "superseded v1 seller recovery selector guard" || exit $?
typeset -g E1R2_SELECTOR_GUARD_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"

typeset -gx STACKPOMDP_E1R2_SELECTOR_TOKEN="${STACKPOMDP_E1R2_SELECTOR_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"
stackpomdp_claim_owned_lock "$E1R2_LOCK" \
  "$STACKPOMDP_E1R2_SELECTOR_TOKEN" "v2 seller threshold-residual selector" || {
  lock_status=$?
  stackpomdp_release_owned_lock "$E1R2_V1_GUARD_LOCK" \
    "$STACKPOMDP_E1R2_SELECTOR_GUARD_TOKEN" \
    "$E1R2_SELECTOR_GUARD_OWNED" \
    "superseded v1 seller recovery selector guard" || :
  exit "$lock_status"
}
typeset -g E1R2_SELECTOR_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"

function release_e1r2_selector_lock() {
  local status=0
  e2_release_transient_lock || status=$?
  stackpomdp_release_owned_lock "$E1R2_LOCK" \
    "$STACKPOMDP_E1R2_SELECTOR_TOKEN" "$E1R2_SELECTOR_LOCK_OWNED" \
    "v2 seller threshold-residual selector" || status=$?
  typeset -g E1R2_SELECTOR_LOCK_OWNED=0
  stackpomdp_release_owned_lock "$E1R2_V1_GUARD_LOCK" \
    "$STACKPOMDP_E1R2_SELECTOR_GUARD_TOKEN" \
    "$E1R2_SELECTOR_GUARD_OWNED" \
    "superseded v1 seller recovery selector guard" || status=$?
  typeset -g E1R2_SELECTOR_GUARD_OWNED=0
  return "$status"
}

function interrupt_e1r2_selector() {
  local status="$1"
  trap - EXIT HUP INT TERM
  e1r2_cancel_active_job
  release_e1r2_selector_lock || :
  exit "$status"
}

trap 'release_e1r2_selector_lock' EXIT
trap 'interrupt_e1r2_selector 129' HUP
trap 'interrupt_e1r2_selector 130' INT
trap 'interrupt_e1r2_selector 143' TERM

e1r2_activate
e1r2_prepare_runtime
e1r2_validate_existing_family

if [[ -f "$E1R2_REPORT" && ! -L "$E1R2_REPORT" ]]; then
  outcome=$(jq -r 'if .passed == true then "passed" elif .passed == false then "failed" else "invalid" end' "$E1R2_REPORT")
  case "$outcome" in
    passed)
      [[ -f "$E1R2_SELECTED" && ! -L "$E1R2_SELECTED" ]] || e1r2_die "passing v2 report lacks selected alias"
      if [[ ! -e "$E1R2_GATE" && ! -L "$E1R2_GATE" ]]; then
        (
          cd "$E1R2_CODE_ROOT"
          "$E1R2_PYTHON" "$E1R2_VALIDATOR" selection \
            --family "$E1R2_FAMILY" \
            --report "$E1R2_REPORT" \
            --selected "$E1R2_SELECTED" \
            --gate-output "$E1R2_GATE"
        )
      fi
      e1r2_validate_existing_gate
      exit 0
      ;;
    failed)
      [[ ! -e "$E1R2_SELECTED" && ! -L "$E1R2_SELECTED" \
          && ! -e "$E1R2_GATE" && ! -L "$E1R2_GATE" ]] || \
        e1r2_die "failed v2 selector retained alias or gate"
      (
        cd "$E1R2_CODE_ROOT"
        "$E1R2_PYTHON" "$E1R2_VALIDATOR" selection \
          --family "$E1R2_FAMILY" \
          --report "$E1R2_REPORT" \
          --selected "$E1R2_SELECTED"
      )
      exit 2
      ;;
    *) e1r2_die "existing v2 report has no Boolean outcome" ;;
  esac
fi

e1r2_refuse_path "$E1R2_SELECTED"
e1r2_refuse_path "$E1R2_GATE"
e1r2_refuse_path "$E1R2_SELECTOR_LOG"

typeset -a checkpoint_arguments
checkpoint_arguments=()
while IFS= read -r candidate; do
  e1r2_wait_for_stable_zip "$candidate"
  checkpoint_arguments+=(--checkpoint "$candidate")
done < <(e1r2_candidate_arguments)

revision=$(jq -r '.code_revision' "$E1R2_ACTIVATION")
export MPLCONFIGDIR=/private/tmp/mpl-stackpomdp-e1-seller-threshold-residual-v2
set +e
(
  set +e
  (
    cd "$E1R2_CODE_ROOT"
    "$E1R2_PYTHON" -u -m replication.atari.evaluate_atari_meta_response_sb3 \
      --role seller \
      --e0b-checkpoint "$E1R2_E0B" \
      "${checkpoint_arguments[@]}" \
      --selected-checkpoint "$E1R2_SELECTED" \
      --screen-seed-start 9000001 \
      --confirmation-seed-start 9100001 \
      --fixed-seed-start 9200001 \
      --timing-seed-start 9300001 \
      --selector-code-revision "$revision" \
      --rom-path "$ROM" \
      --output-dir "$E1_OUTPUT" \
      --run-name "$E1R2_SELECTOR_NAME" \
      --device cpu
  ) 2>&1 | tee "$E1R2_SELECTOR_LOG"
  pipeline_status=(${pipestatus[@]})
  (( pipeline_status[2] == 0 )) || exit "${pipeline_status[2]}"
  exit "${pipeline_status[1]}"
) &
E1R2_ACTIVE_JOB_PID=$!
wait "$E1R2_ACTIVE_JOB_PID"
selector_status=$?
E1R2_ACTIVE_JOB_PID=0
set -e

case "$selector_status" in
  0)
    (
      cd "$E1R2_CODE_ROOT"
      "$E1R2_PYTHON" "$E1R2_VALIDATOR" selection \
        --family "$E1R2_FAMILY" \
        --report "$E1R2_REPORT" \
        --selected "$E1R2_SELECTED" \
        --gate-output "$E1R2_GATE"
    )
    e1r2_validate_existing_gate
    ;;
  2)
    (
      cd "$E1R2_CODE_ROOT"
      "$E1R2_PYTHON" "$E1R2_VALIDATOR" selection \
        --family "$E1R2_FAMILY" \
        --report "$E1R2_REPORT" \
        --selected "$E1R2_SELECTED"
    )
    [[ ! -e "$E1R2_SELECTED" && ! -L "$E1R2_SELECTED" \
        && ! -e "$E1R2_GATE" && ! -L "$E1R2_GATE" ]] || \
      e1r2_die "failed v2 selector retained alias or gate"
    ;;
  *)
    print -u2 "v2 seller selector crashed with status $selector_status"
    exit "$selector_status"
    ;;
esac
exit "$selector_status"
