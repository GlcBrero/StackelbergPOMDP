#!/bin/zsh
set -euo pipefail

# One release-bound, all-six E1 seller selection.  This replaces the legacy
# per-checkpoint diagnostic waiters; only the common screen winner receives one
# fresh confirmation and no lower-ranked fallback is permitted.
source "${0:A:h}/atari_e2_pipeline_common.zsh"
e2_prepare_runtime

typeset -gr SELECTOR_CODE_ROOT="$AUTOMATION_ROOT"
typeset -gr E0B="$CHECKPOINT_ROOT/space_invaders_e0b_ppo_seed1_firefix_retrain_selected.zip"
typeset -gr BASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_seed1_firefix_retrain.zip"
typeset -gr RELEASE="${BASE%.zip}.buyer_gate.json"
typeset -gr FINAL_EVALUATION="${BASE%.zip}.evaluation.json"
typeset -gr RUN_NAME=e1_seller_balanced_all6_selector_v2
typeset -gr REPORT="$E1_OUTPUT/${RUN_NAME}.json"
typeset -gr GATE="${REPORT%.json}.gate.json"
typeset -gr SELECTED="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_seed1_firefix_retrain_selected.zip"
typeset -gr SELECTOR_LOG="$LOG_ROOT/${RUN_NAME}.log"
typeset -gr SELECTOR_LOCK=/private/tmp/stackpomdp-atari-e1-seller-selector.lock
typeset -gra SELLER_STEPS=(400160 800320 1200480 1600640 2000800)
typeset -gx STACKPOMDP_E1_SELLER_SELECTOR_LOCK_TOKEN="${STACKPOMDP_E1_SELLER_SELECTOR_LOCK_TOKEN:-$(hostname)-$$-${EPOCHSECONDS}-${RANDOM}}"

stackpomdp_claim_owned_lock \
  "$SELECTOR_LOCK" "$STACKPOMDP_E1_SELLER_SELECTOR_LOCK_TOKEN" \
  "E1 seller selector" || exit $?
typeset -g SELECTOR_LOCK_OWNED_BY_CALLER="$STACKPOMDP_LOCK_RESULT_OWNED"
function release_selector_lock() {
  stackpomdp_release_owned_lock \
    "$SELECTOR_LOCK" "$STACKPOMDP_E1_SELLER_SELECTOR_LOCK_TOKEN" \
    "$SELECTOR_LOCK_OWNED_BY_CALLER" "E1 seller selector" || return $?
  typeset -g SELECTOR_LOCK_OWNED_BY_CALLER=0
}
trap 'release_selector_lock' EXIT
trap 'release_selector_lock; exit 130' INT
trap 'release_selector_lock; exit 143' TERM

e2_wait_for_file "$RELEASE" "immutable E1 seller-release manifest"
"$PYTHON" "$VALIDATOR" read-e1-seller-release \
  --release-manifest "$RELEASE"
e2_wait_for_stable_zip "$E0B"
e2_wait_for_file "$FINAL_EVALUATION" "completed E1 seller evaluation"

checkpoint_args=()
for step in $SELLER_STEPS; do
  checkpoint="${BASE%.zip}_step${step}.zip"
  e2_wait_for_stable_zip "$checkpoint"
  checkpoint_args+=(--checkpoint "$checkpoint")
done
e2_wait_for_stable_zip "$BASE"
checkpoint_args+=(--checkpoint "$BASE")

if [[ -f "$REPORT" ]]; then
  outcome=$(jq -r 'if .passed == true then "passed" elif .passed == false then "failed" else "invalid" end' "$REPORT")
  case "$outcome" in
    passed)
      if [[ ! -f "$GATE" ]]; then
        "$PYTHON" "$VALIDATOR" write-e1-seller-selection-gate \
          --output "$GATE" \
          --report "$REPORT" \
          --selected "$SELECTED" \
          --release-manifest "$RELEASE" \
          --selector-code-root "$SELECTOR_CODE_ROOT"
      fi
      "$PYTHON" "$VALIDATOR" read-e1-seller-selection-gate --gate "$GATE"
      exit 0
      ;;
    failed)
      [[ ! -e "$SELECTED" && ! -L "$SELECTED" ]] || {
        print -u2 "failed E1 seller report retained a selected alias"
        exit 1
      }
      [[ ! -e "$GATE" && ! -L "$GATE" ]] || {
        print -u2 "failed E1 seller report retained a gate sidecar"
        exit 1
      }
      exit 2
      ;;
    *)
      print -u2 "existing E1 seller report has no Boolean outcome: $REPORT"
      exit 1
      ;;
  esac
fi

e2_refuse_path "$SELECTED"
e2_refuse_path "$GATE"
e2_refuse_path "$SELECTOR_LOG"
export MPLCONFIGDIR=/private/tmp/mpl-stackpomdp-e1-seller-final-selector

set +e
(
  cd "$SELECTOR_CODE_ROOT"
  PYTHONPATH="$SELECTOR_CODE_ROOT" \
    PYTHONNOUSERSITE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    "$PYTHON" -u -m replication.atari.evaluate_atari_meta_response_sb3 \
      --role seller \
      --e0b-checkpoint "$E0B" \
      "${checkpoint_args[@]}" \
      --selected-checkpoint "$SELECTED" \
      --screen-seed-start 3500001 \
      --confirmation-seed-start 3600001 \
      --fixed-seed-start 3700001 \
      --timing-seed-start 3800001 \
      --selector-code-revision "$E2_AUTOMATION_REVISION" \
      --rom-path "$ROM" \
      --output-dir "$E1_OUTPUT" \
      --run-name "$RUN_NAME" \
      --device cpu
) 2>&1 | tee "$SELECTOR_LOG"
selector_status=$pipestatus[1]
set -e

case "$selector_status" in
  0)
    "$PYTHON" "$VALIDATOR" write-e1-seller-selection-gate \
      --output "$GATE" \
      --report "$REPORT" \
      --selected "$SELECTED" \
      --release-manifest "$RELEASE" \
      --selector-code-root "$SELECTOR_CODE_ROOT"
    "$PYTHON" "$VALIDATOR" read-e1-seller-selection-gate --gate "$GATE"
    ;;
  2)
    [[ "$(jq -r '.passed' "$REPORT")" == false ]] || {
      print -u2 "E1 seller selector exited 2 without a failed report"
      exit 1
    }
    [[ ! -e "$SELECTED" && ! -L "$SELECTED" \
        && ! -e "$GATE" && ! -L "$GATE" ]] || {
      print -u2 "failed E1 seller selection retained an alias or gate"
      exit 1
    }
    ;;
  *)
    print -u2 "E1 seller selector crashed with exit status $selector_status"
    exit "$selector_status"
    ;;
esac
exit "$selector_status"
