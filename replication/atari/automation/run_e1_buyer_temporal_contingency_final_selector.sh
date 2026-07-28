#!/bin/zsh
set -euo pipefail

source "${0:A:h}/atari_e1_temporal_contingency_common.zsh"
e1_activate
e1_prepare_runtime
e1_validate_existing_family

if [[ -f "$E1_SELECTOR_REPORT" ]]; then
  set +e
  (
    cd "$E1_CODE_ROOT"
    "$E1_PYTHON" "$E1_VALIDATOR" selection \
      --family "$E1_FAMILY" \
      --report "$E1_SELECTOR_REPORT" \
      --selected "$E1_SELECTED" \
      --gate-output "$E1_SELECTOR_GATE"
  )
  validation_status=$?
  set -e
  (( validation_status == 0 )) || exit "$validation_status"
  [[ "$(jq -r '.passed' "$E1_SELECTOR_REPORT")" == true ]] && exit 0
  exit 2
fi

mkdir -p "$E1_RESULT_ROOT" "$E1_LOG_ROOT"
arguments=()
while IFS= read -r candidate; do
  arguments+=(--checkpoint "$candidate")
done < <(e1_candidates)
e0b=$(jq -r '.e0b_source.path' "$E1_ACTIVATION")
export MPLCONFIGDIR=/private/tmp/mpl-stackpomdp-e1-temporal-selector

set +e
(
  cd "$E1_CODE_ROOT"
  "$E1_PYTHON" -u -m replication.atari.evaluate_atari_meta_response_sb3 \
    --role buyer \
    --e0b-checkpoint "$e0b" \
    "${arguments[@]}" \
    --selected-checkpoint "$E1_SELECTED" \
    --screen-seed-start 6000001 \
    --confirmation-seed-start 6100001 \
    --fixed-seed-start 6200001 \
    --timing-seed-start 6300001 \
    --rom-path "$E1_ROM" \
    --output-dir "$E1_RESULT_ROOT" \
    --run-name "$E1_SELECTOR_NAME" \
    --device cpu
) 2>&1 | tee "$E1_SELECTOR_LOG"
selector_status=$pipestatus[1]
set -e
if (( selector_status != 0 && selector_status != 2 )); then
  e1_die "temporal E1 selector failed with exit status $selector_status"
fi

(
  cd "$E1_CODE_ROOT"
  "$E1_PYTHON" "$E1_VALIDATOR" selection \
    --family "$E1_FAMILY" \
    --report "$E1_SELECTOR_REPORT" \
    --selected "$E1_SELECTED" \
    --gate-output "$E1_SELECTOR_GATE"
)
exit "$selector_status"
