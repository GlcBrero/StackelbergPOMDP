#!/bin/zsh
set -euo pipefail

# Run both E2 roles sequentially on the local CPU.  One global token prevents
# standalone role launchers from racing this pipeline.  Scientific selector
# failure (exit 2) is retained as a result and does not suppress the other role.
source "${0:A:h}/atari_e2_pipeline_common.zsh"
e2_claim_pipeline_lock
trap 'e2_release_active_locks || print -u2 "failed to release an E2 lock"' EXIT
trap 'e2_release_active_locks || print -u2 "failed to release an E2 lock"; exit 129' HUP
trap 'e2_release_active_locks || print -u2 "failed to release an E2 lock"; exit 130' INT
trap 'e2_release_active_locks || print -u2 "failed to release an E2 lock"; exit 143' TERM
e2_prepare_runtime

typeset -gr SUMMARY="$E2_OUTPUT/e2_sequential_orchestration_v1_${E2_NAMESPACE}.json"
typeset -A training_status selector_status

e2_wait_for_both_e1_gates
e2_refuse_path "$SUMMARY"
# Preflight both complete namespaces before the first trainer can mutate one.
e2_refuse_role_pipeline_outputs buyer
e2_refuse_role_pipeline_outputs seller

for role in buyer seller; do
  trainer="$AUTOMATION_DIR/run_atari_clean_e2_${role}_balanced_2m.sh"
  selector="$AUTOMATION_DIR/run_e2_${role}_balanced_final_selector.sh"
  set +e
  "$trainer"
  training_status[$role]=$?
  set -e
  if (( training_status[$role] != 0 )); then
    selector_status[$role]=-1
    print -u2 "E2 $role training failed with exit ${training_status[$role]}; continuing with the other role"
    continue
  fi

  set +e
  "$selector"
  selector_status[$role]=$?
  set -e
  case "${selector_status[$role]}" in
    0)
      print "E2 $role final scientific gate passed"
      ;;
    2)
      print -u2 "E2 $role final scientific gate failed; retaining the immutable result"
      ;;
    *)
      print -u2 "E2 $role selector crashed with exit ${selector_status[$role]}; continuing with the other role"
      ;;
  esac
done

"$PYTHON" "$VALIDATOR" write-e2-orchestration-summary \
  --output "$SUMMARY" \
  --cohort-manifest "$E1_COHORT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --result-root "$E2_OUTPUT" \
  --automation-code-root "$AUTOMATION_ROOT" \
  --buyer-training-exit-code "${training_status[buyer]}" \
  --buyer-selector-exit-code "${selector_status[buyer]}" \
  --seller-training-exit-code "${training_status[seller]}" \
  --seller-selector-exit-code "${selector_status[seller]}"
"$PYTHON" "$VALIDATOR" read-e2-orchestration-summary --summary "$SUMMARY"

operational_failure=0
scientific_failure=0
for role in buyer seller; do
  (( training_status[$role] == 0 )) || operational_failure=1
  case "${selector_status[$role]}" in
    0) ;;
    2) scientific_failure=1 ;;
    *) operational_failure=1 ;;
  esac
done
(( operational_failure == 0 )) || exit 1
(( scientific_failure == 0 )) || exit 2
exit 0
