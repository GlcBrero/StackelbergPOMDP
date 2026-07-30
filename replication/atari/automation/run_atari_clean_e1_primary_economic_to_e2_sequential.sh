#!/bin/zsh
set -euo pipefail

# One unattended local chain: certify the preregistered primary-economic E1
# buyer, train/select the E1 seller it releases, then run both E2 leaders. All
# trainers retain their own live W&B configuration. Exit 2 is a scientific
# gate failure and stops the chain; operational failures retain their status.
source "${0:A:h}/atari_e2_pipeline_common.zsh"
e2_claim_pipeline_lock
typeset -gi PRIMARY_E2_ACTIVE_STAGE_PID=0

function terminate_primary_e2_process_tree() {
  local parent_pid="$1"
  local signal_name="${2:-TERM}"
  local children child
  [[ "$parent_pid" == <-> ]] || return 0
  children=$(pgrep -P "$parent_pid" 2>/dev/null) || children=""
  for child in ${(f)children}; do
    [[ "$child" == <-> ]] || continue
    terminate_primary_e2_process_tree "$child" "$signal_name"
  done
  kill -s "$signal_name" "$parent_pid" 2>/dev/null || :
}

function interrupt_primary_e2_pipeline() {
  local status="$1"
  trap - EXIT HUP INT TERM
  if (( PRIMARY_E2_ACTIVE_STAGE_PID > 0 )); then
    terminate_primary_e2_process_tree "$PRIMARY_E2_ACTIVE_STAGE_PID" TERM
    wait "$PRIMARY_E2_ACTIVE_STAGE_PID" 2>/dev/null || :
    PRIMARY_E2_ACTIVE_STAGE_PID=0
  fi
  e2_release_active_locks || \
    print -u2 "failed to release an E2 lock after interrupt"
  exit "$status"
}

trap 'e2_release_active_locks || print -u2 "failed to release an E2 lock"' EXIT
trap 'interrupt_primary_e2_pipeline 129' HUP
trap 'interrupt_primary_e2_pipeline 130' INT
trap 'interrupt_primary_e2_pipeline 143' TERM

typeset -gr PRIMARY_RELEASE="$AUTOMATION_DIR/run_atari_clean_e1_buyer_primary_economic_release.sh"
typeset -gr SELLER_STAGE="$AUTOMATION_DIR/run_atari_clean_e1_seller_conditioning_recovery.sh"
typeset -gr E2_STAGE="$AUTOMATION_DIR/run_atari_clean_e2_sequential.sh"

function run_required_stage() {
  local label="$1"
  local launcher="$2"
  local stage_status
  [[ -x "$launcher" ]] || {
    print -u2 "$label launcher is unavailable or not executable: $launcher"
    return 1
  }
  print "starting $label"
  set +e
  "$launcher" &
  PRIMARY_E2_ACTIVE_STAGE_PID=$!
  wait "$PRIMARY_E2_ACTIVE_STAGE_PID"
  stage_status=$?
  PRIMARY_E2_ACTIVE_STAGE_PID=0
  set -e
  case "$stage_status" in
    0)
      print "completed $label"
      ;;
    2)
      print -u2 "$label failed its scientific gate; downstream stages remain closed"
      return 2
      ;;
    *)
      print -u2 "$label failed operationally with exit status $stage_status"
      return "$stage_status"
      ;;
  esac
}

run_required_stage "E1 primary-economic buyer confirmation" "$PRIMARY_RELEASE"
run_required_stage "E1 seller conditioning recovery and selection" "$SELLER_STAGE"
run_required_stage "sequential E2 buyer/seller training and selection" "$E2_STAGE"

print "completed the primary-economic E1 -> seller -> E2 local pipeline"
