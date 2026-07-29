#!/bin/zsh
set -euo pipefail

# One unattended local chain: certify the preregistered primary-economic E1
# buyer, train/select the E1 seller it releases, then run both E2 leaders. All
# trainers retain their own live W&B configuration. Exit 2 is a scientific
# gate failure and stops the chain; operational failures retain their status.
source "${0:A:h}/atari_e2_pipeline_common.zsh"
e2_claim_pipeline_lock
trap 'e2_release_pipeline_lock' EXIT

typeset -gr PRIMARY_RELEASE="$AUTOMATION_DIR/run_atari_clean_e1_buyer_primary_economic_release.sh"
typeset -gr SELLER_STAGE="$AUTOMATION_DIR/run_atari_clean_e1_seller_after_buyer_gate.sh"
typeset -gr E2_STAGE="$AUTOMATION_DIR/run_atari_clean_e2_sequential.sh"

function run_required_stage() {
  local label="$1"
  local launcher="$2"
  local status
  [[ -x "$launcher" ]] || {
    print -u2 "$label launcher is unavailable or not executable: $launcher"
    return 1
  }
  print "starting $label"
  set +e
  "$launcher"
  status=$?
  set -e
  case "$status" in
    0)
      print "completed $label"
      ;;
    2)
      print -u2 "$label failed its scientific gate; downstream stages remain closed"
      return 2
      ;;
    *)
      print -u2 "$label failed operationally with exit status $status"
      return "$status"
      ;;
  esac
}

run_required_stage "E1 primary-economic buyer confirmation" "$PRIMARY_RELEASE"
run_required_stage "E1 seller training and selection" "$SELLER_STAGE"
run_required_stage "sequential E2 buyer/seller training and selection" "$E2_STAGE"

print "completed the primary-economic E1 -> seller -> E2 local pipeline"
