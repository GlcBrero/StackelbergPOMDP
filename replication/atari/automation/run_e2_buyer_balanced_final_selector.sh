#!/bin/zsh
set -euo pipefail

source "${0:A:h}/atari_e2_pipeline_common.zsh"
e2_claim_pipeline_lock
trap 'e2_release_pipeline_lock' EXIT
trap 'e2_release_pipeline_lock; exit 129' HUP
trap 'e2_release_pipeline_lock; exit 130' INT
trap 'e2_release_pipeline_lock; exit 143' TERM
e2_prepare_runtime
export MPLCONFIGDIR=/private/tmp/mpl-stackpomdp-e2-buyer-final-selector
e2_run_final_selector buyer
