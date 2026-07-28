#!/bin/zsh
set -euo pipefail

source "${0:A:h}/atari_e2_pipeline_common.zsh"
e2_prepare_runtime
export MPLCONFIGDIR=/private/tmp/mpl-stackpomdp-e2-seller-final-selector
e2_run_final_selector seller
