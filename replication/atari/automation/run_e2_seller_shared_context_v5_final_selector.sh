#!/bin/zsh
set -euo pipefail

export STACKPOMDP_ATARI_E2_PROFILE=v5-shared-context-exposure-v2
exec "${0:A:h}/run_e2_seller_balanced_final_selector.sh"
