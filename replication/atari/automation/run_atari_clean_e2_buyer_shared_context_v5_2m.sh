#!/bin/zsh
set -euo pipefail

export STACKPOMDP_ATARI_E2_PROFILE=v5-shared-context-exposure-v2
exec "${0:A:h}/run_atari_clean_e2_buyer_balanced_2m.sh"
