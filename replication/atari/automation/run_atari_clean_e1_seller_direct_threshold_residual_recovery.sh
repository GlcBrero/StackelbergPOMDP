#!/bin/zsh
set -euo pipefail

# Separately named v3 entrypoint; implementation is shared with the immutable
# v2 chain through an explicit artifact/architecture profile.

export STACKPOMDP_E1_SELLER_RESIDUAL_PROFILE=v3-direct
exec "${0:A:h}/run_atari_clean_e1_seller_threshold_residual_recovery.sh"
