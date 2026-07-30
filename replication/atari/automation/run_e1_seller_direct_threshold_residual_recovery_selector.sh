#!/bin/zsh
set -euo pipefail

# Screen and confirm only the six v3 Uniform-target checkpoints.

export STACKPOMDP_E1_SELLER_RESIDUAL_PROFILE=v3-direct
exec "${0:A:h}/run_e1_seller_threshold_residual_recovery_selector.sh"
