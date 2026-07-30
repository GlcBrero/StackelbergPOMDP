#!/bin/zsh
set -euo pipefail

export STACKPOMDP_E1V5_PROTOCOL=exposure_v2
exec "${0:A:h}/run_e1_seller_shared_context_v5_selector.sh"
