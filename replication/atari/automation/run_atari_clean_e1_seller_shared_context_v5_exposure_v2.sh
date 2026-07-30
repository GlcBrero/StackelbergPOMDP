#!/bin/zsh
set -euo pipefail

# Exposure protocol v2 changes only the fresh preflight length and held-out
# real-ALE seed namespace.  The v5 actor, PPO configuration, sampler, and all
# scientific gates remain byte-identical to the standard-v1 implementation.
export STACKPOMDP_E1V5_PROTOCOL=exposure_v2
exec "${0:A:h}/run_atari_clean_e1_seller_shared_context_v5.sh"
