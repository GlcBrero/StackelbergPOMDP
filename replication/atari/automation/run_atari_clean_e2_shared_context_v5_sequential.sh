#!/bin/zsh
set -euo pipefail

# Both leader roles share one immutable primary-buyer/v5-seller cohort.  The
# generic sequential launcher preserves scientific failures while continuing
# the other role; this wrapper only selects the disjoint v5 authority profile.
export STACKPOMDP_ATARI_E2_PROFILE=v5-shared-context-exposure-v2
exec "${0:A:h}/run_atari_clean_e2_sequential.sh"
