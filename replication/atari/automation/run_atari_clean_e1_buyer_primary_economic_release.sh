#!/bin/zsh
set -euo pipefail

# Preregister and run the one-checkpoint fresh E1 primary-economic holdout.
# This never trains, searches checkpoints, or relaxes the immutable timing
# failure.  A failed confirmation is retained and exits with scientific code 2.

typeset -gr AUTOMATION_DIR="${0:A:h}"
typeset -gr CODE_ROOT="${AUTOMATION_DIR:h:h:h}"
typeset -gr ROOT="${STACKPOMDP_ARTIFACT_ROOT:-/Users/gbrero/active-research/StackelbergPOMDP/code/StackelbergPOMDP}"
typeset -gr PYTHON=/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python
typeset -gr RELEASE="$AUTOMATION_DIR/release_atari_e1_primary_economic.py"
typeset -gr RESULTS="$ROOT/replication/atari/results/e1_selections"
typeset -gr CHECKPOINTS="$ROOT/replication/atari/checkpoints/clean"
typeset -gr PROTOCOL="$RESULTS/e1_buyer_temporal_mix_v1_primary_economic_protocol_v1.json"
typeset -gr REPORT="$RESULTS/e1_buyer_temporal_mix_v1_primary_economic_confirmation_v1.json"
typeset -gr GATE="$RESULTS/e1_buyer_temporal_mix_v1_primary_economic_confirmation_v1.gate.json"
typeset -gr SELECTED="$CHECKPOINTS/meta_buyer_e1_ppo_balanced_temporal_mix_v1_primary_economic_selected.zip"
typeset -gr SOURCE_REPORT="$RESULTS/e1_buyer_balanced_temporal_mix_v1_all6_selector_v2.json"
typeset -gr DIAGNOSTIC="$RESULTS/e1_buyer_temporal_contingency_all6_timing_diagnostic_v1.json"
typeset -gr FAMILY="$RESULTS/e1_buyer_temporal_contingency_family_v1.json"
typeset -gr ACTIVATION="$RESULTS/e1_buyer_temporal_contingency_activation_v1.json"
typeset -gr PREFLIGHT="$RESULTS/e1_buyer_temporal_contingency_preflight_v1.json"
typeset -gr SOURCE="$CHECKPOINTS/meta_buyer_e1_ppo_balanced_temporal_mix_v1_seed1_contingency_step2400960.zip"
typeset -gr E0B="$CHECKPOINTS/space_invaders_e0b_ppo_seed1_firefix_retrain_selected.zip"
typeset -gr ROM="$ROOT/stackelberg_pomdp/atari/roms/space_invaders.bin"

mkdir -p "$RESULTS" "$CHECKPOINTS"
export PYTHONPATH="$CODE_ROOT"
export PYTHONNOUSERSITE=1
export STACKPOMDP_SPACE_INVADERS_ROM="$ROM"
export MPLCONFIGDIR=/private/tmp/mpl-stackpomdp-e1-primary-economic

"$PYTHON" "$RELEASE" write-protocol \
  --source-report "$SOURCE_REPORT" \
  --timing-diagnostic "$DIAGNOSTIC" \
  --training-family "$FAMILY" \
  --activation "$ACTIVATION" \
  --preflight "$PREFLIGHT" \
  --source-checkpoint "$SOURCE" \
  --e0b "$E0B" \
  --rom "$ROM" \
  --code-root "$CODE_ROOT" \
  --output "$PROTOCOL"

if [[ -f "$REPORT" ]]; then
  if [[ "$(jq -r '.passed' "$REPORT")" == true ]]; then
    "$PYTHON" "$RELEASE" finalize-release \
      --protocol "$PROTOCOL" \
      --report "$REPORT" \
      --gate "$GATE" \
      --selected-checkpoint "$SELECTED" \
      --code-root "$CODE_ROOT"
    exit 0
  fi
  "$PYTHON" "$RELEASE" validate-report \
    --protocol "$PROTOCOL" \
    --report "$REPORT" \
    --selected-checkpoint "$SELECTED" \
    --code-root "$CODE_ROOT"
  exit 2
fi

"$PYTHON" "$RELEASE" evaluate \
  --protocol "$PROTOCOL" \
  --report "$REPORT" \
  --gate "$GATE" \
  --selected-checkpoint "$SELECTED" \
  --code-root "$CODE_ROOT" \
  --device cpu
