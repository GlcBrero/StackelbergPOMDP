#!/bin/zsh
set -euo pipefail

# Release the balanced E1 seller only after one immutable final buyer gate.
# Eligible buyers are a strict final uniform family or the preregistered
# temporal contingency with its validated gate sidecar.  Step diagnostics are
# deliberately invisible to this launcher.

source "${0:A:h}/atari_e2_pipeline_common.zsh"
e2_prepare_runtime

typeset -gr SELLER_BASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_seed1_firefix_retrain.zip"
typeset -gr SELLER_RELEASE="${SELLER_BASE%.zip}.buyer_gate.json"
typeset -gr SELLER_LOG="$LOG_ROOT/atari_clean_e1_seller_balanced_seed1_firefix_retrain_2m_local.log"
typeset -gr SELLER_LOCK=/private/tmp/stackpomdp-atari-e1-seller-release.lock

if ! mkdir "$SELLER_LOCK" 2>/dev/null; then
  print -u2 "another E1 seller release/training process is running"
  exit 1
fi
trap 'rmdir "$SELLER_LOCK" 2>/dev/null || true' EXIT INT TERM

if [[ ! -f "$SELLER_RELEASE" ]]; then
  e2_resolve_e1_gate buyer
  "$PYTHON" "$VALIDATOR" write-e1-seller-release \
    --output "$SELLER_RELEASE" \
    --buyer-report "$E1_BUYER_REPORT" \
    --buyer-checkpoint "$E1_BUYER" \
    --buyer-actor-loss-mode "$E1_BUYER_MODE"
fi

release=$(
  "$PYTHON" "$VALIDATOR" read-e1-seller-release \
    --release-manifest "$SELLER_RELEASE"
)
typeset -g E1_BUYER_REPORT=$(print -r -- "$release" | jq -r '.buyer_gate.report')
typeset -g E1_BUYER=$(print -r -- "$release" | jq -r '.buyer_gate.checkpoint')
typeset -g E1_BUYER_MODE=$(print -r -- "$release" | jq -r '.buyer_gate.actor_loss_mode')
typeset -g E1_BUYER_SOURCE_KIND=$(print -r -- "$release" | jq -r '.buyer_gate.source_kind')

typeset -gr E0B="$CHECKPOINT_ROOT/space_invaders_e0b_ppo_seed1_firefix_retrain_selected.zip"
e2_wait_for_stable_zip "$E0B"
e2_refuse_path "$SELLER_BASE"
e2_refuse_path "${SELLER_BASE%.zip}.evaluation.json"
e2_refuse_path "${SELLER_BASE%.zip}.fixed_contexts.csv"
e2_refuse_path "${SELLER_BASE%.zip}.training.jsonl"
e2_refuse_path "$SELLER_LOG"
for step in 400160 800320 1200480 1600640 2000800; do
  e2_refuse_path "${SELLER_BASE%.zip}_step${step}.zip"
done

export MPLCONFIGDIR=/private/tmp/mpl-stackpomdp-e1-seller-balanced
print "strict buyer gate released E1 seller: $E1_BUYER_REPORT ($E1_BUYER_SOURCE_KIND)"
set +e
"$PYTHON" -u -m replication.atari.train_atari_meta_response_sb3 \
  --role seller \
  --e0b-checkpoint "$E0B" \
  --actor-loss-mode balanced \
  --seed 1 \
  --timesteps 2000800 \
  --gameplay-horizon 200 \
  --event-tail-steps 0 \
  --num-envs 4 \
  --start-method spawn \
  --n-steps 205 \
  --batch-size 820 \
  --n-epochs 4 \
  --learning-rate 0.0001 \
  --pretrained-lr-scale 0.1 \
  --entropy-coeff 0.01 \
  --clip-range 0.1 \
  --value-coefficient 0.5 \
  --max-grad-norm 0.5 \
  --noop-max 30 \
  --max-frames 100000 \
  --rom-path "$ROM" \
  --checkpoint-every 400160 \
  --eval-episodes 100 \
  --fixed-eval-episodes 20 \
  --checkpoint "$SELLER_BASE" \
  --wandb \
  --wandb-project StackPOMDP \
  --wandb-group atari_clean_curriculum \
  --wandb-name atari_clean_e1_seller_balanced_seed1_firefix_retrain_2m_local \
  2>&1 | tee "$SELLER_LOG"
status=$pipestatus[1]
set -e
(( status == 0 )) || {
  print -u2 "E1 seller training failed with exit status $status"
  exit "$status"
}

rmdir "$SELLER_LOCK"
trap - EXIT INT TERM
exec "$AUTOMATION_DIR/run_e1_seller_balanced_final_selector.sh"
