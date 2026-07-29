#!/bin/zsh
set -euo pipefail

# Release the balanced E1 seller only after the immutable primary-economic
# buyer gate. Once its protocol exists, a failed or malformed confirmation is
# terminal and no legacy selector can release seller training.

source "${0:A:h}/atari_e2_pipeline_common.zsh"
trap 'e2_release_transient_lock || print -u2 "failed to release a runtime-init lock"' EXIT
trap 'e2_release_transient_lock || print -u2 "failed to release a runtime-init lock"; exit 129' HUP
trap 'e2_release_transient_lock || print -u2 "failed to release a runtime-init lock"; exit 130' INT
trap 'e2_release_transient_lock || print -u2 "failed to release a runtime-init lock"; exit 143' TERM
e2_prepare_runtime
trap - EXIT HUP INT TERM

typeset -gr SELLER_BASE="$CHECKPOINT_ROOT/meta_seller_e1_ppo_balanced_seed1_firefix_retrain.zip"
typeset -gr SELLER_RELEASE="${SELLER_BASE%.zip}.buyer_gate.json"
typeset -gr SELLER_LOG="$LOG_ROOT/atari_clean_e1_seller_balanced_primary_economic_release_seed1_2m_local.log"
typeset -gr SELLER_LOCK=/private/tmp/stackpomdp-atari-e1-seller-release.lock
typeset -gx STACKPOMDP_E1_SELLER_LOCK_TOKEN="${STACKPOMDP_E1_SELLER_LOCK_TOKEN:-$(hostname)-$$-$(date +%s)-${RANDOM}}"

stackpomdp_claim_owned_lock \
  "$SELLER_LOCK" "$STACKPOMDP_E1_SELLER_LOCK_TOKEN" "E1 seller" || exit $?
typeset -g SELLER_LOCK_OWNED_BY_CALLER="$STACKPOMDP_LOCK_RESULT_OWNED"
function release_seller_lock() {
  stackpomdp_release_owned_lock \
    "$SELLER_LOCK" "$STACKPOMDP_E1_SELLER_LOCK_TOKEN" \
    "$SELLER_LOCK_OWNED_BY_CALLER" "E1 seller" || return $?
  typeset -g SELLER_LOCK_OWNED_BY_CALLER=0
}
trap 'release_seller_lock' EXIT
trap 'release_seller_lock; exit 129' HUP
trap 'release_seller_lock; exit 130' INT
trap 'release_seller_lock; exit 143' TERM

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
[[ "$E1_BUYER_SOURCE_KIND" == primary_economic_v1 ]] || {
  print -u2 "refusing seller training without the primary-economic buyer"
  exit 1
}
print "primary-economic buyer gate released E1 seller: $E1_BUYER_REPORT"
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
  --wandb-name atari_clean_e1_seller_balanced_primary_economic_release_seed1_2m_local \
  2>&1 | tee "$SELLER_LOG"
training_status=$pipestatus[1]
set -e
(( training_status == 0 )) || {
  print -u2 "E1 seller training failed with exit status $training_status"
  exit "$training_status"
}

release_seller_lock
trap - EXIT INT TERM
exec "$AUTOMATION_DIR/run_e1_seller_balanced_final_selector.sh"
