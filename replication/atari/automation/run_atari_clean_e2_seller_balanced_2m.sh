#!/bin/zsh
set -euo pipefail

source "${0:A:h}/atari_e2_pipeline_common.zsh"
e2_claim_pipeline_lock
trap 'e2_release_active_locks || print -u2 "failed to release an E2 lock"' EXIT
trap 'e2_release_active_locks || print -u2 "failed to release an E2 lock"; exit 129' HUP
trap 'e2_release_active_locks || print -u2 "failed to release an E2 lock"; exit 130' INT
trap 'e2_release_active_locks || print -u2 "failed to release an E2 lock"; exit 143' TERM
e2_prepare_runtime
export MPLCONFIGDIR="/private/tmp/mpl-stackpomdp-e2-${E2_NAMESPACE}-seller-training"

e2_wait_for_both_e1_gates
e2_role_paths seller

stem="${E2_BASE%.zip}"
e2_refuse_path "$E2_BASE"
e2_refuse_path "${stem}.evaluation.json"
e2_refuse_path "${stem}.provenance.json"
e2_refuse_path "${stem}.training.jsonl"
e2_refuse_path "$E2_INPUT_MANIFEST"
e2_refuse_path "$E2_TRAIN_LOG"
for step in $E2_STEPS; do
  e2_refuse_path "${stem}_step${step}.zip"
done
e2_write_input_manifest seller

set +e
"$PYTHON" -u -m replication.atari.train_atari_stackpomdp_leader_sb3 \
  --leader-role seller \
  --response-checkpoint "$E2_RESPONSE" \
  --leader-e1-checkpoint "$E2_LEADER_E1" \
  --seed 1 \
  --timesteps "$E2_TIMESTEPS" \
  --gameplay-horizon 200 \
  --event-tail-steps 0 \
  --num-envs "$E2_NUM_ENVS" \
  --start-method spawn \
  --n-steps "$E2_N_STEPS" \
  --batch-size "$E2_BATCH_SIZE" \
  --n-epochs 4 \
  --learning-rate 0.0001 \
  --pretrained-lr-scale 0.1 \
  --actor-loss-mode balanced \
  --entropy-coeff 0.01 \
  --clip-range 0.1 \
  --value-coefficient 0.5 \
  --max-grad-norm 0.5 \
  --noop-max 30 \
  --max-frames 100000 \
  --rom-path "$ROM" \
  --checkpoint "$E2_BASE" \
  --checkpoint-every 400000 \
  --eval-episodes 100 \
  --device cpu \
  --wandb \
  --wandb-project StackPOMDP \
  --wandb-group atari_clean_curriculum \
  --wandb-job-type "$E2_WANDB_SELLER_JOB_TYPE" \
  --wandb-name "$E2_RUN_NAME" \
  2>&1 | tee "$E2_TRAIN_LOG"
exit_code=$pipestatus[1]
set -e
if (( exit_code != 0 )); then
  print -u2 "E2 seller training failed with exit status $exit_code"
  exit "$exit_code"
fi

e2_wait_for_stable_zip "$E2_BASE"
"$PYTHON" "$VALIDATOR" e2-output \
  --role seller \
  --checkpoint "$E2_BASE" \
  --response "$E2_RESPONSE" \
  --leader-e1 "$E2_LEADER_E1" \
  --timesteps "$E2_TIMESTEPS" \
  --input-manifest "$E2_INPUT_MANIFEST"
for step in $E2_STEPS; do
  checkpoint="${stem}_step${step}.zip"
  e2_wait_for_stable_zip "$checkpoint"
done
e2_validate_family seller

print "validated complete E2 seller training family: $E2_BASE"
