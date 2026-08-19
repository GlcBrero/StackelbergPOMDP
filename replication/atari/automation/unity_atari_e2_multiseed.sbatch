#!/bin/bash
#SBATCH --job-name=atari-e2-multiseed
#SBATCH --partition=cpu
#SBATCH --array=0-19%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=24:00:00

# The system profile must be loaded before nounset mode on Unity.  Conda is
# optional: a caller may instead provide an executable via STACKPOMDP_PYTHON.
source /etc/profile
if [[ -n "${STACKPOMDP_CONDA_ENV:-}" ]]; then
  module load conda/latest
  conda activate "$STACKPOMDP_CONDA_ENV"
fi
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DEFAULT_CODE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
readonly CODE_ROOT="${STACKPOMDP_CODE_ROOT:-$DEFAULT_CODE_ROOT}"
readonly VARIANT="${STACKPOMDP_E2_VARIANT:-canonical}"
readonly RUN_ROOT="${STACKPOMDP_RUN_ROOT:-$HOME/stackpomdp-atari-e2-$VARIANT}"
readonly INPUT_ROOT="${STACKPOMDP_INPUT_ROOT:-$RUN_ROOT/inputs}"
readonly OUTPUT_ROOT="${STACKPOMDP_OUTPUT_ROOT:-$RUN_ROOT/outputs}"
readonly PYTHON_BIN="${STACKPOMDP_PYTHON:-python}"
readonly TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

[[ -f "$CODE_ROOT/replication/atari/train_atari_stackpomdp_leader_sb3.py" ]] || {
  echo "invalid Atari source root: $CODE_ROOT" >&2
  exit 66
}

case "$VARIANT" in
  canonical)
    if (( TASK_ID < 0 || TASK_ID > 19 )); then
      echo "canonical array index must lie in [0, 19], got $TASK_ID" >&2
      exit 64
    fi
    readonly SEED=$((TASK_ID / 2 + 1))
    if (( TASK_ID % 2 == 0 )); then
      readonly ROLE=buyer
    else
      readonly ROLE=seller
    fi
    readonly FREEZE_GAMEPLAY=0
    ;;
  frozen-buyer)
    if (( TASK_ID < 0 || TASK_ID > 9 )); then
      echo "frozen-buyer array index must lie in [0, 9], got $TASK_ID" >&2
      exit 64
    fi
    readonly SEED=$((TASK_ID + 1))
    readonly ROLE=buyer
    readonly FREEZE_GAMEPLAY=1
    ;;
  *)
    echo "STACKPOMDP_E2_VARIANT must be canonical or frozen-buyer" >&2
    exit 64
    ;;
esac

readonly BUYER_E1="${STACKPOMDP_BUYER_E1:-$INPUT_ROOT/meta_buyer_e1_selected.zip}"
readonly SELLER_E1="${STACKPOMDP_SELLER_E1:-$INPUT_ROOT/meta_seller_e1_selected.zip}"
readonly ROM="${STACKPOMDP_ATARI_ROM:-$INPUT_ROOT/space_invaders.bin}"
readonly BUYER_E1_SHA256="${STACKPOMDP_BUYER_E1_SHA256:-5d19fbb579f04da191ab006c5b9e9dcef574fb840e10544338dcab1b019d9e79}"
readonly SELLER_E1_SHA256="${STACKPOMDP_SELLER_E1_SHA256:-be0e0ceb6b9d666203ce362dd2562d200f19e1c5cdb3127438f43402be124c05}"
readonly ROM_SHA256="${STACKPOMDP_ATARI_ROM_SHA256:-7224b17462b992d67f4e06a3c85f269c9822b06df6015bf038b55f384ced0301}"

if [[ "$ROLE" == buyer ]]; then
  readonly RESPONSE="$SELLER_E1"
  readonly LEADER_E1="$BUYER_E1"
else
  readonly RESPONSE="$BUYER_E1"
  readonly LEADER_E1="$SELLER_E1"
fi

readonly STEM="leader_${ROLE}_e2_${VARIANT}_seed${SEED}"
readonly CHECKPOINT_ROOT="$OUTPUT_ROOT/checkpoints/$VARIANT/$ROLE/seed${SEED}"
readonly CHECKPOINT="$CHECKPOINT_ROOT/${STEM}.zip"
readonly SELECTED="$CHECKPOINT_ROOT/${STEM}_selected.zip"
readonly RESULT_ROOT="$OUTPUT_ROOT/results/$VARIANT/$ROLE/seed${SEED}"
readonly LOG_ROOT="$OUTPUT_ROOT/logs/$VARIANT/$ROLE/seed${SEED}"
readonly TRAIN_LOG="$LOG_ROOT/${STEM}.train.log"
readonly SELECT_LOG="$LOG_ROOT/${STEM}.select.log"
readonly RUN_NAME="atari_e2_${VARIANT}_${ROLE}_seed${SEED}"
readonly SELECT_RUN_NAME="${STEM}_all6_selector"
readonly COMPACT="$RESULT_ROOT/${SELECT_RUN_NAME}.compact.json"
readonly ARCHIVE="$RESULT_ROOT/${SELECT_RUN_NAME}.full.tar.gz"

mkdir -p "$CHECKPOINT_ROOT" "$RESULT_ROOT" "$LOG_ROOT"

function require_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  [[ -f "$path" ]] || { echo "missing immutable input: $path" >&2; exit 66; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 65
  }
}

require_sha256 "$BUYER_E1" "$BUYER_E1_SHA256"
require_sha256 "$SELLER_E1" "$SELLER_E1_SHA256"
require_sha256 "$ROM" "$ROM_SHA256"

if [[ "${STACKPOMDP_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  cd "$CODE_ROOT"
  PYTHONPATH="$CODE_ROOT" "$PYTHON_BIN" -c \
    'import replication.atari.train_atari_stackpomdp_leader_sb3'
  echo "preflight passed: task=$TASK_ID role=$ROLE seed=$SEED variant=$VARIANT"
  exit 0
fi

for path in "$CHECKPOINT" "$SELECTED" "$COMPACT" "$ARCHIVE"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite formal-cohort artifact: $path" >&2
    exit 73
  fi
done

readonly NODE_TMP="$(mktemp -d "${TMPDIR:-/tmp}/stackpomdp-e2-${SLURM_JOB_ID}-${TASK_ID}-XXXXXX")"
trap 'rm -rf "$NODE_TMP"' EXIT
mkdir -p "$NODE_TMP/evaluation" "$NODE_TMP/wandb" "$NODE_TMP/mpl"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$NODE_TMP/mpl"
export WANDB_DIR="$NODE_TMP/wandb"
export WANDB_CACHE_DIR="$NODE_TMP/wandb/cache"
export WANDB_CONFIG_DIR="$NODE_TMP/wandb/config"
export WANDB_START_METHOD=thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

declare -a FREEZE_ARGS=()
if (( FREEZE_GAMEPLAY == 1 )); then
  FREEZE_ARGS+=(--freeze-gameplay-actor)
fi

cd "$CODE_ROOT"
set +e
"$PYTHON_BIN" -u -m replication.atari.train_atari_stackpomdp_leader_sb3 \
  --leader-role "$ROLE" \
  --response-checkpoint "$RESPONSE" \
  --leader-e1-checkpoint "$LEADER_E1" \
  "${FREEZE_ARGS[@]}" \
  --seed "$SEED" --timesteps 2000040 \
  --gameplay-horizon 200 --event-tail-steps 0 \
  --num-envs 4 --start-method spawn --n-steps 210 --batch-size 840 \
  --n-epochs 4 --learning-rate 0.0001 --pretrained-lr-scale 0.1 \
  --actor-loss-mode balanced --entropy-coeff 0.01 --clip-range 0.1 \
  --value-coefficient 0.5 --max-grad-norm 0.5 \
  --noop-max 30 --max-frames 100000 --rom-path "$ROM" \
  --checkpoint "$CHECKPOINT" --checkpoint-every 400000 \
  --eval-episodes 100 --device cpu --wandb \
  --wandb-project StackPOMDP --wandb-group "atari_e2_${VARIANT}_10seed" \
  --wandb-job-type "atari_e2_${VARIANT}_${ROLE}" --wandb-name "$RUN_NAME" \
  2>&1 | tee "$TRAIN_LOG"
train_status=${PIPESTATUS[0]}
set -e
if (( train_status != 0 )); then
  exit "$train_status"
fi

readonly CHECKPOINT_STEM="${CHECKPOINT%.zip}"
declare -a CANDIDATE_ARGS=()
for step in 400680 800520 1200360 1600200 2000040; do
  candidate="${CHECKPOINT_STEM}_step${step}.zip"
  [[ -f "$candidate" ]] || { echo "missing checkpoint: $candidate" >&2; exit 66; }
  CANDIDATE_ARGS+=(--checkpoint "$candidate")
done
CANDIDATE_ARGS+=(--checkpoint "$CHECKPOINT")

set +e
"$PYTHON_BIN" -u -m replication.atari.evaluate_atari_stackpomdp_leader_sb3 \
  --leader-role "$ROLE" --response-checkpoint "$RESPONSE" \
  "${CANDIDATE_ARGS[@]}" --selected-checkpoint "$SELECTED" \
  --screen-episodes 20 --screen-seed-start 4000001 \
  --confirmation-episodes 100 --confirmation-seed-start 5000001 \
  --gameplay-horizon 200 --event-tail-steps 0 \
  --noop-max 30 --max-frames 100000 --rom-path "$ROM" --device cpu \
  --output-dir "$NODE_TMP/evaluation" --run-name "$SELECT_RUN_NAME" \
  2>&1 | tee "$SELECT_LOG"
selector_status=${PIPESTATUS[0]}
set -e

readonly REPORT="$NODE_TMP/evaluation/${SELECT_RUN_NAME}.json"
if [[ -d "$NODE_TMP/evaluation" ]] && find "$NODE_TMP/evaluation" -type f -print -quit | grep -q .; then
  tar -C "$NODE_TMP/evaluation" -czf "$ARCHIVE" .
fi
if [[ -f "$REPORT" ]]; then
  "$PYTHON_BIN" -u -m replication.atari.automation.compact_atari_e2_selection \
    --report "$REPORT" --archive "$ARCHIVE" --output "$COMPACT" \
    --sha256-sidecars
fi
if (( selector_status != 0 )); then
  exit "$selector_status"
fi
[[ -f "$SELECTED" && -f "$COMPACT" && -f "$ARCHIVE" ]] || {
  echo "job completed without all retained artifacts" >&2
  exit 70
}
