#!/bin/zsh

typeset -gr E1_SOURCE_ROOT="${${(%):-%N}:A:h:h:h:h}"
typeset -gr E1_ACTIVE_ROOT="${STACKPOMDP_ACTIVE_ROOT:-/Users/gbrero/active-research/StackelbergPOMDP/code/StackelbergPOMDP}"
typeset -gr E1_PYTHON="${STACKPOMDP_PYTHON:-/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python}"
typeset -gr E1_ROM="$E1_ACTIVE_ROOT/stackelberg_pomdp/atari/roms/space_invaders.bin"
typeset -gr E1_ROM_SHA256=7224b17462b992d67f4e06a3c85f269c9822b06df6015bf038b55f384ced0301
typeset -gr E1_CHECKPOINT_ROOT="$E1_ACTIVE_ROOT/replication/atari/checkpoints/clean"
typeset -gr E1_RESULT_ROOT="$E1_ACTIVE_ROOT/replication/atari/results/e1_selections"
typeset -gr E1_LOG_ROOT="$E1_ACTIVE_ROOT/replication/atari/results/run_logs/clean_20260728"
typeset -gr E1_WANDB_ROOT="$E1_CHECKPOINT_ROOT/wandb_runs"
typeset -gr E1_STANDARD_REPORT="$E1_RESULT_ROOT/e1_buyer_standard_all5_selector_v2.json"
typeset -gr E1_BALANCED_REPORT="$E1_RESULT_ROOT/e1_buyer_balanced_all6_selector_v2.json"
typeset -gr E1_BASE="$E1_CHECKPOINT_ROOT/meta_buyer_e1_ppo_balanced_temporal_mix_v1_seed1_contingency.zip"
typeset -gr E1_ACTIVATION="$E1_RESULT_ROOT/e1_buyer_temporal_contingency_activation_v1.json"
typeset -gr E1_PREFLIGHT="$E1_RESULT_ROOT/e1_buyer_temporal_contingency_preflight_v1.json"
typeset -gr E1_FAMILY="$E1_RESULT_ROOT/e1_buyer_temporal_contingency_family_v1.json"
typeset -gr E1_RUN_NAME=e1_buyer_balanced_temporal_mix_v1_seed1_contingency_2m
typeset -gr E1_SELECTOR_NAME=e1_buyer_balanced_temporal_mix_v1_all6_selector_v2
typeset -gr E1_SELECTOR_REPORT="$E1_RESULT_ROOT/${E1_SELECTOR_NAME}.json"
typeset -gr E1_SELECTED="$E1_CHECKPOINT_ROOT/meta_buyer_e1_ppo_balanced_temporal_mix_v1_seed1_contingency_selected.zip"
typeset -gr E1_TRAIN_LOG="$E1_LOG_ROOT/${E1_RUN_NAME}.log"
typeset -gr E1_SELECTOR_LOG="$E1_LOG_ROOT/${E1_SELECTOR_NAME}.log"
typeset -gr E1_ACTIVATION_LOCK=/private/tmp/stackpomdp-e1-temporal-activation.lock
typeset -gr E1_TRAINING_LOCK=/private/tmp/stackpomdp-e1-temporal-training.lock

E1_CODE_ROOT="$E1_SOURCE_ROOT"
E1_VALIDATOR="$E1_SOURCE_ROOT/replication/atari/automation/validate_atari_e1_temporal_contingency.py"
E1_PREFLIGHT_TOOL="$E1_SOURCE_ROOT/replication/atari/automation/preflight_atari_e1_temporal_contingency.py"

function e1_die() {
  print -u2 -- "$*"
  return 1
}

function e1_require_regular_sha() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || e1_die "missing regular file: $path"
  local actual
  actual=$(shasum -a 256 "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || e1_die "SHA-256 mismatch for $path: $actual"
}

function e1_require_scoped_clean() {
  local dirty
  dirty=$(git -C "$E1_SOURCE_ROOT" status --porcelain -- \
    replication/atari/train_atari_meta_response_sb3.py \
    replication/atari/evaluate_atari_meta_response_sb3.py \
    replication/atari/automation \
    stackelberg_pomdp/atari/e1_sampling.py \
    stackelberg_pomdp/atari/stackpomdp_env.py)
  [[ -z "$dirty" ]] || e1_die "refusing uncommitted temporal-pipeline code:\n$dirty"
}

function e1_activate() {
  e1_require_regular_sha "$E1_ROM" "$E1_ROM_SHA256"
  e1_require_scoped_clean
  if ! mkdir "$E1_ACTIVATION_LOCK" 2>/dev/null; then
    e1_die "another temporal-contingency activation check is running"
  fi
  local exit_code=0
  (
    cd "$E1_SOURCE_ROOT"
    PYTHONPATH=. PYTHONNOUSERSITE=1 \
      STACKPOMDP_SPACE_INVADERS_ROM="$E1_ROM" \
      "$E1_PYTHON" "$E1_VALIDATOR" activation \
        --standard-report "$E1_STANDARD_REPORT" \
        --balanced-report "$E1_BALANCED_REPORT" \
        --rom "$E1_ROM" \
        --base "$E1_BASE" \
        --code-root "$E1_SOURCE_ROOT" \
        --output "$E1_ACTIVATION"
  ) || exit_code=$?
  rmdir "$E1_ACTIVATION_LOCK"
  return "$exit_code"
}

function e1_prepare_runtime() {
  local revision
  revision=$(jq -r '.code_revision' "$E1_ACTIVATION")
  [[ "$revision" == [0-9a-f]## && ${#revision} == 40 ]] || e1_die "invalid activation code revision"
  local runtime="/private/tmp/stackpomdp-e1-temporal-code-${revision[1,12]}"
  if [[ -d "$runtime/.git" || -f "$runtime/.git" ]]; then
    [[ "$(git -C "$runtime" rev-parse HEAD)" == "$revision" ]] || \
      e1_die "existing temporal runtime has another revision: $runtime"
  elif [[ -e "$runtime" ]]; then
    e1_die "temporal runtime path exists but is not a worktree: $runtime"
  else
    git -C "$E1_SOURCE_ROOT" worktree add --detach "$runtime" "$revision"
  fi
  [[ -z "$(git -C "$runtime" status --porcelain)" ]] || \
    e1_die "temporal runtime is dirty: $runtime"
  E1_CODE_ROOT="$runtime"
  E1_VALIDATOR="$runtime/replication/atari/automation/validate_atari_e1_temporal_contingency.py"
  E1_PREFLIGHT_TOOL="$runtime/replication/atari/automation/preflight_atari_e1_temporal_contingency.py"
  export STACKPOMDP_SPACE_INVADERS_ROM="$E1_ROM"
  export PYTHONPATH="$runtime"
  export PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export WANDB_START_METHOD=thread
}

function e1_run_preflight() {
  (
    cd "$E1_CODE_ROOT"
    "$E1_PYTHON" "$E1_PREFLIGHT_TOOL" \
      --activation "$E1_ACTIVATION" \
      --output "$E1_PREFLIGHT" \
      --code-root "$E1_CODE_ROOT" \
      --device cpu
  )
}

function e1_refuse_path() {
  [[ ! -e "$1" ]] || e1_die "refusing to overwrite temporal artifact: $1"
}

function e1_wait_for_stable_zip() {
  local path="$1"
  local first second
  [[ -f "$path" ]] || e1_die "missing expected temporal checkpoint: $path"
  first=$(stat -f %z "$path")
  sleep 4
  second=$(stat -f %z "$path")
  [[ "$first" == "$second" ]] || e1_die "temporal checkpoint is still changing: $path"
  unzip -tq "$path"
}

function e1_candidates() {
  jq -r '.protocol.candidate_paths[]' "$E1_ACTIVATION"
}

function e1_validate_family() {
  local arguments=()
  local candidate
  while IFS= read -r candidate; do
    arguments+=(--candidate "$candidate")
  done < <(e1_candidates)
  (
    cd "$E1_CODE_ROOT"
    "$E1_PYTHON" "$E1_VALIDATOR" training-family \
      --activation "$E1_ACTIVATION" \
      --preflight "$E1_PREFLIGHT" \
      "${arguments[@]}" \
      --output "$E1_FAMILY" \
      --device cpu
  )
}

function e1_validate_existing_family() {
  (
    cd "$E1_CODE_ROOT"
    "$E1_PYTHON" "$E1_VALIDATOR" validate-training-family \
      --family "$E1_FAMILY"
  )
}
