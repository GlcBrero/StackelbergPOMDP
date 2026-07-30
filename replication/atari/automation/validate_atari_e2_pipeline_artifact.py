#!/Users/gbrero/miniconda3/envs/stackelbergPOMDP/bin/python
"""Integrity checks and collision-safe manifests for clean Atari E1 -> E2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


REPOSITORY_ROOT = Path(os.environ.get(
    "STACKPOMDP_CODE_ROOT",
    "/Users/gbrero/active-research/StackelbergPOMDP/code/StackelbergPOMDP",
)).resolve()
E1_EVALUATOR = "clean_atari_e1_selector_v2"
E2_EVALUATOR = "clean_atari_e2_selector_v2"
E2_STEPS = (400_680, 800_520, 1_200_360, 1_600_200, 2_000_040)
E1_UNIFORM_SAMPLER = "uniform"
E1_TEMPORAL_SAMPLER = "temporal-marginal-v1"
E1_TEMPORAL_GATE_KIND = "atari_e1_buyer_temporal_contingency_gate"
E1_TEMPORAL_FAMILY_KIND = "atari_e1_buyer_temporal_contingency_family"
E1_TEMPORAL_ACTIVATION_KIND = "atari_e1_buyer_temporal_contingency_activation"
E1_SELLER_GATE_KIND = "atari_e1_seller_selection_gate"
E1_SELLER_RECOVERY_SOURCE_KIND = "seller_conditioning_recovery_v1"
E1_SELLER_RECOVERY_ACTIVATION_NAME = (
    "e1_seller_conditioning_recovery_activation_v1.json"
)
E1_SELLER_RECOVERY_REPORT_NAME = (
    "e1_seller_conditioning_recovery_all6_selector_v2.json"
)
E1_SELLER_RECOVERY_GATE_NAME = (
    "e1_seller_conditioning_recovery_all6_selector_v2.gate.json"
)
E1_SELLER_RECOVERY_MODULE_NAME = (
    "validate_atari_e1_seller_conditioning_recovery.py"
)
E1_SELLER_THRESHOLD_RESIDUAL_SOURCE_KIND = (
    "seller_conditioning_recovery_v3_direct_threshold_residual_v1"
)
E1_SELLER_THRESHOLD_RESIDUAL_ACTIVATION_NAME = (
    "e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_"
    "activation.json"
)
E1_SELLER_THRESHOLD_RESIDUAL_REPORT_NAME = (
    "e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_"
    "all6_selector_v3.json"
)
E1_SELLER_THRESHOLD_RESIDUAL_GATE_NAME = (
    "e1_seller_conditioning_recovery_v3_direct_threshold_residual_v1_"
    "all6_selector_v3.gate.json"
)
E1_SELLER_THRESHOLD_RESIDUAL_MODULE_NAME = (
    "validate_atari_e1_seller_direct_threshold_residual_recovery.py"
)
E1_PRIMARY_PROTOCOL_NAME = (
    "e1_buyer_temporal_mix_v1_primary_economic_protocol_v1.json"
)
E1_PRIMARY_REPORT_NAME = (
    "e1_buyer_temporal_mix_v1_primary_economic_confirmation_v1.json"
)
E1_PRIMARY_GATE_NAME = (
    "e1_buyer_temporal_mix_v1_primary_economic_confirmation_v1.gate.json"
)
E1_PRIMARY_EVALUATOR = "clean_atari_e1_primary_economic_confirmation_v1"
E1_PRIMARY_PROTOCOL_KIND = (
    "stackpomdp.atari.e1_buyer_primary_economic_protocol.v1"
)
E1_PRIMARY_GATE_KIND = "stackpomdp.atari.e1_buyer_primary_economic_gate.v1"
E1_PRIMARY_SOURCE_KIND = "primary_economic_v1"
E1_PRIMARY_MODULE_NAME = "release_atari_e1_primary_economic.py"
E1_PRIMARY_CHECKPOINT_NAME = (
    "meta_buyer_e1_ppo_balanced_temporal_mix_v1_"
    "primary_economic_selected.zip"
)
E1_PRIMARY_CHECK_NAMES = (
    "random buyer net payoff",
    "low/mid-price mean purchases",
    "low/mid-price mean shots",
    "price 0.5 purchases",
    "price 0.5 shots",
    "price 0.5 purchased bullets used",
    "price 0.5 buyer net payoff",
    "price 1 demand",
    "low-to-high demand drop",
    "largest adjacent demand reversal",
    "price 0.0 buyer net payoff",
    "price 0.1 buyer net payoff",
    "price 0.2 buyer net payoff",
    "price 0.3 buyer net payoff",
    "price 0.4 buyer net payoff",
    "price 0.5 buyer net payoff",
)
E2_ORCHESTRATION_SCHEMA = "stackpomdp.atari.e2_sequential_orchestration.v1"
E2_NAMESPACE = "e1seller_direct_threshold_residual_v3"
CANONICAL_ROM_SHA256 = (
    "7224b17462b992d67f4e06a3c85f269c9822b06df6015bf038b55f384ced0301"
)
CANONICAL_E0B_SHA256 = (
    "3a9ded5c53e10bf0b2215f1223f7bde15ba23d197dd0c651590bcd980c375ca3"
)

sys.path.insert(0, str(REPOSITORY_ROOT))

# Automation remains on the active reviewed branch while E2 policy code is
# deliberately imported from REPOSITORY_ROOT's pinned detached worktree.
AUTOMATION_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict:
    path = Path(path).expanduser()
    if path.is_symlink():
        fail(f"expected a regular, non-symlink JSON file: {path}")
    path = path.resolve()
    if not path.is_file():
        fail(f"expected a regular, non-symlink JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        fail(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    path = Path(path).expanduser()
    if path.is_symlink():
        fail(f"expected a regular, non-symlink file: {path}")
    path = path.resolve()
    if not path.is_file():
        fail(f"expected a regular, non-symlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identities = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
    )
    if identities[0] != identities[1]:
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def atomic_write_new_json(path: Path, value: dict) -> None:
    """Publish a new JSON artifact atomically and never overwrite a peer."""

    path = Path(path).expanduser()
    if os.path.lexists(path):
        fail(f"refusing to overwrite immutable JSON artifact: {path}")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        fail(f"refusing to overwrite immutable JSON artifact: {path}")
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            fail(f"refusing to overwrite immutable JSON artifact: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def validate_zip(path: Path) -> str:
    path = Path(path).expanduser()
    if path.is_symlink():
        fail(f"expected a regular, non-symlink ZIP file: {path}")
    path = path.resolve()
    digest = sha256_file(path)
    with zipfile.ZipFile(path, "r") as archive:
        broken = archive.testzip()
        if broken is not None:
            fail(f"corrupt ZIP member {broken!r} in {path}")
    if sha256_file(path) != digest:
        fail(f"ZIP changed during validation: {path}")
    return digest


def same_path(actual: object, expected: Path, *, label: str) -> None:
    if not isinstance(actual, str):
        fail(f"{label} is not a path string")
    if Path(actual).expanduser().resolve() != expected.expanduser().resolve():
        fail(f"{label} does not name {expected}: {actual}")


def expect_equal(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, observed {actual!r}")


def configured_e2_code_head() -> str:
    """Read the single E2 pin owned by the shared launcher contract."""

    common = Path(__file__).resolve().with_name("atari_e2_pipeline_common.zsh")
    if common.is_symlink() or not common.is_file():
        fail(f"E2 pipeline common file is unavailable or unsafe: {common}")
    matches = re.findall(
        r"^typeset -gr EXPECTED_HEAD=([0-9a-f]{40})$",
        common.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        fail("E2 pipeline common file has no unique full expected HEAD")
    return matches[0]


def validate_code_root() -> str:
    try:
        head = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"cannot validate E2 code root {REPOSITORY_ROOT}: {error}")
    expected = configured_e2_code_head()
    expect_equal(head, expected, label="E2 code HEAD")
    return head


def validate_selector_code_root(path: Path) -> str:
    """Return the exact clean revision used to run the E1 seller selector."""

    path = path.expanduser().resolve()
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git", "-C", str(path), "status", "--porcelain", "--",
                "replication/atari/evaluate_atari_meta_response_sb3.py",
                "replication/atari/train_atari_meta_response_sb3.py",
                "replication/atari/automation",
                "replication/atari/sb3_common.py",
                "stackelberg_pomdp/atari",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"cannot validate selector code root {path}: {error}")
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        fail(f"selector code root has no full git revision: {path}")
    if dirty:
        fail(f"selector code root has uncommitted Atari changes: {dirty}")
    return head


def validate_recorded_revision(revision: object) -> str:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        fail("recorded selector code revision is invalid")
    try:
        subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "cat-file", "-e", f"{revision}^{{commit}}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"recorded selector revision is unavailable: {revision}: {error}")
    return revision


def _canonical_uniform_sampler() -> dict:
    return {
        "mode": E1_UNIFORM_SAMPLER,
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "schedule": "ExactFiveEventSchedule.sample",
        "context": "five independent Uniform(0,1) prices",
        "schedule_context_rngs_independent": False,
        "legacy_sampling_path": True,
    }


def _canonical_temporal_sampler() -> dict:
    return {
        "mode": E1_TEMPORAL_SAMPLER,
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "schedule_stratum_weights": {
            "unconditional": 0.50,
            "early_fifth": 0.25,
            "late_fifth": 0.25,
        },
        "early_fifth_interval_half_open": [120, 160],
        "late_fifth_interval_half_open": [180, 200],
        "conditional_schedule_method": (
            "rejection sample from ExactFiveEventSchedule"
        ),
        "context_stratum_weights": {"uniform": 0.75, "low_prefix": 0.25},
        "uniform_context": "five independent Uniform(0,1) prices",
        "low_prefix_context": {
            "first_four": "four independent Uniform(0,0.25) prices",
            "fifth": "Uniform(0,1) price",
        },
        "schedule_context_rngs_independent": True,
        "legacy_sampling_path": False,
    }


def _e1_sampler_contract(report: dict) -> dict:
    """Classify a final E1 family without silently accepting mixed sampling."""

    family = report.get("training_family", {})
    current = family.get("common_sampler_provenance")
    history = family.get("common_sampler_history")
    if current is None and history is None:
        return {
            "source_kind": "uniform",
            "sampler_mode": E1_UNIFORM_SAMPLER,
            "legacy_inferred": True,
        }
    if not isinstance(current, dict) or not isinstance(history, list):
        fail("E1 sampler provenance/history must be present together")
    mode = current.get("mode")
    if mode == E1_UNIFORM_SAMPLER:
        expect_equal(
            current, _canonical_uniform_sampler(), label="uniform E1 sampler"
        )
        if len(history) != 1:
            fail("uniform E1 gate must have exactly one sampler stage")
        stage = history[0]
        if not isinstance(stage, dict):
            fail("uniform E1 sampler stage is not an object")
        expect_equal(stage.get("start_total_timesteps"), 0, label="uniform sampler start")
        expect_equal(stage.get("sampler"), current, label="uniform sampler history")
        if type(stage.get("inferred_for_legacy_checkpoint")) is not bool:
            fail("uniform E1 sampler stage lacks a Boolean inference flag")
        if not isinstance(stage.get("resume_sources"), list):
            fail("uniform E1 sampler stage lacks resume-source provenance")
        return {
            "source_kind": "uniform",
            "sampler_mode": mode,
            "legacy_inferred": bool(
                family.get("sampler_contract_inferred_for_legacy_checkpoint", False)
            ),
        }
    if mode != E1_TEMPORAL_SAMPLER:
        fail(f"E1 gate has an unsupported sampler mode: {mode!r}")
    expect_equal(
        current, _canonical_temporal_sampler(), label="temporal E1 sampler"
    )
    if len(history) != 2:
        fail("temporal E1 gate must have exactly uniform and temporal stages")
    uniform, temporal = history
    if not isinstance(uniform, dict) or not isinstance(temporal, dict):
        fail("temporal E1 sampler history contains a non-object stage")
    expect_equal(uniform.get("start_total_timesteps"), 0, label="uniform-stage start")
    expect_equal(
        uniform.get("sampler"), _canonical_uniform_sampler(),
        label="temporal lineage uniform stage",
    )
    expect_equal(temporal.get("sampler"), current, label="temporal sampler stage")
    start = temporal.get("start_total_timesteps")
    if type(start) is not int or start <= 0:
        fail("temporal sampler stage has an invalid start timestep")
    if temporal.get("inferred_for_legacy_checkpoint") is not False:
        fail("temporal sampler stage cannot be legacy-inferred")
    sources = temporal.get("resume_sources")
    if not isinstance(sources, list) or len(sources) != 1:
        fail("temporal sampler stage must bind exactly one resume source")
    source = sources[0]
    if not isinstance(source, dict):
        fail("temporal resume source is not an object")
    expect_equal(source.get("resume_total_timesteps"), start, label="temporal resume step")
    expect_equal(source.get("training_total_timesteps"), start, label="temporal parent step")
    digest = source.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        fail("temporal resume source has no valid SHA-256")
    if not isinstance(source.get("path"), str) or not source["path"]:
        fail("temporal resume source has no path")
    return {
        "source_kind": "temporal_contingency",
        "sampler_mode": mode,
        "legacy_inferred": False,
    }


def _strict_e1_selection(
        report: dict, report_path: Path, checkpoint: Path, digest: str,
) -> list[str]:
    """Require one common all-six screen and confirmation of its winner only."""

    protocol = report.get("protocol", {})
    for key, expected in {
        "screen_episodes": 20,
        "confirmation_episodes": 100,
        "outer_transitions": 205,
    }.items():
        expect_equal(protocol.get(key), expected, label=f"E1 protocol {key}")
    if protocol.get("confirmation_policy") is not None:
        expect_equal(
            protocol.get("confirmation_policy"),
            "screen_winner_only_no_fallback",
            label="E1 confirmation policy",
        )
    environment = report.get("environment", {})
    for key, expected in {
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "rom_sha256": CANONICAL_ROM_SHA256,
    }.items():
        expect_equal(environment.get(key), expected, label=f"E1 environment {key}")

    screen = report.get("screen", {})
    common = screen.get("common_pairing", {})
    if common.get("passed") is not True:
        fail("E1 final gate has no valid common screen")
    expect_equal(common.get("candidates_checked"), 6, label="E1 screen candidates")
    pairs = common.get("seed_context_pairs")
    if not isinstance(pairs, list) or len(pairs) != 20:
        fail("E1 common screen must use exactly 20 paired episodes")
    results = screen.get("results")
    ranking = report.get("ranking")
    if not isinstance(results, list) or len(results) != 6:
        fail("E1 final gate must screen exactly six candidates")
    if not isinstance(ranking, list) or len(ranking) != 6:
        fail("E1 final gate must rank exactly six candidates")
    screen_hashes = []
    for result in results:
        metadata = result.get("metadata", {})
        path = Path(metadata.get("path", "")).expanduser().resolve()
        candidate_digest = metadata.get("sha256")
        expect_equal(
            validate_zip(path), candidate_digest, label="E1 screen candidate SHA-256"
        )
        if result.get("protocol", {}).get("passed") is not True:
            fail("E1 screen candidate failed mechanics")
        rows = result.get("episode_rows")
        if not isinstance(rows, list) or len(rows) != 20:
            fail("E1 screen candidate does not contain 20 episodes")
        screen_hashes.append(candidate_digest)
    if len(set(screen_hashes)) != 6:
        fail("E1 final family is not byte-distinct")
    expect_equal(
        report.get("immutable_evaluation", {}).get("candidate_sha256"),
        screen_hashes,
        label="E1 immutable candidate order",
    )
    ranked_hashes = [row.get("checkpoint_sha256") for row in ranking]
    if set(ranked_hashes) != set(screen_hashes):
        fail("E1 ranking does not cover the common screen")
    expect_equal(
        [row.get("rank") for row in ranking], list(range(1, 7)),
        label="E1 ranking order",
    )
    winner = ranking[0].get("checkpoint_sha256")
    if ranking[0].get("mechanically_valid") is not True:
        fail("E1 screen winner is mechanically invalid")
    expect_equal(winner, digest, label="E1 selected screen winner")

    attempts = report.get("confirmation_attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        fail("E1 gate must confirm only the screen winner; fallback is forbidden")
    attempt = attempts[0]
    expect_equal(
        attempt.get("metadata", {}).get("sha256"), winner,
        label="E1 confirmed screen winner",
    )
    if attempt.get("behavioral_gate", {}).get("passed") is not True:
        fail("E1 screen winner did not pass fresh confirmation")
    random_rows = attempt.get("random", {}).get("episode_rows")
    if not isinstance(random_rows, list) or len(random_rows) != 100:
        fail("E1 confirmation must contain exactly 100 fresh episodes")

    selection = report.get("selection")
    if selection is not None:
        if not isinstance(selection, dict):
            fail("E1 selection record is malformed")
        expect_equal(selection.get("fallback_allowed"), False, label="E1 fallback flag")
        expect_equal(
            selection.get("screen_selected_checkpoint_sha256"), winner,
            label="E1 screen-selected hash",
        )
        expect_equal(
            selection.get("selected_checkpoint_sha256"), winner,
            label="E1 selected hash",
        )
    same_path(
        report.get("artifacts", {}).get("json"), report_path,
        label="E1 report self-artifact",
    )
    return screen_hashes


def _validate_canonical_seller_selection_protocol(report: dict) -> None:
    protocol = report.get("protocol", {})
    expected_protocol = {
        "screen_episodes": 20,
        "screen_seed_start": 3_500_001,
        "confirmation_episodes": 100,
        "confirmation_seed_start": 3_600_001,
        "fixed_context_episodes": 20,
        "fixed_context_seed_start": 3_700_001,
        "confirmation_policy": "screen_winner_only_no_fallback",
    }
    for key, expected in expected_protocol.items():
        expect_equal(
            protocol.get(key), expected,
            label=f"canonical E1 seller protocol {key}",
        )
    validate_recorded_revision(
        report.get("immutable_evaluation", {}).get("selector_code_revision")
    )
    expect_equal(
        report.get("immutable_evaluation", {}).get("e0b_sha256"),
        CANONICAL_E0B_SHA256,
        label="canonical E1 seller E0b SHA-256",
    )
    results = report.get("screen", {}).get("results")
    if not isinstance(results, list) or len(results) != 6:
        fail("canonical E1 seller selection must screen six candidates")
    expect_equal(
        [row.get("metadata", {}).get("training_timesteps") for row in results],
        [400_160, 800_320, 1_200_480, 1_600_640, 2_000_800, 2_000_800],
        label="canonical E1 seller candidate timesteps",
    )
    for row in results:
        expect_equal(
            row.get("metadata", {}).get("e0b_source_provenance", {}).get(
                "sha256"
            ),
            CANONICAL_E0B_SHA256,
            label="canonical E1 seller candidate E0b",
        )
    pairs = report.get("screen", {}).get("common_pairing", {}).get(
        "seed_context_pairs"
    )
    expect_equal(
        [row.get("evaluation_seed") for row in pairs or []],
        list(range(3_500_001, 3_500_021)),
        label="canonical E1 seller screen seeds",
    )
    attempts = report.get("confirmation_attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        fail("canonical E1 seller selection must have one confirmation")
    random_rows = attempts[0].get("random", {}).get("episode_rows")
    expect_equal(
        [row.get("evaluation_seed") for row in random_rows or []],
        list(range(3_600_001, 3_600_101)),
        label="canonical E1 seller confirmation seeds",
    )
    fixed = attempts[0].get("fixed_contexts")
    if not isinstance(fixed, list) or len(fixed) != 11:
        fail("canonical E1 seller selection must evaluate eleven fixed contexts")
    expect_equal(
        [round(float(row.get("opponent_value")), 6) for row in fixed],
        [round(index / 10.0, 6) for index in range(11)],
        label="canonical E1 seller fixed-context grid",
    )
    fixed_seeds = list(range(3_700_001, 3_700_021))
    for row in fixed:
        expect_equal(
            [episode.get("evaluation_seed") for episode in row.get("episode_rows", [])],
            fixed_seeds,
            label="canonical E1 seller fixed-context seeds",
        )


def _validate_temporal_gate_support(
        report_path: Path, checkpoint: Path, digest: str, report: dict,
        candidate_hashes: list[str],
) -> dict:
    """Recheck the temporal gate sidecar, activation, and exact family bytes."""

    gate_path = report_path.with_name(f"{report_path.stem}.gate.json")
    gate = load_json(gate_path)
    expect_equal(gate.get("schema_version"), 1, label="temporal gate schema")
    expect_equal(gate.get("kind"), E1_TEMPORAL_GATE_KIND, label="temporal gate kind")
    if gate.get("passed") is not True:
        fail("temporal E1 support gate did not pass")
    expect_equal(gate.get("role"), "buyer", label="temporal gate role")
    expect_equal(gate.get("actor_loss_mode"), "balanced", label="temporal gate mode")
    gate_report = gate.get("report", {})
    same_path(gate_report.get("path"), report_path, label="temporal gate report")
    expect_equal(gate_report.get("sha256"), sha256_file(report_path), label="temporal report SHA-256")
    selected = gate.get("selected_checkpoint", {})
    same_path(selected.get("path"), checkpoint, label="temporal selected checkpoint")
    expect_equal(selected.get("sha256"), digest, label="temporal selected SHA-256")

    family_record = gate.get("training_family", {})
    family_path = Path(family_record.get("path", "")).expanduser().resolve()
    expect_equal(
        family_record.get("sha256"), sha256_file(family_path),
        label="temporal family SHA-256",
    )
    family = load_json(family_path)
    expect_equal(family.get("schema_version"), 1, label="temporal family schema")
    expect_equal(family.get("kind"), E1_TEMPORAL_FAMILY_KIND, label="temporal family kind")
    if family.get("passed") is not True:
        fail("temporal training family did not pass integrity validation")
    expect_equal(family.get("candidate_sha256"), candidate_hashes, label="temporal family candidates")
    expect_equal(
        family_record.get("candidate_sha256"), candidate_hashes,
        label="temporal gate candidate hashes",
    )
    metadata = family.get("candidate_metadata")
    if not isinstance(metadata, list) or len(metadata) != 6:
        fail("temporal family must contain six candidate metadata records")
    current = report["training_family"]["common_sampler_provenance"]
    history = report["training_family"]["common_sampler_history"]
    for expected_digest, item in zip(candidate_hashes, metadata):
        expect_equal(item.get("sha256"), expected_digest, label="temporal candidate hash")
        expect_equal(
            validate_zip(Path(item.get("path", ""))), expected_digest,
            label="temporal candidate bytes",
        )
        expect_equal(
            item.get("atari_e1_sampler_provenance"), current,
            label="temporal candidate sampler",
        )
        expect_equal(
            item.get("atari_e1_sampler_history"), history,
            label="temporal candidate sampler history",
        )

    activation_record = gate.get("activation", {})
    activation_path = Path(activation_record.get("path", "")).expanduser().resolve()
    expect_equal(
        activation_record.get("sha256"), sha256_file(activation_path),
        label="temporal activation SHA-256",
    )
    activation = load_json(activation_path)
    expect_equal(activation.get("schema_version"), 1, label="temporal activation schema")
    expect_equal(
        activation.get("kind"), E1_TEMPORAL_ACTIVATION_KIND,
        label="temporal activation kind",
    )
    expect_equal(
        activation.get("activation_condition"),
        {
            "standard_uniform_all_six_failed": True,
            "balanced_uniform_all_six_failed": True,
        },
        label="temporal activation condition",
    )
    revision = activation.get("code_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        fail("temporal activation has no full code revision")
    expect_equal(
        activation_record.get("code_revision"), revision,
        label="temporal gate code revision",
    )
    expect_equal(
        family.get("activation", {}).get("sha256"), sha256_file(activation_path),
        label="temporal family activation hash",
    )
    same_path(
        family.get("activation", {}).get("path"), activation_path,
        label="temporal family activation path",
    )
    activation_protocol = activation.get("protocol", {})
    expect_equal(
        activation_protocol.get("training_sampler"), current,
        label="temporal activation training sampler",
    )
    expect_equal(
        activation_protocol.get("evaluation_sampler"), _canonical_uniform_sampler(),
        label="temporal activation evaluation sampler",
    )
    expect_equal(activation_protocol.get("candidate_count"), 6, label="temporal candidate count")
    expect_equal(
        activation_protocol.get("additional_timesteps"), 2_000_800,
        label="temporal additional training budget",
    )
    expect_equal(
        activation_protocol.get("checkpoint_interval"), 400_160,
        label="temporal checkpoint interval",
    )
    expect_equal(
        [str(Path(path).expanduser().resolve()) for path in activation_protocol.get("candidate_paths", [])],
        [str(Path(item["path"]).expanduser().resolve()) for item in metadata],
        label="temporal preregistered candidate paths",
    )
    expect_equal(
        activation_protocol.get("confirmation", {}).get("fallback_allowed"),
        False,
        label="temporal activation fallback flag",
    )
    expect_equal(
        activation_protocol.get("screen"),
        {"episodes": 20, "seed_start": 6_000_001},
        label="temporal activation screen",
    )
    expect_equal(
        activation_protocol.get("confirmation"),
        {
            "episodes": 100,
            "seed_start": 6_100_001,
            "fallback_allowed": False,
        },
        label="temporal activation confirmation",
    )
    expect_equal(
        activation_protocol.get("fixed_seed_start"), 6_200_001,
        label="temporal fixed-grid seed",
    )
    expect_equal(
        activation_protocol.get("timing_seed_start"), 6_300_001,
        label="temporal paired-timing seed",
    )
    failure_reports = activation.get("uniform_failure_reports", {})
    if not isinstance(failure_reports, dict) or set(failure_reports) != {
        "standard", "balanced"
    }:
        fail("temporal activation must bind standard and balanced failures")
    for label, item in failure_reports.items():
        path = Path(item.get("path", "")).expanduser().resolve()
        expect_equal(sha256_file(path), item.get("sha256"), label=f"{label} failure-report SHA-256")
        expect_equal(item.get("candidate_count"), 6, label=f"{label} failed family size")
        expect_equal(item.get("sampler", {}).get("mode"), E1_UNIFORM_SAMPLER, label=f"{label} failed sampler")
        expect_equal(
            item.get("actor_loss_mode"), label,
            label=f"{label} failed actor-loss mode",
        )
        expect_equal(
            item.get("strict_screen_winner_passed"), False,
            label=f"{label} strict screen-winner outcome",
        )
        if type(item.get("reported_passed")) is not bool:
            fail(f"{label} failed report has no recorded legacy outcome")
        expect_equal(
            item.get("legacy_fallback_selected"), item["reported_passed"],
            label=f"{label} legacy fallback record",
        )
    resume = activation.get("resume_source", {})
    temporal_source = history[1]["resume_sources"][0]
    for key in ("path", "sha256", "training_total_timesteps"):
        expect_equal(
            resume.get(key), temporal_source.get(key),
            label=f"temporal activation resume {key}",
        )
    resume_path = Path(resume.get("path", "")).expanduser().resolve()
    expect_equal(
        validate_zip(resume_path), resume.get("sha256"),
        label="temporal activation resume bytes",
    )
    expect_equal(
        activation_protocol.get("expected_total_timesteps"),
        resume.get("training_total_timesteps") + 2_000_800,
        label="temporal expected final timestep",
    )
    e0b = activation.get("e0b_source", {})
    e0b_path = Path(e0b.get("path", "")).expanduser().resolve()
    expect_equal(
        validate_zip(e0b_path), e0b.get("sha256"),
        label="temporal activation E0b bytes",
    )
    rom = activation.get("rom", {})
    rom_path = Path(rom.get("path", "")).expanduser().resolve()
    expect_equal(
        sha256_file(rom_path), CANONICAL_ROM_SHA256,
        label="temporal activation ROM bytes",
    )
    expect_equal(
        rom.get("sha256"), CANONICAL_ROM_SHA256,
        label="temporal activation ROM record",
    )
    preflight_record = family.get("preflight", {})
    preflight_path = Path(preflight_record.get("path", "")).expanduser().resolve()
    expect_equal(
        sha256_file(preflight_path), preflight_record.get("sha256"),
        label="temporal preflight SHA-256",
    )
    preflight = load_json(preflight_path)
    if preflight.get("passed") is not True:
        fail("temporal preflight did not pass")
    expect_equal(
        preflight.get("activation_sha256"), sha256_file(activation_path),
        label="temporal preflight activation",
    )
    expect_equal(gate.get("sampler", {}).get("current"), current, label="temporal gate sampler")
    expect_equal(gate.get("sampler", {}).get("history"), history, label="temporal gate history")
    gate_selection = gate.get("selection", {})
    for key in (
        "screen_selected_checkpoint_sha256",
        "confirmed_checkpoint_sha256",
        "selected_checkpoint_sha256",
    ):
        expect_equal(gate_selection.get(key), digest, label=f"temporal gate {key}")
    expect_equal(gate_selection.get("confirmation_attempts"), 1, label="temporal confirmations")
    expect_equal(
        gate_selection.get("confirmation_policy"),
        "screen_winner_only_no_fallback",
        label="temporal confirmation policy",
    )
    expect_equal(gate_selection.get("fallback_allowed"), False, label="temporal fallback flag")
    return {
        "temporal_gate": {
            "path": str(gate_path), "sha256": sha256_file(gate_path),
        },
        "training_family": {
            "path": str(family_path), "sha256": sha256_file(family_path),
        },
        "activation": {
            "path": str(activation_path), "sha256": sha256_file(activation_path),
            "code_revision": revision,
        },
        "preflight": {
            "path": str(preflight_path), "sha256": sha256_file(preflight_path),
        },
    }


def _run_active_seller_validator(
        module_name: str, arguments: list[str], *, label: str,
) -> dict:
    """Run one active immutable seller validator outside pinned E2 imports."""

    module_path = (
        Path(__file__).expanduser().parent / module_name
    )
    if module_path.is_symlink() or not module_path.resolve().is_file():
        fail(f"{label} validator is unavailable or unsafe: {module_path}")
    module_path = module_path.resolve()
    environment = os.environ.copy()
    environment.pop("STACKPOMDP_CODE_ROOT", None)
    environment["PYTHONPATH"] = str(AUTOMATION_SOURCE_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, str(module_path), *arguments],
            cwd=str(AUTOMATION_SOURCE_ROOT),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "")
        fail(f"{label} validation failed: {stderr or error}")
    values = []
    for line in completed.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("value"), dict):
            values.append(record["value"])
    if len(values) != 1:
        fail(f"{label} validator emitted no unique JSON result")
    return values[0]


def _run_seller_recovery_validator(arguments: list[str]) -> dict:
    return _run_active_seller_validator(
        E1_SELLER_RECOVERY_MODULE_NAME,
        arguments,
        label="seller-recovery v1",
    )


def _run_seller_threshold_residual_validator(arguments: list[str]) -> dict:
    return _run_active_seller_validator(
        E1_SELLER_THRESHOLD_RESIDUAL_MODULE_NAME,
        arguments,
        label="seller threshold-residual recovery v2",
    )


def canonical_seller_threshold_residual_architecture() -> dict:
    from stackelberg_pomdp.atari.stackpomdp_policy import (
        direct_threshold_residual_architecture_provenance,
    )

    return direct_threshold_residual_architecture_provenance(
        state_features=64
    )


def _validate_seller_threshold_residual_gate(args: argparse.Namespace) -> dict:
    """Normalize the authoritative v2 gate for generic E2 manifests."""

    if args.role != "seller" or args.actor_loss_mode != "balanced":
        fail("the seller threshold-residual recovery is balanced-seller only")
    report = Path(args.report).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if report.name != E1_SELLER_THRESHOLD_RESIDUAL_REPORT_NAME:
        fail(f"unexpected threshold-residual report name: {report.name}")
    gate_path = report.with_name(E1_SELLER_THRESHOLD_RESIDUAL_GATE_NAME)
    gate = _run_seller_threshold_residual_validator([
        "validate-selection-gate",
        "--gate", str(gate_path),
        "--report", str(report),
        "--selected", str(checkpoint),
    ])
    digest = validate_zip(checkpoint)
    expect_equal(
        gate.get("selected_checkpoint", {}).get("sha256"),
        digest,
        label="threshold-residual selected SHA-256",
    )
    architecture = canonical_seller_threshold_residual_architecture()
    expect_equal(
        gate.get("economic_architecture"),
        architecture,
        label="threshold-residual seller architecture",
    )
    candidate_hashes = gate.get("training_family", {}).get("candidate_sha256")
    if not isinstance(candidate_hashes, list) or len(candidate_hashes) != 6:
        fail("threshold-residual seller gate does not bind six candidates")

    from replication.atari.train_atari_stackpomdp_leader_sb3 import (
        checkpoint_policy_metadata,
    )

    metadata = checkpoint_policy_metadata(
        checkpoint, device="cpu", label="selected v3 E1 seller"
    )
    expect_equal(metadata.get("sha256"), digest, label="loaded v3 seller SHA")
    expect_equal(
        metadata.get("economic_architecture"),
        architecture,
        label="loaded v3 seller architecture",
    )
    training_code_revision = metadata.get("e1_training_code_revision")
    expect_equal(
        training_code_revision,
        gate.get("activation", {}).get("code_revision"),
        label="v3 seller checkpoint versus activation training revision",
    )
    expect_equal(
        gate.get("e1_training_code_revision"),
        training_code_revision,
        label="v3 seller gate training revision",
    )
    from replication.atari.train_atari_meta_response_sb3 import (
        direct_threshold_initialization_contract,
    )

    initialization = direct_threshold_initialization_contract(
        state_features=64, economic_hidden=64
    )
    expect_equal(
        metadata.get("direct_threshold_initialization"),
        initialization,
        label="loaded v3 seller direct-threshold initialization",
    )
    expect_equal(
        gate.get("direct_threshold_initialization"),
        initialization,
        label="v3 seller gate direct-threshold initialization",
    )
    support = {
        "seller_threshold_residual_recovery_gate": {
            "path": str(gate_path), "sha256": sha256_file(gate_path),
        },
        "training_family": dict(gate["training_family"]),
        "activation": dict(gate["activation"]),
        "prerequisite_v1_failure": dict(gate["prerequisite_v1_failure"]),
        "warmup_probe": dict(gate["warmup_probe"]),
        "seller_release": dict(gate["seller_release"]),
        "economic_architecture": architecture,
        "direct_threshold_initialization": initialization,
        "e1_training_code_revision": training_code_revision,
    }
    return {
        "kind": "e1_gate",
        "role": "seller",
        "report": str(report),
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "actor_loss_mode": "balanced",
        "source_kind": E1_SELLER_THRESHOLD_RESIDUAL_SOURCE_KIND,
        "sampler_mode": E1_UNIFORM_SAMPLER,
        "support_artifacts": support,
        "candidate_sha256": candidate_hashes,
        "passed": True,
    }


def _validate_seller_recovery_gate(args: argparse.Namespace) -> dict:
    """Normalize the authoritative recovery gate for generic E2 manifests."""

    if args.role != "seller" or args.actor_loss_mode != "balanced":
        fail("the seller-conditioning recovery gate is balanced-seller only")
    report = Path(args.report).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if report.name != E1_SELLER_RECOVERY_REPORT_NAME:
        fail(f"unexpected seller-recovery report name: {report.name}")
    gate_path = report.with_name(E1_SELLER_RECOVERY_GATE_NAME)
    gate = _run_seller_recovery_validator([
        "validate-selection-gate",
        "--gate", str(gate_path),
        "--report", str(report),
        "--selected", str(checkpoint),
    ])
    digest = validate_zip(checkpoint)
    expect_equal(
        gate.get("selected_checkpoint", {}).get("sha256"),
        digest,
        label="seller-recovery selected SHA-256",
    )
    candidate_hashes = gate.get("training_family", {}).get(
        "candidate_sha256"
    )
    if not isinstance(candidate_hashes, list) or len(candidate_hashes) != 6:
        fail("seller-recovery gate does not bind six candidates")
    support = {
        "seller_conditioning_recovery_gate": {
            "path": str(gate_path), "sha256": sha256_file(gate_path),
        },
        "training_family": dict(gate["training_family"]),
        "activation": dict(gate["activation"]),
        "prerequisite_failure": dict(gate["prerequisite_failure"]),
        "warmup_probe": dict(gate["warmup_probe"]),
        "seller_release": dict(gate["seller_release"]),
    }
    return {
        "kind": "e1_gate",
        "role": "seller",
        "report": str(report),
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "actor_loss_mode": "balanced",
        "source_kind": E1_SELLER_RECOVERY_SOURCE_KIND,
        "sampler_mode": E1_UNIFORM_SAMPLER,
        "support_artifacts": support,
        "candidate_sha256": candidate_hashes,
        "passed": True,
    }


def _run_primary_economic_gate_validator(
        *, protocol_path: Path, report_path: Path, gate_path: Path,
        selected_checkpoint: Path,
) -> dict:
    """Validate the active E1 release in a process isolated from pinned E2.

    E2 deliberately imports policy code from the detached commit named by the
    shared launcher contract.  The release validator and its evaluator
    dependencies are newer active automation, so importing them into this
    process would either fail or mix scientific runtimes.  A short child
    process receives the active source root explicitly; this process and every
    subsequent E2 import remain pinned.
    """

    module_path = Path(__file__).expanduser().parent / E1_PRIMARY_MODULE_NAME
    if module_path.is_symlink():
        fail(
            "primary-economic release validator is unavailable or unsafe: "
            f"{module_path}"
        )
    module_path = module_path.resolve()
    if not module_path.is_file():
        fail(
            "primary-economic release validator is unavailable or unsafe: "
            f"{module_path}"
        )
    environment = os.environ.copy()
    environment.pop("STACKPOMDP_CODE_ROOT", None)
    environment["PYTHONPATH"] = str(AUTOMATION_SOURCE_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [
                sys.executable, str(module_path), "validate-gate",
                "--protocol", str(protocol_path),
                "--report", str(report_path),
                "--gate", str(gate_path),
                "--selected-checkpoint", str(selected_checkpoint),
                "--code-root", str(AUTOMATION_SOURCE_ROOT),
            ],
            cwd=str(AUTOMATION_SOURCE_ROOT),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "")
        fail(f"primary-economic release validation failed: {stderr or error}")
    records = []
    for line in completed.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
                isinstance(record, dict)
                and record.get("kind") == "primary_economic_gate"
                and isinstance(record.get("value"), dict)
        ):
            records.append(record["value"])
    if len(records) != 1:
        fail("primary-economic validator emitted no unique JSON result")
    value = records[0]
    if not isinstance(value, dict):
        fail("primary-economic validator result is not an object")
    return value


def _primary_checkpoint_from_records(*, report: dict, gate: dict) -> Path:
    """Read one exact selected path from the report/gate publication pair."""

    alias = report.get("selected_alias")
    selected = gate.get("selected_checkpoint")
    if not isinstance(alias, dict) or not isinstance(selected, dict):
        fail("primary report/gate has no selected-checkpoint records")
    alias_raw = alias.get("pinned_path")
    selected_raw = selected.get("path")
    if not isinstance(alias_raw, str) or not isinstance(selected_raw, str):
        fail("primary report/gate selected paths are not strings")
    alias_path = Path(alias_raw).expanduser()
    selected_path = Path(selected_raw).expanduser()
    if alias_path.is_symlink() or selected_path.is_symlink():
        fail("primary selected checkpoint cannot be a symlink")
    alias_path = alias_path.resolve()
    selected_path = selected_path.resolve()
    expect_equal(
        alias_path, selected_path,
        label="primary report versus gate selected path",
    )
    if alias_path.name != E1_PRIMARY_CHECKPOINT_NAME:
        fail(f"unexpected primary selected-checkpoint name: {alias_path.name}")
    alias_digest = alias.get("sha256")
    selected_digest = selected.get("sha256")
    if (
            not isinstance(alias_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", alias_digest)
    ):
        fail("primary report selected alias has no valid SHA-256")
    expect_equal(
        alias_digest, selected_digest,
        label="primary report versus gate selected SHA-256",
    )
    return alias_path


def _validate_primary_economic_gate(args: argparse.Namespace) -> dict:
    """Validate and normalize the authoritative one-checkpoint buyer gate."""

    if args.role != "buyer" or args.actor_loss_mode != "balanced":
        fail("the primary-economic E1 gate is balanced-buyer only")
    report_path = Path(args.report).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    if report_path.is_symlink():
        fail(f"primary-economic report cannot be a symlink: {report_path}")
    if checkpoint.is_symlink():
        fail(f"primary selected checkpoint cannot be a symlink: {checkpoint}")
    report_path = report_path.resolve()
    checkpoint = checkpoint.resolve()
    if report_path.name != E1_PRIMARY_REPORT_NAME:
        fail(f"unexpected primary-economic report name: {report_path.name}")
    protocol_path = report_path.with_name(E1_PRIMARY_PROTOCOL_NAME)
    gate_path = report_path.with_name(E1_PRIMARY_GATE_NAME)
    report = load_json(report_path)
    expect_equal(report.get("passed"), True, label="primary buyer outcome")
    expect_equal(report.get("role"), "buyer", label="primary buyer role")
    expect_equal(
        report.get("evaluator"), E1_PRIMARY_EVALUATOR,
        label="primary buyer evaluator",
    )
    gate_record = load_json(gate_path)
    published_checkpoint = _primary_checkpoint_from_records(
        report=report, gate=gate_record,
    )
    if published_checkpoint != checkpoint:
        fail(
            "requested primary checkpoint differs from the report/gate alias: "
            f"{checkpoint} != {published_checkpoint}"
        )

    gate = _run_primary_economic_gate_validator(
        protocol_path=protocol_path,
        report_path=report_path,
        gate_path=gate_path,
        selected_checkpoint=checkpoint,
    )
    if not isinstance(gate, dict):
        fail("primary-economic validate_gate did not return an object")
    expect_equal(gate.get("kind"), E1_PRIMARY_GATE_KIND, label="primary gate kind")
    expect_equal(gate.get("passed"), True, label="primary gate outcome")
    expect_equal(gate.get("role"), "buyer", label="primary gate role")
    expect_equal(
        gate.get("actor_loss_mode"), "balanced", label="primary gate actor loss"
    )
    expect_equal(
        gate.get("sampler_mode"), E1_TEMPORAL_SAMPLER,
        label="primary gate sampler",
    )
    expect_equal(
        gate.get("source_kind"), E1_PRIMARY_SOURCE_KIND,
        label="primary gate source kind",
    )
    digest = validate_zip(checkpoint)

    from replication.atari.train_atari_stackpomdp_leader_sb3 import (
        checkpoint_policy_metadata,
    )

    metadata = checkpoint_policy_metadata(
        checkpoint, device="cpu", label="selected primary-economic E1 buyer"
    )
    expect_equal(metadata.get("sha256"), digest, label="loaded primary E1 SHA-256")
    expect_equal(metadata.get("economic_role"), "buyer", label="loaded primary role")
    expect_equal(
        metadata.get("economic_input_mode"), "full",
        label="loaded primary economic input mode",
    )
    expect_equal(
        metadata.get("actor_loss_mode"), "balanced",
        label="loaded primary actor-loss mode",
    )
    return {
        "kind": "e1_gate",
        "role": "buyer",
        "report": str(report_path),
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "actor_loss_mode": "balanced",
        "source_kind": E1_PRIMARY_SOURCE_KIND,
        "sampler_mode": E1_TEMPORAL_SAMPLER,
        "support_artifacts": {
            "primary_economic_protocol": {
                "path": str(protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "primary_economic_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
            "primary_economic_gate": {
                "path": str(gate_path),
                "sha256": sha256_file(gate_path),
            },
        },
        "candidate_sha256": [digest],
        "passed": True,
    }


def _validate_e1_gate_core(
        args: argparse.Namespace, *, require_seller_support: bool,
) -> dict:
    report_path = Path(args.report).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    report = load_json(report_path)
    if report.get("passed") is not True:
        fail("E1 selector report is not strictly passing")
    expect_equal(report.get("role"), args.role, label="E1 report role")
    expect_equal(
        report.get("evaluator"), E1_EVALUATOR, label="E1 evaluator"
    )
    alias = report.get("selected_alias")
    if not isinstance(alias, dict):
        fail("passing E1 report has no selected_alias object")
    same_path(
        alias.get("pinned_path"), checkpoint, label="E1 selected alias"
    )
    digest = validate_zip(checkpoint)
    expect_equal(alias.get("sha256"), digest, label="E1 alias SHA-256")
    candidate_hashes = _strict_e1_selection(
        report, report_path, checkpoint, digest
    )
    if args.role == "seller":
        _validate_canonical_seller_selection_protocol(report)
    sampler = _e1_sampler_contract(report)
    if sampler["source_kind"] == "temporal_contingency":
        if args.role != "buyer" or args.actor_loss_mode != "balanced":
            fail("temporal E1 contingency is valid only for the balanced buyer")
        support_artifacts = _validate_temporal_gate_support(
            report_path, checkpoint, digest, report, candidate_hashes
        )
    elif args.role == "seller" and require_seller_support:
        if args.actor_loss_mode != "balanced":
            fail("the downstream E1 seller gate must use balanced actor loss")
        support_artifacts = _validate_seller_selection_gate_support(
            report_path, checkpoint, digest, candidate_hashes
        )
    else:
        support_artifacts = {}

    config = report.get("training_family", {}).get(
        "common_training_config"
    )
    if not isinstance(config, dict):
        fail("E1 report has no common training configuration")
    expected_config = {
        "algorithm": "PPO",
        "actor_loss_mode": args.actor_loss_mode,
        "seed": 1,
        "n_steps": 205,
        "batch_size": 820,
        "n_epochs": 4,
        "learning_rate": 1.0e-4,
        "pretrained_lr_scale": 0.1,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "entropy_coefficient": 0.01,
        "clip_range_at_start": 0.1,
        "value_coefficient": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "visual_features": 512,
        "state_features": 64,
        "economic_hidden": 64,
        "critic_hidden": 256,
        "policy_class": (
            "stackelberg_pomdp.atari.stackpomdp_policy."
            "StackPOMDPAtariPolicy"
        ),
    }
    for key, expected in expected_config.items():
        expect_equal(config.get(key), expected, label=f"E1 config {key}")

    from replication.atari.train_atari_stackpomdp_leader_sb3 import (
        checkpoint_policy_metadata,
    )

    metadata = checkpoint_policy_metadata(
        checkpoint, device="cpu", label=f"selected E1 {args.role}"
    )
    expect_equal(metadata.get("sha256"), digest, label="loaded E1 SHA-256")
    expect_equal(
        metadata.get("economic_role"), args.role, label="loaded E1 role"
    )
    expect_equal(
        metadata.get("economic_input_mode"),
        "full",
        label="loaded E1 economic input mode",
    )
    expect_equal(
        metadata.get("actor_loss_mode"),
        args.actor_loss_mode,
        label="loaded E1 actor loss mode",
    )
    return {
        "kind": "e1_gate",
        "role": args.role,
        "report": str(report_path),
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "actor_loss_mode": args.actor_loss_mode,
        "source_kind": sampler["source_kind"],
        "sampler_mode": sampler["sampler_mode"],
        "support_artifacts": support_artifacts,
        "candidate_sha256": candidate_hashes,
        "passed": True,
    }


def validate_e1_gate(args: argparse.Namespace) -> dict:
    if (
            args.role == "seller"
            and Path(args.report).expanduser().resolve().name
            == E1_SELLER_THRESHOLD_RESIDUAL_REPORT_NAME
    ):
        return _validate_seller_threshold_residual_gate(args)
    if (
            args.role == "seller"
            and Path(args.report).expanduser().resolve().name
            == E1_SELLER_RECOVERY_REPORT_NAME
    ):
        return _validate_seller_recovery_gate(args)
    if (
            args.role == "buyer"
            and Path(args.report).expanduser().resolve().name
            == E1_PRIMARY_REPORT_NAME
    ):
        return _validate_primary_economic_gate(args)
    return _validate_e1_gate_core(args, require_seller_support=True)


def _report_selected_screen_winner(report: dict) -> bool:
    """Return False only for a coherent legacy lower-rank fallback pass."""

    ranking = report.get("ranking")
    alias = report.get("selected_alias")
    attempts = report.get("confirmation_attempts")
    if (
            not isinstance(ranking, list) or len(ranking) != 6
            or not isinstance(alias, dict)
            or not isinstance(attempts, list) or not attempts
    ):
        fail("passing E1 report lacks final selection records")
    ranked_hashes = [row.get("checkpoint_sha256") for row in ranking]
    selected = alias.get("sha256")
    attempted = [row.get("metadata", {}).get("sha256") for row in attempts]
    if selected == ranked_hashes[0]:
        if attempted != [ranked_hashes[0]]:
            fail("screen-winner E1 pass contains extra confirmation attempts")
        return True
    if selected not in ranked_hashes[1:]:
        fail("legacy E1 fallback selected a checkpoint outside the ranking")
    selected_rank = ranked_hashes.index(selected)
    if attempted != ranked_hashes[:selected_rank + 1]:
        fail("legacy E1 fallback confirmations are not a ranking prefix")
    if any(
            row.get("behavioral_gate", {}).get("passed") is not False
            for row in attempts[:-1]
    ) or attempts[-1].get("behavioral_gate", {}).get("passed") is not True:
        fail("legacy E1 fallback outcomes are inconsistent")
    return False


def _validate_primary_protocol_header(path: Path) -> dict:
    """Fail early on a malformed authoritative protocol while evaluation runs."""

    value = load_json(path)
    expect_equal(value.get("schema_version"), 1, label="primary protocol schema")
    expect_equal(
        value.get("kind"), E1_PRIMARY_PROTOCOL_KIND,
        label="primary protocol kind",
    )
    expect_equal(value.get("role"), "buyer", label="primary protocol role")
    expect_equal(
        value.get("evaluator"), E1_PRIMARY_EVALUATOR,
        label="primary protocol evaluator",
    )
    expect_equal(
        value.get("source_kind"), E1_PRIMARY_SOURCE_KIND,
        label="primary protocol source kind",
    )
    expect_equal(
        value.get("sampler_mode"), E1_TEMPORAL_SAMPLER,
        label="primary protocol sampler",
    )
    expect_equal(
        value.get("economic_check_names"), list(E1_PRIMARY_CHECK_NAMES),
        label="primary protocol economic checks",
    )
    candidate = value.get("candidate_policy", {})
    for key, expected in {
        "eligible_count": 1,
        "candidate_search": False,
        "fallback_allowed": False,
    }.items():
        expect_equal(candidate.get(key), expected, label=f"primary candidate {key}")
    holdout = value.get("holdout", {})
    expect_equal(holdout.get("timing_evaluation_run"), False, label="timing holdout")
    random_holdout = holdout.get("random", {})
    for key, expected in {
        "episodes": 100,
        "seed_start": 8_000_001,
        "seed_end": 8_000_100,
    }.items():
        expect_equal(
            random_holdout.get(key), expected, label=f"primary random {key}"
        )
    fixed = holdout.get("fixed_grid", {})
    for key, expected in {
        "episodes_per_value": 20,
        "seed_start": 8_100_001,
        "seed_end": 8_100_020,
        "values": [value / 10.0 for value in range(11)],
        "shared_seeds": True,
        "event_steps": [20, 50, 80, 110, 140],
    }.items():
        expect_equal(fixed.get(key), expected, label=f"primary fixed {key}")
    revision = value.get("evaluator_code_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        fail("primary protocol has no full evaluator revision")
    return value


def _discover_authoritative_primary_buyer_gate(
        *, output_dir: Path, override_report: str | None,
) -> dict | None:
    """Resolve the primary buyer protocol or return None before activation.

    The exact protocol file is the activation marker.  Once it exists, legacy
    all-six and timing-contingency reports are permanently ineligible.  A
    still-running primary confirmation returns ``found=False``; a published
    negative or malformed artifact aborts instead of silently falling back.
    """

    protocol_path = output_dir / E1_PRIMARY_PROTOCOL_NAME
    report_path = output_dir / E1_PRIMARY_REPORT_NAME
    gate_path = output_dir / E1_PRIMARY_GATE_NAME
    primary_paths_exist = any(
        os.path.lexists(path) for path in (protocol_path, report_path, gate_path)
    )
    if not os.path.lexists(protocol_path):
        if primary_paths_exist:
            fail(
                "primary-economic artifacts exist without their authoritative "
                f"protocol: {protocol_path}"
            )
        return None

    # Reject malformed JSON/symlinks immediately even while the confirmation
    # process is still running. Full protocol semantics are rechecked by the
    # transitive stage-1 validator when the gate is published.
    _validate_primary_protocol_header(protocol_path)
    if override_report:
        override = Path(override_report).expanduser().resolve()
        if override != report_path.resolve():
            fail(
                "the active primary-economic protocol forbids overriding the "
                f"buyer report: {override}"
            )
    if os.path.lexists(gate_path) and not os.path.lexists(report_path):
        fail("primary-economic gate exists before its required report")
    if not os.path.lexists(report_path):
        return {
            "kind": "e1_gate_discovery",
            "found": False,
            "authoritative_source": E1_PRIMARY_SOURCE_KIND,
            "state": "confirmation_pending",
        }

    report = load_json(report_path)
    passed = report.get("passed")
    if type(passed) is not bool:
        fail("authoritative primary-economic report has no Boolean outcome")
    expect_equal(report.get("role"), "buyer", label="primary report role")
    expect_equal(
        report.get("evaluator"), E1_PRIMARY_EVALUATOR,
        label="primary report evaluator",
    )
    if passed is False:
        fail("authoritative primary-economic buyer confirmation failed")
    if not os.path.lexists(gate_path):
        return {
            "kind": "e1_gate_discovery",
            "found": False,
            "authoritative_source": E1_PRIMARY_SOURCE_KIND,
            "state": "gate_publication_pending",
        }

    gate = load_json(gate_path)
    checkpoint = _primary_checkpoint_from_records(report=report, gate=gate)
    validated = validate_e1_gate(argparse.Namespace(
        report=str(report_path),
        checkpoint=str(checkpoint),
        role="buyer",
        actor_loss_mode="balanced",
    ))
    return {
        "kind": "e1_gate_discovery",
        "found": True,
        "role": "buyer",
        "report": validated["report"],
        "checkpoint": validated["checkpoint"],
        "checkpoint_sha256": validated["sha256"],
        "actor_loss_mode": validated["actor_loss_mode"],
        "source_kind": validated["source_kind"],
        "sampler_mode": validated["sampler_mode"],
        "support_artifacts": validated["support_artifacts"],
    }


def _discover_authoritative_seller_threshold_residual_gate(
        *, output_dir: Path, override_report: str | None,
) -> dict | None:
    """Make a published v2 activation authoritative over all older sellers."""

    activation_path = output_dir / E1_SELLER_THRESHOLD_RESIDUAL_ACTIVATION_NAME
    report_path = output_dir / E1_SELLER_THRESHOLD_RESIDUAL_REPORT_NAME
    gate_path = output_dir / E1_SELLER_THRESHOLD_RESIDUAL_GATE_NAME
    paths_exist = any(
        os.path.lexists(path)
        for path in (activation_path, report_path, gate_path)
    )
    if not os.path.lexists(activation_path):
        if paths_exist:
            fail(
                "seller threshold-residual artifacts exist without their "
                f"authoritative activation: {activation_path}"
            )
        return None
    _run_seller_threshold_residual_validator([
        "validate-activation", "--activation", str(activation_path),
    ])
    if override_report:
        override = Path(override_report).expanduser().resolve()
        if override != report_path:
            fail(
                "the active seller threshold-residual recovery forbids "
                f"overriding its report: {override}"
            )
    if os.path.lexists(gate_path) and not os.path.lexists(report_path):
        fail("threshold-residual gate exists before its required report")
    if not os.path.lexists(report_path):
        return {
            "kind": "e1_gate_discovery",
            "found": False,
            "authoritative_source": E1_SELLER_THRESHOLD_RESIDUAL_SOURCE_KIND,
            "state": "threshold_residual_training_or_selection_pending",
        }
    report = load_json(report_path)
    passed = report.get("passed")
    if type(passed) is not bool:
        fail("authoritative threshold-residual report has no Boolean outcome")
    expect_equal(report.get("role"), "seller", label="threshold-residual role")
    expect_equal(
        report.get("evaluator"), E1_EVALUATOR,
        label="threshold-residual evaluator",
    )
    if passed is False:
        fail("authoritative seller threshold-residual confirmation failed")
    if not os.path.lexists(gate_path):
        return {
            "kind": "e1_gate_discovery",
            "found": False,
            "authoritative_source": E1_SELLER_THRESHOLD_RESIDUAL_SOURCE_KIND,
            "state": "threshold_residual_gate_publication_pending",
        }
    gate = load_json(gate_path)
    selected = Path(
        gate.get("selected_checkpoint", {}).get("path", "")
    ).expanduser().resolve()
    validated = validate_e1_gate(argparse.Namespace(
        report=str(report_path), checkpoint=str(selected), role="seller",
        actor_loss_mode="balanced",
    ))
    return {
        "kind": "e1_gate_discovery",
        "found": True,
        "role": "seller",
        "report": validated["report"],
        "checkpoint": validated["checkpoint"],
        "checkpoint_sha256": validated["sha256"],
        "actor_loss_mode": validated["actor_loss_mode"],
        "source_kind": validated["source_kind"],
        "sampler_mode": validated["sampler_mode"],
        "support_artifacts": validated["support_artifacts"],
    }


def _discover_authoritative_seller_recovery_gate(
        *, output_dir: Path, override_report: str | None,
) -> dict | None:
    """Hide all older sellers once the recovery activation is published."""

    activation_path = output_dir / E1_SELLER_RECOVERY_ACTIVATION_NAME
    report_path = output_dir / E1_SELLER_RECOVERY_REPORT_NAME
    gate_path = output_dir / E1_SELLER_RECOVERY_GATE_NAME
    paths_exist = any(
        os.path.lexists(path)
        for path in (activation_path, report_path, gate_path)
    )
    if not os.path.lexists(activation_path):
        if paths_exist:
            fail(
                "seller-recovery artifacts exist without their authoritative "
                f"activation: {activation_path}"
            )
        return None
    _run_seller_recovery_validator([
        "validate-activation", "--activation", str(activation_path),
    ])
    if override_report:
        override = Path(override_report).expanduser().resolve()
        if override != report_path:
            fail(
                "the active seller-conditioning recovery forbids overriding "
                f"the seller report: {override}"
            )
    if os.path.lexists(gate_path) and not os.path.lexists(report_path):
        fail("seller-recovery gate exists before its required report")
    if not os.path.lexists(report_path):
        return {
            "kind": "e1_gate_discovery",
            "found": False,
            "authoritative_source": E1_SELLER_RECOVERY_SOURCE_KIND,
            "state": "recovery_training_or_selection_pending",
        }
    report = load_json(report_path)
    passed = report.get("passed")
    if type(passed) is not bool:
        fail("authoritative seller-recovery report has no Boolean outcome")
    expect_equal(report.get("role"), "seller", label="seller-recovery role")
    expect_equal(
        report.get("evaluator"), E1_EVALUATOR,
        label="seller-recovery evaluator",
    )
    if passed is False:
        fail("authoritative seller-conditioning recovery confirmation failed")
    if not os.path.lexists(gate_path):
        return {
            "kind": "e1_gate_discovery",
            "found": False,
            "authoritative_source": E1_SELLER_RECOVERY_SOURCE_KIND,
            "state": "recovery_gate_publication_pending",
        }
    gate = load_json(gate_path)
    selected = Path(
        gate.get("selected_checkpoint", {}).get("path", "")
    ).expanduser().resolve()
    validated = validate_e1_gate(argparse.Namespace(
        report=str(report_path), checkpoint=str(selected), role="seller",
        actor_loss_mode="balanced",
    ))
    return {
        "kind": "e1_gate_discovery",
        "found": True,
        "role": "seller",
        "report": validated["report"],
        "checkpoint": validated["checkpoint"],
        "checkpoint_sha256": validated["sha256"],
        "actor_loss_mode": validated["actor_loss_mode"],
        "source_kind": validated["source_kind"],
        "sampler_mode": validated["sampler_mode"],
        "support_artifacts": validated["support_artifacts"],
    }


def discover_e1_gate(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.role == "seller":
        threshold_residual = (
            _discover_authoritative_seller_threshold_residual_gate(
                output_dir=output_dir,
                override_report=args.override_report,
            )
        )
        if threshold_residual is not None:
            return threshold_residual
        recovery = _discover_authoritative_seller_recovery_gate(
            output_dir=output_dir, override_report=args.override_report,
        )
        if recovery is not None:
            return recovery
    if args.role == "buyer":
        primary = _discover_authoritative_primary_buyer_gate(
            output_dir=output_dir, override_report=args.override_report,
        )
        if primary is not None:
            return primary
    if args.override_report:
        override = Path(args.override_report).expanduser().resolve()
        if not override.exists():
            return {"kind": "e1_gate_discovery", "found": False}
        reports = [override]
    else:
        reports = sorted(output_dir.glob(f"e1_{args.role}_*selector_v2.json"))
        reports = [
            path for path in reports
            if not re.search(r"_step[0-9]+_selector_v2[.]json$", path.name)
        ]

    passing = []
    for report_path in reports:
        report = load_json(report_path)
        if report.get("passed") is not True:
            if args.override_report:
                fail(f"explicit E1 gate report is not passing: {report_path}")
            continue
        if report.get("role") != args.role:
            fail(f"E1 gate report has the wrong role: {report_path}")
        sampler_mode = report.get("training_family", {}).get(
            "common_sampler_provenance", {}
        ).get("mode")
        if sampler_mode == E1_TEMPORAL_SAMPLER:
            gate_path = report_path.with_name(f"{report_path.stem}.gate.json")
            if not gate_path.is_file():
                # The selector publishes its immutable JSON before the
                # contingency validator can publish the gate sidecar.  Treat
                # this narrow interval as not ready, never as an eligible gate.
                continue
        if args.role == "seller":
            gate_path = report_path.with_name(f"{report_path.stem}.gate.json")
            if not gate_path.is_file():
                # The report and selected alias are published before the
                # release-bound gate sidecar.  Never discover the transient
                # report alone as a downstream response policy.
                continue
        if not _report_selected_screen_winner(report):
            # Legacy selectors could report success after searching below the
            # screen winner.  That immutable result is a valid negative input
            # to the temporal contingency, but never a downstream gate.
            continue
        config = report.get("training_family", {}).get(
            "common_training_config", {}
        )
        mode = config.get("actor_loss_mode")
        if mode not in ("balanced", "standard"):
            fail(f"E1 gate has an unsupported actor-loss mode: {report_path}")
        if args.require_mode and mode != args.require_mode:
            if args.override_report:
                fail(
                    f"explicit E1 gate must use {args.require_mode}, got {mode}"
                )
            continue
        alias = report.get("selected_alias")
        if (
                not isinstance(alias, dict)
                or not isinstance(alias.get("pinned_path"), str)
        ):
            fail(f"passing E1 gate has no selected checkpoint: {report_path}")
        checkpoint = Path(alias["pinned_path"]).expanduser().resolve()
        validated = validate_e1_gate(argparse.Namespace(
            report=str(report_path),
            checkpoint=str(checkpoint),
            role=args.role,
            actor_loss_mode=mode,
        ))
        passing.append({
            **validated,
            "priority": 0 if mode == "balanced" else 1,
        })

    if not passing:
        return {"kind": "e1_gate_discovery", "found": False}
    passing.sort(key=lambda row: (row["priority"], row["report"]))
    best_priority = passing[0]["priority"]
    equally_preferred = [
        row for row in passing if row["priority"] == best_priority
    ]
    if len(equally_preferred) > 1 and not args.override_report:
        paths = [row["report"] for row in equally_preferred]
        fail(
            "multiple equally preferred passing E1 gates require an explicit "
            f"override: {paths}"
        )
    chosen = passing[0]
    return {
        "kind": "e1_gate_discovery",
        "found": True,
        "role": args.role,
        "report": chosen["report"],
        "checkpoint": chosen["checkpoint"],
        "checkpoint_sha256": chosen["sha256"],
        "actor_loss_mode": chosen["actor_loss_mode"],
        "source_kind": chosen["source_kind"],
        "sampler_mode": chosen["sampler_mode"],
        "support_artifacts": chosen["support_artifacts"],
    }


def validate_e1_gate_record(gate: object, *, role: str) -> dict:
    """Revalidate one immutable cohort record, including temporal support."""

    if not isinstance(gate, dict):
        fail(f"E1 gate cohort has no {role} record")
    report = Path(gate.get("report", "")).expanduser().resolve()
    checkpoint = Path(gate.get("checkpoint", "")).expanduser().resolve()
    expect_equal(
        sha256_file(report), gate.get("report_sha256"),
        label=f"{role} gate-report SHA-256",
    )
    expect_equal(
        validate_zip(checkpoint), gate.get("checkpoint_sha256"),
        label=f"{role} checkpoint SHA-256",
    )
    validated = validate_e1_gate(argparse.Namespace(
        report=str(report),
        checkpoint=str(checkpoint),
        role=role,
        actor_loss_mode=gate.get("actor_loss_mode"),
    ))
    for key in ("source_kind", "sampler_mode", "support_artifacts"):
        expect_equal(
            gate.get(key), validated[key], label=f"{role} gate {key}"
        )
    return validated


def validated_pipeline_inputs(path: Path, *, role: str) -> dict:
    value = load_json(path)
    expect_equal(
        value.get("schema"),
        "stackpomdp.atari.e2_pipeline_inputs.v2",
        label="E2 pipeline-input schema",
    )
    expect_equal(value.get("code_head"), validate_code_root(), label="code HEAD")
    expect_equal(value.get("leader_role"), role, label="input leader role")
    expect_equal(
        value.get("training_actor_loss_mode"),
        "balanced",
        label="E2 training actor-loss mode",
    )
    gates = value.get("e1_gates")
    if not isinstance(gates, dict):
        fail("E2 pipeline-input manifest has no E1 gates")
    for gate_role in ("buyer", "seller"):
        gate = gates.get(gate_role)
        validate_e1_gate_record(gate, role=gate_role)
    cohort_path = Path(value.get("e1_gate_cohort", "")).expanduser().resolve()
    expect_equal(
        sha256_file(cohort_path),
        value.get("e1_gate_cohort_sha256"),
        label="shared E1 gate-cohort SHA-256",
    )
    cohort = validated_e1_gate_cohort(cohort_path)
    expect_equal(
        gates, cohort["e1_gates"], label="shared E1 gate-cohort records"
    )
    return value


def validated_e1_gate_cohort(path: Path) -> dict:
    value = load_json(path)
    expect_equal(
        value.get("schema"),
        "stackpomdp.atari.e2_e1_gate_cohort.v3",
        label="E1 gate-cohort schema",
    )
    expect_equal(value.get("code_head"), validate_code_root(), label="code HEAD")
    gates = value.get("e1_gates")
    if not isinstance(gates, dict) or set(gates) != {"buyer", "seller"}:
        fail("E1 gate cohort must contain exactly buyer and seller records")
    validated = {
        role: validate_e1_gate_record(gate, role=role)
        for role, gate in gates.items()
    }
    if validated["buyer"]["source_kind"] != E1_PRIMARY_SOURCE_KIND:
        fail("E2 gate cohort requires the authoritative primary-economic buyer")
    if gates["seller"]["actor_loss_mode"] != "balanced":
        fail("E2 gate cohort requires the balanced E1 seller")
    _validate_authoritative_seller_gate_record(gates["seller"])
    _validate_cohort_seller_release_binding(value, gates)
    return value


def _validate_authoritative_seller_gate_record(gate: dict) -> dict | None:
    """Reject a cached seller as soon as a newer recovery is activated."""

    seller_report = Path(gate.get("report", "")).expanduser().resolve()
    output_dir = seller_report.parent
    authoritative = _discover_authoritative_seller_threshold_residual_gate(
        output_dir=output_dir,
        override_report=None,
    )
    if authoritative is not None:
        label = "threshold-residual recovery v2"
    else:
        authoritative = _discover_authoritative_seller_recovery_gate(
            output_dir=output_dir,
            override_report=None,
        )
        label = "recovery v1"
    if authoritative is None:
        return None
    if (
            not isinstance(authoritative, dict)
            or authoritative.get("found") is not True
    ):
        fail(
            f"active seller {label} has not released an authoritative E2 gate"
        )
    comparison = {
        "report": authoritative.get("report"),
        "report_sha256": sha256_file(authoritative.get("report", "")),
        "checkpoint": authoritative.get("checkpoint"),
        "checkpoint_sha256": authoritative.get("checkpoint_sha256"),
        "actor_loss_mode": authoritative.get("actor_loss_mode"),
        "source_kind": authoritative.get("source_kind"),
        "sampler_mode": authoritative.get("sampler_mode"),
        "support_artifacts": authoritative.get("support_artifacts"),
    }
    expect_equal(
        gate,
        comparison,
        label=f"cohort seller versus authoritative {label}",
    )
    return authoritative


def _validate_cohort_seller_release_binding(value: dict, gates: dict) -> dict:
    """Prove E2's buyer is exactly the buyer used to release its seller."""

    record = value.get("seller_release")
    if not isinstance(record, dict):
        fail("E1 gate cohort has no seller-release record")
    release_path = Path(record.get("path", "")).expanduser().resolve()
    expect_equal(
        record.get("sha256"), sha256_file(release_path),
        label="cohort seller-release SHA-256",
    )
    release = validated_e1_seller_release(release_path)
    expect_equal(
        gates.get("buyer"), release.get("buyer_gate"),
        label="cohort buyer versus seller-training buyer",
    )
    seller_support = gates.get("seller", {}).get("support_artifacts", {})
    expect_equal(
        seller_support.get("seller_release"), record,
        label="cohort seller gate versus seller release",
    )
    return release


def write_e1_gate_cohort(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    if os.path.lexists(output):
        fail(f"refusing to overwrite shared E1 gate cohort: {output}")
    entries = {}
    for role in ("buyer", "seller"):
        report = Path(getattr(args, f"{role}_report")).expanduser().resolve()
        checkpoint = Path(
            getattr(args, f"{role}_checkpoint")
        ).expanduser().resolve()
        mode = getattr(args, f"{role}_actor_loss_mode")
        validated = validate_e1_gate(argparse.Namespace(
            report=str(report), checkpoint=str(checkpoint), role=role,
            actor_loss_mode=mode,
        ))
        entries[role] = {
            "report": str(report),
            "report_sha256": sha256_file(report),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": validated["sha256"],
            "actor_loss_mode": mode,
            "source_kind": validated["source_kind"],
            "sampler_mode": validated["sampler_mode"],
            "support_artifacts": validated["support_artifacts"],
        }
    if entries["seller"]["actor_loss_mode"] != "balanced":
        fail("E2 gate cohort requires the balanced E1 seller")
    if entries["buyer"]["source_kind"] != E1_PRIMARY_SOURCE_KIND:
        fail("E2 gate cohort requires the authoritative primary-economic buyer")
    _validate_authoritative_seller_gate_record(entries["seller"])
    seller_release = entries["seller"]["support_artifacts"].get(
        "seller_release"
    )
    if not isinstance(seller_release, dict):
        fail("E2 seller gate does not bind an immutable seller release")
    release_path = Path(seller_release.get("path", "")).expanduser().resolve()
    expect_equal(
        seller_release.get("sha256"), sha256_file(release_path),
        label="seller gate release SHA-256",
    )
    release = validated_e1_seller_release(release_path)
    expect_equal(
        entries["buyer"], release.get("buyer_gate"),
        label="E2 buyer versus seller-training buyer",
    )
    result = {
        "schema": "stackpomdp.atari.e2_e1_gate_cohort.v3",
        "code_root": str(REPOSITORY_ROOT),
        "code_head": validate_code_root(),
        "buyer_authority": E1_PRIMARY_SOURCE_KIND,
        "seller_release": seller_release,
        "e1_gates": entries,
    }
    atomic_write_new_json(output, result)
    return {"kind": "e1_gate_cohort", "path": str(output), **result}


def read_e1_gate_cohort(args: argparse.Namespace) -> dict:
    path = Path(args.cohort_manifest).expanduser().resolve()
    return {
        "kind": "e1_gate_cohort",
        "path": str(path),
        **validated_e1_gate_cohort(path),
    }


def write_e1_seller_release(args: argparse.Namespace) -> dict:
    """Pin the exact primary-economic buyer that authorizes seller training."""

    output = Path(args.output).expanduser().resolve()
    if os.path.lexists(output):
        fail(f"refusing to overwrite E1 seller-release manifest: {output}")
    report = Path(args.buyer_report).expanduser().resolve()
    checkpoint = Path(args.buyer_checkpoint).expanduser().resolve()
    validated = validate_e1_gate(argparse.Namespace(
        report=str(report),
        checkpoint=str(checkpoint),
        role="buyer",
        actor_loss_mode=args.buyer_actor_loss_mode,
    ))
    if validated["source_kind"] != E1_PRIMARY_SOURCE_KIND:
        fail("E1 seller training requires the authoritative primary-economic buyer")
    gate = {
        "report": str(report),
        "report_sha256": sha256_file(report),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": validated["sha256"],
        "actor_loss_mode": args.buyer_actor_loss_mode,
        "source_kind": validated["source_kind"],
        "sampler_mode": validated["sampler_mode"],
        "support_artifacts": validated["support_artifacts"],
    }
    result = {
        "schema": "stackpomdp.atari.e1_seller_release.v2",
        "code_root": str(REPOSITORY_ROOT),
        "code_head": validate_code_root(),
        "seller_training_actor_loss_mode": "balanced",
        "buyer_gate": gate,
    }
    atomic_write_new_json(output, result)
    return {"kind": "e1_seller_release", "path": str(output), **result}


def validated_e1_seller_release(path: Path) -> dict:
    value = load_json(path)
    expect_equal(
        value.get("schema"), "stackpomdp.atari.e1_seller_release.v2",
        label="E1 seller-release schema",
    )
    expect_equal(value.get("code_head"), validate_code_root(), label="code HEAD")
    expect_equal(
        value.get("seller_training_actor_loss_mode"), "balanced",
        label="E1 seller training actor-loss mode",
    )
    buyer = validate_e1_gate_record(value.get("buyer_gate"), role="buyer")
    if buyer["source_kind"] != E1_PRIMARY_SOURCE_KIND:
        fail("E1 seller release is not bound to the primary-economic buyer")
    return value


def read_e1_seller_release(args: argparse.Namespace) -> dict:
    path = Path(args.release_manifest).expanduser().resolve()
    return {
        "kind": "e1_seller_release",
        "path": str(path),
        **validated_e1_seller_release(path),
    }


def _validate_seller_selection_gate_support(
        report_path: Path,
        checkpoint: Path,
        digest: str,
        candidate_hashes: list[str],
) -> dict:
    """Recheck the immutable release-bound E1 seller selection gate."""

    gate_path = report_path.with_name(f"{report_path.stem}.gate.json")
    gate = load_json(gate_path)
    expect_equal(gate.get("schema_version"), 1, label="seller gate schema")
    expect_equal(gate.get("kind"), E1_SELLER_GATE_KIND, label="seller gate kind")
    if gate.get("passed") is not True:
        fail("E1 seller selection gate did not pass")
    expect_equal(gate.get("role"), "seller", label="seller gate role")
    expect_equal(gate.get("actor_loss_mode"), "balanced", label="seller gate mode")

    report_record = gate.get("report", {})
    same_path(report_record.get("path"), report_path, label="seller gate report")
    expect_equal(
        report_record.get("sha256"), sha256_file(report_path),
        label="seller report SHA-256",
    )
    selected_record = gate.get("selected_checkpoint", {})
    same_path(
        selected_record.get("path"), checkpoint,
        label="seller gate selected checkpoint",
    )
    expect_equal(
        selected_record.get("sha256"), digest,
        label="seller selected SHA-256",
    )
    expect_equal(
        gate.get("candidate_sha256"), candidate_hashes,
        label="seller gate candidate hashes",
    )

    release_record = gate.get("seller_release", {})
    release_path = Path(release_record.get("path", "")).expanduser().resolve()
    expect_equal(
        release_record.get("sha256"), sha256_file(release_path),
        label="seller-release SHA-256",
    )
    validated_e1_seller_release(release_path)
    revision = validate_recorded_revision(
        gate.get("selector", {}).get("code_revision")
    )
    report_revision = load_json(report_path).get(
        "immutable_evaluation", {}
    ).get("selector_code_revision")
    expect_equal(
        revision, report_revision,
        label="seller gate evaluator revision",
    )
    expect_equal(
        gate.get("selector", {}).get("evaluator"), E1_EVALUATOR,
        label="seller selector evaluator",
    )
    return {
        "seller_selection_gate": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
        },
        "seller_release": {
            "path": str(release_path),
            "sha256": sha256_file(release_path),
        },
        "selector_code_revision": revision,
    }


def write_e1_seller_selection_gate(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    report = Path(args.report).expanduser().resolve()
    selected = Path(args.selected).expanduser().resolve()
    release = Path(args.release_manifest).expanduser().resolve()
    expected_output = report.with_name(f"{report.stem}.gate.json")
    if output != expected_output:
        fail(f"seller gate must be adjacent to its report: {expected_output}")
    if os.path.lexists(output):
        fail(f"refusing to overwrite E1 seller selection gate: {output}")

    validated = _validate_e1_gate_core(
        argparse.Namespace(
            report=str(report), role="seller", checkpoint=str(selected),
            actor_loss_mode="balanced",
        ),
        require_seller_support=False,
    )
    validated_e1_seller_release(release)
    selector_revision = validate_selector_code_root(
        Path(args.selector_code_root)
    )
    expect_equal(
        selector_revision,
        load_json(report).get("immutable_evaluation", {}).get(
            "selector_code_revision"
        ),
        label="seller selector code revision",
    )
    result = {
        "schema_version": 1,
        "kind": E1_SELLER_GATE_KIND,
        "passed": True,
        "role": "seller",
        "actor_loss_mode": "balanced",
        "report": {
            "path": str(report),
            "sha256": sha256_file(report),
        },
        "selected_checkpoint": {
            "path": str(selected),
            "sha256": validated["sha256"],
        },
        "candidate_sha256": validated["candidate_sha256"],
        "seller_release": {
            "path": str(release),
            "sha256": sha256_file(release),
        },
        "selector": {
            "code_revision": selector_revision,
            "evaluator": E1_EVALUATOR,
            "confirmation_policy": "screen_winner_only_no_fallback",
        },
    }
    atomic_write_new_json(output, result)
    support = _validate_seller_selection_gate_support(
        report, selected, validated["sha256"], validated["candidate_sha256"]
    )
    return {"kind": "e1_seller_selection_gate", **result, **support}


def read_e1_seller_selection_gate(args: argparse.Namespace) -> dict:
    gate = load_json(Path(args.gate).expanduser().resolve())
    report = Path(gate.get("report", {}).get("path", "")).expanduser().resolve()
    selected = Path(
        gate.get("selected_checkpoint", {}).get("path", "")
    ).expanduser().resolve()
    validated = _validate_e1_gate_core(
        argparse.Namespace(
            report=str(report), role="seller", checkpoint=str(selected),
            actor_loss_mode="balanced",
        ),
        require_seller_support=False,
    )
    support = _validate_seller_selection_gate_support(
        report, selected, validated["sha256"], validated["candidate_sha256"]
    )
    return {"kind": "e1_seller_selection_gate", **gate, **support}


def validate_rom(args: argparse.Namespace) -> dict:
    path = Path(args.rom).expanduser().resolve()
    digest = sha256_file(path)
    expect_equal(digest, args.sha256.lower(), label="canonical Atari ROM SHA-256")
    return {
        "kind": "canonical_rom",
        "path": str(path),
        "sha256": digest,
        "passed": True,
    }


def write_e2_input_manifest(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    if os.path.lexists(output):
        fail(f"refusing to overwrite E2 input manifest: {output}")
    cohort_path = Path(args.cohort_manifest).expanduser().resolve()
    cohort = validated_e1_gate_cohort(cohort_path)
    entries = cohort["e1_gates"]
    response_role = "seller" if args.role == "buyer" else "buyer"
    result = {
        "schema": "stackpomdp.atari.e2_pipeline_inputs.v2",
        "code_root": str(REPOSITORY_ROOT),
        "code_head": validate_code_root(),
        "leader_role": args.role,
        "frozen_response_role": response_role,
        "same_role_initialization_role": args.role,
        "training_actor_loss_mode": "balanced",
        "e1_gate_cohort": str(cohort_path),
        "e1_gate_cohort_sha256": sha256_file(cohort_path),
        "protocol": {
            "retained_step_timesteps": list(E2_STEPS),
            "include_post_update_final_base": True,
            "candidate_count": 6,
            "common_screen_episodes": 20,
            "common_screen_seed_start": 4_000_001,
            "confirmation_episodes": 100,
            "confirmation_seed_start": 5_000_001,
            "confirmation_policy": "screen_winner_only_no_fallback",
        },
        "e1_gates": entries,
    }
    atomic_write_new_json(output, result)
    return {"kind": "e2_input_manifest", "path": str(output), **result}


def read_e2_input_manifest(args: argparse.Namespace) -> dict:
    path = Path(args.input_manifest).expanduser().resolve()
    value = validated_pipeline_inputs(path, role=args.role)
    return {"kind": "e2_input_manifest", "path": str(path), **value}


def inspect_e2_checkpoint(args: argparse.Namespace) -> tuple[dict, str]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    response = Path(args.response).expanduser().resolve()
    leader_e1 = Path(args.leader_e1).expanduser().resolve()
    digest = validate_zip(checkpoint)
    response_digest = validate_zip(response)
    leader_e1_digest = validate_zip(leader_e1)
    inputs = validated_pipeline_inputs(
        Path(args.input_manifest).expanduser().resolve(), role=args.role
    )

    from replication.atari.evaluate_atari_stackpomdp_leader_sb3 import (
        load_e2_checkpoint,
    )
    from replication.atari.sb3_common import model_actor_loss_mode
    from replication.atari.train_atari_stackpomdp_leader_sb3 import (
        E2_PROVENANCE_ATTRIBUTE,
        checkpoint_policy_metadata,
        validate_e2_provenance_manifest,
    )

    model = load_e2_checkpoint(
        checkpoint,
        leader_role=args.role,
        device="cpu",
        expected_sha256=digest,
    )
    try:
        expect_equal(
            int(getattr(model, "num_timesteps", -1)),
            args.timesteps,
            label="E2 num_timesteps",
        )
        expect_equal(
            model_actor_loss_mode(model),
            "balanced",
            label="E2 actor loss mode",
        )
        manifest = validate_e2_provenance_manifest(
            getattr(model, E2_PROVENANCE_ATTRIBUTE, None)
        )
    finally:
        del model

    scientific = manifest.get("scientific_config", {})
    environment = scientific.get("environment", {})
    protocol = scientific.get("protocol", {})
    optimization = scientific.get("optimization", {})
    leader_policy = scientific.get("leader_policy", {})
    expect_equal(scientific.get("stage"), "e2", label="E2 stage")
    expect_equal(
        scientific.get("leader_role"), args.role, label="E2 leader role"
    )
    expect_equal(
        scientific.get("follower_role"),
        "seller" if args.role == "buyer" else "buyer",
        label="E2 follower role",
    )
    for key, expected in {
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "fixed_event_steps": None,
        "noop_max": 30,
        "frame_skip": 4,
        "frame_stack": 4,
        "episodic_life": True,
        "clip_game_rewards": True,
        "max_frames": 100_000,
    }.items():
        expect_equal(environment.get(key), expected, label=f"E2 env {key}")
    for key, expected in {
        "trade_events": 5,
        "query_transitions": 5,
        "cached_trade_replays": 5,
        "outer_episode_transitions": 210,
        "policy_action_cache": True,
        "leader_economic_input": "event_only",
        "response_economic_input": "full",
        "response_algorithm": "frozen_meta_policy",
    }.items():
        expect_equal(protocol.get(key), expected, label=f"E2 protocol {key}")
    for key, expected in {
        "algorithm": "PPO",
        "actor_loss_mode": "balanced",
        "seed": 1,
        "num_envs": 4,
        "start_method": "spawn",
        "n_steps": 210,
        "batch_size": 840,
        "n_epochs": 4,
        "learning_rate": 1.0e-4,
        "pretrained_lr_scale": 0.1,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "clip_range": 0.1,
        "entropy_coefficient": 0.01,
        "value_coefficient": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
    }.items():
        expect_equal(
            optimization.get(key), expected, label=f"E2 optimization {key}"
        )
    expect_equal(
        leader_policy.get("economic_role"), args.role, label="E2 policy role"
    )
    expect_equal(
        leader_policy.get("economic_input_mode"),
        "event_only",
        label="E2 policy input mode",
    )
    artifacts = manifest.get("artifacts", {})
    fresh_response = checkpoint_policy_metadata(
        response, device="cpu", label="E2 frozen E1 response"
    )
    fresh_leader_e1 = checkpoint_policy_metadata(
        leader_e1, device="cpu", label="E2 same-role E1 initialization"
    )
    expect_equal(
        artifacts.get("frozen_response"),
        {
            "sha256": fresh_response["sha256"],
            "policy": fresh_response["policy_metadata"],
        },
        label="E2 embedded frozen-response policy provenance",
    )
    expect_equal(
        artifacts.get("same_role_e1_initialization"),
        {
            "sha256": fresh_leader_e1["sha256"],
            "policy": fresh_leader_e1["policy_metadata"],
        },
        label="E2 embedded same-role policy provenance",
    )
    expect_equal(
        artifacts.get("frozen_response", {}).get("sha256"),
        response_digest,
        label="E2 frozen-response SHA-256",
    )
    expect_equal(
        artifacts.get("same_role_e1_initialization", {}).get("sha256"),
        leader_e1_digest,
        label="E2 E1-initialization SHA-256",
    )
    response_role = "seller" if args.role == "buyer" else "buyer"
    expect_equal(
        inputs["e1_gates"][response_role]["checkpoint_sha256"],
        response_digest,
        label="chosen response in pipeline-input manifest",
    )
    expect_equal(
        inputs["e1_gates"][args.role]["checkpoint_sha256"],
        leader_e1_digest,
        label="chosen initialization in pipeline-input manifest",
    )
    expect_equal(
        artifacts.get("frozen_response", {}).get("policy", {}).get(
            "actor_loss_mode"
        ),
        inputs["e1_gates"][response_role]["actor_loss_mode"],
        label="embedded response actor-loss mode",
    )
    expect_equal(
        artifacts.get("same_role_e1_initialization", {}).get("policy", {}).get(
            "actor_loss_mode"
        ),
        inputs["e1_gates"][args.role]["actor_loss_mode"],
        label="embedded initialization actor-loss mode",
    )
    return manifest, digest


def validate_e2_checkpoint(args: argparse.Namespace) -> dict:
    manifest, digest = inspect_e2_checkpoint(args)
    return {
        "kind": "e2_checkpoint",
        "role": args.role,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "sha256": digest,
        "timesteps": args.timesteps,
        "provenance_fingerprint": manifest["fingerprint_sha256"],
        "passed": True,
    }


def validate_e2_output(args: argparse.Namespace) -> dict:
    manifest, digest = inspect_e2_checkpoint(args)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    evaluation_path = checkpoint.with_name(
        f"{checkpoint.stem}.evaluation.json"
    )
    provenance_path = checkpoint.with_name(
        f"{checkpoint.stem}.provenance.json"
    )
    training_path = checkpoint.with_name(f"{checkpoint.stem}.training.jsonl")
    evaluation = load_json(evaluation_path)
    provenance = load_json(provenance_path)
    expect_equal(provenance, manifest, label="E2 provenance sidecar")
    expect_equal(
        evaluation.get("e2_provenance_manifest"),
        manifest,
        label="E2 evaluation provenance",
    )
    if not isinstance(evaluation.get("summary"), dict):
        fail("E2 evaluation has no summary")
    if not training_path.is_file() or training_path.is_symlink():
        fail(f"E2 training JSONL is absent or not regular: {training_path}")
    rows = []
    with training_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail(f"invalid training JSONL line {number}: {error}")
            if not isinstance(row, dict):
                fail(f"training JSONL line {number} is not an object")
            rows.append(row)
    if not rows:
        fail("E2 training JSONL is empty")
    observed_steps = [
        int(row["train/total_timesteps"])
        for row in rows
        if "train/total_timesteps" in row
    ]
    if not observed_steps or max(observed_steps) != args.timesteps:
        fail(
            "E2 local training trace does not end at the expected timestep: "
            f"expected {args.timesteps}, max={max(observed_steps, default=None)}"
        )
    return {
        "kind": "e2_output",
        "role": args.role,
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "timesteps": args.timesteps,
        "evaluation": str(evaluation_path),
        "provenance": str(provenance_path),
        "training_log": str(training_path),
        "provenance_fingerprint": manifest["fingerprint_sha256"],
        "passed": True,
    }


def validate_e2_family(args: argparse.Namespace) -> dict:
    if len(args.step_checkpoint) != len(E2_STEPS):
        fail(
            f"E2 family requires exactly {len(E2_STEPS)} retained step ZIPs"
        )
    manifests = []
    hashes = []
    candidates = []
    for timesteps, checkpoint in zip(E2_STEPS, args.step_checkpoint):
        local = argparse.Namespace(
            role=args.role,
            checkpoint=checkpoint,
            response=args.response,
            leader_e1=args.leader_e1,
            timesteps=timesteps,
            input_manifest=args.input_manifest,
        )
        manifest, digest = inspect_e2_checkpoint(local)
        manifests.append(manifest)
        hashes.append(digest)
        candidates.append({
            "kind": "retained_step",
            "timesteps": timesteps,
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "sha256": digest,
        })
    final_args = argparse.Namespace(
        role=args.role,
        checkpoint=args.base_checkpoint,
        response=args.response,
        leader_e1=args.leader_e1,
        timesteps=2_000_040,
        input_manifest=args.input_manifest,
    )
    final_manifest, final_digest = inspect_e2_checkpoint(final_args)
    manifests.append(final_manifest)
    hashes.append(final_digest)
    candidates.append({
        "kind": "post_update_final_base",
        "timesteps": 2_000_040,
        "checkpoint": str(Path(args.base_checkpoint).expanduser().resolve()),
        "sha256": final_digest,
    })
    if len(set(hashes)) != 6:
        fail("E2 selection family does not contain six byte-distinct candidates")
    fingerprints = {manifest["fingerprint_sha256"] for manifest in manifests}
    lineages = {manifest["run_lineage_id"] for manifest in manifests}
    identities = {
        manifest["scientific_identity_sha256"] for manifest in manifests
    }
    if len(fingerprints) != 1 or len(lineages) != 1 or len(identities) != 1:
        fail("E2 candidates do not belong to one exact training lineage/config")
    return {
        "kind": "e2_family",
        "role": args.role,
        "candidate_count": 6,
        "candidates": candidates,
        "provenance_fingerprint": next(iter(fingerprints)),
        "run_lineage_id": next(iter(lineages)),
        "passed": True,
    }


def validate_e2_report(args: argparse.Namespace) -> dict:
    report_path = Path(args.report).expanduser().resolve()
    selected = Path(args.selected).expanduser().resolve()
    response = Path(args.response).expanduser().resolve()
    report = load_json(report_path)
    expect_equal(report.get("schema_version"), 2, label="E2 report schema")
    passed = report.get("passed")
    if type(passed) is not bool:
        fail("E2 report passed field is not Boolean")
    if args.expect != "either":
        expect_equal(
            passed, args.expect == "passed", label="E2 selection outcome"
        )
    expect_equal(report.get("evaluator"), E2_EVALUATOR, label="E2 evaluator")
    environment = report.get("environment_config", {})
    expect_equal(
        environment.get("leader_role"), args.role, label="E2 report role"
    )
    expect_equal(
        environment.get("gameplay_horizon"), 200, label="E2 report horizon"
    )
    expect_equal(
        environment.get("event_tail_steps"), 0, label="E2 report tail"
    )
    response_digest = validate_zip(response)
    inputs = validated_pipeline_inputs(
        Path(args.input_manifest).expanduser().resolve(), role=args.role
    )
    same_path(
        report.get("response_checkpoint"),
        response,
        label="E2 report response checkpoint",
    )
    expect_equal(
        report.get("response_checkpoint_sha256"),
        response_digest,
        label="E2 report response SHA-256",
    )
    response_role = "seller" if args.role == "buyer" else "buyer"
    expect_equal(
        inputs["e1_gates"][response_role]["checkpoint_sha256"],
        response_digest,
        label="E2 report chosen response",
    )
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("E2 report does not declare its artifact set")
    same_path(
        artifacts.get("report_json"), report_path, label="E2 report artifact"
    )
    for label, raw in artifacts.items():
        if not isinstance(raw, str):
            fail(f"E2 artifact path {label} is not a string")
        artifact = Path(raw).expanduser().resolve()
        if not artifact.is_file() or artifact.is_symlink():
            fail(f"E2 artifact {label} is absent or not regular: {artifact}")

    if len(args.candidate) != 6:
        fail("E2 report validation requires exactly six declared candidates")
    expected_candidates = []
    for raw in args.candidate:
        path = Path(raw).expanduser().resolve()
        expected_candidates.append({
            "path": path,
            "sha256": validate_zip(path),
        })
    expected_hashes = [item["sha256"] for item in expected_candidates]
    if len(set(expected_hashes)) != 6:
        fail("declared E2 family is not byte-distinct")
    screen = report.get("screen")
    if not isinstance(screen, dict):
        fail("E2 report has no common screen")
    for key, expected in {
        "episodes_per_checkpoint": 20,
        "seed_start": 4_000_001,
        "seed_end": 4_000_020,
    }.items():
        expect_equal(screen.get(key), expected, label=f"E2 screen {key}")
    pairing = screen.get("common_seed_schedule_check")
    if not isinstance(pairing, dict) or pairing.get("passed") is not True:
        fail("E2 common screen seed/schedule check did not pass")
    expect_equal(pairing.get("episodes"), 20, label="common screen episodes")
    pairs = pairing.get("seed_schedule_pairs")
    if not isinstance(pairs, list) or [
        row.get("evaluation_seed") for row in pairs
    ] != list(range(4_000_001, 4_000_021)):
        fail("E2 common screen does not contain the exact seed schedule")
    results = screen.get("checkpoint_results")
    if not isinstance(results, list) or len(results) != 6:
        fail("E2 common screen must contain exactly six checkpoint results")
    observed = []
    fingerprints = set()
    for result in results:
        if not isinstance(result, dict):
            fail("E2 screen result is not an object")
        path = Path(result.get("checkpoint_path", "")).expanduser().resolve()
        observed.append((path, result.get("checkpoint_sha256")))
        fingerprints.add(result.get("e2_provenance_fingerprint"))
        expect_equal(
            result.get("response_checkpoint_sha256"),
            response_digest,
            label="screen response SHA-256",
        )
        expect_equal(result.get("seed_start"), 4_000_001, label="screen seed")
        expect_equal(result.get("seed_end"), 4_000_020, label="screen seed end")
    expected_pairs = {
        (item["path"], item["sha256"]) for item in expected_candidates
    }
    if set(observed) != expected_pairs or len(set(observed)) != 6:
        fail("E2 report screen does not match the exact declared family")
    if len(fingerprints) != 1 or None in fingerprints:
        fail("E2 screen candidates do not share one provenance fingerprint")

    selection = report.get("selection")
    if not isinstance(selection, dict):
        fail("E2 report has no selection record")
    screen_winner = selection.get("screen_selected_checkpoint_sha256")
    if screen_winner is not None and screen_winner not in expected_hashes:
        fail("E2 screen winner is outside the declared family")
    confirmation = report.get("confirmation")
    attempts = report.get("confirmation_attempts")
    if not isinstance(attempts, list) or len(attempts) > 1:
        fail("E2 selector attempted confirmation fallback")
    if confirmation is None:
        if attempts or screen_winner is not None:
            fail("E2 report omitted confirmation for a screen winner")
    else:
        if len(attempts) != 1 or attempts[0] != confirmation:
            fail("E2 report must retain exactly its sole confirmation attempt")
        for key, expected in {
            "screen_rank": 1,
            "episodes": 100,
            "seed_start": 5_000_001,
            "seed_end": 5_000_100,
            "disjoint_from_screen": True,
        }.items():
            expect_equal(
                confirmation.get(key), expected, label=f"confirmation {key}"
            )
        confirmed = confirmation.get("result")
        if not isinstance(confirmed, dict):
            fail("E2 confirmation has no result")
        expect_equal(
            confirmed.get("checkpoint_sha256"),
            screen_winner,
            label="confirmed screen-winner SHA-256",
        )
        expect_equal(
            confirmed.get("seed_start"), 5_000_001,
            label="confirmation result seed start",
        )
        expect_equal(
            confirmed.get("seed_end"), 5_000_100,
            label="confirmation result seed end",
        )
        checks = confirmation.get("checks")
        if not isinstance(checks, dict):
            fail("E2 confirmation has no screen-match checks")
        expect_equal(
            checks.get("checkpoint_hash_matches_screen"),
            True,
            label="confirmation winner identity",
        )
    selected_hash = selection.get("selected_checkpoint_sha256")
    if passed:
        expect_equal(selected_hash, screen_winner, label="selected winner hash")
        if confirmation is None or confirmation.get("checks", {}).get(
                "passed"
        ) is not True:
            fail("passing E2 report has no passing sole confirmation")
    else:
        expect_equal(selected_hash, None, label="failed selected hash")

    alias = report.get("selected_alias")
    if passed:
        if not isinstance(alias, dict):
            fail("passing E2 report has no selected alias")
        same_path(
            alias.get("selected_checkpoint_path"),
            selected,
            label="E2 selected alias",
        )
        expect_equal(
            alias.get("created_by_selection_run"),
            True,
            label="E2 alias ownership",
        )
        expect_equal(
            alias.get("copy_verified"), True, label="E2 alias copy check"
        )
        digest = validate_zip(selected)
        expect_equal(
            alias.get("checkpoint_sha256"), digest, label="E2 alias SHA-256"
        )
    else:
        expect_equal(alias, None, label="failed E2 alias")
        if os.path.lexists(selected):
            fail(f"failed E2 gate unexpectedly left a selected alias: {selected}")

    return {
        "kind": "e2_report",
        "role": args.role,
        "report": str(report_path),
        "selected": str(selected) if passed else None,
        "passed": passed,
    }


def _e2_orchestration_role_record(
        *, role: str, training_exit_code: int, selector_exit_code: int,
        checkpoint_root: Path, result_root: Path, cohort_path: Path,
) -> dict:
    if type(training_exit_code) is not int or type(selector_exit_code) is not int:
        fail(f"E2 {role} orchestration statuses must be integers")
    if training_exit_code != 0:
        if selector_exit_code != -1:
            fail(f"E2 {role} selector must be unrun after training failure")
        return {
            "training_exit_code": training_exit_code,
            "selector_exit_code": None,
            "training_completed": False,
            "selection_outcome": "not_run",
        }

    stem = checkpoint_root / (
        f"leader_{role}_e2_ppo_balanced_seed1_firefix_retrain_"
        f"{E2_NAMESPACE}"
    )
    base = stem.with_suffix(".zip")
    input_manifest = stem.with_suffix(".pipeline_inputs.json")
    inputs = validated_pipeline_inputs(input_manifest, role=role)
    same_path(
        inputs.get("e1_gate_cohort"), cohort_path,
        label=f"E2 {role} orchestration cohort",
    )
    candidate_paths = [
        stem.with_name(f"{stem.name}_step{step}.zip") for step in E2_STEPS
    ] + [base]
    candidate_hashes = [validate_zip(path) for path in candidate_paths]
    response_role = "seller" if role == "buyer" else "buyer"
    response = inputs["e1_gates"][response_role]["checkpoint"]
    leader_e1 = inputs["e1_gates"][role]["checkpoint"]
    validate_e2_output(argparse.Namespace(
        role=role,
        checkpoint=str(base),
        response=response,
        leader_e1=leader_e1,
        timesteps=2_000_040,
        input_manifest=str(input_manifest),
    ))
    validate_e2_family(argparse.Namespace(
        role=role,
        response=response,
        leader_e1=leader_e1,
        input_manifest=str(input_manifest),
        step_checkpoint=[str(path) for path in candidate_paths[:-1]],
        base_checkpoint=str(base),
    ))

    run_name = f"e2_{role}_balanced_all6_selector_v2_{E2_NAMESPACE}"
    report = result_root / f"{run_name}.json"
    selected = checkpoint_root / (
        f"leader_{role}_e2_ppo_balanced_seed1_firefix_retrain_"
        f"{E2_NAMESPACE}_selected.zip"
    )
    if selector_exit_code in (0, 2):
        validate_e2_report(argparse.Namespace(
            role=role,
            report=str(report),
            selected=str(selected),
            response=response,
            input_manifest=str(input_manifest),
            candidate=[str(path) for path in candidate_paths],
            expect="passed" if selector_exit_code == 0 else "failed",
        ))
        report_record = {"path": str(report), "sha256": sha256_file(report)}
        selected_record = (
            {"path": str(selected), "sha256": validate_zip(selected)}
            if selector_exit_code == 0 else None
        )
        outcome = "passed" if selector_exit_code == 0 else "failed"
    else:
        report_record = (
            {"path": str(report), "sha256": sha256_file(report)}
            if report.is_file() and not report.is_symlink() else None
        )
        selected_record = None
        outcome = "crashed"

    return {
        "training_exit_code": 0,
        "selector_exit_code": selector_exit_code,
        "training_completed": True,
        "selection_outcome": outcome,
        "input_manifest": {
            "path": str(input_manifest),
            "sha256": sha256_file(input_manifest),
        },
        "candidate_checkpoints": [
            {"path": str(path), "sha256": digest}
            for path, digest in zip(candidate_paths, candidate_hashes)
        ],
        "report": report_record,
        "selected_checkpoint": selected_record,
    }


def build_e2_orchestration_summary(args: argparse.Namespace) -> dict:
    cohort_path = Path(args.cohort_manifest).expanduser().resolve()
    validated_e1_gate_cohort(cohort_path)
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    roles = {}
    for role in ("buyer", "seller"):
        roles[role] = _e2_orchestration_role_record(
            role=role,
            training_exit_code=getattr(args, f"{role}_training_exit_code"),
            selector_exit_code=getattr(args, f"{role}_selector_exit_code"),
            checkpoint_root=checkpoint_root,
            result_root=result_root,
            cohort_path=cohort_path,
        )
    completed = all(
        row["training_completed"]
        and row["selection_outcome"] in ("passed", "failed")
        for row in roles.values()
    )
    all_passed = all(
        row["selection_outcome"] == "passed" for row in roles.values()
    )
    recorded_revision = getattr(args, "recorded_automation_revision", None)
    automation_revision = (
        validate_recorded_revision(recorded_revision)
        if recorded_revision is not None
        else validate_selector_code_root(Path(args.automation_code_root))
    )
    return {
        "schema": E2_ORCHESTRATION_SCHEMA,
        "code_head": validate_code_root(),
        "automation_code_revision": automation_revision,
        "checkpoint_root": str(checkpoint_root),
        "result_root": str(result_root),
        "e1_gate_cohort": {
            "path": str(cohort_path),
            "sha256": sha256_file(cohort_path),
        },
        "execution_order": ["buyer", "seller"],
        "roles": roles,
        "orchestration_completed": completed,
        "all_scientific_gates_passed": all_passed,
        "passed": all_passed,
    }


def write_e2_orchestration_summary(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    if os.path.lexists(output):
        fail(f"refusing to overwrite E2 orchestration summary: {output}")
    result = build_e2_orchestration_summary(args)
    atomic_write_new_json(output, result)
    return {"kind": "e2_orchestration_summary", "path": str(output), **result}


def validate_e2_orchestration_summary(args: argparse.Namespace) -> dict:
    path = Path(args.summary).expanduser().resolve()
    value = load_json(path)
    expect_equal(
        value.get("schema"), E2_ORCHESTRATION_SCHEMA,
        label="E2 orchestration-summary schema",
    )
    roles = value.get("roles", {})
    if not isinstance(roles, dict) or set(roles) != {"buyer", "seller"}:
        fail("E2 orchestration summary must contain both roles")
    reconstructed = argparse.Namespace(
        cohort_manifest=value.get("e1_gate_cohort", {}).get("path"),
        checkpoint_root=value.get("checkpoint_root"),
        result_root=value.get("result_root"),
        buyer_training_exit_code=roles["buyer"].get("training_exit_code"),
        buyer_selector_exit_code=(
            -1 if roles["buyer"].get("selector_exit_code") is None
            else roles["buyer"].get("selector_exit_code")
        ),
        seller_training_exit_code=roles["seller"].get("training_exit_code"),
        seller_selector_exit_code=(
            -1 if roles["seller"].get("selector_exit_code") is None
            else roles["seller"].get("selector_exit_code")
        ),
        recorded_automation_revision=value.get("automation_code_revision"),
    )
    expected = build_e2_orchestration_summary(reconstructed)
    expect_equal(value, expected, label="E2 orchestration-summary contents")
    return {"kind": "e2_orchestration_summary", "path": str(path), **value}


def add_e2_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--role", choices=("buyer", "seller"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--leader-e1", required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--input-manifest", required=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    e1 = subparsers.add_parser("e1-gate")
    e1.add_argument("--report", required=True)
    e1.add_argument("--role", choices=("buyer", "seller"), required=True)
    e1.add_argument("--checkpoint", required=True)
    e1.add_argument(
        "--actor-loss-mode", choices=("balanced", "standard"), required=True
    )
    e1.set_defaults(handler=validate_e1_gate)

    discovery = subparsers.add_parser("discover-e1-gate")
    discovery.add_argument("--role", choices=("buyer", "seller"), required=True)
    discovery.add_argument("--output-dir", required=True)
    discovery.add_argument("--override-report")
    discovery.add_argument("--require-mode", choices=("balanced", "standard"))
    discovery.set_defaults(handler=discover_e1_gate)

    cohort = subparsers.add_parser("write-e1-gate-cohort")
    cohort.add_argument("--output", required=True)
    for role in ("buyer", "seller"):
        cohort.add_argument(f"--{role}-report", required=True)
        cohort.add_argument(f"--{role}-checkpoint", required=True)
        cohort.add_argument(
            f"--{role}-actor-loss-mode",
            choices=("balanced", "standard"),
            required=True,
        )
    cohort.set_defaults(handler=write_e1_gate_cohort)

    read_cohort = subparsers.add_parser("read-e1-gate-cohort")
    read_cohort.add_argument("--cohort-manifest", required=True)
    read_cohort.set_defaults(handler=read_e1_gate_cohort)

    release = subparsers.add_parser("write-e1-seller-release")
    release.add_argument("--output", required=True)
    release.add_argument("--buyer-report", required=True)
    release.add_argument("--buyer-checkpoint", required=True)
    release.add_argument(
        "--buyer-actor-loss-mode",
        choices=("balanced", "standard"),
        required=True,
    )
    release.set_defaults(handler=write_e1_seller_release)

    read_release = subparsers.add_parser("read-e1-seller-release")
    read_release.add_argument("--release-manifest", required=True)
    read_release.set_defaults(handler=read_e1_seller_release)

    seller_gate = subparsers.add_parser("write-e1-seller-selection-gate")
    seller_gate.add_argument("--output", required=True)
    seller_gate.add_argument("--report", required=True)
    seller_gate.add_argument("--selected", required=True)
    seller_gate.add_argument("--release-manifest", required=True)
    seller_gate.add_argument("--selector-code-root", required=True)
    seller_gate.set_defaults(handler=write_e1_seller_selection_gate)

    read_seller_gate = subparsers.add_parser(
        "read-e1-seller-selection-gate"
    )
    read_seller_gate.add_argument("--gate", required=True)
    read_seller_gate.set_defaults(handler=read_e1_seller_selection_gate)

    rom = subparsers.add_parser("validate-rom")
    rom.add_argument("--rom", required=True)
    rom.add_argument("--sha256", required=True)
    rom.set_defaults(handler=validate_rom)

    inputs = subparsers.add_parser("write-e2-input-manifest")
    inputs.add_argument("--role", choices=("buyer", "seller"), required=True)
    inputs.add_argument("--output", required=True)
    inputs.add_argument("--cohort-manifest", required=True)
    inputs.set_defaults(handler=write_e2_input_manifest)

    read_inputs = subparsers.add_parser("read-e2-input-manifest")
    read_inputs.add_argument("--role", choices=("buyer", "seller"), required=True)
    read_inputs.add_argument("--input-manifest", required=True)
    read_inputs.set_defaults(handler=read_e2_input_manifest)

    e2 = subparsers.add_parser("e2-checkpoint")
    add_e2_checkpoint_arguments(e2)
    e2.set_defaults(handler=validate_e2_checkpoint)

    output = subparsers.add_parser("e2-output")
    add_e2_checkpoint_arguments(output)
    output.set_defaults(handler=validate_e2_output)

    family = subparsers.add_parser("e2-family")
    family.add_argument("--role", choices=("buyer", "seller"), required=True)
    family.add_argument("--response", required=True)
    family.add_argument("--leader-e1", required=True)
    family.add_argument("--input-manifest", required=True)
    family.add_argument("--step-checkpoint", action="append", required=True)
    family.add_argument("--base-checkpoint", required=True)
    family.set_defaults(handler=validate_e2_family)

    report = subparsers.add_parser("e2-report")
    report.add_argument("--report", required=True)
    report.add_argument("--role", choices=("buyer", "seller"), required=True)
    report.add_argument("--selected", required=True)
    report.add_argument("--response", required=True)
    report.add_argument("--input-manifest", required=True)
    report.add_argument("--candidate", action="append", required=True)
    report.add_argument(
        "--expect", choices=("passed", "failed", "either"), required=True
    )
    report.set_defaults(handler=validate_e2_report)

    summary = subparsers.add_parser("write-e2-orchestration-summary")
    summary.add_argument("--output", required=True)
    summary.add_argument("--cohort-manifest", required=True)
    summary.add_argument("--checkpoint-root", required=True)
    summary.add_argument("--result-root", required=True)
    summary.add_argument("--automation-code-root", required=True)
    for role in ("buyer", "seller"):
        summary.add_argument(
            f"--{role}-training-exit-code", type=int, required=True
        )
        summary.add_argument(
            f"--{role}-selector-exit-code", type=int, required=True
        )
    summary.set_defaults(handler=write_e2_orchestration_summary)

    read_summary = subparsers.add_parser("read-e2-orchestration-summary")
    read_summary.add_argument("--summary", required=True)
    read_summary.set_defaults(handler=validate_e2_orchestration_summary)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_code_root()
    result = args.handler(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
