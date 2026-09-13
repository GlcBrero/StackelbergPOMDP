import json
import hashlib
import math
from pathlib import Path
import subprocess
import sys

import pytest

from replication.atari.automation.aggregate_atari_e2_multiseed import (
    EXPECTED_SEEDS,
    numeric_summary,
    parse_args,
)


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "replication/atari/automation/unity_atari_e2_multiseed.sbatch"


def test_array_covers_both_paper_roles_and_ten_seeds():
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-19%8" in text
    assert "SEED=$((TASK_ID / 2 + 1))" in text
    assert "TASK_ID % 2 == 0" in text
    assert "--timesteps 2000040" in text
    assert "--screen-seed-start 4000001" in text
    assert "--confirmation-seed-start 5000001" in text


def test_cluster_launcher_has_no_personal_absolute_path():
    text = SBATCH.read_text(encoding="utf-8")
    assert "/Users/" not in text
    assert "/home/" not in text
    assert "STACKPOMDP_CODE_ROOT" in text
    assert "STACKPOMDP_RUN_ROOT" in text


def test_policy_level_sem_uses_ten_means_not_pooled_episodes():
    rows = [{"mean_leader_payoff": float(seed)} for seed in EXPECTED_SEEDS]
    result = numeric_summary(rows)["mean_leader_payoff"]
    assert result["n_policies"] == 10
    assert result["mean"] == pytest.approx(5.5)
    assert result["sem_across_policies"] == pytest.approx(
        3.0276503540974917 / math.sqrt(10)
    )


def test_aggregator_accepts_one_completed_role():
    args = parse_args([
        "--input-dir", "inputs",
        "--output", "aggregate.json",
        "--roles", "buyer",
    ])
    assert args.roles == ("buyer",)


def test_compactor_removes_transition_rows(tmp_path):
    environment_config = {
        "leader_role": "buyer",
        "rom_path": "/work/gianluca_brero_uri_edu/roms/space_invaders.bin",
        "rom_sha256": "0" * 64,
    }
    environment_hash = hashlib.sha256(json.dumps(
        environment_config,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")).hexdigest()
    report = {
        "passed": True,
        "evaluator": "test",
        "created_at_utc": "now",
        "environment_config": environment_config,
        "environment_config_sha256": environment_hash,
        "response_checkpoint_sha256": "b" * 64,
        "response_metadata": {
            "checkpoint_path": (
                "/Users/researcher/project/meta_seller_selected.zip"
            ),
        },
        "screen": {
            "episodes_per_checkpoint": 20,
            "seed_start": 4_000_001,
            "seed_end": 4_000_020,
            "common_seed_schedule_check": {"passed": True},
            "checkpoint_results": [{
                "checkpoint_id": "candidate",
                "checkpoint_path": (
                    "/work/gianluca_brero_uri_edu/run/candidate.zip"
                ),
                "checkpoint_sha256": "c" * 64,
                "environment_config_sha256": environment_hash,
                "e2_provenance_fingerprint": "d" * 64,
                "e2_provenance_manifest": {
                    "fingerprint_sha256": "d" * 64,
                    "scientific_identity_sha256": "e" * 64,
                },
                "summary": {"mean_leader_payoff": 1.0},
                "protocol": {"passed": True},
                "economic_gate": {"passed": True},
                "transition_rows": [{"large": "discarded"}],
            }],
        },
        "selection": {
            "ranking_rows": [{
                "checkpoint_path": "/home/researcher/run/candidate.zip",
                "checkpoint_sha256": "c" * 64,
            }],
            "screen_selected_checkpoint_sha256": "c" * 64,
            "selected_checkpoint_sha256": "c" * 64,
        },
        "selected_alias": {
            "selected_checkpoint_path": (
                "/work/gianluca_brero_uri_edu/run/selected.zip"
            ),
            "checkpoint_sha256": "c" * 64,
        },
        "confirmation": {
            "screen_rank": 1,
            "episodes": 100,
            "seed_start": 5_000_001,
            "seed_end": 5_000_100,
            "disjoint_from_screen": True,
            "checks": {"passed": True},
            "result": {
                "checkpoint_id": "candidate",
                "checkpoint_path": "/Users/researcher/run/candidate.zip",
                "checkpoint_sha256": "c" * 64,
                "environment_config_sha256": environment_hash,
                "e2_provenance_fingerprint": "d" * 64,
                "e2_provenance_manifest": {
                    "fingerprint_sha256": "d" * 64,
                    "scientific_identity_sha256": "e" * 64,
                },
                "summary": {"mean_leader_payoff": 1.0},
                "protocol": {"passed": True},
                "economic_gate": {"passed": True},
                "transition_rows": [{"large": "discarded"}],
            },
        },
    }
    source = tmp_path / "full.json"
    output = tmp_path / "compact.json"
    archive = tmp_path / "full.tar.gz"
    source.write_text(json.dumps(report), encoding="utf-8")
    archive.write_bytes(b"archive")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "replication.atari.automation.compact_atari_e2_selection",
            "--report",
            str(source),
            "--output",
            str(output),
            "--archive",
            str(archive),
            "--sha256-sidecars",
        ],
        cwd=ROOT,
        check=True,
    )
    compact = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(compact)
    assert "transition_rows" not in serialized
    assert "/work/" not in serialized
    assert "/Users/" not in serialized
    assert "/home/" not in serialized
    assert "gianluca_brero_uri_edu" not in serialized
    assert compact["full_report"]["path_at_selection"] == source.name
    assert compact["full_report"]["archive_path"] == archive.name
    assert compact["environment_config"]["rom_path"] == (
        "space_invaders.bin"
    )
    portable_environment_hash = hashlib.sha256(json.dumps(
        compact["environment_config"],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")).hexdigest()
    assert compact["environment_config_sha256"] == portable_environment_hash
    assert compact["screen"]["checkpoint_results"][0][
        "environment_config_sha256"
    ] == portable_environment_hash
    assert compact["confirmation"]["result"][
        "environment_config_sha256"
    ] == portable_environment_hash
    assert compact["screen"]["checkpoint_results"][0][
        "checkpoint_sha256"
    ] == "c" * 64
    assert compact["confirmation"]["result"][
        "e2_provenance_fingerprint"
    ] == "d" * 64
    assert compact["confirmation"]["result"]["e2_provenance_manifest"] == {
        "fingerprint_sha256": "d" * 64,
        "scientific_identity_sha256": "e" * 64,
    }
    assert output.with_name(f"{output.name}.sha256").read_text() == (
        f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n"
    )
    assert archive.with_name(f"{archive.name}.sha256").read_text() == (
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"
    )
    assert compact["confirmation"]["result"]["summary"] == {
        "mean_leader_payoff": 1.0
    }

    # Normalizing retained compact reports is in-place safe and idempotent;
    # it does not require the original large evaluator report or archive.
    before = output.read_bytes()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "replication.atari.automation.compact_atari_e2_selection",
            "--compact-input",
            str(output),
            "--output",
            str(output),
            "--sha256-sidecars",
        ],
        cwd=ROOT,
        check=True,
    )
    assert output.read_bytes() == before
    assert output.with_name(f"{output.name}.sha256").read_text() == (
        f"{hashlib.sha256(before).hexdigest()}  {output.name}\n"
    )
