from argparse import Namespace
import copy
import json
from pathlib import Path

import pytest

from replication.atari.automation import (
    validate_atari_e2_pipeline_artifact as validator,
)


def test_shared_e1_cohort_is_collision_safe_and_requires_balanced_seller(
        monkeypatch, tmp_path,
):
    reports = {}
    checkpoints = {}
    for role in ("buyer", "seller"):
        reports[role] = tmp_path / f"{role}.json"
        checkpoints[role] = tmp_path / f"{role}.zip"
        reports[role].write_text("{}", encoding="utf-8")
        checkpoints[role].write_bytes(role.encode())

    monkeypatch.setattr(validator, "validate_code_root", lambda: "7" * 40)
    monkeypatch.setattr(
        validator, "sha256_file",
        lambda path: ("a" if Path(path).suffix == ".json" else "b") * 64,
    )
    monkeypatch.setattr(validator, "validate_zip", lambda path: "b" * 64)
    monkeypatch.setattr(
        validator,
        "validate_e1_gate",
        lambda args: {"sha256": "b" * 64, "role": args.role, "passed": True},
    )
    output = tmp_path / "cohort.json"
    args = Namespace(
        output=str(output),
        buyer_report=str(reports["buyer"]),
        buyer_checkpoint=str(checkpoints["buyer"]),
        buyer_actor_loss_mode="standard",
        seller_report=str(reports["seller"]),
        seller_checkpoint=str(checkpoints["seller"]),
        seller_actor_loss_mode="balanced",
    )
    result = validator.write_e1_gate_cohort(args)
    assert result["e1_gates"]["buyer"]["actor_loss_mode"] == "standard"
    assert result["e1_gates"]["seller"]["actor_loss_mode"] == "balanced"
    assert validator.validated_e1_gate_cohort(output)["e1_gates"] == (
        result["e1_gates"]
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        validator.write_e1_gate_cohort(args)

    wrong = Namespace(**{**vars(args),
        "output": str(tmp_path / "wrong.json"),
        "seller_actor_loss_mode": "standard",
    })
    with pytest.raises(RuntimeError, match="balanced E1 seller"):
        validator.write_e1_gate_cohort(wrong)
    assert not Path(wrong.output).exists()


def _report_fixture(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    artifact_path = tmp_path / "screen.csv"
    response = tmp_path / "response.zip"
    selected = tmp_path / "selected.zip"
    candidates = [tmp_path / f"candidate_{index}.zip" for index in range(6)]
    for path in [report_path, artifact_path, response, *candidates]:
        path.write_bytes(b"fixture")
    hashes = {
        path.resolve(): f"{index + 1:064x}"
        for index, path in enumerate(candidates)
    }
    response_hash = "f" * 64
    hashes[response.resolve()] = response_hash
    screen_results = [{
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": hashes[path.resolve()],
        "e2_provenance_fingerprint": "e" * 64,
        "response_checkpoint_sha256": response_hash,
        "seed_start": 4_000_001,
        "seed_end": 4_000_020,
    } for path in candidates]
    winner = screen_results[0]["checkpoint_sha256"]
    confirmation = {
        "screen_rank": 1,
        "episodes": 100,
        "seed_start": 5_000_001,
        "seed_end": 5_000_100,
        "disjoint_from_screen": True,
        "result": {
            "checkpoint_sha256": winner,
            "seed_start": 5_000_001,
            "seed_end": 5_000_100,
        },
        "checks": {
            "checkpoint_hash_matches_screen": True,
            "passed": False,
        },
    }
    report = {
        "schema_version": 2,
        "evaluator": validator.E2_EVALUATOR,
        "passed": False,
        "environment_config": {
            "leader_role": "seller",
            "gameplay_horizon": 200,
            "event_tail_steps": 0,
        },
        "response_checkpoint": str(response.resolve()),
        "response_checkpoint_sha256": response_hash,
        "artifacts": {
            "report_json": str(report_path.resolve()),
            "screen_csv": str(artifact_path.resolve()),
        },
        "screen": {
            "episodes_per_checkpoint": 20,
            "seed_start": 4_000_001,
            "seed_end": 4_000_020,
            "common_seed_schedule_check": {
                "passed": True,
                "episodes": 20,
                "seed_schedule_pairs": [
                    {"evaluation_seed": seed, "event_steps": [1, 2, 3, 4, 5]}
                    for seed in range(4_000_001, 4_000_021)
                ],
            },
            "checkpoint_results": screen_results,
        },
        "selection": {
            "screen_selected_checkpoint_sha256": winner,
            "selected_checkpoint_sha256": None,
        },
        "confirmation": confirmation,
        "confirmation_attempts": [confirmation],
        "selected_alias": None,
    }
    monkeypatch.setattr(validator, "load_json", lambda path: report)
    monkeypatch.setattr(
        validator, "validate_zip", lambda path: hashes[Path(path).resolve()]
    )
    monkeypatch.setattr(
        validator,
        "validated_pipeline_inputs",
        lambda path, role: {
            "e1_gates": {"buyer": {"checkpoint_sha256": response_hash}}
        },
    )
    args = Namespace(
        report=str(report_path),
        role="seller",
        selected=str(selected),
        response=str(response),
        input_manifest=str(tmp_path / "inputs.json"),
        candidate=[str(path) for path in candidates],
        expect="failed",
    )
    return report, args, winner


def test_e2_report_rejects_multiple_confirmation_attempts(monkeypatch, tmp_path):
    report, args, _ = _report_fixture(tmp_path, monkeypatch)
    report["confirmation_attempts"].append(copy.deepcopy(report["confirmation"]))
    with pytest.raises(RuntimeError, match="confirmation fallback"):
        validator.validate_e2_report(args)


def test_e2_report_rejects_confirmation_of_nonwinner(monkeypatch, tmp_path):
    report, args, winner = _report_fixture(tmp_path, monkeypatch)
    assert report["confirmation"]["result"]["checkpoint_sha256"] == winner
    report["confirmation"]["result"]["checkpoint_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="confirmed screen-winner"):
        validator.validate_e2_report(args)
