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
    release = tmp_path / "seller_release.json"
    release.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(validator, "validate_code_root", lambda: "7" * 40)
    monkeypatch.setattr(
        validator, "sha256_file",
        lambda path: ("a" if Path(path).suffix == ".json" else "b") * 64,
    )
    monkeypatch.setattr(validator, "validate_zip", lambda path: "b" * 64)
    release_record = {
        "path": str(release.resolve()), "sha256": "a" * 64,
    }

    def validate_gate(args):
        return {
            "sha256": "b" * 64,
            "role": args.role,
            "source_kind": (
                validator.E1_PRIMARY_SOURCE_KIND
                if args.role == "buyer" else "uniform"
            ),
            "sampler_mode": (
                validator.E1_TEMPORAL_SAMPLER
                if args.role == "buyer" else "uniform"
            ),
            "support_artifacts": (
                {} if args.role == "buyer"
                else {"seller_release": release_record}
            ),
            "passed": True,
        }

    monkeypatch.setattr(validator, "validate_e1_gate", validate_gate)
    output = tmp_path / "cohort.json"
    args = Namespace(
        output=str(output),
        buyer_report=str(reports["buyer"]),
        buyer_checkpoint=str(checkpoints["buyer"]),
        buyer_actor_loss_mode="balanced",
        seller_report=str(reports["seller"]),
        seller_checkpoint=str(checkpoints["seller"]),
        seller_actor_loss_mode="balanced",
    )
    expected_buyer = {
        "report": str(reports["buyer"].resolve()),
        "report_sha256": "a" * 64,
        "checkpoint": str(checkpoints["buyer"].resolve()),
        "checkpoint_sha256": "b" * 64,
        "actor_loss_mode": "balanced",
        "source_kind": validator.E1_PRIMARY_SOURCE_KIND,
        "sampler_mode": validator.E1_TEMPORAL_SAMPLER,
        "support_artifacts": {},
    }
    monkeypatch.setattr(
        validator, "validated_e1_seller_release",
        lambda path: {"buyer_gate": expected_buyer},
    )
    result = validator.write_e1_gate_cohort(args)
    assert result["schema"] == "stackpomdp.atari.e2_e1_gate_cohort.v3"
    assert result["e1_gates"]["buyer"] == expected_buyer
    assert result["e1_gates"]["seller"]["actor_loss_mode"] == "balanced"
    assert result["seller_release"] == release_record
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


def test_strict_e1_gate_forbids_confirmation_fallback(monkeypatch, tmp_path):
    report_path = tmp_path / "buyer.json"
    checkpoint = tmp_path / "selected.zip"
    candidates = [tmp_path / f"candidate_{index}.zip" for index in range(6)]
    for path in [checkpoint, *candidates]:
        path.write_bytes(path.name.encode())
    hashes = {
        path.resolve(): f"{index + 1:064x}"
        for index, path in enumerate(candidates)
    }
    winner = hashes[candidates[0].resolve()]
    hashes[checkpoint.resolve()] = winner
    report = {
        "protocol": {
            "screen_episodes": 20,
            "confirmation_episodes": 100,
            "outer_transitions": 205,
            "confirmation_policy": "screen_winner_only_no_fallback",
        },
        "environment": {
            "gameplay_horizon": 200,
            "event_tail_steps": 0,
            "rom_sha256": validator.CANONICAL_ROM_SHA256,
        },
        "screen": {
            "common_pairing": {
                "passed": True,
                "candidates_checked": 6,
                "seed_context_pairs": [{} for _ in range(20)],
            },
            "results": [{
                "metadata": {"path": str(path), "sha256": hashes[path.resolve()]},
                "protocol": {"passed": True},
                "episode_rows": [{} for _ in range(20)],
            } for path in candidates],
        },
        "ranking": [{
            "rank": index + 1,
            "checkpoint_sha256": hashes[path.resolve()],
            "mechanically_valid": True,
        } for index, path in enumerate(candidates)],
        "immutable_evaluation": {
            "candidate_sha256": [hashes[path.resolve()] for path in candidates]
        },
        "confirmation_attempts": [{
            "metadata": {"sha256": winner},
            "behavioral_gate": {"passed": True},
            "random": {"episode_rows": [{} for _ in range(100)]},
        }],
        "selection": {
            "fallback_allowed": False,
            "screen_selected_checkpoint_sha256": winner,
            "selected_checkpoint_sha256": winner,
        },
        "artifacts": {"json": str(report_path)},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        validator, "validate_zip", lambda path: hashes[Path(path).resolve()]
    )
    assert validator._strict_e1_selection(
        report, report_path, checkpoint, winner
    ) == report["immutable_evaluation"]["candidate_sha256"]

    report["confirmation_attempts"].append(
        copy.deepcopy(report["confirmation_attempts"][0])
    )
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        validator._strict_e1_selection(
            report, report_path, checkpoint, winner
        )


def test_temporal_sampler_requires_exact_two_stage_lineage():
    report = {
        "training_family": {
            "common_sampler_provenance": validator._canonical_temporal_sampler(),
            "common_sampler_history": [
                {
                    "start_total_timesteps": 0,
                    "sampler": validator._canonical_uniform_sampler(),
                    "inferred_for_legacy_checkpoint": True,
                    "resume_sources": [],
                },
                {
                    "start_total_timesteps": 1_600_640,
                    "sampler": validator._canonical_temporal_sampler(),
                    "inferred_for_legacy_checkpoint": False,
                    "resume_sources": [{
                        "path": "/tmp/parent.zip",
                        "sha256": "a" * 64,
                        "training_total_timesteps": 1_600_640,
                        "resume_total_timesteps": 1_600_640,
                    }],
                },
            ],
        }
    }
    assert validator._e1_sampler_contract(report)["source_kind"] == (
        "temporal_contingency"
    )
    report["training_family"]["common_sampler_history"].append({})
    with pytest.raises(RuntimeError, match="exactly uniform and temporal"):
        validator._e1_sampler_contract(report)


def test_temporal_report_is_not_discoverable_before_gate_sidecar(
        monkeypatch, tmp_path,
):
    report_path = tmp_path / "e1_buyer_temporal_selector_v2.json"
    report_path.write_text(json.dumps({
        "passed": True,
        "role": "buyer",
        "training_family": {
            "common_sampler_provenance": {
                "mode": validator.E1_TEMPORAL_SAMPLER,
            },
            "common_training_config": {"actor_loss_mode": "balanced"},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(
        validator,
        "validate_e1_gate",
        lambda args: pytest.fail("an unpublished temporal report was validated"),
    )
    result = validator.discover_e1_gate(Namespace(
        role="buyer",
        output_dir=str(tmp_path),
        override_report=None,
        require_mode=None,
    ))
    assert result == {"kind": "e1_gate_discovery", "found": False}


def test_legacy_lower_rank_pass_is_not_a_downstream_gate(
        monkeypatch, tmp_path,
):
    report_path = tmp_path / "e1_buyer_uniform_selector_v2.json"
    hashes = [f"{index + 1:064x}" for index in range(6)]
    report_path.write_text(json.dumps({
        "passed": True,
        "role": "buyer",
        "selected_alias": {
            "pinned_path": "selected.zip", "sha256": hashes[1],
        },
        "ranking": [{"checkpoint_sha256": digest} for digest in hashes],
        "confirmation_attempts": [
            {
                "metadata": {"sha256": hashes[0]},
                "behavioral_gate": {"passed": False},
            },
            {
                "metadata": {"sha256": hashes[1]},
                "behavioral_gate": {"passed": True},
            },
        ],
        "training_family": {
            "common_training_config": {"actor_loss_mode": "balanced"},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(
        validator,
        "validate_e1_gate",
        lambda args: pytest.fail("a lower-rank fallback was treated as a gate"),
    )
    result = validator.discover_e1_gate(Namespace(
        role="buyer",
        output_dir=str(tmp_path),
        override_report=None,
        require_mode=None,
    ))
    assert result["found"] is False


def test_durable_seller_release_uses_only_final_strict_gate_discovery():
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "replication/atari/automation"
        / "run_atari_clean_e1_seller_after_buyer_gate.sh"
    ).read_text(encoding="utf-8")
    assert "e2_resolve_e1_gate buyer" in launcher
    assert "write-e1-seller-release" in launcher
    assert "step${step}_selector" not in launcher
    assert "--actor-loss-mode balanced" in launcher
    assert "stackpomdp_claim_owned_lock" in launcher
    assert "STACKPOMDP_E1_SELLER_LOCK_TOKEN" in launcher
    assert "stackpomdp_release_owned_lock" in launcher
    assert "rmdir \"$SELLER_LOCK\"" not in launcher


def test_seller_release_requires_primary_economic_buyer(
        monkeypatch, tmp_path,
):
    report = tmp_path / "buyer.json"
    checkpoint = tmp_path / "buyer.zip"
    report.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"buyer")
    monkeypatch.setattr(validator, "validate_code_root", lambda: "7" * 40)
    monkeypatch.setattr(validator, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(validator, "validate_e1_gate", lambda args: {
        "sha256": "b" * 64,
        "source_kind": "uniform",
        "sampler_mode": "uniform",
        "support_artifacts": {},
    })
    args = Namespace(
        output=str(tmp_path / "release.json"),
        buyer_report=str(report),
        buyer_checkpoint=str(checkpoint),
        buyer_actor_loss_mode="balanced",
    )
    with pytest.raises(RuntimeError, match="primary-economic buyer"):
        validator.write_e1_seller_release(args)
    assert not Path(args.output).exists()


def test_cohort_rejects_buyer_different_from_seller_training_release(
        monkeypatch, tmp_path,
):
    release_path = tmp_path / "release.json"
    release_path.write_text("{}", encoding="utf-8")
    buyer = {
        "report": "/buyer/report.json",
        "report_sha256": "1" * 64,
        "checkpoint": "/buyer/selected.zip",
        "checkpoint_sha256": "2" * 64,
        "actor_loss_mode": "balanced",
        "source_kind": validator.E1_PRIMARY_SOURCE_KIND,
        "sampler_mode": validator.E1_TEMPORAL_SAMPLER,
        "support_artifacts": {},
    }
    release_record = {
        "path": str(release_path.resolve()), "sha256": "a" * 64,
    }
    gates = {
        "buyer": buyer,
        "seller": {
            "support_artifacts": {"seller_release": release_record},
        },
    }
    monkeypatch.setattr(validator, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(
        validator, "validated_e1_seller_release",
        lambda path: {
            "buyer_gate": {
                **buyer, "checkpoint_sha256": "3" * 64,
            }
        },
    )
    with pytest.raises(RuntimeError, match="seller-training buyer"):
        validator._validate_cohort_seller_release_binding(
            {"seller_release": release_record}, gates
        )


def test_temporal_support_binds_activation_family_and_selected_bytes(
        monkeypatch, tmp_path,
):
    candidates = [tmp_path / f"candidate_{index}.zip" for index in range(6)]
    for index, path in enumerate(candidates):
        path.write_bytes(f"candidate-{index}".encode())
    hashes = [validator.sha256_file(path) for path in candidates]
    selected = tmp_path / "selected.zip"
    selected.write_bytes(candidates[0].read_bytes())
    resume = tmp_path / "resume.zip"
    resume.write_bytes(b"resume")
    e0b = tmp_path / "e0b.zip"
    e0b.write_bytes(b"e0b")
    rom = tmp_path / "space_invaders.bin"
    rom.write_bytes(b"rom")
    monkeypatch.setattr(
        validator, "CANONICAL_ROM_SHA256", validator.sha256_file(rom)
    )
    monkeypatch.setattr(
        validator, "validate_zip", lambda path: validator.sha256_file(path)
    )

    current = validator._canonical_temporal_sampler()
    history = [
        {
            "start_total_timesteps": 0,
            "sampler": validator._canonical_uniform_sampler(),
            "inferred_for_legacy_checkpoint": True,
            "resume_sources": [],
        },
        {
            "start_total_timesteps": 1_600_640,
            "sampler": current,
            "inferred_for_legacy_checkpoint": False,
            "resume_sources": [{
                "path": str(resume),
                "sha256": validator.sha256_file(resume),
                "training_total_timesteps": 1_600_640,
                "resume_total_timesteps": 1_600_640,
            }],
        },
    ]
    failed = {}
    for mode in ("standard", "balanced"):
        path = tmp_path / f"{mode}_failed.json"
        path.write_text("{}", encoding="utf-8")
        failed[mode] = {
            "path": str(path),
            "sha256": validator.sha256_file(path),
            "actor_loss_mode": mode,
            "candidate_count": 6,
            "sampler": {"mode": "uniform"},
            "reported_passed": False,
            "strict_screen_winner_passed": False,
            "legacy_fallback_selected": False,
        }
    activation_path = tmp_path / "activation.json"
    activation = {
        "schema_version": 1,
        "kind": validator.E1_TEMPORAL_ACTIVATION_KIND,
        "activation_condition": {
            "standard_uniform_all_six_failed": True,
            "balanced_uniform_all_six_failed": True,
        },
        "code_revision": "a" * 40,
        "uniform_failure_reports": failed,
        "resume_source": {
            "path": str(resume),
            "sha256": validator.sha256_file(resume),
            "training_total_timesteps": 1_600_640,
        },
        "e0b_source": {
            "path": str(e0b), "sha256": validator.sha256_file(e0b),
        },
        "rom": {
            "path": str(rom), "sha256": validator.sha256_file(rom),
        },
        "protocol": {
            "training_sampler": current,
            "evaluation_sampler": validator._canonical_uniform_sampler(),
            "additional_timesteps": 2_000_800,
            "checkpoint_interval": 400_160,
            "expected_total_timesteps": 3_601_440,
            "candidate_count": 6,
            "candidate_paths": [str(path) for path in candidates],
            "screen": {"episodes": 20, "seed_start": 6_000_001},
            "confirmation": {
                "episodes": 100,
                "seed_start": 6_100_001,
                "fallback_allowed": False,
            },
            "fixed_seed_start": 6_200_001,
            "timing_seed_start": 6_300_001,
        },
    }
    activation_path.write_text(json.dumps(activation), encoding="utf-8")
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps({
        "passed": True,
        "activation_sha256": validator.sha256_file(activation_path),
    }), encoding="utf-8")
    family_path = tmp_path / "family.json"
    family = {
        "schema_version": 1,
        "kind": validator.E1_TEMPORAL_FAMILY_KIND,
        "passed": True,
        "activation": {
            "path": str(activation_path),
            "sha256": validator.sha256_file(activation_path),
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": validator.sha256_file(preflight_path),
        },
        "candidate_sha256": hashes,
        "candidate_metadata": [{
            "path": str(path),
            "sha256": digest,
            "atari_e1_sampler_provenance": current,
            "atari_e1_sampler_history": history,
        } for path, digest in zip(candidates, hashes)],
    }
    family_path.write_text(json.dumps(family), encoding="utf-8")
    report_path = tmp_path / "temporal_selector_v2.json"
    report = {
        "training_family": {
            "common_sampler_provenance": current,
            "common_sampler_history": history,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    gate_path = report_path.with_name(f"{report_path.stem}.gate.json")
    gate_path.write_text(json.dumps({
        "schema_version": 1,
        "kind": validator.E1_TEMPORAL_GATE_KIND,
        "passed": True,
        "role": "buyer",
        "actor_loss_mode": "balanced",
        "report": {
            "path": str(report_path),
            "sha256": validator.sha256_file(report_path),
        },
        "selected_checkpoint": {
            "path": str(selected), "sha256": hashes[0],
        },
        "training_family": {
            "path": str(family_path),
            "sha256": validator.sha256_file(family_path),
            "candidate_sha256": hashes,
        },
        "activation": {
            "path": str(activation_path),
            "sha256": validator.sha256_file(activation_path),
            "code_revision": "a" * 40,
        },
        "sampler": {"current": current, "history": history},
        "selection": {
            "screen_selected_checkpoint_sha256": hashes[0],
            "confirmed_checkpoint_sha256": hashes[0],
            "selected_checkpoint_sha256": hashes[0],
            "confirmation_attempts": 1,
            "confirmation_policy": "screen_winner_only_no_fallback",
            "fallback_allowed": False,
        },
    }), encoding="utf-8")

    support = validator._validate_temporal_gate_support(
        report_path, selected, hashes[0], report, hashes
    )
    assert support["temporal_gate"]["sha256"] == validator.sha256_file(
        gate_path
    )
    assert support["activation"]["code_revision"] == "a" * 40


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
