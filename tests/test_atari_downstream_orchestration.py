from argparse import Namespace
import json
from pathlib import Path

import pytest

from replication.atari import evaluate_atari_meta_response_sb3 as e1_evaluator
from replication.atari.automation import (
    validate_atari_e2_pipeline_artifact as validator,
)


def _canonical_seller_report(revision="a" * 40):
    screen_seeds = list(range(3_500_001, 3_500_021))
    confirmation_seeds = list(range(3_600_001, 3_600_101))
    fixed_seeds = list(range(3_700_001, 3_700_021))
    timesteps = [400_160, 800_320, 1_200_480, 1_600_640, 2_000_800, 2_000_800]
    return {
        "protocol": {
            "screen_episodes": 20,
            "screen_seed_start": 3_500_001,
            "confirmation_episodes": 100,
            "confirmation_seed_start": 3_600_001,
            "fixed_context_episodes": 20,
            "fixed_context_seed_start": 3_700_001,
            "confirmation_policy": "screen_winner_only_no_fallback",
        },
        "immutable_evaluation": {
            "selector_code_revision": revision,
            "e0b_sha256": validator.CANONICAL_E0B_SHA256,
        },
        "screen": {
            "common_pairing": {
                "seed_context_pairs": [
                    {"evaluation_seed": seed} for seed in screen_seeds
                ],
            },
            "results": [{
                "metadata": {
                    "training_timesteps": timestep,
                    "e0b_source_provenance": {
                        "sha256": validator.CANONICAL_E0B_SHA256,
                    },
                },
            } for timestep in timesteps],
        },
        "confirmation_attempts": [{
            "random": {
                "episode_rows": [
                    {"evaluation_seed": seed} for seed in confirmation_seeds
                ],
            },
            "fixed_contexts": [{
                "opponent_value": index / 10.0,
                "episode_rows": [
                    {"evaluation_seed": seed} for seed in fixed_seeds
                ],
            } for index in range(11)],
        }],
    }


def test_e1_selected_alias_uses_gate_schema(tmp_path):
    source = tmp_path / "source.zip"
    selected = tmp_path / "selected.zip"
    source.write_bytes(b"checkpoint")
    alias = e1_evaluator.atomic_copy_no_overwrite(source, selected)
    assert alias == {
        "pinned_path": str(selected.resolve()),
        "sha256": e1_evaluator.checkpoint_sha256(selected),
    }


def test_e1_report_failure_rolls_back_new_alias(monkeypatch, tmp_path):
    source = tmp_path / "source.zip"
    selected = tmp_path / "selected.zip"
    report_path = tmp_path / "report.json"
    unrelated = tmp_path / "unrelated.txt"
    source.write_bytes(b"checkpoint")
    unrelated.write_text("keep", encoding="utf-8")
    args = Namespace(selected_checkpoint=str(selected))

    monkeypatch.setattr(e1_evaluator, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        e1_evaluator, "artifact_paths", lambda local: {"json": report_path}
    )

    def select(local):
        return {
            "selected_alias": e1_evaluator.atomic_copy_no_overwrite(
                source, local.selected_checkpoint
            )
        }

    def fail_persist(report, local):
        report_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("forced persistence failure")

    monkeypatch.setattr(e1_evaluator, "run_selection", select)
    monkeypatch.setattr(e1_evaluator, "persist_report", fail_persist)
    with pytest.raises(RuntimeError, match="forced persistence failure"):
        e1_evaluator.main([])
    assert not selected.exists()
    assert not report_path.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_canonical_seller_protocol_binds_seeds_family_and_revision(monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(
        validator, "validate_recorded_revision", lambda value: value
    )
    report = _canonical_seller_report(revision)
    validator._validate_canonical_seller_selection_protocol(report)
    report["screen"]["common_pairing"]["seed_context_pairs"][0][
        "evaluation_seed"
    ] += 1
    with pytest.raises(RuntimeError, match="screen seeds"):
        validator._validate_canonical_seller_selection_protocol(report)


def test_passing_seller_report_is_hidden_until_gate_sidecar(
        monkeypatch, tmp_path,
):
    report = tmp_path / "e1_seller_balanced_all6_selector_v2.json"
    report.write_text(json.dumps({
        "passed": True,
        "role": "seller",
        "training_family": {
            "common_sampler_provenance": {"mode": "uniform"},
            "common_training_config": {"actor_loss_mode": "balanced"},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(
        validator,
        "validate_e1_gate",
        lambda args: pytest.fail("seller report was visible before its sidecar"),
    )
    discovered = validator.discover_e1_gate(Namespace(
        role="seller",
        output_dir=str(tmp_path),
        override_report=None,
        require_mode="balanced",
    ))
    assert discovered == {"kind": "e1_gate_discovery", "found": False}


def test_seller_gate_binds_release_report_alias_and_revision(
        monkeypatch, tmp_path,
):
    revision = "a" * 40
    report = tmp_path / "e1_seller_balanced_all6_selector_v2.json"
    selected = tmp_path / "selected.zip"
    release = tmp_path / "seller.buyer_gate.json"
    selected.write_bytes(b"selected")
    release.write_text("{}", encoding="utf-8")
    report.write_text(json.dumps({
        "immutable_evaluation": {"selector_code_revision": revision},
    }), encoding="utf-8")
    digest = validator.sha256_file(selected)
    candidates = [f"{index + 1:064x}" for index in range(6)]
    gate = report.with_name(f"{report.stem}.gate.json")
    validator.atomic_write_new_json(gate, {
        "schema_version": 1,
        "kind": validator.E1_SELLER_GATE_KIND,
        "passed": True,
        "role": "seller",
        "actor_loss_mode": "balanced",
        "report": {
            "path": str(report.resolve()),
            "sha256": validator.sha256_file(report),
        },
        "selected_checkpoint": {
            "path": str(selected.resolve()), "sha256": digest,
        },
        "candidate_sha256": candidates,
        "seller_release": {
            "path": str(release.resolve()),
            "sha256": validator.sha256_file(release),
        },
        "selector": {
            "code_revision": revision,
            "evaluator": validator.E1_EVALUATOR,
        },
    })
    monkeypatch.setattr(
        validator, "validated_e1_seller_release", lambda path: {}
    )
    monkeypatch.setattr(
        validator, "validate_recorded_revision", lambda value: value
    )
    support = validator._validate_seller_selection_gate_support(
        report, selected, digest, candidates
    )
    assert support["selector_code_revision"] == revision
    release.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="seller-release SHA-256"):
        validator._validate_seller_selection_gate_support(
            report, selected, digest, candidates
        )


def test_atomic_json_publication_never_overwrites(tmp_path):
    output = tmp_path / "gate.json"
    validator.atomic_write_new_json(output, {"value": 1})
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        validator.atomic_write_new_json(output, {"value": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    assert not list(tmp_path.glob(".gate.json.*.tmp"))


def test_incomplete_orchestration_summary_is_immutable_and_revalidated(
        monkeypatch, tmp_path,
):
    cohort = tmp_path / "cohort.json"
    cohort.write_text("{}", encoding="utf-8")
    revision = "b" * 40
    monkeypatch.setattr(
        validator, "validated_e1_gate_cohort", lambda path: {}
    )
    monkeypatch.setattr(validator, "validate_code_root", lambda: "7" * 40)
    monkeypatch.setattr(
        validator, "validate_selector_code_root", lambda path: revision
    )
    monkeypatch.setattr(
        validator, "validate_recorded_revision", lambda value: value
    )
    output = tmp_path / "summary.json"
    args = Namespace(
        output=str(output),
        cohort_manifest=str(cohort),
        checkpoint_root=str(tmp_path / "checkpoints"),
        result_root=str(tmp_path / "results"),
        automation_code_root=str(tmp_path / "code"),
        buyer_training_exit_code=17,
        buyer_selector_exit_code=-1,
        seller_training_exit_code=23,
        seller_selector_exit_code=-1,
    )
    result = validator.write_e2_orchestration_summary(args)
    assert result["orchestration_completed"] is False
    assert result["all_scientific_gates_passed"] is False
    assert result["passed"] is False
    validator.validate_e2_orchestration_summary(
        Namespace(summary=str(output))
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    value["passed"] = True
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary contents"):
        validator.validate_e2_orchestration_summary(
            Namespace(summary=str(output))
        )


def test_master_is_sequential_preflights_both_roles_and_isolates_failures():
    root = Path(__file__).resolve().parents[1]
    automation = root / "replication/atari/automation"
    master = (automation / "run_atari_clean_e2_sequential.sh").read_text(
        encoding="utf-8"
    )
    assert master.index("e2_refuse_role_pipeline_outputs buyer") < master.index(
        "for role in buyer seller"
    )
    assert master.index("e2_refuse_role_pipeline_outputs seller") < master.index(
        "for role in buyer seller"
    )
    assert "selector_status[$role]=-1" in master
    assert "continuing with the other role" in master
    assert "write-e2-orchestration-summary" in master
    for name in (
        "run_atari_clean_e2_buyer_balanced_2m.sh",
        "run_atari_clean_e2_seller_balanced_2m.sh",
        "run_e2_buyer_balanced_final_selector.sh",
        "run_e2_seller_balanced_final_selector.sh",
    ):
        script = (automation / name).read_text(encoding="utf-8")
        assert script.index("e2_claim_pipeline_lock") < script.index(
            "e2_prepare_runtime"
        )


def test_seller_selector_is_one_pinned_all_six_run():
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "replication/atari/automation"
        / "run_e1_seller_balanced_final_selector.sh"
    ).read_text(encoding="utf-8")
    assert "SELLER_STEPS=(400160 800320 1200480 1600640 2000800)" in script
    assert "--selector-code-revision" in script
    assert "write-e1-seller-selection-gate" in script
    assert "read-e1-seller-release" in script
    assert "for step in $SELLER_STEPS" in script
    assert "rank" not in script.lower() or "no lower-ranked fallback" in script
