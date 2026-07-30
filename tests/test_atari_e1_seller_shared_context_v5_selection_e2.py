"""Regression tests for seller-v5 selection and its versioned E2 authority."""

from argparse import Namespace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari import train_atari_stackpomdp_leader_sb3 as leader_trainer
from replication.atari.automation import (
    validate_atari_e1_seller_shared_context_v5 as diagnostics,
)
from replication.atari.automation import (
    validate_atari_e1_seller_shared_context_v5_selection as selection,
)
from replication.atari.automation import (
    validate_atari_e2_pipeline_artifact as e2,
)
from replication.atari.automation import (
    validate_atari_e1_seller_conditioning_recovery as recovery,
)


ROOT = Path(__file__).resolve().parents[1]
AUTOMATION = ROOT / "replication/atari/automation"
SELECTION_COMMON = (
    AUTOMATION / "atari_e1_seller_shared_context_v5_selection_common.zsh"
)
SELECTION_LAUNCHER = (
    AUTOMATION / "run_e1_seller_shared_context_v5_selector.sh"
)
SELECTION_EXPOSURE_WRAPPER = (
    AUTOMATION
    / "run_e1_seller_shared_context_v5_exposure_v2_selector.sh"
)
E2_COMMON = AUTOMATION / "atari_e2_pipeline_common.zsh"
E2_V5_WRAPPERS = (
    AUTOMATION / "run_atari_clean_e2_buyer_shared_context_v5_2m.sh",
    AUTOMATION / "run_atari_clean_e2_seller_shared_context_v5_2m.sh",
    AUTOMATION / "run_e2_buyer_shared_context_v5_final_selector.sh",
    AUTOMATION / "run_e2_seller_shared_context_v5_final_selector.sh",
    AUTOMATION / "run_atari_clean_e2_shared_context_v5_sequential.sh",
)
REAL_CANONICAL_SELLER_RELEASE = Path(
    "/Users/gbrero/active-research/StackelbergPOMDP/code/StackelbergPOMDP/"
    "replication/atari/checkpoints/clean/"
    "meta_seller_e1_ppo_balanced_seed1_firefix_retrain.buyer_gate.json"
)
REAL_EXPOSURE_V2_DIAGNOSTICS_GATE = Path(
    "/Users/gbrero/active-research/StackelbergPOMDP/code/StackelbergPOMDP/"
    "replication/atari/results/e1_selections/"
    "e1_seller_conditioning_recovery_v5_shared_context_exposure_v2_"
    "diagnostics_gate.json"
)


@pytest.fixture(autouse=True)
def _restore_global_protocols():
    """Keep mutable protocol modules independent of test order."""

    try:
        yield
    finally:
        selection.configure_protocol(diagnostics.STANDARD_PROTOCOL)
        e2.configure_e2_profile(e2.E2_PROFILE_V3)


def _formal_paths(root):
    base = root / selection.FORMAL_CHECKPOINT_NAME
    steps = [
        base.with_name(f"{base.stem}_step{step}.zip")
        for step in diagnostics.FORMAL_STEP_TIMESTEPS
    ]
    return [*steps, base]


def test_v5_selection_protocol_is_six_candidates_with_disjoint_holdouts():
    assert selection.FORMAL_CANDIDATE_TIMESTEPS == (
        400_160,
        800_320,
        1_200_480,
        1_600_640,
        2_000_800,
        2_000_800,
    )
    assert selection.SELECTION_PROTOCOLS == {
        "standard_v1": {
            "screen_seed_start": 12_000_001,
            "confirmation_seed_start": 12_100_001,
            "fixed_seed_start": 12_200_001,
            "timing_seed_start": 12_300_001,
        },
        "exposure_v2": {
            "screen_seed_start": 13_000_001,
            "confirmation_seed_start": 13_100_001,
            "fixed_seed_start": 13_200_001,
            "timing_seed_start": 13_300_001,
        },
    }
    selection.configure_protocol(diagnostics.EXPOSURE_PROTOCOL)
    assert selection.SOURCE_KIND == e2.E1_SELLER_V5_SOURCE_KIND
    assert selection.REPORT_NAME == e2.E1_SELLER_V5_REPORT_NAME
    assert selection.SELECTION_GATE_NAME == e2.E1_SELLER_V5_GATE_NAME


@pytest.mark.skipif(
    not REAL_EXPOSURE_V2_DIAGNOSTICS_GATE.is_file(),
    reason="current exposure-v2 diagnostics gate is unavailable",
)
def test_selector_accepts_real_180_runtime_gate_and_short_wandb_job_type():
    """Exercise the validator bridge and formal contract used by the run."""

    selection.configure_protocol(diagnostics.EXPOSURE_PROTOCOL)
    gate = diagnostics.validate_gate(REAL_EXPOSURE_V2_DIAGNOSTICS_GATE)
    assert gate["code_revision"] == (
        "180c84f51005d2f36ae86f1cd7319c0e71249065"
    )
    assert gate["diagnostic_evidence_revision"] == (
        diagnostics.EXPOSURE_V2_DIAGNOSTIC_REVISION
    )
    assert gate["validator_revision_bridge"] == {
        "applied": True,
        "reason": "float32_gradient_norm_validator_tolerance",
        "evidence_revision": diagnostics.EXPOSURE_V2_DIAGNOSTIC_REVISION,
        "runtime_revision": gate["code_revision"],
        "changed_paths": list(diagnostics.VALIDATOR_REPAIR_ALLOWED_PATHS),
        "training_implementation_changed": False,
        "gradient_norm_absolute_tolerance": diagnostics.V5_GRADIENT_NORM_ATOL,
    }
    assert gate["formal_release"]["wandb"]["job_type"] == (
        "atari_e1_seller_v5_exposure_v2_formal"
    )
    assert e2.E2_PROFILES[e2.E2_PROFILE_V5]["code_head"] == (
        gate["code_revision"]
    )


def test_formal_family_binds_exact_ordered_six_and_rejects_duplicate_bytes(
        monkeypatch, tmp_path,
):
    selection.configure_protocol(diagnostics.EXPOSURE_PROTOCOL)
    candidate_paths = _formal_paths(tmp_path)
    console = tmp_path / "formal.log"
    console.write_text("complete\n", encoding="utf-8")
    gate_path = tmp_path / selection.DIAGNOSTICS_GATE_NAME
    gate_path.write_text("{}\n", encoding="utf-8")
    training_revision = "a" * 40
    selector_revision = "b" * 40
    formal = {
        "checkpoint": str(candidate_paths[-1].resolve()),
        "candidate_paths": [str(path.resolve()) for path in candidate_paths],
    }
    prerequisite = lambda name: {
        "path": str((tmp_path / name).resolve()),
        "sha256": name[0] * 64,
    }
    gate = {
        "code_revision": training_revision,
        "formal_release": formal,
        "prerequisites": {
            "e0b": prerequisite("e0b.zip"),
            "buyer_release": prerequisite("release.json"),
            "rom": prerequisite("rom.bin"),
        },
    }
    hashes = {
        str(path.resolve()): f"{index + 1:064x}"
        for index, path in enumerate(candidate_paths)
    }

    monkeypatch.setattr(selection, "_git_scoped_clean", lambda root: None)
    monkeypatch.setattr(
        selection, "_git_revision", lambda root: selector_revision
    )
    monkeypatch.setattr(diagnostics, "validate_gate", lambda path: gate)
    monkeypatch.setattr(
        selection,
        "_candidate_metadata",
        lambda path, **kwargs: {
            "path": str(Path(path).resolve()),
            "sha256": hashes[str(Path(path).resolve())],
            "training_timesteps": kwargs["timesteps"],
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_validate_trace",
        lambda path, **kwargs: {"path": str(Path(path).resolve())},
    )
    monkeypatch.setattr(
        selection,
        "_formal_evaluation",
        lambda path, **kwargs: {"path": str(Path(path).resolve())},
    )
    monkeypatch.setattr(selection, "sha256_file", lambda path: "f" * 64)
    base = candidate_paths[-1]
    args = Namespace(
        code_root=str(tmp_path),
        diagnostics_gate=str(gate_path),
        candidate=[str(path) for path in candidate_paths],
        training_log=str(base.with_name(f"{base.stem}.training.jsonl")),
        evaluation=str(base.with_name(f"{base.stem}.evaluation.json")),
        console_log=str(console),
    )

    family = selection.build_formal_family(args)
    assert [row["path"] for row in family["candidate_metadata"]] == (
        formal["candidate_paths"]
    )
    assert [row["training_timesteps"] for row in family["candidate_metadata"]] == (
        list(selection.FORMAL_CANDIDATE_TIMESTEPS)
    )
    assert family["candidate_sha256"] == [
        hashes[path] for path in formal["candidate_paths"]
    ]
    assert family["training_code_revision"] == training_revision
    assert family["selector_code_revision"] == selector_revision

    hashes[formal["candidate_paths"][1]] = hashes[formal["candidate_paths"][0]]
    with pytest.raises(ValueError, match="six byte-distinct"):
        selection.build_formal_family(args)


def _minimal_passing_selection(monkeypatch, tmp_path):
    selection.configure_protocol(diagnostics.EXPOSURE_PROTOCOL)
    candidate_paths = _formal_paths(tmp_path)
    hashes = [f"{index + 1:064x}" for index in range(6)]
    metadata = [
        {
            "path": str(path.resolve()),
            "sha256": digest,
            "training_timesteps": timesteps,
        }
        for path, digest, timesteps in zip(
            candidate_paths, hashes, selection.FORMAL_CANDIDATE_TIMESTEPS
        )
    ]
    family = {
        "selector_code_revision": "c" * 40,
        "candidate_sha256": hashes,
        "candidate_metadata": metadata,
        "rom": {"path": str((tmp_path / "rom.bin").resolve())},
    }
    winner = hashes[0]
    selected = tmp_path / selection.SELECTED_NAME
    selected.write_bytes(b"selected")
    report_path = tmp_path / selection.REPORT_NAME
    conditioning = {"warmup_gate": {"passed": True}}
    behavioral_gate = {
        "name": "seller_shared_context_behavioral_gate_v5",
        "passed": True,
        "conditioning_probe_gate": conditioning["warmup_gate"],
    }
    attempt_metadata = dict(metadata[0])
    common_pairing = {"passed": True, "candidates_checked": 6}
    ranking = [
        {
            "rank": index + 1,
            "checkpoint_sha256": digest,
            "mechanically_valid": True,
        }
        for index, digest in enumerate(hashes)
    ]
    random_result = {"episode_rows": [{"event_steps": [1, 2, 3, 4, 5]}]}
    ablated_result = {
        "episode_rows": [{"event_steps": [1, 2, 3, 4, 5]}]
    }
    report = {
        "passed": True,
        "role": "seller",
        "evaluator": selection.EVALUATOR,
        "protocol": {
            "screen_episodes": 20,
            "screen_seed_start": selection.SCREEN_SEED_START,
            "confirmation_episodes": 100,
            "confirmation_seed_start": selection.CONFIRMATION_SEED_START,
            "fixed_context_episodes": 20,
            "fixed_context_seed_start": selection.FIXED_SEED_START,
            "confirmation_policy": "screen_winner_only_no_fallback",
        },
        "immutable_evaluation": {
            "selector_code_revision": family["selector_code_revision"],
            "e0b_sha256": diagnostics.CANONICAL_E0B_SHA256,
            "rom_sha256": diagnostics.CANONICAL_ROM_SHA256,
            "candidate_sha256": hashes,
        },
        "environment": selection.canonical_selector_environment(family),
        "screen": {
            "results": [{"metadata": item} for item in metadata],
            "common_pairing": common_pairing,
        },
        "training_family": {"directory": "/family", "stem": "formal"},
        "ranking": ranking,
        "confirmation_attempts": [{
            "metadata": attempt_metadata,
            "random": random_result,
            "fixed_contexts": [{} for _ in range(11)],
            "context_ablated_random": ablated_result,
            "forced_constant_controls": {
                str(value): {}
                for value in evaluator.CANONICAL_FIXED_VALUES
            },
            "conditioning_probe": conditioning,
            "behavioral_gate": behavioral_gate,
        }],
        "selection": {
            "fallback_allowed": False,
            "screen_selected_checkpoint_sha256": winner,
            "selected_checkpoint_sha256": winner,
        },
        "selected_alias": {"pinned_path": str(selected.resolve())},
        "artifacts": {"json": str(report_path.resolve())},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(
        selection, "validate_formal_family", lambda path: family
    )
    monkeypatch.setattr(selection, "load_json", lambda path: report)
    monkeypatch.setattr(
        evaluator,
        "validate_common_screen",
        lambda *args, **kwargs: common_pairing,
    )
    monkeypatch.setattr(selection, "_validate_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        selection, "_validate_fixed_results", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluator, "checkpoint_family", lambda path: ("/family", "formal")
    )
    monkeypatch.setattr(evaluator, "common_training_family", lambda rows: {})
    monkeypatch.setattr(
        evaluator, "rank_candidates", lambda rows: (rows[0], ranking)
    )
    monkeypatch.setattr(
        evaluator,
        "load_candidate",
        lambda *args, **kwargs: (SimpleNamespace(), {"sha256": winner}),
    )
    monkeypatch.setattr(
        selection.probe,
        "collect_conditioning_report",
        lambda model: conditioning,
    )
    monkeypatch.setattr(
        evaluator,
        "seller_v5_behavioral_gate",
        lambda **kwargs: {
            "name": "seller_shared_context_behavioral_gate_v5",
            "passed": True,
        },
    )
    monkeypatch.setattr(selection, "sha256_file", lambda path: winner)
    args = Namespace(
        family=str(tmp_path / selection.FAMILY_NAME),
        report=str(report_path),
        selected=str(selected),
    )
    return report, args, winner


def test_selection_confirms_only_screen_winner_and_forbids_fallback(
        monkeypatch, tmp_path,
):
    report, args, winner = _minimal_passing_selection(monkeypatch, tmp_path)
    assert selection.validate_selection(args)["selection"] == {
        "fallback_allowed": False,
        "screen_selected_checkpoint_sha256": winner,
        "selected_checkpoint_sha256": winner,
    }

    report["confirmation_attempts"].append(report["confirmation_attempts"][0])
    with pytest.raises(ValueError, match="confirm only the screen winner"):
        selection.validate_selection(args)
    report["confirmation_attempts"].pop()
    report["confirmation_attempts"][0]["metadata"] = {
        **report["confirmation_attempts"][0]["metadata"],
        "sha256": report["ranking"][1]["checkpoint_sha256"],
    }
    with pytest.raises(ValueError, match="confirm only the screen winner"):
        selection.validate_selection(args)


def test_failed_selection_cannot_publish_gate(monkeypatch, tmp_path):
    gate = tmp_path / selection.SELECTION_GATE_NAME
    result = selection.write_or_validate_selection_gate(
        Namespace(gate_output=str(gate)), {"passed": False}
    )
    assert result is None
    assert not gate.exists()

    gate.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="retained a gate"):
        selection.write_or_validate_selection_gate(
            Namespace(gate_output=str(gate)), {"passed": False}
        )


def test_v5_gate_normalization_binds_architecture_release_and_diagnostics(
        monkeypatch, tmp_path,
):
    e2.configure_e2_profile(e2.E2_PROFILE_V5)
    report = tmp_path / e2.E1_SELLER_V5_REPORT_NAME
    gate_path = tmp_path / e2.E1_SELLER_V5_GATE_NAME
    checkpoint = tmp_path / "seller-v5-selected.zip"
    diagnostics_path = tmp_path / e2.E1_SELLER_V5_DIAGNOSTICS_GATE_NAME
    release_path = tmp_path / "buyer-release.json"
    for path in (report, gate_path, checkpoint, diagnostics_path, release_path):
        path.write_bytes(path.name.encode())
    digest = "1" * 64
    gate_digest = "2" * 64
    diagnostics_digest = "3" * 64
    release_digest = "4" * 64
    revision = e2.E2_PROFILES[e2.E2_PROFILE_V5]["code_head"]
    architecture = e2.canonical_seller_v5_architecture()
    initialization = e2.canonical_seller_v5_initialization()
    gate = {
        "selected_checkpoint": {
            "path": str(checkpoint.resolve()), "sha256": digest,
        },
        "source_kind": e2.E1_SELLER_V5_SOURCE_KIND,
        "economic_architecture": architecture,
        "shared_context_initialization": initialization,
        "training_family": {
            "candidate_sha256": [f"{index + 1:064x}" for index in range(6)]
        },
        "e1_training_code_revision": revision,
        "diagnostics_gate": {
            "path": str(diagnostics_path.resolve()),
            "sha256": diagnostics_digest,
        },
        "seller_release": {
            "path": str(release_path.resolve()), "sha256": release_digest,
        },
        "selector_code_revision": "6" * 40,
        "selected_conditioning_probe": {"passed": True},
        "selected_behavioral_gate": {"passed": True},
    }
    captured = {}

    def validate_v5(arguments):
        captured["selection"] = arguments
        return gate

    monkeypatch.setattr(e2, "_run_seller_v5_validator", validate_v5)
    monkeypatch.setattr(
        e2,
        "_run_seller_v5_diagnostics_validator",
        lambda arguments: {"code_revision": revision},
    )
    monkeypatch.setattr(e2, "validate_zip", lambda path: digest)
    monkeypatch.setattr(
        e2, "validate_recorded_revision", lambda value: value
    )
    monkeypatch.setattr(
        e2,
        "sha256_file",
        lambda path: {
            gate_path.resolve(): gate_digest,
            diagnostics_path.resolve(): diagnostics_digest,
            release_path.resolve(): release_digest,
        }[Path(path).resolve()],
    )
    monkeypatch.setattr(
        leader_trainer,
        "checkpoint_policy_metadata",
        lambda *args, **kwargs: {
            "sha256": digest,
            "economic_architecture": architecture,
            "shared_context_initialization": initialization,
            "e1_training_code_revision": revision,
        },
    )
    monkeypatch.setattr(
        e2, "validated_e1_seller_release", lambda path: {"passed": True}
    )

    normalized = e2._validate_seller_v5_gate(Namespace(
        role="seller",
        actor_loss_mode="balanced",
        report=str(report),
        checkpoint=str(checkpoint),
    ))
    assert captured["selection"] == [
        "validate-selection-gate",
        "--gate", str(gate_path.resolve()),
        "--report", str(report.resolve()),
        "--selected", str(checkpoint.resolve()),
    ]
    assert normalized["source_kind"] == e2.E1_SELLER_V5_SOURCE_KIND
    assert normalized["candidate_sha256"] == (
        gate["training_family"]["candidate_sha256"]
    )
    assert normalized["support_artifacts"]["seller_release"] == (
        gate["seller_release"]
    )
    assert normalized["support_artifacts"][
        "seller_shared_context_v5_selection_gate"
    ] == {"path": str(gate_path), "sha256": gate_digest}


def test_historical_seller_release_is_not_reinterpreted_as_current_e2_code(
        monkeypatch,
):
    historical_head = recovery.PINNED_E2_CODE_HEAD
    value = {
        "schema": "stackpomdp.atari.e1_seller_release.v2",
        "code_head": historical_head,
        "seller_training_actor_loss_mode": "balanced",
        "buyer_gate": {"source_kind": e2.E1_PRIMARY_SOURCE_KIND},
    }
    monkeypatch.setattr(
        recovery, "validate_seller_release", lambda path: value
    )
    monkeypatch.setattr(
        e2,
        "validate_e1_gate_record",
        lambda gate, role: {
            "source_kind": e2.E1_PRIMARY_SOURCE_KIND,
        },
    )
    monkeypatch.setattr(
        e2,
        "validate_code_root",
        lambda: pytest.fail("historical release compared with current E2 HEAD"),
    )
    assert e2.validated_e1_seller_release(Path("/immutable/release.json")) == value


@pytest.mark.skipif(
    not REAL_CANONICAL_SELLER_RELEASE.is_file()
    or not e2.E1_PRIMARY_AUTHORITY_CODE_ROOT.is_dir(),
    reason="canonical local Atari release artifacts are unavailable",
)
def test_current_v5_runtime_accepts_real_historical_buyer_seller_prerequisites(
        monkeypatch,
):
    """Exercise both immutable E1 prerequisites through the v5 contract."""

    e2.configure_e2_profile(e2.E2_PROFILE_V5)
    current_v5_head = e2.E2_PROFILES[e2.E2_PROFILE_V5]["code_head"]
    monkeypatch.setattr(e2, "validate_code_root", lambda: current_v5_head)
    release = e2.validated_e1_seller_release(REAL_CANONICAL_SELLER_RELEASE)
    assert release["code_head"] == recovery.PINNED_E2_CODE_HEAD
    assert release["code_head"] != current_v5_head
    assert release["buyer_gate"]["source_kind"] == e2.E1_PRIMARY_SOURCE_KIND
    assert release["buyer_gate"]["checkpoint_sha256"] == (
        "5d19fbb579f04da191ab006c5b9e9dcef574fb840e10544338dcab1b019d9e79"
    )
    assert e2.E1_PRIMARY_AUTHORITY_CODE_ROOT.name.endswith("dedab3e")


def test_v5_authority_is_fail_closed_from_diagnostics_to_publication(
        monkeypatch, tmp_path,
):
    e2.configure_e2_profile(e2.E2_PROFILE_V5)
    assert e2._discover_authoritative_seller_v5_gate(
        output_dir=tmp_path, override_report=None
    ) is None

    report = tmp_path / e2.E1_SELLER_V5_REPORT_NAME
    report.write_text(json.dumps({"passed": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="without.*diagnostics gate"):
        e2._discover_authoritative_seller_v5_gate(
            output_dir=tmp_path, override_report=None
        )

    diagnostics_path = tmp_path / e2.E1_SELLER_V5_DIAGNOSTICS_GATE_NAME
    diagnostics_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        e2, "_run_seller_v5_diagnostics_validator", lambda arguments: {}
    )
    report.unlink()
    pending = e2._discover_authoritative_seller_v5_gate(
        output_dir=tmp_path, override_report=None
    )
    assert pending == {
        "kind": "e1_gate_discovery",
        "found": False,
        "authoritative_source": e2.E1_SELLER_V5_SOURCE_KIND,
        "state": "formal_training_or_selection_pending",
    }

    report.write_text(json.dumps({
        "passed": True,
        "role": "seller",
        "evaluator": e2.E1_EVALUATOR,
    }), encoding="utf-8")
    pending = e2._discover_authoritative_seller_v5_gate(
        output_dir=tmp_path, override_report=None
    )
    assert pending["state"] == "selection_gate_publication_pending"

    gate_path = tmp_path / e2.E1_SELLER_V5_GATE_NAME
    checkpoint = tmp_path / "selected.zip"
    gate_path.write_text(json.dumps({
        "selected_checkpoint": {"path": str(checkpoint.resolve())}
    }), encoding="utf-8")
    normalized = {
        "report": str(report.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "sha256": "a" * 64,
        "actor_loss_mode": "balanced",
        "source_kind": e2.E1_SELLER_V5_SOURCE_KIND,
        "sampler_mode": "uniform",
        "support_artifacts": {"gate": "bound"},
    }
    monkeypatch.setattr(e2, "validate_e1_gate", lambda args: normalized)
    discovered = e2._discover_authoritative_seller_v5_gate(
        output_dir=tmp_path, override_report=None
    )
    assert discovered == {
        "kind": "e1_gate_discovery",
        "found": True,
        "role": "seller",
        "report": normalized["report"],
        "checkpoint": normalized["checkpoint"],
        "checkpoint_sha256": normalized["sha256"],
        "actor_loss_mode": "balanced",
        "source_kind": e2.E1_SELLER_V5_SOURCE_KIND,
        "sampler_mode": "uniform",
        "support_artifacts": {"gate": "bound"},
    }


def test_cached_old_seller_cannot_bypass_active_pending_v5(monkeypatch):
    monkeypatch.setattr(
        e2,
        "_discover_authoritative_seller_v5_gate",
        lambda **kwargs: {"found": False, "state": "selection_pending"},
    )
    monkeypatch.setattr(
        e2,
        "_discover_authoritative_seller_threshold_residual_gate",
        lambda **kwargs: pytest.fail("older authority was consulted"),
    )
    with pytest.raises(RuntimeError, match="has not released"):
        e2._validate_authoritative_seller_gate_record({
            "report": "/tmp/old-seller.json"
        })


def test_v5_cohort_requires_exact_buyer_used_to_train_seller(
        monkeypatch, tmp_path,
):
    release_path = tmp_path / "seller-release.json"
    release_path.write_text("{}\n", encoding="utf-8")
    release_record = {
        "path": str(release_path.resolve()), "sha256": "a" * 64,
    }
    buyer_gate = {"checkpoint_sha256": "b" * 64}
    seller_gate = {
        "support_artifacts": {"seller_release": release_record}
    }
    value = {"seller_release": release_record}
    monkeypatch.setattr(e2, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(
        e2,
        "validated_e1_seller_release",
        lambda path: {"buyer_gate": buyer_gate},
    )
    assert e2._validate_cohort_seller_release_binding(
        value, {"buyer": buyer_gate, "seller": seller_gate}
    ) == {"buyer_gate": buyer_gate}

    with pytest.raises(RuntimeError, match="seller-training buyer"):
        e2._validate_cohort_seller_release_binding(
            value,
            {
                "buyer": {"checkpoint_sha256": "c" * 64},
                "seller": seller_gate,
            },
        )


def test_e2_profiles_preserve_v3_and_add_disjoint_v5_contract():
    e2.configure_e2_profile(e2.E2_PROFILE_V3)
    assert e2.E2_NAMESPACE == "e1seller_direct_threshold_residual_v3"
    assert e2.E2_COHORT_SCHEMA == "stackpomdp.atari.e2_e1_gate_cohort.v3"
    assert e2.E2_INPUT_SCHEMA == "stackpomdp.atari.e2_pipeline_inputs.v2"
    assert e2.configured_e2_code_head() == (
        "87fc165000517e874881cac63850133b9982de7f"
    )

    e2.configure_e2_profile(e2.E2_PROFILE_V5)
    assert e2.E2_NAMESPACE == "e1seller_shared_context_v5_exposure_v2"
    assert e2.E1_REQUIRED_SELLER_SOURCE_KIND == e2.E1_SELLER_V5_SOURCE_KIND
    assert e2.E2_COHORT_SCHEMA == "stackpomdp.atari.e2_e1_gate_cohort.v4"
    assert e2.E2_INPUT_SCHEMA == "stackpomdp.atari.e2_pipeline_inputs.v3"
    assert e2.E2_ORCHESTRATION_SCHEMA.endswith(".v2")
    assert e2.configured_e2_code_head() == (
        "180c84f51005d2f36ae86f1cd7319c0e71249065"
    )


def test_v5_orchestration_summary_records_profile_and_namespace(
        monkeypatch, tmp_path,
):
    e2.configure_e2_profile(e2.E2_PROFILE_V5)
    cohort = tmp_path / "cohort.json"
    cohort.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(e2, "validated_e1_gate_cohort", lambda path: {})
    monkeypatch.setattr(
        e2,
        "_e2_orchestration_role_record",
        lambda *, role, **kwargs: {
            "role": role,
            "training_completed": True,
            "selection_outcome": "passed",
        },
    )
    monkeypatch.setattr(e2, "validate_selector_code_root", lambda path: "d" * 40)
    monkeypatch.setattr(
        e2,
        "validate_code_root",
        lambda: e2.E2_PROFILES[e2.E2_PROFILE_V5]["code_head"],
    )
    monkeypatch.setattr(e2, "sha256_file", lambda path: "e" * 64)
    result = e2.build_e2_orchestration_summary(Namespace(
        cohort_manifest=str(cohort),
        checkpoint_root=str(tmp_path / "checkpoints"),
        result_root=str(tmp_path / "results"),
        automation_code_root=str(ROOT),
        buyer_training_exit_code=0,
        buyer_selector_exit_code=0,
        seller_training_exit_code=0,
        seller_selector_exit_code=0,
        recorded_automation_revision=None,
    ))
    assert result["schema"] == (
        "stackpomdp.atari.e2_sequential_orchestration.v2"
    )
    assert result["e2_profile"] == e2.E2_PROFILE_V5
    assert result["e2_namespace"] == e2.E2_NAMESPACE


def test_v5_shell_namespace_job_types_wrappers_and_both_role_mappings():
    common = E2_COMMON.read_text(encoding="utf-8")
    assert (
        "E2_PIPELINE_LOCK=/private/tmp/stackpomdp-atari-e2-shared-context-"
        "v5-exposure-v2-sequential.lock"
    ) in common
    assert (
        "E2_WANDB_BUYER_JOB_TYPE="
        "atari_e2_shared_context_v5_exposure_v2_buyer_leader"
    ) in common
    assert (
        "E2_WANDB_SELLER_JOB_TYPE="
        "atari_e2_shared_context_v5_exposure_v2_seller_leader"
    ) in common
    assert "e2_e1_gate_cohort_${E2_NAMESPACE}.json" in common
    assert "status --short --untracked-files=no" not in common
    assert common.count('status --short)') >= 2
    assert "E1_PRIMARY_AUTHORITY_EXPECTED_HEAD=" in common
    assert "E1_PRIMARY_AUTHORITY_CODE_ROOT=" in common
    for wrapper in E2_V5_WRAPPERS:
        text = wrapper.read_text(encoding="utf-8")
        assert "STACKPOMDP_ATARI_E2_PROFILE=v5-shared-context-exposure-v2" in text
        assert text.count("exec ") == 1

    program = "\n".join((
        'source "$COMMON"',
        'E1_BUYER="/exact/buyer.zip"',
        'E1_SELLER="/exact/seller.zip"',
        'E1_BUYER_MODE="balanced"',
        'E1_SELLER_MODE="balanced"',
        'E1_BUYER_SOURCE_KIND="primary_economic_v1"',
        f'E1_SELLER_SOURCE_KIND="{e2.E1_SELLER_V5_SOURCE_KIND}"',
        'e2_role_paths buyer',
        'print -r -- "buyer:$E2_RESPONSE:$E2_LEADER_E1"',
        'e2_role_paths seller',
        'print -r -- "seller:$E2_RESPONSE:$E2_LEADER_E1"',
    ))
    result = subprocess.run(
        ["zsh", "-c", program],
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
            "COMMON": str(E2_COMMON),
            "STACKPOMDP_ATARI_E2_PROFILE": e2.E2_PROFILE_V5,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "buyer:/exact/seller.zip:/exact/buyer.zip",
        "seller:/exact/buyer.zip:/exact/seller.zip",
    ]


def test_selection_launchers_use_exposure_namespace_and_no_fallback():
    common = SELECTION_COMMON.read_text(encoding="utf-8")
    launcher = SELECTION_LAUNCHER.read_text(encoding="utf-8")
    wrapper = SELECTION_EXPOSURE_WRAPPER.read_text(encoding="utf-8")
    assert "STACKPOMDP_E1V5_PROTOCOL=exposure_v2" in wrapper
    assert launcher.index("e1v5_prepare_runtime") < launcher.index(
        "e1v5s_prepare_runtime"
    )
    assert "E1V5S_DIAGNOSTICS_VALIDATOR" in common
    assert (
        '"$E1V5_PYTHON" "$E1V5S_DIAGNOSTICS_VALIDATOR"' in launcher
    )
    assert (
        '"$E1V5_PYTHON" "$E1V5_VALIDATOR"' not in launcher
    )
    assert "${candidate_args[@]/--candidate/--checkpoint}" in launcher
    assert "screen winner failed fresh confirmation; E2 remains closed" in launcher
    assert "selection_gate" not in common.lower() or "validate-selection-gate" in common
    assert "--wandb" not in launcher
    assert "--resume" not in launcher
