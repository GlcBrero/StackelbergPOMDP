from argparse import Namespace
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace
import time
import zipfile

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


def _primary_protocol():
    return {
        "schema_version": 1,
        "kind": validator.E1_PRIMARY_PROTOCOL_KIND,
        "role": "buyer",
        "evaluator": validator.E1_PRIMARY_EVALUATOR,
        "evaluator_code_revision": "a" * 40,
        "source_kind": validator.E1_PRIMARY_SOURCE_KIND,
        "sampler_mode": validator.E1_TEMPORAL_SAMPLER,
        "economic_check_names": list(validator.E1_PRIMARY_CHECK_NAMES),
        "candidate_policy": {
            "eligible_count": 1,
            "candidate_search": False,
            "fallback_allowed": False,
        },
        "holdout": {
            "random": {
                "episodes": 100,
                "seed_start": 8_000_001,
                "seed_end": 8_000_100,
            },
            "fixed_grid": {
                "episodes_per_value": 20,
                "seed_start": 8_100_001,
                "seed_end": 8_100_020,
                "values": [value / 10.0 for value in range(11)],
                "shared_seeds": True,
                "event_steps": [20, 50, 80, 110, 140],
            },
            "timing_evaluation_run": False,
        },
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


def test_primary_protocol_is_authoritative_pending_and_fail_closed(
        monkeypatch, tmp_path,
):
    protocol = tmp_path / validator.E1_PRIMARY_PROTOCOL_NAME
    report = tmp_path / validator.E1_PRIMARY_REPORT_NAME
    gate = tmp_path / validator.E1_PRIMARY_GATE_NAME
    legacy = tmp_path / "e1_buyer_balanced_all6_selector_v2.json"
    legacy.write_text(json.dumps({"passed": True}), encoding="utf-8")
    protocol.write_text(json.dumps(_primary_protocol()), encoding="utf-8")
    args = Namespace(
        role="buyer", output_dir=str(tmp_path), override_report=None,
        require_mode=None,
    )

    pending = validator.discover_e1_gate(args)
    assert pending["found"] is False
    assert pending["authoritative_source"] == validator.E1_PRIMARY_SOURCE_KIND

    malformed = _primary_protocol()
    malformed["holdout"]["random"]["seed_start"] += 1
    protocol.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="primary random seed_start"):
        validator.discover_e1_gate(args)
    protocol.write_text(json.dumps(_primary_protocol()), encoding="utf-8")

    report.write_text(json.dumps({
        "passed": False,
        "role": "buyer",
        "evaluator": validator.E1_PRIMARY_EVALUATOR,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="confirmation failed"):
        validator.discover_e1_gate(args)

    passing_report = {
        "passed": True,
        "role": "buyer",
        "evaluator": validator.E1_PRIMARY_EVALUATOR,
    }
    report.write_text(json.dumps(passing_report), encoding="utf-8")
    assert validator.discover_e1_gate(args)["state"] == (
        "gate_publication_pending"
    )

    automation_root = tmp_path / "automation_worktree"
    artifact_root = tmp_path / "main_artifact_root"
    checkpoint = artifact_root / validator.E1_PRIMARY_CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"primary")
    digest = "a" * 64
    passing_report["selected_alias"] = {
        "pinned_path": str(checkpoint.resolve()), "sha256": digest,
    }
    report.write_text(json.dumps(passing_report), encoding="utf-8")
    gate.write_text(json.dumps({
        "selected_checkpoint": {
            "path": str(checkpoint.resolve()), "sha256": digest,
        },
    }), encoding="utf-8")
    monkeypatch.setattr(validator, "AUTOMATION_SOURCE_ROOT", automation_root)
    monkeypatch.setattr(validator, "validate_e1_gate", lambda local: {
        "report": str(report.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "sha256": digest,
        "actor_loss_mode": "balanced",
        "source_kind": validator.E1_PRIMARY_SOURCE_KIND,
        "sampler_mode": validator.E1_TEMPORAL_SAMPLER,
        "support_artifacts": {"primary_economic_gate": {}},
    })
    discovered = validator.discover_e1_gate(args)
    assert discovered["found"] is True
    assert discovered["checkpoint"] == str(checkpoint.resolve())
    assert not str(checkpoint).startswith(str(automation_root))

    with pytest.raises(RuntimeError, match="forbids overriding"):
        validator.discover_e1_gate(Namespace(
            **{**vars(args), "override_report": str(legacy)}
        ))


def test_orphan_primary_report_never_falls_back(monkeypatch, tmp_path):
    report = tmp_path / validator.E1_PRIMARY_REPORT_NAME
    report.write_text(json.dumps({
        "passed": True,
        "role": "buyer",
        "evaluator": validator.E1_PRIMARY_EVALUATOR,
    }), encoding="utf-8")
    monkeypatch.setattr(
        validator, "validate_e1_gate",
        lambda args: pytest.fail("orphan primary report reached validation"),
    )
    with pytest.raises(RuntimeError, match="without their authoritative protocol"):
        validator.discover_e1_gate(Namespace(
            role="buyer", output_dir=str(tmp_path), override_report=None,
            require_mode=None,
        ))


def test_primary_validation_subprocess_separates_active_and_pinned_code(
        monkeypatch, tmp_path,
):
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(
            stdout=json.dumps({
                "kind": "primary_economic_gate",
                "passed": True,
                "value": {"kind": validator.E1_PRIMARY_GATE_KIND},
            }) + "\n"
        )

    monkeypatch.setattr(validator.subprocess, "run", run)
    monkeypatch.setenv("STACKPOMDP_CODE_ROOT", "/private/tmp/pinned-e2")
    # The helper checks that its active sibling exists before spawning.
    monkeypatch.setattr(
        validator, "E1_PRIMARY_MODULE_NAME",
        Path(validator.__file__).name,
    )
    result = validator._run_primary_economic_gate_validator(
        protocol_path=tmp_path / "protocol.json",
        report_path=tmp_path / "report.json",
        gate_path=tmp_path / "gate.json",
        selected_checkpoint=tmp_path / "selected.zip",
    )
    assert result["kind"] == validator.E1_PRIMARY_GATE_KIND
    assert observed["cwd"] == str(validator.AUTOMATION_SOURCE_ROOT)
    assert observed["env"]["PYTHONPATH"] == str(
        validator.AUTOMATION_SOURCE_ROOT
    )
    assert observed["command"][-2:] == [
        "--code-root", str(validator.AUTOMATION_SOURCE_ROOT),
    ]
    assert "STACKPOMDP_CODE_ROOT" not in observed["env"]


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


def test_validator_rejects_symlinks_before_resolution(tmp_path):
    target_json = tmp_path / "target.json"
    target_json.write_text("{}", encoding="utf-8")
    linked_json = tmp_path / "linked.json"
    linked_json.symlink_to(target_json)
    with pytest.raises(RuntimeError, match="non-symlink JSON"):
        validator.load_json(linked_json)
    with pytest.raises(RuntimeError, match="non-symlink file"):
        validator.sha256_file(linked_json)

    target_zip = tmp_path / "target.zip"
    with zipfile.ZipFile(target_zip, "w") as archive:
        archive.writestr("member", b"value")
    linked_zip = tmp_path / "linked.zip"
    linked_zip.symlink_to(target_zip)
    with pytest.raises(RuntimeError, match="non-symlink ZIP"):
        validator.validate_zip(linked_zip)

    publication = tmp_path / "publication.json"
    publication.symlink_to(target_json)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        validator.atomic_write_new_json(publication, {"changed": True})
    assert target_json.read_text(encoding="utf-8") == "{}"


def test_primary_gate_rejects_symlinked_cli_paths(tmp_path):
    report_target = tmp_path / validator.E1_PRIMARY_REPORT_NAME
    report_target.write_text("{}", encoding="utf-8")
    report_link = tmp_path / "report-link.json"
    report_link.symlink_to(report_target)
    checkpoint_target = tmp_path / validator.E1_PRIMARY_CHECKPOINT_NAME
    checkpoint_target.write_bytes(b"not reached")
    checkpoint_link = tmp_path / "checkpoint-link.zip"
    checkpoint_link.symlink_to(checkpoint_target)

    arguments = Namespace(
        role="buyer",
        actor_loss_mode="balanced",
        report=str(report_link),
        checkpoint=str(checkpoint_target),
    )
    with pytest.raises(RuntimeError, match="report cannot be a symlink"):
        validator._validate_primary_economic_gate(arguments)

    arguments.report = str(report_target)
    arguments.checkpoint = str(checkpoint_link)
    with pytest.raises(RuntimeError, match="checkpoint cannot be a symlink"):
        validator._validate_primary_economic_gate(arguments)


def _run_lock_helper(common, lock, token):
    program = """
source "$COMMON"
stackpomdp_claim_owned_lock "$LOCK" "$TOKEN" "test" || exit $?
print -r -- "owned=$STACKPOMDP_LOCK_RESULT_OWNED"
if [[ "$STACKPOMDP_LOCK_RESULT_OWNED" == 1 ]]; then
  stackpomdp_release_owned_lock "$LOCK" "$TOKEN" 1 "test"
fi
"""
    return subprocess.run(
        ["zsh", "-c", program],
        env={
            **os.environ,
            "COMMON": str(common),
            "LOCK": str(lock),
            "TOKEN": token,
        },
        capture_output=True,
        text=True,
    )


def test_owned_lock_reclaims_only_old_empty_directories(tmp_path):
    common = (
        Path(__file__).resolve().parents[1]
        / "replication/atari/automation/atari_e2_pipeline_common.zsh"
    )
    lock = tmp_path / "old-empty.lock"
    lock.mkdir()
    old = time.time() - 120
    os.utime(lock, (old, old))
    result = _run_lock_helper(common, lock, "new-owner")
    assert result.returncode == 0, result.stderr
    assert "reclaimed stable empty test lock" in result.stdout
    assert "owned=1" in result.stdout
    assert not lock.exists()

    young = tmp_path / "young-empty.lock"
    young.mkdir()
    result = _run_lock_helper(common, young, "new-owner")
    assert result.returncode != 0
    assert "younger than 60 seconds" in result.stderr
    assert young.is_dir()


@pytest.mark.parametrize(
    "owner_kind", ["live", "dead", "reentrant_dead", "foreign", "malformed"]
)
def test_owned_lock_refuses_unsafe_recorded_owners(tmp_path, owner_kind):
    common = (
        Path(__file__).resolve().parents[1]
        / "replication/atari/automation/atari_e2_pipeline_common.zsh"
    )
    lock = tmp_path / f"{owner_kind}.lock"
    lock.mkdir()
    host = socket.gethostname()
    if owner_kind == "live":
        record = f"other\t{os.getpid()}\t{host}\tcreated\n"
    elif owner_kind == "dead":
        record = f"other\t99999999\t{host}\tcreated\n"
    elif owner_kind == "reentrant_dead":
        record = f"same-token\t99999999\t{host}\tcreated\n"
    elif owner_kind == "foreign":
        record = "other\t99999999\tother-host\tcreated\n"
    else:
        record = "same-token\n"
    (lock / "owner.tsv").write_text(record, encoding="utf-8")
    result = _run_lock_helper(common, lock, "same-token")
    assert result.returncode != 0
    assert lock.is_dir()
    assert (lock / "owner.tsv").read_text(encoding="utf-8") == record
    if owner_kind == "dead":
        assert "descendants may still be active" in result.stderr
    if owner_kind == "reentrant_dead":
        assert "reentrant token names absent PID" in result.stderr
    if owner_kind == "foreign":
        assert "foreign host" in result.stderr
    if owner_kind == "malformed":
        assert "malformed owner metadata" in result.stderr


def test_owned_lock_allows_valid_nested_token_without_releasing_parent(tmp_path):
    common = (
        Path(__file__).resolve().parents[1]
        / "replication/atari/automation/atari_e2_pipeline_common.zsh"
    )
    lock = tmp_path / "nested.lock"
    lock.mkdir()
    token = "shared-token"
    record = f"{token}\t{os.getpid()}\t{socket.gethostname()}\tcreated\n"
    (lock / "owner.tsv").write_text(record, encoding="utf-8")
    result = _run_lock_helper(common, lock, token)
    assert result.returncode == 0, result.stderr
    assert "owned=0" in result.stdout
    assert (lock / "owner.tsv").read_text(encoding="utf-8") == record


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


def test_primary_master_chains_all_gates_and_preserves_live_wandb():
    root = Path(__file__).resolve().parents[1]
    automation = root / "replication/atari/automation"
    master = (
        automation
        / "run_atari_clean_e1_primary_economic_to_e2_sequential.sh"
    ).read_text(encoding="utf-8")
    buyer_stage = master.index(
        'run_required_stage "E1 primary-economic buyer confirmation"'
    )
    seller_stage = master.index(
        'run_required_stage "E1 seller training and selection"'
    )
    e2_stage = master.index(
        'run_required_stage "sequential E2 buyer/seller training and selection"'
    )
    assert master.index("e2_claim_pipeline_lock") < buyer_stage
    assert buyer_stage < seller_stage < e2_stage
    assert "return 2" in master
    assert "downstream stages remain closed" in master

    common = (automation / "atari_e2_pipeline_common.zsh").read_text(
        encoding="utf-8"
    )
    assert "CODE_ROOT=/private/tmp/stackpomdp-e2-code-7a193ba" in common
    assert "EXPECTED_HEAD=7a193ba14b91f6ab116da29ff288e3e577d73b88" in common
    assert "WANDB_MODE=online" in common
    assert '"E2 pipeline" || return $?' in common
    assert "write-e1-gate-cohort" in common
    assert common.index("write-e1-gate-cohort") < common.index(
        '"E1 cohort initialization"; then'
    )
    for name in (
        "run_atari_clean_e1_seller_after_buyer_gate.sh",
        "run_atari_clean_e2_buyer_balanced_2m.sh",
        "run_atari_clean_e2_seller_balanced_2m.sh",
    ):
        assert "--wandb" in (automation / name).read_text(encoding="utf-8")


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
    assert "stackpomdp_claim_owned_lock" in script
    assert "STACKPOMDP_E1_SELLER_SELECTOR_LOCK_TOKEN" in script
    assert "stackpomdp_release_owned_lock" in script
    assert 'rmdir "$SELECTOR_LOCK"' not in script
