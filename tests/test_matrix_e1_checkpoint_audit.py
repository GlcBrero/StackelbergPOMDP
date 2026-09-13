import hashlib
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_audit():
    path = REPO_ROOT / "replication/matrix_ablations/audit_e1_checkpoints.py"
    spec = importlib.util.spec_from_file_location(
        "matrix_e1_checkpoint_audit", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_contract(
        path,
        checkpoint,
        config,
        row,
):
    profile_hash = canonical_sha256(config["profile"])
    summary = {
        "mean_regret": row["mean_regret"],
        "max_regret": row["max_regret"],
        "optimal_commitments": row["optimal_commitments"],
        "commitments": row["commitments"],
    }
    contract = {
        "schema_version": 2,
        "kind": "matrix_meta_follower",
        "algorithm": "REINFORCE",
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "response_game": {
            "profile_id": config["profile_id"],
            "profile": config["profile"],
            "profile_sha256": profile_hash,
            "memory_mode": config["memory_mode"],
            "episode_length": config["episode_length"],
            "query_states": config["query_states"],
            "training_reward_offset": config["reward_offset"],
        },
        "metadata": {
            "evaluation": summary,
            "seed": config["seed"],
            "training_config_sha256": canonical_sha256(config),
            "response_training": config["reinforce_protocol"],
            "completed_updates": row["completed_updates"],
            "follower_gradient_samples": row[
                "follower_gradient_samples"
            ],
            "executed_equivalent_steps": row[
                "executed_equivalent_steps"
            ],
        },
    }
    write_json(path, contract)


def make_completed_run(root):
    run_dir = root / "meta_follower/joint/reinforce/seed1-test"
    run_dir.mkdir(parents=True)
    profile = {
        "profile_id": "paper_joint_v1",
        "memory_mode": "joint",
        "state_labels": ["Start", "CC", "CD", "DC", "DD"],
        "state_index_formula": "0=Start; 1+2*leader_action+follower_action",
        "query_state_order": [0, 1, 2, 3, 4],
        "context_encoding_version": "mixed_radix_v1",
        "training_reward_offset": -4.0,
        "reporting_transform": "identity",
        "interpretation": "test",
    }
    rollout = {
        "collection_target_env_steps": 100,
        "episodes_per_update": 10,
        "executed_env_steps_per_update": 100,
        "query_env_steps_per_update": 50,
        "follower_gradient_samples_per_update": 50,
    }
    cadence = {
        "update_interval": 1,
        "follower_gradient_samples_interval": 50,
        "executed_env_steps_interval": 100,
    }
    config = {
        "schema_version": 2,
        "stage": "meta_follower",
        "profile_id": profile["profile_id"],
        "profile": profile,
        "matrix": "prisoners_dilemma",
        "memory_mode": "joint",
        "algorithm": "REINFORCE",
        "seed": 1,
        "learning_rate": 0.02,
        "timesteps": 100,
        "n_steps": 50,
        "episode_length": 5,
        "query_states": [0, 1, 2, 3, 4],
        "reward_offset": -4.0,
        "reinforce_protocol": {
            "loss": "negative_mean_logp_times_reward_to_go",
            "rollout_geometry": rollout,
            "checkpoint_geometry": cadence,
        },
        "run_dir": "remote/results/meta_follower/joint/reinforce/seed1-test",
    }
    config_hash = canonical_sha256(config)
    checkpoints = [
        run_dir / "meta_follower_update0001_follower50.zip",
        run_dir / "model.zip",
    ]
    for index, checkpoint in enumerate(checkpoints, start=1):
        checkpoint.write_bytes("checkpoint {}".format(index).encode("ascii"))
    rows = [
        {
            "schema_version": 2,
            "profile_id": profile["profile_id"],
            "seed": 1,
            "learning_rate": 0.02,
            "completed_updates": 1,
            "follower_gradient_samples": 50,
            "executed_equivalent_steps": 100,
            "training_config_sha256": config_hash,
            "checkpoint_semantics": "post_optimizer_update",
            "mean_regret": 0.125,
            "max_regret": 1.0,
            "optimal_commitments": 31,
            "commitments": 32,
            "checkpoint_filename": checkpoints[0].name,
            "checkpoint_sha256": sha256(checkpoints[0]),
            "contract_filename": (
                "meta_follower_update0001_follower50.response_contract.json"
            ),
        },
        {
            "schema_version": 2,
            "profile_id": profile["profile_id"],
            "seed": 1,
            "learning_rate": 0.02,
            "completed_updates": 2,
            "follower_gradient_samples": 100,
            "executed_equivalent_steps": 200,
            "training_config_sha256": config_hash,
            "checkpoint_semantics": "post_optimizer_update",
            "mean_regret": 0.0,
            "max_regret": 0.0,
            "optimal_commitments": 32,
            "commitments": 32,
            "checkpoint_filename": "model.zip",
            "checkpoint_sha256": sha256(checkpoints[1]),
            "contract_filename": "response_contract.json",
        },
    ]
    contract_paths = [run_dir / row["contract_filename"] for row in rows]
    for row, checkpoint, contract_path in zip(rows, checkpoints, contract_paths):
        make_contract(contract_path, checkpoint, config, row)
        row["contract_sha256"] = sha256(contract_path)

    ledger_path = run_dir / "checkpoint_evaluations.jsonl"
    ledger_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    evaluation_path = run_dir / "evaluation.json"
    write_json(evaluation_path, {
        "schema_version": 2,
        "config": config,
        "summary": {
            "mean_regret": 0.0,
            "max_regret": 0.0,
            "optimal_commitments": 32,
            "commitments": 32,
        },
    })
    write_json(run_dir / "config.json", config)

    files = {
        "checkpoint_evaluations": ledger_path,
        "evaluation": evaluation_path,
        "checkpoint_update0001": checkpoints[0],
        "checkpoint_contract_update0001": contract_paths[0],
        "checkpoint_final": checkpoints[1],
        "checkpoint_contract_final": contract_paths[1],
        "model": checkpoints[1],
        "response_contract": contract_paths[1],
    }
    manifest = {
        "schema_version": 2,
        "status": "completed",
        "config_sha256": config_hash,
        "artifacts": {
            name: {
                # Exercise relocation: the audit must not follow this stale
                # cluster path and must instead bind to the local run dir.
                "path": "/home/user/remote-results/{}".format(path.name),
                "sha256": sha256(path),
            }
            for name, path in files.items()
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir, rows


def test_audit_validates_all_checkpoints_and_reports_strict_pass(tmp_path):
    audit = load_audit()
    run_dir, _ = make_completed_run(tmp_path)

    report = audit.audit_root(tmp_path)

    assert report["status"] == "valid"
    assert report["completed_run_count"] == 1
    assert report["checkpoint_count"] == 2
    assert report["strict_pass_count"] == 1
    assert report["has_strict_pass"]
    run = report["runs"][0]
    assert run["run_dir"] == str(run_dir.resolve())
    assert run["valid"]
    assert [row["strict_pass"] for row in run["checkpoints"]] == [False, True]
    assert run["strict_passes"][0]["checkpoint_filename"] == "model.zip"


def test_audit_rejects_checkpoint_tampering_and_withholds_pass(tmp_path):
    audit = load_audit()
    run_dir, _ = make_completed_run(tmp_path)
    (run_dir / "meta_follower_update0001_follower50.zip").write_bytes(
        b"tampered"
    )

    report = audit.audit_root(tmp_path)

    assert report["status"] == "invalid"
    assert report["invalid_run_count"] == 1
    assert not report["has_strict_pass"]
    errors = " ".join(report["runs"][0]["errors"])
    row_errors = " ".join(report["runs"][0]["checkpoints"][0]["errors"])
    assert "artifact" in errors
    assert "checkpoint row SHA256 mismatch" in row_errors


def test_audit_rejects_profile_and_counter_misalignment_even_if_rehashed(
        tmp_path
):
    audit = load_audit()
    run_dir, rows = make_completed_run(tmp_path)
    rows[1]["profile_id"] = "other_profile"
    rows[1]["follower_gradient_samples"] = 99
    ledger = run_dir / "checkpoint_evaluations.jsonl"
    ledger.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["checkpoint_evaluations"]["sha256"] = sha256(ledger)
    write_json(manifest_path, manifest)

    report = audit.audit_root(tmp_path)

    assert report["status"] == "invalid"
    assert not report["has_strict_pass"]
    row_errors = " ".join(report["runs"][0]["checkpoints"][1]["errors"])
    assert "profile_id does not match" in row_errors
    assert "follower-step counter is misaligned" in row_errors


def test_cli_atomically_writes_same_compact_report_it_prints(
        tmp_path, capsys
):
    audit = load_audit()
    make_completed_run(tmp_path)
    output = tmp_path / "reports/audit.json"

    exit_code = audit.main([
        "--root", str(tmp_path),
        "--output", str(output),
    ])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())
    assert printed == written
    assert written["status"] == "valid"
    assert not list(output.parent.glob("*.tmp"))
