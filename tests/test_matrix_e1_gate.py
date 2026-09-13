import hashlib
import importlib.util
import json
from pathlib import Path

from stackelberg_pomdp.matrix_ablations.profiles import PAPER_JOINT_V1


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_gate():
    path = REPO_ROOT / "replication/matrix_ablations/gate_e1.py"
    spec = importlib.util.spec_from_file_location("matrix_e1_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(root, seeds=(1,)):
    records = []
    for seed in seeds:
        key = "e1.seed{}".format(seed)
        records.append({
            "key": key,
            "stage": "meta-follower",
            "seed": seed,
            "match": {
                "sweep_id": "test-sweep",
                "record_key": key,
                "stage": "meta_follower",
                "matrix": "prisoners_dilemma",
                "memory_mode": "joint",
                "algorithm": "PPO",
                "seed": seed,
                "timesteps": 200,
                "learning_rate": 0.002,
            },
        })
    plan = {
        "schema_version": 1,
        "sweep_id": "test-sweep",
        "seeds": list(seeds),
        "records": records,
    }
    path = root / "plan.json"
    write_json(path, plan)
    return path, plan


def make_completed_e1(root, plan_path, plan, seed=1, attempt=0, regret=0.1):
    run_dir = root / "runs/meta_follower/joint/ppo" / (
        "seed{}-attempt{}".format(seed, attempt)
    )
    run_dir.mkdir(parents=True)
    key = "e1.seed{}".format(seed)
    config = {
        "schema_version": 2,
        "stage": "meta_follower",
        "profile_id": PAPER_JOINT_V1.profile_id,
        "profile": PAPER_JOINT_V1.to_dict(),
        "matrix": "prisoners_dilemma",
        "memory_mode": "joint",
        "algorithm": "PPO",
        "seed": seed,
        "timesteps": 200,
        "learning_rate": 0.002,
        "episode_length": 5,
        "query_states": [0, 1, 2, 3, 4],
        "payoffs": [
            [[4.0, 3.0], [2.0, 4.0]],
            [[3.0, 1.0], [1.0, 2.0]],
        ],
        "reward_offset": -4.0,
        "sweep_id": plan["sweep_id"],
        "record_key": key,
        "sweep_plan": str(plan_path),
        "sweep_plan_sha256": sha256(plan_path),
        "attempt": attempt,
        "run_dir": str(run_dir),
    }
    model_path = run_dir / "model.zip"
    model_path.write_bytes("model seed {} attempt {}".format(seed, attempt).encode())
    evaluation = {
        "schema_version": 2,
        "config": config,
        "summary": {"max_regret": regret},
    }
    evaluation_path = run_dir / "evaluation.json"
    write_json(evaluation_path, evaluation)
    gate = load_gate()
    contract = {
        "schema_version": 2,
        "kind": "matrix_meta_follower",
        "algorithm": "PPO",
        "checkpoint_filename": "model.zip",
        "checkpoint_sha256": sha256(model_path),
        "response_game": {
            "profile_id": PAPER_JOINT_V1.profile_id,
            "profile": PAPER_JOINT_V1.to_dict(),
            "memory_mode": "joint",
            "episode_length": 5,
            "query_states": [0, 1, 2, 3, 4],
            "training_reward_offset": -4.0,
            "centered_follower_payoffs": [[-1.0, 0.0], [-3.0, -2.0]],
        },
        "metadata": {
            "seed": seed,
            "evaluation": {"max_regret": regret},
            "training_config_sha256": gate.config_sha256(config),
        },
    }
    contract_path = run_dir / "response_contract.json"
    write_json(contract_path, contract)
    manifest = {
        "schema_version": 2,
        "status": "completed",
        "config_sha256": gate.config_sha256(config),
        "artifacts": {
            "model": {"path": str(model_path), "sha256": sha256(model_path)},
            "evaluation": {
                "path": str(evaluation_path),
                "sha256": sha256(evaluation_path),
            },
            "response_contract": {
                "path": str(contract_path),
                "sha256": sha256(contract_path),
            },
        },
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def test_gate_passes_one_valid_completed_e1_per_planned_seed(tmp_path):
    gate = load_gate()
    plan_path, plan = make_plan(tmp_path, seeds=(1, 2))
    make_completed_e1(tmp_path, plan_path, plan, seed=1)
    make_completed_e1(tmp_path, plan_path, plan, seed=2)

    report = gate.run_gate(tmp_path)

    assert report["passed"]
    assert [row["seed"] for row in report["e1"]] == [1, 2]
    assert all(row["checkpoint_sha256"] for row in report["e1"])
    assert json.loads((tmp_path / "e1_gate.json").read_text())["passed"]
    assert not (tmp_path / "e1_gate.json.tmp").exists()


def test_contract_payoffs_use_player_axis_of_joint_action_tensor(tmp_path):
    gate = load_gate()
    plan_path, plan = make_plan(tmp_path)
    make_completed_e1(tmp_path, plan_path, plan)

    report = gate.run_gate(tmp_path)

    assert report["passed"]
    config = json.loads(next(
        (tmp_path / "runs/meta_follower").glob("**/config.json")
    ).read_text())
    assert gate._centered_follower_payoffs(config, []) == [
        [-1.0, 0.0], [-3.0, -2.0]
    ]


def test_gate_rejects_duplicate_completed_attempts(tmp_path):
    gate = load_gate()
    plan_path, plan = make_plan(tmp_path)
    make_completed_e1(tmp_path, plan_path, plan, attempt=0)
    make_completed_e1(tmp_path, plan_path, plan, attempt=1)

    report = gate.run_gate(tmp_path)

    assert not report["passed"]
    assert "found 2" in report["e1"][0]["errors"][0]


def test_gate_rejects_tampered_checkpoint_sha(tmp_path):
    gate = load_gate()
    plan_path, plan = make_plan(tmp_path)
    run_dir = make_completed_e1(tmp_path, plan_path, plan)
    (run_dir / "model.zip").write_bytes(b"tampered")

    report = gate.run_gate(tmp_path)

    assert not report["passed"]
    errors = " ".join(report["e1"][0]["errors"])
    assert "manifest" in errors
    assert "response contract" in errors


def test_cli_fails_and_writes_report_when_regret_exceeds_threshold(tmp_path):
    gate = load_gate()
    plan_path, plan = make_plan(tmp_path)
    make_completed_e1(tmp_path, plan_path, plan, regret=0.251)

    exit_code = gate.main(["--sweep-root", str(tmp_path)])

    assert exit_code == 1
    report = json.loads((tmp_path / "e1_gate.json").read_text())
    assert not report["passed"]
    assert "exceeds threshold" in " ".join(report["e1"][0]["errors"])
