import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_PATH = REPO_ROOT / "replication/matrix_ablations/plot.py"
SPEC = importlib.util.spec_from_file_location("matrix_plot", PLOT_PATH)
matrix_plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix_plot)


def _canonical_hash(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_run(root, *, seed, implementation="ray_rllib_es_2_0_1"):
    run_dir = root / "runs" / "es" / "hidden" / "seed{}".format(seed)
    run_dir.mkdir(parents=True)
    config = {
        "stage": "leader",
        "profile_id": "paper_joint_v1",
        "experiment": "hidden_queries",
        "condition": "hidden",
        "matrix": "modified_pd",
        "algorithm": "ES",
        "seed": seed,
        "episode_length": 5,
        "learning_rate": None,
        "es_protocol": {
            "implementation": implementation,
            "stepsize": 0.01,
        },
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    rows = [
        {
            "row_type": "training_iteration",
            "iteration": iteration,
            "global_step": iteration * 10_000 + seed * 10,
            "measured_timesteps_total": iteration * 10_000 + seed * 10,
            "native_episode_reward_mean": reward * 5,
            "native_leader_reward_per_stage": reward,
        }
        for iteration, reward in ((1, -1.5 + seed / 10), (2, -1.2 + seed / 10))
    ]
    progress_path = run_dir / "progress.jsonl"
    progress_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "status": "completed",
        "config_sha256": _canonical_hash(config),
        "artifacts": {
            "progress": {"sha256": matrix_plot.sha256(progress_path)},
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_plot_collects_only_native_rllib_es_and_aligns_by_iteration(tmp_path):
    _write_run(tmp_path, seed=1)
    _write_run(tmp_path, seed=2)
    _write_run(tmp_path, seed=3, implementation="obsolete_custom_es")

    runs, history, _ = matrix_plot.collect_runs(tmp_path)
    assert set(runs["seed"]) == {1, 2}
    assert set(history["plot_step"]) == {1, 2}
    assert set(history["plot_step_unit"]) == {"es_iterations"}
    assert set(history["evaluation_metric"]) == {
        "native_episode_reward_mean_per_stage"
    }
    assert history["measured_timesteps_total"].nunique() == 4

    matrix_plot.validate_history(history, {1, 2}, allow_incomplete=False)
    summary = matrix_plot.summarize(history)
    assert set(summary["n_independent_seeds"]) == {2}
    assert np.allclose(summary["sem"], 0.05)


def test_plot_cli_has_no_es_backend_selector():
    parser = matrix_plot.build_parser()
    assert not any(
        "--es-backend" in action.option_strings for action in parser._actions
    )
