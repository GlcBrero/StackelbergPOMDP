import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_trend():
    path = REPO_ROOT / "replication/matrix_ablations/trend.py"
    spec = importlib.util.spec_from_file_location("matrix_ablation_trend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, payload, allow_nan=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def leader_record(experiment, matrix, algorithm, condition, seed):
    key = ".".join((
        experiment, matrix, algorithm.lower(), condition, "seed{}".format(seed)
    ))
    match = {
        "sweep_id": "test-sweep",
        "record_key": key,
        "stage": "leader",
        "experiment": experiment,
        "matrix": matrix,
        "algorithm": algorithm,
        "condition": condition,
        "seed": seed,
    }
    return {"key": key, "stage": "leader", "seed": seed, "match": match}


def write_plan(root, records):
    path = root / "plan.json"
    write_json(path, {
        "schema_version": 1,
        "sweep_id": "test-sweep",
        "records": records,
    })
    return path


def write_attempt(
        trend, plan_path, record, attempt, status, value=None,
        nonfinite=False,
):
    run_dir = (
        plan_path.parent / "runs" / record["key"] / "attempt{}".format(attempt)
    )
    config = {
        **record["match"],
        "attempt": attempt,
        "run_dir": str(run_dir),
        "sweep_plan": str(plan_path),
        "sweep_plan_sha256": trend.file_sha256(plan_path),
    }
    manifest = {
        "schema_version": 2,
        "status": status,
        "config_sha256": trend.canonical_sha256(config),
    }
    write_json(run_dir / "config.json", config)
    if status == "completed":
        evaluation = {
            "schema_version": 2,
            "config": config,
            "per_stage_summary": {
                "n": 10,
                "mean": float("nan") if nonfinite else value,
                "std": 0.0,
                "sem": 0.0,
            },
        }
        evaluation_path = run_dir / "evaluation.json"
        write_json(evaluation_path, evaluation, allow_nan=nonfinite)
        manifest["artifacts"] = {
            "evaluation": {
                "path": str(evaluation_path),
                "sha256": trend.file_sha256(evaluation_path),
            }
        }
    write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def assert_value_error(function, expected_text):
    try:
        function()
    except ValueError as exc:
        assert expected_text in str(exc)
    else:
        raise AssertionError("expected ValueError containing {!r}".format(
            expected_text
        ))


def test_summary_collapses_attempts_and_uses_paired_sample_sem(tmp_path):
    trend = load_trend()
    records = []
    hidden = {}
    for condition in ("observed", "hidden"):
        for seed in (1, 2):
            record = leader_record(
                "hidden_queries", "modified_pd", "A2C", condition, seed
            )
            records.append(record)
            hidden[condition, seed] = record
    phase_visible = leader_record(
        "phase_observability", "prisoners_dilemma", "A2C", "visible", 1
    )
    phase_hidden = leader_record(
        "phase_observability", "prisoners_dilemma", "A2C", "hidden", 1
    )
    reset = leader_record(
        "q_reset", "battle_of_the_sexes", "A2C", "reset", 1
    )
    ongoing = leader_record(
        "q_reset", "battle_of_the_sexes", "A2C", "ongoing", 1
    )
    records.extend((phase_visible, phase_hidden, reset, ongoing))
    plan_path = write_plan(tmp_path / "test-sweep", records)

    write_attempt(trend, plan_path, hidden["observed", 1], 0, "failed")
    write_attempt(trend, plan_path, hidden["observed", 1], 1, "completed", 1.0)
    write_attempt(trend, plan_path, hidden["observed", 2], 0, "completed", 3.0)
    write_attempt(trend, plan_path, hidden["hidden", 1], 0, "completed", 0.0)
    write_attempt(trend, plan_path, hidden["hidden", 2], 0, "completed", 0.0)
    write_attempt(trend, plan_path, phase_visible, 0, "running")
    write_attempt(trend, plan_path, reset, 0, "failed")

    summary = trend.build_summary(plan_path)
    counts = summary["status_counts_by_stage"]["leader"]
    assert counts == {
        "planned": 8,
        "completed": 4,
        "failed": 1,
        "running": 1,
        "missing": 2,
    }
    selected = next(
        row for row in summary["records"]
        if row["record_key"] == hidden["observed", 1]["key"]
    )
    assert selected["attempts"] == 2
    assert selected["selected_attempt"] == 1

    observed = next(
        row for row in summary["final_per_stage"]
        if row["experiment"] == "hidden_queries"
        and row["condition"] == "observed"
    )
    assert observed["mean"] == 2.0
    assert observed["std"] == 2.0 ** 0.5
    assert observed["sem"] == 1.0
    delta = summary["condition_deltas"]["hidden_queries"][0]
    assert delta["definition"] == "observed - hidden"
    assert delta["paired_seeds"] == [1, 2]
    assert delta["mean"] == 2.0
    assert delta["sem"] == 1.0


def test_condition_deltas_cover_all_paper_comparisons():
    trend = load_trend()
    rows = []

    def add(experiment, matrix, algorithm, condition, seed, value):
        rows.append({
            "stage": "leader",
            "status": "completed",
            "experiment": experiment,
            "matrix": matrix,
            "algorithm": algorithm,
            "condition": condition,
            "seed": seed,
            "per_stage_mean": value,
        })

    for algorithm in ("A2C", "PPO", "ES"):
        add("hidden_queries", "modified_pd", algorithm, "observed", 1, 2.0)
        add("hidden_queries", "modified_pd", algorithm, "hidden", 1, 1.0)
    add("phase_observability", "prisoners_dilemma", "A2C", "visible", 1, 0.0)
    add("phase_observability", "prisoners_dilemma", "A2C", "hidden", 1, -1.0)
    add("q_reset", "battle_of_the_sexes", "A2C", "reset", 1, 2.0)
    add("q_reset", "battle_of_the_sexes", "A2C", "ongoing", 1, 1.5)
    for matrix in ("coordination_zero", "coordination_penalized"):
        add("response_reward", matrix, "A2C", "excluded", 1, 2.0)
        add("response_reward", matrix, "A2C", "included", 1, 1.0)

    result = trend.condition_deltas(rows)
    hidden = {row["algorithm"]: row for row in result["hidden_queries"]}
    assert set(hidden) == {"A2C", "PPO", "ES"}
    assert hidden["A2C"]["definition"] == "observed - hidden"
    assert hidden["PPO"]["mean"] == 1.0
    assert hidden["ES"]["kind"] == "gap"
    assert hidden["ES"]["absolute_mean_gap"] == 1.0
    assert result["phase_observability"][0]["definition"] == "visible - hidden"
    assert result["q_reset"][0]["definition"] == "reset - ongoing"
    response = result["response_reward"]
    assert len(response) == 2
    assert len({row["id"] for row in response}) == 2
    assert all(row["definition"] == "excluded - included" for row in response)


def test_rejects_evaluation_hash_corruption(tmp_path):
    trend = load_trend()
    record = leader_record(
        "q_reset", "battle_of_the_sexes", "A2C", "reset", 1
    )
    plan_path = write_plan(tmp_path / "test-sweep", [record])
    run_dir = write_attempt(
        trend, plan_path, record, 0, "completed", value=1.0
    )
    with (run_dir / "evaluation.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    assert_value_error(
        lambda: trend.build_summary(plan_path), "evaluation hash mismatch"
    )


def test_rejects_nonfinite_completed_evaluation(tmp_path):
    trend = load_trend()
    record = leader_record(
        "q_reset", "battle_of_the_sexes", "A2C", "reset", 1
    )
    plan_path = write_plan(tmp_path / "test-sweep", [record])
    write_attempt(
        trend, plan_path, record, 0, "completed", value=0.0, nonfinite=True
    )
    assert_value_error(lambda: trend.build_summary(plan_path), "non-finite value")


def test_atomic_report_writer_replaces_complete_json(tmp_path):
    trend = load_trend()
    path = tmp_path / "trend.json"
    path.write_text("old", encoding="utf-8")
    trend.write_json_atomic(path, {"mean": 1.0})
    assert json.loads(path.read_text(encoding="utf-8")) == {"mean": 1.0}
    assert list(tmp_path.glob("*.tmp")) == []
