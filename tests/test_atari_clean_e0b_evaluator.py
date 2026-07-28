from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from replication.atari import evaluate_atari_e0b_sb3 as evaluator


def _episode_row(
        episode,
        *,
        seed=None,
        event_steps=(0, 20, 40, 60, 80),
        reward=5.0,
        shots=5,
        checkpoint_id="policy",
        checkpoint_path="/tmp/policy.zip",
        scenario="random",
):
    final_ammo = 5 - int(shots)
    return {
        "evaluation_episode": int(episode),
        "evaluation_seed": int(seed if seed is not None else 100 + episode),
        "evaluation_return": float(reward),
        "evaluation_steps": 205,
        "stage": "e0b",
        "gameplay_steps": 200,
        "trade_transitions": 5,
        "outer_transition_count": 205,
        "event_steps": list(event_steps),
        "fifth_event_step": int(event_steps[-1]),
        "fifth_event_fraction": float(event_steps[-1]) / 200.0,
        "game_reward": float(reward),
        "payments": 0.0,
        "episode_reward": float(reward),
        "shots_fired": int(shots),
        "free_transfers": 5,
        "final_ammo": final_ammo,
        "bullet_accounting_error": 0,
        "emulator_step_calls": 200,
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": checkpoint_path,
        "scenario": scenario,
    }


def _checkpoint_result(path, rows):
    path = str(path)
    return {
        "checkpoint_id": Path(path).stem,
        "checkpoint_path": path,
        "scenarios": [{
            "scenario": rows[0]["scenario"],
            "episode_rows": rows,
        }],
    }


def test_protocol_audit_and_timing_strata_retain_late_horizon_censoring():
    rows = [
        _episode_row(0, event_steps=(0, 5, 10, 20, 30), reward=5, shots=5),
        _episode_row(1, event_steps=(0, 20, 40, 80, 100), reward=4, shots=4),
        _episode_row(2, event_steps=(10, 40, 80, 120, 180), reward=3, shots=3),
    ]
    protocol = evaluator.audit_e0b_protocol(
        rows, required_episodes=3, gameplay_horizon=200
    )
    assert protocol["passed"]
    assert protocol["violations"] == []
    assert protocol["max_abs_bullet_accounting_error"] == 0.0
    assert protocol["max_abs_payment"] == 0.0
    assert protocol["max_abs_return_error"] == 0.0

    timing = evaluator.timing_strata(rows, gameplay_horizon=200)
    assert [
        row["fifth_event_step"]
        for row in timing["by_fifth_event_step"]
    ] == [30, 100, 180]
    bands = {
        row["timing_band"]: row
        for row in timing["by_timing_band"]
    }
    assert bands["early"]["episodes"] == 1
    assert bands["middle"]["episodes"] == 1
    assert bands["late"]["episodes"] == 1
    assert bands["early"]["mean_game_reward"] == 5.0
    assert bands["middle"]["mean_shots_fired"] == 4.0
    assert bands["late"]["mean_final_ammo"] == 2.0


def test_protocol_audit_reports_transfer_payment_and_schedule_failures():
    row = _episode_row(0)
    row.update({
        "free_transfers": 4,
        "payments": 0.2,
        "event_steps": [0, 20, 20, 60, 80],
        "evaluation_return": 4.5,
    })
    protocol = evaluator.audit_e0b_protocol(
        [row], required_episodes=1, gameplay_horizon=200
    )
    assert not protocol["passed"]
    fields = {violation["field"] for violation in protocol["violations"]}
    assert {
        "free_transfers",
        "payments",
        "event_steps",
        "evaluation_return == game_reward",
    } <= fields


def test_paired_comparison_requires_identical_seed_schedule_pairs():
    left_rows = [
        _episode_row(
            0,
            seed=700,
            reward=4,
            shots=4,
            checkpoint_id="a",
            checkpoint_path="/tmp/a.zip",
        ),
        _episode_row(
            1,
            seed=701,
            reward=5,
            shots=5,
            checkpoint_id="a",
            checkpoint_path="/tmp/a.zip",
        ),
    ]
    right_rows = deepcopy(left_rows)
    for row in right_rows:
        row["checkpoint_id"] = "b"
        row["checkpoint_path"] = "/tmp/b.zip"
    right_rows[0].update({
        "game_reward": 5.0,
        "evaluation_return": 5.0,
        "episode_reward": 5.0,
        "shots_fired": 5,
        "final_ammo": 0,
    })
    comparison = evaluator.paired_comparison(
        _checkpoint_result("/tmp/a.zip", left_rows),
        _checkpoint_result("/tmp/b.zip", right_rows),
    )
    assert comparison["same_seed_and_schedule_pairs"]
    assert comparison["summary"]["paired_episodes"] == 2
    assert comparison["summary"]["mean_delta_game_reward_b_minus_a"] == 0.5
    assert comparison["summary"]["checkpoint_b_reward_win_rate"] == 0.5

    right_rows[0]["event_steps"][-1] = 81
    with pytest.raises(RuntimeError, match="different event schedules"):
        evaluator.paired_comparison(
            _checkpoint_result("/tmp/a.zip", left_rows),
            _checkpoint_result("/tmp/b.zip", right_rows),
        )


def test_run_evaluations_pairs_random_and_multiple_fixed_schedules(
        tmp_path, monkeypatch
):
    checkpoints = [tmp_path / "a.zip", tmp_path / "b.zip"]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"placeholder")
    args = SimpleNamespace(
        checkpoint=[str(path) for path in checkpoints],
        episodes=3,
        seed_start=900,
        gameplay_horizon=200,
        event_tail_steps=0,
        fixed_event_steps=((0, 20, 40, 60, 80), (10, 50, 90, 130, 170)),
        noop_max=0,
        max_frames=100_000,
        rom_path=None,
        device="cpu",
    )

    class _Model:
        def __init__(self, path):
            self.path = Path(path)
            self.num_timesteps = 123
            self.policy = SimpleNamespace(
                economic_role="gameplay",
                economic_input_mode="full",
                pretrained_lr_scale=0.1,
            )

    monkeypatch.setattr(
        evaluator,
        "load_clean_checkpoint",
        lambda path, *, device: _Model(path),
    )
    monkeypatch.setattr(
        evaluator,
        "make_e0b_env",
        lambda args, *, seed, fixed_event_steps=None: SimpleNamespace(
            seed=seed,
            fixed_event_steps=fixed_event_steps,
        ),
    )
    calls = []

    def fake_evaluate_model(model, env_factory, *, episodes):
        environments = [env_factory(episode) for episode in range(episodes)]
        calls.append((model.path.stem, [
            (env.seed, env.fixed_event_steps) for env in environments
        ]))
        rows = []
        for episode, env in enumerate(environments):
            schedule = env.fixed_event_steps or (0, 20, 40, 60, 80 + episode)
            rows.append(_episode_row(
                episode,
                seed=env.seed,
                event_steps=schedule,
                checkpoint_id=model.path.stem,
                checkpoint_path=str(model.path),
            ))
            rows[-1].pop("evaluation_seed")
            rows[-1].pop("checkpoint_id")
            rows[-1].pop("checkpoint_path")
            rows[-1].pop("scenario")
            rows[-1].pop("fifth_event_step")
            rows[-1].pop("fifth_event_fraction")
        return {"summary": {}, "episode_rows": rows}

    monkeypatch.setattr(evaluator, "evaluate_model", fake_evaluate_model)
    report = evaluator.run_evaluations(args)

    assert report["all_protocol_checks_passed"]
    assert len(calls) == 6
    first_policy_calls = [values for policy, values in calls if policy == "a"]
    second_policy_calls = [values for policy, values in calls if policy == "b"]
    assert first_policy_calls == second_policy_calls
    assert all(
        [seed for seed, _ in values] == [900, 901, 902]
        for values in first_policy_calls
    )
    assert report["paired_comparison"]["same_seed_and_schedule_pairs"]
    assert report["paired_comparison"]["summary"]["paired_episodes"] == 9
    assert [
        result["checkpoint_sha256"]
        for result in report["checkpoint_results"]
    ] == [
        evaluator.checkpoint_sha256(path) for path in checkpoints
    ]


def test_artifacts_use_dedicated_names_and_refuse_overwrite(tmp_path):
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"placeholder")
    row = _episode_row(
        0,
        checkpoint_id="checkpoint",
        checkpoint_path=str(checkpoint),
    )
    scenario = {
        "scenario": "random",
        "fixed_event_steps": None,
        "summary": evaluator.outcome_summary([row]),
        "protocol": evaluator.audit_e0b_protocol(
            [row], required_episodes=1, gameplay_horizon=200
        ),
        "timing": evaluator.timing_strata([row], gameplay_horizon=200),
        "episode_rows": [row],
    }
    report = {
        "schema_version": 1,
        "evaluator": "test",
        "config": {"seed_start": 100, "seed_end": 100},
        "all_protocol_checks_passed": True,
        "checkpoint_results": [{
            "checkpoint_id": "checkpoint",
            "checkpoint_path": str(checkpoint),
            "scenarios": [scenario],
        }],
        "paired_comparison": None,
    }
    output_dir = tmp_path / "audits"
    with pytest.raises(ValueError, match="training evaluation filename"):
        evaluator.write_evaluation_artifacts(
            report,
            output_dir=checkpoint.parent,
            run_name="checkpoint.evaluation",
        )
    written = evaluator.write_evaluation_artifacts(
        report, output_dir=output_dir, run_name="explicit_audit"
    )
    assert Path(written["artifacts"]["report_json"]).is_file()
    assert Path(written["artifacts"]["episode_rows_csv"]).is_file()
    assert Path(written["artifacts"]["timing_rows_csv"]).is_file()
    assert not checkpoint.with_name("checkpoint.evaluation.json").exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.write_evaluation_artifacts(
            report, output_dir=output_dir, run_name="explicit_audit"
        )


def test_parser_accepts_repeated_fixed_schedules_and_rejects_duplicates():
    args = evaluator.parse_args([
        "--checkpoint", "a.zip",
        "--fixed-event-steps", "0,20,40,60,80",
        "--fixed-event-steps", "10,50,90,130,170",
    ])
    assert args.fixed_event_steps == (
        (0, 20, 40, 60, 80),
        (10, 50, 90, 130, 170),
    )
    with pytest.raises(SystemExit):
        evaluator.parse_args([
            "--checkpoint", "a.zip",
            "--fixed-event-steps", "0,20,40,60,80",
            "--fixed-event-steps", "0,20,40,60,80",
        ])
