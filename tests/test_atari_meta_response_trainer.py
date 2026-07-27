from pathlib import Path
from types import SimpleNamespace

import pytest

from replication.atari import train_atari_meta_response_sb3 as trainer


def _result(score, *, timesteps=250):
    return {
        "metadata": {"model_timesteps": int(timesteps)},
        "fixed_context_table": [],
        "random_context_summary": {
            "episodes": 2,
            "mean_controlled_reward": float(score),
        },
        "episodes": [],
    }


def _callback_args(tmp_path, game_checkpoint, *, resume=None):
    return SimpleNamespace(
        role="buyer",
        seed=7,
        resume=resume,
        output=tmp_path / "meta_buyer.evaluation.json",
        eval_seed=200_001,
        game_checkpoint=game_checkpoint,
        log_every=100,
        checkpoint_every=100,
        eval_every=250,
    )


class _FakePolicy:
    def __init__(self, fingerprint):
        self.gameplay_fingerprint = fingerprint


class _FakeModel:
    def __init__(self, fingerprint, env=None):
        self.policy = _FakePolicy(fingerprint)
        self.num_timesteps = 0
        self.saved = []
        self._env = env

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.saved.append(path)
        path.write_bytes(f"save-{len(self.saved)}".encode())

    def learn(self, **_kwargs):
        raise AssertionError("eval-only must not call learn")

    def get_env(self):
        return self._env


class _FakeVecEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_step_and_best_paths_preserve_evaluation_suffix(tmp_path):
    checkpoint = tmp_path / "response.zip"
    evaluation = tmp_path / "response.evaluation.json"

    assert trainer.step_checkpoint_path(checkpoint, 250).name == (
        "response_step250.zip"
    )
    assert trainer.step_evaluation_path(evaluation, 250).name == (
        "response_step250.evaluation.json"
    )
    assert trainer.best_checkpoint_path(checkpoint).name == "response_best.zip"
    assert trainer.best_evaluation_path(evaluation).name == (
        "response_best.evaluation.json"
    )


def test_periodic_archives_use_paired_seed_and_keep_deterministic_best(
        monkeypatch, tmp_path
):
    game_checkpoint = tmp_path / "e0.zip"
    game_checkpoint.write_bytes(b"frozen-e0")
    fingerprint = trainer.sha256_file(game_checkpoint)
    args = _callback_args(tmp_path, game_checkpoint)
    checkpoint = tmp_path / "meta_buyer.zip"
    callback = trainer.MetaResponseTrainingCallback(
        args,
        checkpoint,
        protected_state={},
        expected_fingerprint=fingerprint,
    )
    model = _FakeModel(fingerprint)
    callback.model = model
    callback.num_timesteps = model.num_timesteps = 250
    monkeypatch.setattr(trainer, "assert_gameplay_unchanged", lambda *_a, **_k: None)
    eval_seeds = []
    scores = iter((2.0, 2.0))

    def fake_evaluate(_model, _args, *, eval_seed):
        eval_seeds.append(eval_seed)
        return _result(next(scores), timesteps=_model.num_timesteps)

    monkeypatch.setattr(trainer, "evaluate_response", fake_evaluate)

    callback._periodic_evaluation()
    callback.num_timesteps = model.num_timesteps = 500
    callback._periodic_evaluation()

    assert eval_seeds == [args.eval_seed, args.eval_seed]
    assert (tmp_path / "meta_buyer_step250.zip").is_file()
    assert (tmp_path / "meta_buyer_step500.zip").is_file()
    assert (tmp_path / "meta_buyer_step250.evaluation.json").is_file()
    assert (tmp_path / "meta_buyer_step500.evaluation.json").is_file()
    assert (tmp_path / "meta_buyer_best.zip").is_file()
    manifest = trainer.json.loads(
        (tmp_path / "meta_buyer.selection.json").read_text()
    )
    assert manifest["best_score"] == 2.0
    assert manifest["best_timestep"] == 250
    assert manifest["controlled_role"] == "buyer"
    assert manifest["game_checkpoint_sha256"] == fingerprint
    assert manifest["selection_eval_seed"] == args.eval_seed
    assert manifest["selection_metric"] == trainer.SELECTION_METRIC
    assert set(manifest["source_files_sha256"]) == set(
        trainer.REPRODUCIBILITY_SOURCE_FILES
    )
    assert manifest["source_bundle_sha256"] == trainer.source_bundle_sha256(
        manifest["source_files_sha256"]
    )


def test_resume_restores_best_selection_and_rejects_unpaired_seed(
        monkeypatch, tmp_path
):
    game_checkpoint = tmp_path / "e0.zip"
    game_checkpoint.write_bytes(b"frozen-e0")
    fingerprint = trainer.sha256_file(game_checkpoint)
    checkpoint = tmp_path / "meta_buyer.zip"
    checkpoint.write_bytes(b"resume")
    best = tmp_path / "meta_buyer_best.zip"
    best.write_bytes(b"best")
    best_eval = tmp_path / "meta_buyer_best.evaluation.json"
    best_eval.write_text("{}")
    manifest = {
        "controlled_role": "buyer",
        "game_checkpoint_sha256": fingerprint,
        "selection_metric": trainer.SELECTION_METRIC,
        "selection_rule": trainer.SELECTION_RULE,
        "selection_eval_seed": 200_001,
        "best_score": 3.5,
        "best_timestep": 750,
        "best_checkpoint": str(best),
        "best_evaluation": str(best_eval),
    }
    trainer._write_json_atomic(
        trainer.selection_manifest_path(checkpoint), manifest
    )
    args = _callback_args(tmp_path, game_checkpoint, resume=checkpoint)
    callback = trainer.MetaResponseTrainingCallback(
        args, checkpoint, {}, fingerprint
    )
    callback._restore_selection()
    assert callback.best_score == 3.5
    assert callback.best_timestep == 750

    args.eval_seed += 1
    incompatible = trainer.MetaResponseTrainingCallback(
        args, checkpoint, {}, fingerprint
    )
    with pytest.raises(ValueError, match="selection metadata"):
        incompatible._restore_selection()


def test_eval_only_requires_resume(tmp_path):
    game_checkpoint = tmp_path / "e0.zip"
    game_checkpoint.write_bytes(b"e0")
    args = trainer.parse_args([
        "--eval-only",
        "--game-checkpoint",
        str(game_checkpoint),
        "--no-wandb",
    ])
    with pytest.raises(ValueError, match="requires --resume"):
        trainer.validate_args(args)


def test_eval_only_regenerates_bundle_without_learning_or_saving(
        monkeypatch, tmp_path
):
    game_checkpoint = tmp_path / "e0.zip"
    resume = tmp_path / "response.zip"
    output = tmp_path / "audit.evaluation.json"
    game_checkpoint.write_bytes(b"e0")
    resume.write_bytes(b"response")
    fingerprint = trainer.sha256_file(game_checkpoint)
    vec_env = _FakeVecEnv()
    model = _FakeModel(fingerprint, env=vec_env)
    model.num_timesteps = 12_345
    monkeypatch.setattr(trainer, "make_training_vec_env", lambda _args: vec_env)
    monkeypatch.setattr(
        trainer, "build_or_load_model", lambda _args, _env: model
    )
    monkeypatch.setattr(
        trainer, "protected_gameplay_state", lambda _policy: {}
    )
    monkeypatch.setattr(
        trainer, "assert_gameplay_unchanged", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        trainer,
        "evaluate_response",
        lambda _model, _args, *, eval_seed: _result(
            1.25, timesteps=_model.num_timesteps
        ),
    )

    trainer.main([
        "--role",
        "buyer",
        "--eval-only",
        "--resume",
        str(resume),
        "--game-checkpoint",
        str(game_checkpoint),
        "--output",
        str(output),
        "--no-wandb",
    ])

    assert output.is_file()
    payload = trainer.json.loads(output.read_text())
    assert payload["metadata"]["evaluation_kind"] == "eval_only"
    assert payload["metadata"]["evaluated_checkpoint"] == str(resume)
    assert payload["metadata"]["response_role"] == "buyer"
    assert payload["metadata"]["selection_eval_seed"] == 200_001
    assert set(payload["metadata"]["source_files_sha256"]) == set(
        trainer.REPRODUCIBILITY_SOURCE_FILES
    )
    assert payload["metadata"]["source_bundle_sha256"] == (
        trainer.source_bundle_sha256(
            payload["metadata"]["source_files_sha256"]
        )
    )
    assert model.saved == []
    assert vec_env.closed


def test_random_summary_reports_trade_timing_diagnostics():
    numeric = {
        field: 0.0 for field in trainer.EPISODE_NUMERIC_FIELDS
    }
    row = {
        "controlled_reward": 1.0,
        "gameplay_horizon": 100,
        **numeric,
        **{f"response_{index}": 0.5 for index in range(1, 6)},
        **{f"context_{index}": 0.4 for index in range(1, 6)},
        "events": [
            {
                "event_index": index,
                "game_step": step,
                "accepted": accepted,
            }
            for index, (step, accepted) in enumerate(
                ((10, True), (30, True), (50, False), (70, False), (90, False))
            )
        ],
    }

    summary = trainer.aggregate_rows([row])

    assert summary["mean_event_1_step"] == 10.0
    assert summary["event_1_acceptance_rate"] == 1.0
    assert summary["event_5_acceptance_rate"] == 0.0
    assert summary["mean_accepted_normalized_game_time"] == pytest.approx(0.2)
    assert summary["mean_rejected_normalized_game_time"] == pytest.approx(0.7)
    assert summary["early_acceptance_rate"] == 1.0
    assert summary["middle_acceptance_rate"] == 0.0
    assert summary["late_acceptance_rate"] == 0.0
