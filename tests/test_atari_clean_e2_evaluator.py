from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from replication.atari import evaluate_atari_stackpomdp_leader_sb3 as evaluator
from replication.atari import train_atari_stackpomdp_leader_sb3 as trainer
from stackelberg_pomdp.atari.protocol import (
    CACHED_TRADE_REPLAY,
    GAMEPLAY,
    LEADER_QUERY,
)


def _episode_row(episode=0, *, checkpoint_hash="a" * 64, trace="b" * 64):
    events = []
    for event, game_step in enumerate((0, 1, 2, 3, 6)):
        events.append({
            "event_index": event,
            "game_step": game_step,
            "price": 0.2,
            "threshold": 0.5,
            "accepted": True,
            "seller_ammo_before": 1,
            "buyer_ammo_before": 0,
            "seller_ammo_after": 0,
            "buyer_ammo_after": 1,
        })
    return {
        "phase": "screen",
        "checkpoint_path": "/tmp/candidate.zip",
        "checkpoint_sha256": checkpoint_hash,
        "response_sha256": "c" * 64,
        "environment_config_sha256": "d" * 64,
        "evaluation_episode": episode,
        "evaluation_seed": 100 + episode,
        "evaluation_return": 1.0,
        "evaluation_steps": 17,
        "terminal_episode_summary": {"r": 1.0, "l": 17},
        "leader_role": "seller",
        "follower_role": "buyer",
        "event_steps": [0, 1, 2, 3, 6],
        "events": events,
        "gameplay_transitions": 7,
        "trade_transitions": 5,
        "reward_transition_count": 12,
        "query_transitions": 5,
        "outer_transition_count": 17,
        "cache_hits": 5,
        "bullets_arrived": 5,
        "purchases": 5,
        "payments": 1.0,
        "seller_game_reward": 0.0,
        "buyer_game_reward": 5.0,
        "seller_reward": 1.0,
        "buyer_reward": 4.0,
        "leader_reward": 1.0,
        "seller_shots_fired": 0,
        "buyer_shots_fired": 5,
        "seller_final_ammo": 0,
        "buyer_final_ammo": 0,
        "seller_emulator_step_calls": 7,
        "buyer_emulator_step_calls": 7,
        "seller_bullet_error": 0,
        "buyer_bullet_error": 0,
        "seller_payoff_error": 0.0,
        "buyer_payoff_error": 0.0,
        "query_actions": [[0.0, 0.2]] * 5,
        "query_trace_sha256": trace,
        "leader_commitment": [0.2] * 5,
        "follower_actions": [[1.0, 0.5]] * 5,
        "response_algorithm": "frozen_meta_policy",
    }


def _transition(
        episode,
        index,
        substep,
        *,
        reward=0.0,
        query_index=None,
        event_index=None,
        actor_hash="game",
        requested_action=(0.0, 0.2),
        done=False,
):
    canonical_event = query_index if substep == LEADER_QUERY else event_index
    actor_state = (
        evaluator.canonical_leader_state(canonical_event).tolist()
        if substep in (LEADER_QUERY, CACHED_TRADE_REPLAY)
        else [0.0] * 14
    )
    return {
        "phase": "screen",
        "checkpoint_path": "/tmp/candidate.zip",
        "checkpoint_sha256": "a" * 64,
        "response_sha256": "c" * 64,
        "environment_config_sha256": "d" * 64,
        "evaluation_episode": episode,
        "evaluation_seed": 100 + episode,
        "transition_index": index,
        "substep_type": substep,
        "reward": reward,
        "cumulative_return": 0.0,
        "done": done,
        "is_reward_phase": substep != LEADER_QUERY,
        "reward_generated": True,
        "emulator_advanced": substep == GAMEPLAY,
        "action_credit": {
            LEADER_QUERY: [0.0, 1.0],
            GAMEPLAY: [1.0, 0.0],
            CACHED_TRADE_REPLAY: [0.0, 0.0],
        }[substep],
        "actor_state": actor_state,
        "action_mask": [1.0] * 6,
        "actor_image_nonzero": 0,
        "actor_observation_sha256": actor_hash,
        "requested_action": list(requested_action),
        "query_index": query_index,
        "event_index": event_index,
        "cache_hit": substep == CACHED_TRADE_REPLAY,
        "leader_executed_action": (
            list(requested_action) if substep == CACHED_TRADE_REPLAY else None
        ),
        "follower_action": (
            [1.0, 0.5] if substep == CACHED_TRADE_REPLAY else None
        ),
        "game_step": None,
        "next_event": None,
    }


def _valid_evaluation(episodes=1):
    episode_rows = []
    transitions = []
    decisions = []
    schedule = (0, 1, 2, 3, 6)
    for episode in range(episodes):
        episode_rows.append(_episode_row(episode))
        index = 0
        for event in range(5):
            row = _transition(
                episode,
                index,
                LEADER_QUERY,
                query_index=event,
                actor_hash=f"query-{event}",
            )
            transitions.append(row)
            decisions.append(row)
            index += 1
        next_event = 0
        for game_step in range(7):
            if next_event < 5 and schedule[next_event] == game_step:
                row = _transition(
                    episode,
                    index,
                    CACHED_TRADE_REPLAY,
                    reward=0.2,
                    event_index=next_event,
                    actor_hash=f"query-{next_event}",
                )
                transitions.append(row)
                decisions.append(row)
                index += 1
                next_event += 1
            transitions.append(_transition(
                episode,
                index,
                GAMEPLAY,
                requested_action=(0.0, 0.5),
                done=(game_step == 6),
            ))
            index += 1
    return {
        "episode_rows": episode_rows,
        "transition_rows": transitions,
        "decision_rows": decisions,
    }


def _candidate(
        name,
        *,
        mean,
        median=None,
        minimum=None,
        std=0.0,
        timesteps=100,
        passed=True,
        digest=None,
):
    digest = digest or (name[0] * 64)
    return {
        "checkpoint_id": name,
        "checkpoint_path": f"/tmp/{name}.zip",
        "checkpoint_sha256": digest,
        "training_total_timesteps": timesteps,
        "summary": {
            "episodes": 20,
            "mean_leader_payoff": mean,
            "median_leader_payoff": mean if median is None else median,
            "min_leader_payoff": mean if minimum is None else minimum,
            "max_leader_payoff": mean,
            "std_leader_payoff": std,
        },
        "protocol": {"passed": passed, "violations": [] if passed else [{}]},
    }


def test_exact_210_analogue_protocol_audits_query_cache_trade_and_payoffs():
    evaluation = _valid_evaluation(episodes=2)
    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=2,
        leader_role="seller",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert protocol["passed"]
    assert protocol["violations"] == []
    assert protocol["single_query_trace"]
    assert protocol["single_leader_commitment"]
    assert protocol["leader_commitment"] == [0.2] * 5


def test_protocol_rejects_cache_mismatch_and_payoff_accounting_error():
    evaluation = _valid_evaluation()
    replay = next(
        row for row in evaluation["decision_rows"]
        if row["substep_type"] == CACHED_TRADE_REPLAY
    )
    replay["requested_action"] = [0.0, 0.7]
    evaluation["episode_rows"][0]["seller_reward"] = 1.1

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=1,
        leader_role="seller",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert not protocol["passed"]
    fields = {row["field"] for row in protocol["violations"]}
    assert "seller payoff identity" in fields
    assert "event 0 requested action cache identity" in fields


def test_protocol_binds_buyer_threshold_and_seller_response_price():
    evaluation = _valid_evaluation()
    episode = evaluation["episode_rows"][0]
    episode.update({
        "leader_role": "buyer",
        "follower_role": "seller",
        "leader_reward": 4.0,
        "evaluation_return": 4.0,
        "terminal_episode_summary": {"r": 4.0, "l": 17},
        "query_actions": [[0.0, 0.5]] * 5,
        "leader_commitment": [0.5] * 5,
        "follower_actions": [[1.0, 0.2]] * 5,
    })
    gameplay = [
        row for row in evaluation["transition_rows"]
        if row["substep_type"] == GAMEPLAY
    ]
    gameplay[-1]["reward"] = 5.0
    for row in evaluation["transition_rows"]:
        if row["substep_type"] in (LEADER_QUERY, CACHED_TRADE_REPLAY):
            row["requested_action"] = [0.0, 0.5]
        if row["substep_type"] == CACHED_TRADE_REPLAY:
            row["leader_executed_action"] = [0.0, 0.5]
            row["reward"] = -0.2

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=1,
        leader_role="buyer",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert protocol["passed"]


def test_protocol_rejects_multiple_deterministic_commitments_or_traces():
    evaluation = _valid_evaluation(episodes=2)
    evaluation["episode_rows"][1]["leader_commitment"][4] = 0.3
    evaluation["episode_rows"][1]["query_actions"][4][1] = 0.3
    evaluation["episode_rows"][1]["query_trace_sha256"] = "e" * 64

    protocol = evaluator.audit_e2_protocol(
        evaluation,
        required_episodes=2,
        leader_role="seller",
        gameplay_horizon=7,
        fixed_event_steps=(0, 1, 2, 3, 6),
    )

    assert not protocol["passed"]
    assert not protocol["single_query_trace"]
    assert not protocol["single_leader_commitment"]
    assert not protocol["single_full_query_action_trace"]


def test_selection_excludes_invalid_then_applies_documented_tiebreaks():
    invalid_high = _candidate("invalid", mean=100.0, passed=False, digest="f" * 64)
    unstable = _candidate(
        "unstable", mean=4.0, median=4.0, minimum=2.0, std=1.0,
        digest="e" * 64,
    )
    robust_late = _candidate(
        "robust_late", mean=4.0, median=4.0, minimum=3.0, std=0.5,
        timesteps=200, digest="d" * 64,
    )
    robust_early = _candidate(
        "robust_early", mean=4.0, median=4.0, minimum=3.0, std=0.5,
        timesteps=100, digest="c" * 64,
    )

    ranked = evaluator.rank_candidates([
        invalid_high, unstable, robust_late, robust_early
    ])

    assert ranked["selected_checkpoint_sha256"] == "c" * 64
    assert ranked["eligible_checkpoints"] == 3
    assert ranked["ranking_rows"][0]["checkpoint_id"] == "robust_early"
    assert ranked["ranking_rows"][-1]["eligible"] is False
    assert ranked["selection_rule"] == list(evaluator.SELECTION_RULE)


def test_common_screen_requires_identical_seed_schedule_pairs():
    left = {
        "response_checkpoint_sha256": "c" * 64,
        "environment_config_sha256": "d" * 64,
        "e2_provenance_fingerprint": "e" * 64,
        "seed_start": 100,
        "seed_end": 101,
        "episode_rows": [
            _episode_row(0),
            _episode_row(1),
        ]
    }
    right = deepcopy(left)
    assert evaluator.validate_common_screen([left, right])["passed"]

    right["episode_rows"][1]["event_steps"][-1] = 5
    with pytest.raises(RuntimeError, match="different event schedules"):
        evaluator.validate_common_screen([left, right])

    duplicate = deepcopy(left)
    duplicate["episode_rows"][1]["evaluation_seed"] = (
        duplicate["episode_rows"][0]["evaluation_seed"]
    )
    with pytest.raises(RuntimeError, match="duplicate evaluation seeds"):
        evaluator.validate_common_screen([duplicate])

    other_provenance = deepcopy(left)
    other_provenance["e2_provenance_fingerprint"] = "f" * 64
    with pytest.raises(RuntimeError, match="different E2 scientific provenance"):
        evaluator.validate_common_screen([left, other_provenance])


def test_selected_alias_is_exact_and_never_overwrites(tmp_path):
    source = tmp_path / "step200.zip"
    source.write_bytes(b"checkpoint bytes")
    target = tmp_path / "selected.zip"

    copied = evaluator.atomic_copy_no_overwrite(source, target)

    assert target.read_bytes() == source.read_bytes()
    assert copied["copy_verified"]
    assert copied["checkpoint_sha256"] == evaluator.checkpoint_sha256(source)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.atomic_copy_no_overwrite(source, target)


def test_model_loader_rejects_checkpoint_mutation_during_load(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.write_bytes(b"validated bytes")

    def mutate(path, *, device):
        del path, device
        checkpoint.write_bytes(b"changed during load")
        return object()

    monkeypatch.setattr(evaluator.ScaledLearningRatePPO, "load", mutate)
    with pytest.raises(RuntimeError, match="changed while"):
        evaluator._load_model(checkpoint, device="cpu")


def test_confirmation_requires_same_hash_trace_and_commitment():
    screen = {
        "checkpoint_sha256": "a" * 64,
        "e2_provenance_fingerprint": "c" * 64,
        "protocol": {
            "passed": True,
            "query_trace_sha256": "b" * 64,
            "leader_commitment": [0.2] * 5,
            "full_query_actions": [[0.0, 0.2]] * 5,
        },
    }
    confirmation = deepcopy(screen)
    assert evaluator.confirmation_matches_screen(screen, confirmation)["passed"]

    confirmation["protocol"]["leader_commitment"][0] = 0.3
    check = evaluator.confirmation_matches_screen(screen, confirmation)
    assert not check["passed"]
    assert not check["leader_commitment_matches_screen"]


def test_artifacts_retain_all_rows_and_refuse_collisions(tmp_path):
    evaluation = _valid_evaluation()
    result = {
        "checkpoint_id": "candidate",
        "checkpoint_path": "/tmp/candidate.zip",
        "checkpoint_sha256": "a" * 64,
        "training_total_timesteps": 100,
        "economic_role": "seller",
        "economic_input_mode": "event_only",
        "phase": "screen",
        "seed_start": 100,
        "seed_end": 100,
        "summary": evaluator.leader_outcome_summary(evaluation["episode_rows"]),
        "protocol": {"passed": True, "violations": []},
        **evaluation,
    }
    ranking = evaluator.rank_candidates([result])
    report = {
        "environment_config": {"leader_role": "seller"},
        "screen": {
            "seed_start": 100,
            "seed_end": 100,
            "checkpoint_results": [result],
        },
        "selection": ranking,
        "confirmation": None,
        "passed": False,
    }

    written = evaluator.write_selection_artifacts(
        report, output_dir=tmp_path, run_name="audit"
    )

    artifacts = written["artifacts"]
    assert Path(artifacts["report_json"]).is_file()
    assert Path(artifacts["screen_episodes_csv"]).is_file()
    assert Path(artifacts["screen_transitions_csv"]).is_file()
    assert Path(artifacts["screen_decisions_csv"]).is_file()
    assert Path(artifacts["screen_events_csv"]).is_file()
    assert len(written["screen"]["checkpoint_results"][0]["transition_rows"]) == 17
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.write_selection_artifacts(
            report, output_dir=tmp_path, run_name="audit"
        )


def test_parser_defaults_to_common_20_screen_and_disjoint_100_confirmation(
        tmp_path):
    args = evaluator.parse_args([
        "--leader-role", "seller",
        "--response-checkpoint", str(tmp_path / "buyer.zip"),
        "--checkpoint", str(tmp_path / "step100.zip"),
        "--selected-checkpoint", str(tmp_path / "selected.zip"),
    ])
    assert args.screen_episodes == 20
    assert args.confirmation_episodes == 100
    assert not evaluator._ranges_overlap(
        args.screen_seed_start,
        args.screen_episodes,
        args.confirmation_seed_start,
        args.confirmation_episodes,
    )

    with pytest.raises(SystemExit):
        evaluator.parse_args([
            "--leader-role", "seller",
            "--response-checkpoint", "buyer.zip",
            "--checkpoint", "step.zip",
            "--selected-checkpoint", "selected.zip",
            "--screen-seed-start", "100",
            "--screen-episodes", "20",
            "--confirmation-seed-start", "110",
            "--confirmation-episodes", "100",
        ])

    with pytest.raises(SystemExit):
        evaluator.parse_args([
            "--leader-role", "seller",
            "--response-checkpoint", "buyer.zip",
            "--checkpoint", "step.zip",
            "--selected-checkpoint", "selected.zip",
            "--gameplay-horizon", "199",
        ])


def test_run_selection_rechecks_canonical_counts_and_disjoint_seeds():
    base = SimpleNamespace(
        screen_episodes=20,
        screen_seed_start=100,
        confirmation_episodes=100,
        confirmation_seed_start=1_000,
        gameplay_horizon=200,
    )
    with pytest.raises(ValueError, match="exactly 20"):
        evaluator.run_selection(SimpleNamespace(**{
            **vars(base), "screen_episodes": 19,
        }))
    with pytest.raises(ValueError, match="exactly 100"):
        evaluator.run_selection(SimpleNamespace(**{
            **vars(base), "confirmation_episodes": 99,
        }))
    with pytest.raises(ValueError, match="must be disjoint"):
        evaluator.run_selection(SimpleNamespace(**{
            **vars(base), "confirmation_seed_start": 110,
        }))
    with pytest.raises(ValueError, match="canonical 200-step"):
        evaluator.run_selection(SimpleNamespace(**{
            **vars(base), "gameplay_horizon": 199,
        }))


def test_environment_hash_binds_role_response_contract_and_schedule(tmp_path):
    args = SimpleNamespace(
        leader_role="buyer",
        gameplay_horizon=200,
        event_tail_steps=0,
        fixed_event_steps=None,
        noop_max=30,
        max_frames=100_000,
        rom_path=None,
    )
    config = evaluator.environment_config(args)
    digest = evaluator.environment_config_sha256(config)
    changed = dict(config)
    changed["noop_max"] = 0

    assert config["follower_role"] == "seller"
    assert config["leader_action_cache"] is True
    assert Path(config["rom_path"]).is_file()
    assert len(config["rom_sha256"]) == 64
    assert len(digest) == 64
    assert digest != evaluator.environment_config_sha256(changed)


def test_candidate_provenance_requires_exact_response_and_evaluation_config():
    config = {
        "leader_role": "seller",
        "follower_role": "buyer",
        "gameplay_horizon": 200,
        "event_tail_steps": 0,
        "fixed_event_steps": None,
        "seller_game_reward_scale": 0.1,
        "buyer_game_reward_scale": 1.0,
        "noop_max": 30,
        "frame_skip": 4,
        "frame_stack": 4,
        "episodic_life": True,
        "clip_game_rewards": True,
        "max_frames": 100_000,
        "rom_sha256": "1" * 64,
    }
    response_hash = "a" * 64
    policy = SimpleNamespace(
        economic_role="seller",
        economic_input_mode="event_only",
        visual_features=512,
        state_features=64,
        economic_hidden=64,
        critic_hidden=256,
        pretrained_lr_scale=0.1,
        game_action_count=6,
    )
    leader_policy = {
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "economic_role": "seller",
        "economic_input_mode": "event_only",
        "visual_features": 512,
        "state_features": 64,
        "economic_hidden": 64,
        "critic_hidden": 256,
        "pretrained_lr_scale": 0.1,
        "game_action_count": 6,
    }
    scientific_config = {
        "leader_role": "seller",
        "follower_role": "buyer",
        "environment": {"seed": 1, **{
            key: config[key]
            for key in (
                "gameplay_horizon",
                "event_tail_steps",
                "fixed_event_steps",
                "seller_game_reward_scale",
                "buyer_game_reward_scale",
                "noop_max",
                "frame_skip",
                "frame_stack",
                "episodic_life",
                "clip_game_rewards",
                "max_frames",
                "rom_sha256",
            )
        }},
        "leader_policy": leader_policy,
        "implementation": trainer.e2_implementation_provenance(),
        "protocol": {
            "trade_events": 5,
            "query_transitions": 5,
            "cached_trade_replays": 5,
            "outer_episode_transitions": 210,
            "policy_action_cache": True,
            "leader_economic_input": "event_only",
            "response_economic_input": "full",
            "response_algorithm": "frozen_meta_policy",
        },
    }
    artifacts = {
        "frozen_response": {
            "sha256": response_hash,
            "policy": {
                "economic_role": "buyer",
                "economic_input_mode": "full",
            },
        },
    }
    identity = {
        "scientific_config": scientific_config,
        "artifacts": artifacts,
    }
    unsigned = {
        "schema": trainer.E2_PROVENANCE_SCHEMA,
        "version": trainer.E2_PROVENANCE_VERSION,
        "run_lineage_id": "0" * 32,
        "scientific_identity_sha256": trainer._canonical_sha256(identity),
        **identity,
    }
    manifest = {
        **unsigned,
        "fingerprint_sha256": trainer._canonical_sha256(unsigned),
    }
    model = SimpleNamespace(e2_provenance_manifest=manifest, policy=policy)

    validated = evaluator.validate_candidate_provenance(
        model, response_hash=response_hash, config=config
    )
    assert validated["fingerprint_sha256"] == manifest["fingerprint_sha256"]

    with pytest.raises(ValueError, match="different frozen E1 response"):
        evaluator.validate_candidate_provenance(
            model, response_hash="b" * 64, config=config
        )
    changed = {**config, "noop_max": 0}
    with pytest.raises(ValueError, match="environment provenance"):
        evaluator.validate_candidate_provenance(
            model, response_hash=response_hash, config=changed
        )

    policy.critic_hidden = 128
    with pytest.raises(ValueError, match="policy architecture"):
        evaluator.validate_candidate_provenance(
            model, response_hash=response_hash, config=config
        )


def test_run_selection_screens_every_candidate_before_copy_then_confirms(
        tmp_path, monkeypatch):
    first = tmp_path / "step100.zip"
    second = tmp_path / "step200.zip"
    response = tmp_path / "buyer_e1.zip"
    selected = tmp_path / "selected.zip"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    response.write_bytes(b"response")
    args = SimpleNamespace(
        leader_role="seller",
        response_checkpoint=str(response),
        checkpoint=[str(first), str(second)],
        selected_checkpoint=str(selected),
        screen_episodes=20,
        screen_seed_start=4_000_001,
        confirmation_episodes=100,
        confirmation_seed_start=5_000_001,
        gameplay_horizon=200,
        event_tail_steps=0,
        fixed_event_steps=None,
        noop_max=30,
        max_frames=100_000,
        rom_path=None,
        device="cpu",
    )

    class _Policy:
        economic_role = "buyer"
        economic_input_mode = "full"

    response_model = SimpleNamespace(
        policy=_Policy(), num_timesteps=500
    )
    monkeypatch.setattr(
        evaluator, "load_e1_response", lambda *a, **k: response_model
    )

    def fake_load(path, *, leader_role, device):
        del leader_role, device
        return SimpleNamespace(
            policy=SimpleNamespace(
                economic_role="seller", economic_input_mode="event_only"
            ),
            num_timesteps=100 if "100" in str(path) else 200,
        )

    monkeypatch.setattr(evaluator, "load_e2_checkpoint", fake_load)
    calls = []

    def fake_evaluate(
            model,
            checkpoint,
            *,
            args,
            response_model,
            response_hash,
            config,
            config_hash,
            episodes,
            seed_start,
            phase,
    ):
        del args, response_model, config
        path = Path(checkpoint)
        digest = evaluator.checkpoint_sha256(path)
        calls.append((phase, path.name, episodes, seed_start, selected.exists()))
        if phase == "screen":
            assert not selected.exists()
        else:
            assert path == selected
            assert selected.exists()
        score = 2.0 if path.read_bytes() == b"second" else 1.0
        rows = [
            {
                "evaluation_seed": seed_start + episode,
                "event_steps": [0, 20, 40, 60, 80],
            }
            for episode in range(episodes)
        ]
        return {
            "checkpoint_id": path.stem,
            "checkpoint_path": str(path),
            "checkpoint_sha256": digest,
            "response_checkpoint_sha256": response_hash,
            "environment_config_sha256": config_hash,
            "e2_provenance_fingerprint": "e" * 64,
            "training_total_timesteps": model.num_timesteps,
            "economic_role": "seller",
            "economic_input_mode": "event_only",
            "phase": phase,
            "seed_start": seed_start,
            "seed_end": seed_start + episodes - 1,
            "summary": {
                "episodes": episodes,
                "mean_leader_payoff": score,
                "median_leader_payoff": score,
                "std_leader_payoff": 0.0,
                "min_leader_payoff": score,
                "max_leader_payoff": score,
            },
            "protocol": {
                "passed": True,
                "violations": [],
                "query_trace_sha256": "f" * 64,
                "leader_commitment": [0.5] * 5,
                "full_query_actions": [[0.0, 0.5]] * 5,
            },
            "episode_rows": rows,
            "transition_rows": [],
            "decision_rows": [],
        }

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", fake_evaluate)
    report = evaluator.run_selection(args)

    assert [call[0] for call in calls] == ["screen", "screen", "confirmation"]
    assert all(not call[4] for call in calls[:2])
    assert calls[2][4]
    assert selected.read_bytes() == b"second"
    assert report["passed"]
    assert report["confirmation"]["disjoint_from_screen"]
    assert report["selection"]["selected_checkpoint_sha256"] == (
        evaluator.checkpoint_sha256(second)
    )
