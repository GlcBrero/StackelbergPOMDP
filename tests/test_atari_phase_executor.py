"""Atari uses the common executor, including internal reward trade replays."""
import numpy as np
import pytest
from experiments.atari._support.protocol import ACTOR_STATE, ACTION_CREDIT, CACHED_TRADE_REPLAY, LEADER_QUERY
from experiments.atari._support.wrappers.meta_follower import make_stackpomdp_atari_leader_env
from stackelberg_pomdp.stackelberg import StackelbergPOMDPEnv
from atari_fixtures import _FakeSide, _FrozenFollower, _bilateral_config


def make_env():
    from types import SimpleNamespace
    class Side(_FakeSide):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ammo_wrapper = SimpleNamespace(fire_action_indices=(1, 4, 5))
    return make_stackpomdp_atari_leader_env(leader_role="seller", config=_bilateral_config(),
        response_model_factory=lambda path, device: _FrozenFollower("buyer"), side_factory=Side)


def test_atari_complete_episode_uses_only_fresh_decisions():
    env = make_env()
    assert isinstance(env, StackelbergPOMDPEnv)
    from stackelberg_pomdp import POMG
    assert isinstance(env.game, POMG)
    env.game.capture_transitions = True
    observation = env.reset()
    done, total, draws = False, 0., 0
    while not done:
        if draws < 5:
            assert env.game.core.seller.step_calls == 0
            np.testing.assert_array_equal(observation[ACTION_CREDIT], [0, 1])
            action = np.asarray([0, .1 * (draws + 1)], dtype=np.float32)
        else:
            np.testing.assert_array_equal(observation[ACTION_CREDIT], [1, 0])
            action = np.asarray([0, 0], dtype=np.float32)
        observation, reward, done, info = env.step(action)
        draws += 1
        total += reward
    assert draws == 5 + 7
    assert info["cache_hits"] == 5 and info["commitment_entries"] == 5
    assert info["response_policy_queries"] == 5 and info["reward_policy_queries"] == 7
    assert info["outer_transition_count"] == 5 + 7 + 5
    assert total == pytest.approx(info["leader_reward"])
    records = env.game.transition_records
    assert len(records) == 17
    queries = [r for r in records if r["substep_type"] == LEADER_QUERY]
    replays = [r for r in records if r["substep_type"] == CACHED_TRADE_REPLAY]
    for query, replay in zip(queries, replays):
        assert query["actor_observation_sha256"] == replay["actor_observation_sha256"]
        assert query["requested_action"] == replay["requested_action"]
    env.close()


def test_atari_collector_and_phase_balanced_update():
    from experiments.atari._support.policy import StackPOMDPAtariPolicy
    from experiments.atari._support.training import PhaseBalancedPPO
    env = make_env()
    model = PhaseBalancedPPO(StackPOMDPAtariPolicy, env, n_steps=12, batch_size=12,
        n_epochs=1, gamma=1., gae_lambda=1., seed=9,
        policy_kwargs=dict(economic_role="seller", economic_input_mode="event_only"))
    model.learn(12)
    assert model.last_rollout_completed_episodes == 1
    assert model.rollout_buffer.buffer_size == 12
    np.testing.assert_array_equal(model.rollout_buffer.actor_credit, 1)
    assert not hasattr(model.policy, "obs_action_map")
    env.close()


def test_evaluator_retains_cached_transitions_for_protocol_audit():
    from experiments.atari._support.evaluation.rollouts import evaluate_e2_model
    from experiments.atari._support.evaluation.protocol_audit import audit_e2_protocol
    class Model:
        def predict(self, observation, deterministic=True):
            return np.asarray([0., .3], dtype=np.float32), None
    evaluation = evaluate_e2_model(Model(), lambda episode: make_env(), episodes=1, seed_start=11,
        checkpoint_path="fixture", checkpoint_hash="test", response_hash="test", config_hash="test",
        provenance_fingerprint="test", phase="test",
        economic_intervention=dict(intervention_id="fixed", manifest_sha256="0" * 64,
                                   economic_commitment=[.1, .2, .3, .4, .5]))
    audit = audit_e2_protocol(evaluation, required_episodes=1, leader_role="seller", gameplay_horizon=7,
                             fixed_event_steps=(0, 1, 2, 3, 6))
    assert audit["passed"], audit
