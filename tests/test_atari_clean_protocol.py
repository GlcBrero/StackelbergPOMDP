from collections import OrderedDict
from types import SimpleNamespace

import gym
import numpy as np
import torch

from stackelberg_pomdp.atari.protocol import (
    ACTION_CREDIT,
    ACTION_MASK,
    ACTOR_OBSERVATION_FIELDS,
    ACTOR_STATE,
    ACTOR_STATE_DIM,
    CACHED_TRADE_REPLAY,
    CRITIC_PREFIX,
    CRITIC_STATE,
    CRITIC_STATE_DIM,
    EVENT_SLICE,
    FOLLOWER_TRADE,
    GAMEPLAY,
    IMAGE,
    LEADER_QUERY,
    actor_state,
    canonical_leader_state,
    observation,
    observation_space,
    action_space,
    actor_observation,
    validate_action,
)
from stackelberg_pomdp.atari.query_trace import LeaderQuery, QueryTraceError
from stackelberg_pomdp.policies.atari.composite import (
    GatedCompositeAtariDistribution,
    StackPOMDPAtariPolicy,
)
from stackelberg_pomdp.wrappers.atari.preprocessing import (
    AmmoLedger,
    MaxAndSkipWrapper,
    ScarceAmmoWrapper,
)


def _spaces():
    image = gym.spaces.Box(
        0, 255, shape=(84, 84, 4), dtype=np.uint8
    )
    return observation_space(image, 6), action_space(6)


def test_published_checkpoint_policy_import_remains_compatible():
    from stackelberg_pomdp.atari.stackpomdp_policy import (
        StackPOMDPAtariPolicy as PublishedCheckpointPolicy,
    )

    assert PublishedCheckpointPolicy is StackPOMDPAtariPolicy


def _policy(*, event_only=False):
    observations, actions = _spaces()
    return StackPOMDPAtariPolicy(
        observations,
        actions,
        lambda _: 3.0e-4,
        economic_role="seller",
        economic_input_mode="event_only" if event_only else "full",
    )


def _observation(*, state, kind, image_value=0):
    return observation(
        image=np.full((84, 84, 4), image_value, dtype=np.uint8),
        state=state,
        action_mask=np.ones(6, dtype=np.float32),
        decision_kind=kind,
        critic_state=np.zeros(CRITIC_STATE_DIM, dtype=np.float32),
    )


def _batch(*observations):
    return OrderedDict(
        (
            key,
            torch.as_tensor(np.stack([item[key] for item in observations])),
        )
        for key in observations[0]
    )


def test_actor_fields_and_critic_prefix_form_one_canonical_contract():
    assert ACTOR_OBSERVATION_FIELDS == (IMAGE, ACTOR_STATE, ACTION_MASK)
    assert CRITIC_STATE.startswith(CRITIC_PREFIX)
    assert ACTION_CREDIT.startswith(CRITIC_PREFIX)

    state = canonical_leader_state(0)
    complete = _observation(state=state, kind=LEADER_QUERY)
    actor = actor_observation(complete)
    assert tuple(actor) == ACTOR_OBSERVATION_FIELDS
    assert not any(name.startswith(CRITIC_PREFIX) for name in actor)

    query = LeaderQuery.capture(0, complete, [0.0, 0.25])
    assert tuple(
        field.name for field in query.observation_fields
    ) == ACTOR_OBSERVATION_FIELDS
    complete["undeclared_actor_field"] = np.zeros(1, dtype=np.float32)
    with np.testing.assert_raises_regex(QueryTraceError, "canonical actor fields"):
        LeaderQuery.capture(0, complete, [0.0, 0.25])


def test_validate_action_clips_both_coordinates_and_checks_shape():
    np.testing.assert_array_equal(
        validate_action([-2.0, 1.5], game_action_count=6),
        np.array([0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        validate_action([9.0, -0.5], game_action_count=6),
        np.array([5.0, 0.0], dtype=np.float32),
    )
    with np.testing.assert_raises_regex(ValueError, "game_action, economic"):
        validate_action([1.0], game_action_count=6)
    with np.testing.assert_raises_regex(ValueError, "must be positive"):
        validate_action([1.0, 0.5], game_action_count=0)


def test_canonical_actor_state_is_exactly_four_plus_five_plus_five():
    values = actor_state(
        ammo_fraction=0.4,
        projectile_active=1.0,
        normalized_time=0.25,
        trade_mode=0.0,
        event_index=2,
        opponent_commitment=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    assert values.shape == (ACTOR_STATE_DIM,) == (14,)
    np.testing.assert_array_equal(values[EVENT_SLICE], [0, 0, 1, 0, 0])

    canonical = canonical_leader_state(2)
    np.testing.assert_array_equal(canonical[:4], [0, 0, 0, 1])
    np.testing.assert_array_equal(canonical[EVENT_SLICE], [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(canonical[9:], np.zeros(5))


def test_action_credit_separates_game_query_and_cached_replay_gradients():
    logits = torch.tensor(
        [[0.2, -0.1], [0.2, -0.1], [0.2, -0.1]],
        requires_grad=True,
    )
    alpha = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
    beta = torch.tensor([3.0, 3.0, 3.0], requires_grad=True)
    credit = torch.tensor([
        [1.0, 0.0],  # gameplay
        [0.0, 1.0],  # original query
        [0.0, 0.0],  # cached reward-trade replay
    ])
    distribution = GatedCompositeAtariDistribution(
        game_logits=logits,
        economic_alpha=alpha,
        economic_beta=beta,
        action_credit=credit,
    )
    actions = torch.tensor([[0.0, 0.4], [0.0, 0.4], [0.0, 0.4]])
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()
    assert log_prob[2].item() == 0.0
    assert entropy[2].item() == 0.0
    log_prob.sum().backward()
    assert torch.count_nonzero(logits.grad[0]) > 0
    assert torch.count_nonzero(logits.grad[1:]) == 0
    assert alpha.grad[0].item() == 0.0
    assert beta.grad[0].item() == 0.0
    assert alpha.grad[1].abs().item() > 0.0
    assert beta.grad[1].abs().item() > 0.0
    assert alpha.grad[2].item() == 0.0
    assert beta.grad[2].item() == 0.0


def test_event_only_leader_economic_output_ignores_every_other_input():
    torch.manual_seed(11)
    policy = _policy(event_only=True)
    first_state = actor_state(
        ammo_fraction=0.0,
        projectile_active=0.0,
        normalized_time=0.0,
        trade_mode=1.0,
        event_index=3,
        opponent_commitment=np.zeros(5),
    )
    second_state = actor_state(
        ammo_fraction=1.0,
        projectile_active=1.0,
        normalized_time=0.91,
        trade_mode=0.0,
        event_index=3,
        opponent_commitment=np.linspace(0.1, 0.9, 5),
    )
    batch = _batch(
        _observation(state=first_state, kind=LEADER_QUERY, image_value=0),
        _observation(state=second_state, kind=LEADER_QUERY, image_value=255),
    )
    with torch.no_grad():
        distribution = policy._distribution(batch)
    torch.testing.assert_close(
        distribution.economic.mean[0],
        distribution.economic.mean[1],
        rtol=0,
        atol=0,
    )


def test_cache_reuses_full_query_action_but_replay_has_zero_log_probability():
    torch.manual_seed(17)
    policy = _policy(event_only=True)
    policy.fix_policy_actions()
    state = canonical_leader_state(1)
    query = _observation(state=state, kind=LEADER_QUERY)
    replay = _observation(state=state, kind=CACHED_TRADE_REPLAY)
    replay[CRITIC_STATE].fill(7.0)
    query_batch = _batch(query)
    replay_batch = _batch(replay)

    query_action, _, query_log_prob = policy.forward(
        query_batch, deterministic=False
    )
    replay_action, _, replay_log_prob = policy.forward(
        replay_batch, deterministic=False
    )
    torch.testing.assert_close(replay_action, query_action, rtol=0, atol=0)
    assert len(policy.obs_action_map) == 1
    assert query_log_prob.abs().item() > 0.0
    assert replay_log_prob.item() == 0.0


def test_cache_covers_gameplay_and_clears_only_completed_vector_rows():
    torch.manual_seed(23)
    policy = _policy()
    policy.fix_policy_actions()
    state = actor_state(
        ammo_fraction=1.0,
        projectile_active=0.0,
        normalized_time=0.2,
        trade_mode=0.0,
        event_index=0,
        opponent_commitment=np.zeros(5),
    )
    item = _observation(state=state, kind=GAMEPLAY)
    batch = _batch(item, item)
    first, _, _ = policy.forward(batch, deterministic=False)
    second, _, _ = policy.forward(batch, deterministic=False)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert {key[0] for key in policy.obs_action_map} == {0, 1}
    policy.clear_obs_action_map(rows=[1])
    assert {key[0] for key in policy.obs_action_map} == {0}


class _FakeALE:
    def __init__(self):
        self.ram = np.full(256, 0xF6, dtype=np.uint8)

    def getRAM(self):
        return self.ram


class _RawFireEnv(gym.Env):
    def __init__(self, *, fire_on_attempt=None):
        super().__init__()
        self.ale = _FakeALE()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(2, 2, 1), dtype=np.uint8
        )
        self.actions = []
        self.fire_on_attempt = fire_on_attempt
        self.fire_attempts = 0

    def get_action_meanings(self):
        return ["NOOP", "FIRE", "RIGHT", "RIGHTFIRE"]

    def reset(self):
        self.actions = []
        self.fire_attempts = 0
        self.ale.ram[...] = 0xF6
        return np.zeros((2, 2, 1), dtype=np.uint8)

    def step(self, action):
        self.actions.append(int(action))
        if int(action) in (1, 3):
            self.fire_attempts += 1
            if self.fire_attempts == self.fire_on_attempt:
                self.ale.ram[
                    ScarceAmmoWrapper.PROJECTILE_RAM_SLOTS[0]
                ] = ScarceAmmoWrapper.NEW_PROJECTILE_RAM_VALUE
        return (
            np.zeros((2, 2, 1), dtype=np.uint8),
            0.0,
            False,
            {"ale.lives": 3},
        )


def test_max_skip_retries_fire_until_projectile_then_removes_fire():
    raw = _RawFireEnv(fire_on_attempt=2)
    scarce = ScarceAmmoWrapper(
        raw, ledger=AmmoLedger(initial_ammo=5, capacity=5)
    )
    env = MaxAndSkipWrapper(scarce, skip=4)
    env.reset()
    _, _, _, info = env.step(3)  # RIGHTFIRE
    assert raw.actions == [3, 3, 2, 2]
    assert info["shots_fired_this_step"] == 1
    assert scarce.ledger.value == 4


def test_max_skip_retries_unregistered_fire_through_decision_window():
    raw = _RawFireEnv()
    scarce = ScarceAmmoWrapper(
        raw, ledger=AmmoLedger(initial_ammo=5, capacity=5)
    )
    env = MaxAndSkipWrapper(scarce, skip=4)
    env.reset()
    _, _, _, info = env.step(3)
    assert raw.actions == [3, 3, 3, 3]
    assert info["shots_fired_this_step"] == 0
    assert scarce.ledger.value == 5


def test_active_projectile_blocks_fire_before_the_raw_step():
    raw = _RawFireEnv()
    scarce = ScarceAmmoWrapper(
        raw, ledger=AmmoLedger(initial_ammo=5, capacity=5)
    )
    scarce.reset()
    raw.ale.ram[ScarceAmmoWrapper.PROJECTILE_RAM_SLOTS[0]] = 0x55
    _, _, _, info = scarce.step(1)
    assert raw.actions == [0]
    assert info["blocked_fire_this_step"]
