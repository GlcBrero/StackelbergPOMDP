"""Full five-query plus bilateral-game Atari Stackelberg-POMDP environment."""

from collections import OrderedDict
from pathlib import Path

import gym
import numpy as np

from stackelberg_pomdp.atari.protocol import (
    CACHED_TRADE_REPLAY,
    CRITIC_STATE_DIM,
    FOLLOWER_TRADE,
    GAMEPLAY,
    LEADER_QUERY,
    NUM_TRADE_EVENTS,
    TERMINAL,
    action_space,
    actor_observation,
    actor_state,
    assert_actor_observations_equal,
    canonical_leader_state,
    observation,
    observation_space,
)
from stackelberg_pomdp.atari.query_trace import LeaderQueryTrace
from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    ROLES,
    SELLER,
    BilateralAtariConfig,
    DualAtariTradeCore,
)


class FullTraceStackPOMDPAtariLeaderEnv(gym.Env):
    """Expose every E2 query, gameplay, and trade transition to PPO.

    Query ``j`` and reward-game trade ``j`` have identical actor-visible
    observations.  They differ only in the critic-prefixed action-credit mask:
    the query is the economic decision, while the cached trade is execution of
    that prior decision and therefore contributes no second actor term.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            *,
            leader_role,
            response_checkpoint=None,
            config=None,
            core_factory=DualAtariTradeCore,
            response_model_factory=None,
            side_factory=None,
            env_factory=None,
            device="cpu",
    ):
        super().__init__()
        if leader_role not in ROLES:
            raise ValueError(f"leader_role must be one of {sorted(ROLES)}")
        self.leader_role = leader_role
        self.follower_role = BUYER if leader_role == SELLER else SELLER
        self.config = (config or BilateralAtariConfig()).resolved()
        self.rng = np.random.default_rng(self.config.seed + 193_939)
        core_kwargs = {}
        if side_factory is not None:
            core_kwargs["side_factory"] = side_factory
        if env_factory is not None:
            core_kwargs["env_factory"] = env_factory
        self.core = core_factory(self.config, **core_kwargs)
        self.action_space = action_space(self.core.game_action_count)
        self.observation_space = observation_space(
            self.core.image_space, self.core.game_action_count
        )
        self.dummy_image = np.zeros(
            self.core.image_space.shape, dtype=self.core.image_space.dtype
        )
        self.canonical_action_mask = np.ones(
            self.core.game_action_count, dtype=np.float32
        )
        fire_indices = tuple(
            int(index)
            for index in self.core.side(self.leader_role)
            .ammo_wrapper.fire_action_indices
        )
        self.canonical_action_mask[list(fire_indices)] = 0.0

        if response_model_factory is None:
            if response_checkpoint is None:
                raise ValueError("response_checkpoint is required")
            from stable_baselines3 import PPO

            path = Path(response_checkpoint).expanduser()
            if not path.is_file() and Path(f"{path}.zip").is_file():
                path = Path(f"{path}.zip")
            if not path.is_file():
                raise FileNotFoundError(
                    f"meta-response checkpoint does not exist: {path}"
                )
            self.response_model = PPO.load(str(path), device=device)
        else:
            self.response_model = response_model_factory(
                response_checkpoint, device=device
            )
        response_role = getattr(
            getattr(self.response_model, "policy", None),
            "economic_role",
            None,
        )
        if response_role != self.follower_role:
            raise ValueError(
                f"{leader_role} leader requires a {self.follower_role} "
                f"meta-response, got {response_role!r}"
            )
        policy = getattr(self.response_model, "policy", None)
        if policy is not None:
            policy.set_training_mode(False)
            for parameter in policy.parameters():
                parameter.requires_grad = False

        self.query_index = 0
        self._query_observations = []
        self._query_actions = []
        self.query_trace = None
        self.follower_full_actions = np.zeros(
            (NUM_TRADE_EVENTS, 2), dtype=np.float32
        )
        self._reward_started = False
        self._done = False
        self._last_observation = None
        self.query_transitions = 0
        self.gameplay_transitions = 0
        self.trade_transitions = 0
        self.cache_hits = 0
        self._canonical_actor_observations = tuple(
            actor_observation(self._canonical_observation(index, LEADER_QUERY))
            for index in range(NUM_TRADE_EVENTS)
        )

    def seed(self, seed=None):
        seed = self.config.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(seed + 193_939)
        return [seed]

    @property
    def leader_query_trace(self):
        return self.query_trace

    def _critic_state(self, *, phase, event_index=0):
        values = np.zeros(CRITIC_STATE_DIM, dtype=np.float32)
        values[:8] = (
            float(phase),
            float(self.query_index) / NUM_TRADE_EVENTS,
            float(event_index) / (NUM_TRADE_EVENTS - 1),
            float(self.query_trace is not None),
            float(self.query_transitions) / NUM_TRADE_EVENTS,
            float(self.gameplay_transitions) / self.config.gameplay_horizon,
            float(self.trade_transitions) / NUM_TRADE_EVENTS,
            float(self._reward_started),
        )
        for index, action in enumerate(self._query_actions):
            start = 8 + 2 * index
            values[start:start + 2] = action
        if not self._reward_started:
            return values
        core = self.core
        values[18:27] = (
            float(core.game_step) / self.config.gameplay_horizon,
            float(core.next_event) / NUM_TRADE_EVENTS,
            float(core.at_event),
            float(core.seller.ammo) / NUM_TRADE_EVENTS,
            float(core.buyer.ammo) / NUM_TRADE_EVENTS,
            float(core.transfers) / NUM_TRADE_EVENTS,
            float(core.payments),
            float(core.seller_payoff),
            float(core.buyer_payoff),
        )
        values[27:32] = (
            np.asarray(core.event_steps, dtype=np.float32)
            / self.config.gameplay_horizon
        )
        return values

    def _canonical_observation(self, event_index, decision_kind):
        return observation(
            image=self.dummy_image,
            state=canonical_leader_state(event_index),
            action_mask=self.canonical_action_mask,
            decision_kind=decision_kind,
            critic_state=self._critic_state(
                phase=float(self._reward_started), event_index=event_index
            ),
        )

    def _query_observation(self, event_index):
        return self._canonical_observation(event_index, LEADER_QUERY)

    def _trade_observation(self, event_index):
        result = self._canonical_observation(
            event_index, CACHED_TRADE_REPLAY
        )
        assert_actor_observations_equal(
            result, self._canonical_actor_observations[event_index]
        )
        return result

    def canonical_reward_observation(self, event_index):
        return actor_observation(self._trade_observation(event_index))

    def _live_observation(self, role, *, follower, decision_kind):
        side = self.core.side(role)
        trade_mode = self.core.at_event and not self._done
        event_index = (
            self.core.next_event
            if self.core.next_event < NUM_TRADE_EVENTS
            else None
        )
        commitment = (
            self.query_trace.economic_commitment
            if follower and self.query_trace is not None
            else np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        )
        return observation(
            image=self.dummy_image if trade_mode else side.image,
            state=actor_state(
                ammo_fraction=float(side.ammo) / NUM_TRADE_EVENTS,
                projectile_active=float(side.projectile_active),
                normalized_time=float(self.core.game_step)
                / self.config.gameplay_horizon,
                trade_mode=float(trade_mode),
                event_index=event_index,
                opponent_commitment=commitment,
            ),
            action_mask=side.action_mask,
            decision_kind=decision_kind,
            critic_state=self._critic_state(
                phase=1.0,
                event_index=min(self.core.next_event, NUM_TRADE_EVENTS - 1),
            ),
        )

    def _leader_gameplay_observation(self):
        kind = TERMINAL if self._done else GAMEPLAY
        return self._live_observation(
            self.leader_role, follower=False, decision_kind=kind
        )

    def _follower_observation(self, *, trade):
        return self._live_observation(
            self.follower_role,
            follower=True,
            decision_kind=FOLLOWER_TRADE if trade else GAMEPLAY,
        )

    def _next_observation(self):
        if self._done:
            return self._leader_gameplay_observation()
        if not self._reward_started:
            return self._query_observation(self.query_index)
        if self.core.at_event:
            return self._trade_observation(self.core.next_event)
        return self._leader_gameplay_observation()

    @staticmethod
    def _copy(values):
        return OrderedDict(
            (key, np.array(value, copy=True)) for key, value in values.items()
        )

    def _emit(self, values):
        self._last_observation = self._copy(values)
        return self._copy(values)

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.seed(seed)
        self.query_index = 0
        self._query_observations = []
        self._query_actions = []
        self.query_trace = None
        self.follower_full_actions.fill(0.0)
        self._reward_started = False
        self._done = False
        self.query_transitions = 0
        self.gameplay_transitions = 0
        self.trade_transitions = 0
        self.cache_hits = 0
        return self._emit(self._query_observation(0))

    def _validated_action(self, action):
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (2,):
            raise ValueError("leader action must be [game_action, economic]")
        return np.array([
            np.clip(values[0], 0.0, self.core.game_action_count - 1),
            np.clip(values[1], 0.0, 1.0),
        ], dtype=np.float32)

    def _query_step(self, action):
        completed = self.query_index
        self._query_observations.append(actor_observation(self._last_observation))
        self._query_actions.append(np.array(action, copy=True))
        self.query_index += 1
        self.query_transitions += 1
        if self.query_index == NUM_TRADE_EVENTS:
            self.query_trace = LeaderQueryTrace.capture(
                self._query_observations, self._query_actions
            )
            self.core.reset(seed=int(self.rng.integers(0, 2 ** 31 - 1)))
            self._reward_started = True
        info = {
            "is_query": True,
            "is_reward_phase": False,
            "substep_type": LEADER_QUERY,
            "query_index": int(completed),
            "query_count": int(self.query_index),
            "reward_game_started": bool(self._reward_started),
        }
        return self._emit(self._next_observation()), 0.0, False, info

    def _predict_follower(self, *, trade):
        values = self._follower_observation(trade=trade)
        action, _ = self.response_model.predict(values, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (2,):
            raise RuntimeError("meta-follower must return two action coordinates")
        return np.array([
            np.clip(action[0], 0.0, self.core.game_action_count - 1),
            np.clip(action[1], 0.0, 1.0),
        ], dtype=np.float32)

    def _validate_cache_hit(self, event_index, action):
        if self.query_trace is None:
            raise RuntimeError("reward trade reached before trace finalization")
        query_observation, query_action = self.query_trace.reconstruct()[event_index]
        assert_actor_observations_equal(self._last_observation, query_observation)
        if not np.allclose(action, query_action, rtol=0.0, atol=1.0e-7):
            raise RuntimeError(
                f"leader cache miss at event {event_index}: "
                f"{action.tolist()} != {query_action.tolist()}"
            )
        self.cache_hits += 1
        return np.array(query_action, copy=True)

    def _leader_reward(self):
        return float(
            self.core.seller_payoff
            if self.leader_role == SELLER
            else self.core.buyer_payoff
        )

    def _trade_step(self, action):
        event_index = int(self.core.next_event)
        cached = self._validate_cache_hit(event_index, action)
        follower = self._predict_follower(trade=True)
        self.follower_full_actions[event_index] = follower
        if self.leader_role == SELLER:
            price, threshold = float(cached[1]), float(follower[1])
        else:
            price, threshold = float(follower[1]), float(cached[1])
        previous = self._leader_reward()
        seller_calls = self.core.seller.step_calls
        buyer_calls = self.core.buyer.step_calls
        event = self.core.trade(price=price, threshold=threshold)
        if (
                seller_calls != self.core.seller.step_calls
                or buyer_calls != self.core.buyer.step_calls
        ):
            raise RuntimeError("an Atari emulator advanced during trade")
        reward = self._leader_reward() - previous
        self.trade_transitions += 1
        info = self._transition_info(CACHED_TRADE_REPLAY, reward)
        info.update({
            "event_index": event_index,
            "trade_event": dict(event),
            "cache_hit": True,
            "leader_executed_action": tuple(float(value) for value in cached),
            "follower_action": tuple(float(value) for value in follower),
            "emulator_advanced": False,
        })
        return self._emit(self._next_observation()), float(reward), False, info

    def _gameplay_step(self, action):
        follower = self._predict_follower(trade=False)
        if self.leader_role == SELLER:
            seller_action, buyer_action = action[0], follower[0]
        else:
            seller_action, buyer_action = follower[0], action[0]
        previous = self._leader_reward()
        transition = self.core.step_gameplay(
            seller_action=seller_action,
            buyer_action=buyer_action,
        )
        reward = self._leader_reward() - previous
        self.gameplay_transitions += 1
        self._done = bool(self.core.done)
        if self._done:
            if self.core.next_event != NUM_TRADE_EVENTS:
                raise RuntimeError("E2 horizon ended before all five trades")
            self.core.assert_accounting()
        info = self._transition_info(GAMEPLAY, reward)
        info.update({
            "gameplay": transition,
            "leader_executed_game_action": int(np.rint(action[0])),
            "follower_game_action": int(np.rint(follower[0])),
            "emulator_advanced": True,
        })
        if self._done:
            info.update(self._episode_info())
            info["episode"] = {
                "r": self._leader_reward(),
                "l": info["outer_transition_count"],
                **self._episode_info(),
            }
        return self._emit(self._next_observation()), float(reward), self._done, info

    def _transition_info(self, substep_type, reward):
        return {
            "is_query": False,
            "is_reward_phase": True,
            "substep_type": substep_type,
            "leader_reward_delta": float(reward),
            "game_step": int(self.core.game_step),
            "next_event": int(self.core.next_event),
            "query_transitions": int(self.query_transitions),
            "gameplay_transitions": int(self.gameplay_transitions),
            "trade_transitions": int(self.trade_transitions),
        }

    def _episode_info(self):
        core = self.core
        if self.query_trace is None:
            raise RuntimeError("terminal episode is missing its query trace")
        return {
            "leader_role": self.leader_role,
            "follower_role": self.follower_role,
            "query_actions": tuple(
                tuple(float(value) for value in action)
                for action in self.query_trace.full_action_matrix()
            ),
            "query_trace_sha256": self.query_trace.sha256,
            "leader_commitment": tuple(
                float(value) for value in self.query_trace.economic_commitment
            ),
            "follower_actions": tuple(
                tuple(float(value) for value in action)
                for action in self.follower_full_actions
            ),
            "event_steps": tuple(int(value) for value in core.event_steps),
            "events": tuple(dict(event) for event in core.events),
            "outer_transition_count": int(
                self.query_transitions
                + self.gameplay_transitions
                + self.trade_transitions
            ),
            "query_transitions": int(self.query_transitions),
            "gameplay_transitions": int(self.gameplay_transitions),
            "trade_transitions": int(self.trade_transitions),
            "cache_hits": int(self.cache_hits),
            "bullets_arrived": int(core.bullets_arrived),
            "purchases": int(core.transfers),
            "payments": float(core.payments),
            "seller_game_reward": float(core.seller.game_reward),
            "buyer_game_reward": float(core.buyer.game_reward),
            "seller_reward": float(core.seller_payoff),
            "buyer_reward": float(core.buyer_payoff),
            "leader_reward": self._leader_reward(),
            "seller_shots_fired": int(core.seller.shots_fired),
            "buyer_shots_fired": int(core.buyer.shots_fired),
            "seller_final_ammo": int(core.seller.ammo),
            "buyer_final_ammo": int(core.buyer.ammo),
            **core.accounting(),
        }

    def step(self, action):
        if self._done:
            raise RuntimeError("step called after E2 outer episode termination")
        values = self._validated_action(action)
        if not self._reward_started:
            return self._query_step(values)
        if self.core.at_event:
            return self._trade_step(values)
        return self._gameplay_step(values)

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")
        seller = self.core.seller.env.render(mode=mode)
        buyer = self.core.buyer.env.render(mode=mode)
        return np.concatenate([seller, buyer], axis=1)

    def close(self):
        self.core.close()


__all__ = ["FullTraceStackPOMDPAtariLeaderEnv"]
