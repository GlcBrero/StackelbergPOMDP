"""Frozen neural PI response and generic StackPOMDP composition for Atari E2.

The module deliberately separates three concerns:

* :class:`BilateralAtariRewardEnv` owns Atari and bilateral-trade dynamics;
* :class:`FrozenMetaPolicyResponse` retains the exact leader query trace and
  conditions a frozen E1 follower policy on its declared economic statistic;
* :class:`AtariMetaFollowerWrapper` adapts that response to the generic
  :class:`~stackelberg_pomdp.gym_envs.envs.wrappers.FollowerWrapper` contract.

The existing game-agnostic ``StackPOMDPWrapper`` then owns response/reward
phase management.  All five leader queries remain in the PPO rollout.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

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
from stackelberg_pomdp.gym_envs.envs.wrappers import (
    FollowerWrapper,
    StackPOMDPWrapper,
)


def _copy_observation(values):
    return OrderedDict(
        (key, np.array(value, copy=True)) for key, value in values.items()
    )


class BilateralAtariRewardEnv(gym.Env):
    """Two-player Atari reward game with no StackPOMDP phase logic.

    ``step`` receives complete seller and buyer actions.  This is the only E2
    layer that advances ALE, transfers bullets, or computes economic payoffs.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            *,
            leader_role,
            config=None,
            core_factory=DualAtariTradeCore,
            side_factory=None,
            env_factory=None,
    ):
        super().__init__()
        if leader_role not in ROLES:
            raise ValueError(f"leader_role must be one of {sorted(ROLES)}")
        self.leader_role = leader_role
        self.follower_role = BUYER if leader_role == SELLER else SELLER
        self.config = (config or BilateralAtariConfig()).resolved()
        core_kwargs = {}
        if side_factory is not None:
            core_kwargs["side_factory"] = side_factory
        if env_factory is not None:
            core_kwargs["env_factory"] = env_factory
        self.core = core_factory(self.config, **core_kwargs)

        role_action_space = action_space(self.core.game_action_count)
        role_observation_space = observation_space(
            self.core.image_space, self.core.game_action_count
        )
        self.action_space = gym.spaces.Dict(OrderedDict(
            (role, role_action_space) for role in (SELLER, BUYER)
        ))
        self.observation_space = gym.spaces.Dict(OrderedDict(
            (role, role_observation_space) for role in (SELLER, BUYER)
        ))
        self.dummy_image = np.zeros(
            self.core.image_space.shape, dtype=self.core.image_space.dtype
        )
        self._done = False
        self.gameplay_transitions = 0
        self.trade_transitions = 0

    @staticmethod
    def _validated_action(action, game_action_count):
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (2,):
            raise ValueError("Atari action must be [game_action, economic]")
        return np.array([
            np.clip(values[0], 0.0, game_action_count - 1),
            np.clip(values[1], 0.0, 1.0),
        ], dtype=np.float32)

    def _validated_joint_action(self, actions):
        if not isinstance(actions, Mapping) or set(actions) != ROLES:
            raise ValueError("joint Atari action must contain seller and buyer")
        return {
            role: self._validated_action(
                actions[role], self.core.game_action_count
            )
            for role in (SELLER, BUYER)
        }

    def role_observation(
            self,
            role,
            *,
            opponent_commitment=None,
            decision_kind=None,
            critic_state=None,
    ):
        """Return one live role observation without response-phase context."""

        side = self.core.side(role)
        trade_mode = self.core.at_event and not self._done
        event_index = (
            self.core.next_event
            if self.core.next_event < NUM_TRADE_EVENTS
            else None
        )
        if decision_kind is None:
            decision_kind = (
                TERMINAL
                if self._done
                else FOLLOWER_TRADE if trade_mode else GAMEPLAY
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
                opponent_commitment=opponent_commitment,
            ),
            action_mask=side.action_mask,
            decision_kind=decision_kind,
            critic_state=critic_state,
        )

    def _observations(self):
        return OrderedDict(
            (role, self.role_observation(role)) for role in (SELLER, BUYER)
        )

    def reset(self, *, seed=None, options=None):
        del options
        self.core.reset(seed=seed)
        self._done = False
        self.gameplay_transitions = 0
        self.trade_transitions = 0
        return self._observations()

    def leader_reward(self):
        return float(
            self.core.seller_payoff
            if self.leader_role == SELLER
            else self.core.buyer_payoff
        )

    def episode_info(self):
        core = self.core
        return {
            "leader_role": self.leader_role,
            "follower_role": self.follower_role,
            "event_steps": tuple(int(value) for value in core.event_steps),
            "events": tuple(dict(event) for event in core.events),
            "gameplay_transitions": int(self.gameplay_transitions),
            "trade_transitions": int(self.trade_transitions),
            "reward_transition_count": int(
                self.gameplay_transitions + self.trade_transitions
            ),
            "bullets_arrived": int(core.bullets_arrived),
            "purchases": int(core.transfers),
            "payments": float(core.payments),
            "seller_game_reward": float(core.seller.game_reward),
            "buyer_game_reward": float(core.buyer.game_reward),
            "seller_reward": float(core.seller_payoff),
            "buyer_reward": float(core.buyer_payoff),
            "leader_reward": self.leader_reward(),
            "seller_shots_fired": int(core.seller.shots_fired),
            "buyer_shots_fired": int(core.buyer.shots_fired),
            "seller_final_ammo": int(core.seller.ammo),
            "buyer_final_ammo": int(core.buyer.ammo),
            **core.accounting(),
        }

    def step(self, actions):
        if self._done:
            raise RuntimeError("reward-game step called after Atari termination")
        values = self._validated_joint_action(actions)
        previous = {
            SELLER: float(self.core.seller_payoff),
            BUYER: float(self.core.buyer_payoff),
        }
        if self.core.at_event:
            seller_calls = self.core.seller.step_calls
            buyer_calls = self.core.buyer.step_calls
            event = self.core.trade(
                price=float(values[SELLER][1]),
                threshold=float(values[BUYER][1]),
            )
            if (
                    seller_calls != self.core.seller.step_calls
                    or buyer_calls != self.core.buyer.step_calls
            ):
                raise RuntimeError("an Atari emulator advanced during trade")
            self.trade_transitions += 1
            detail = {
                "substep_type": "trade",
                "trade_event": dict(event),
                "emulator_advanced": False,
            }
        else:
            transition = self.core.step_gameplay(
                seller_action=values[SELLER][0],
                buyer_action=values[BUYER][0],
            )
            self.gameplay_transitions += 1
            detail = {
                "substep_type": GAMEPLAY,
                "gameplay": transition,
                "emulator_advanced": True,
            }

        rewards = {
            SELLER: float(self.core.seller_payoff - previous[SELLER]),
            BUYER: float(self.core.buyer_payoff - previous[BUYER]),
        }
        self._done = bool(self.core.done)
        if self._done:
            if self.core.next_event != NUM_TRADE_EVENTS:
                raise RuntimeError("Atari horizon ended before all five trades")
            self.core.assert_accounting()
        info = {
            **detail,
            "reward_generated": True,
            "utilities": dict(rewards),
            "surplus": float(rewards[self.leader_role]),
            "game_step": int(self.core.game_step),
            "next_event": int(self.core.next_event),
        }
        if self._done:
            info.update(self.episode_info())
        return self._observations(), rewards, self._done, info

    def reward_phase_length(self, default_length):
        del default_length
        return int(self.config.gameplay_horizon + NUM_TRADE_EVENTS)

    def max_reward_phase_length(self, default_length):
        return self.reward_phase_length(default_length)

    def max_subepisode_transitions(self):
        return 1

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("only rgb_array mode is supported")
        seller = self.core.seller.env.render(mode=mode)
        buyer = self.core.buyer.env.render(mode=mode)
        return np.concatenate([seller, buyer], axis=1)

    def close(self):
        self.core.close()


class FrozenMetaPolicyResponse:
    """Exact five-query PI response implemented by a frozen E1 policy.

    E1 is the meta-learning stage.  E2 performs no follower learning: it
    captures the exact ordered leader trace, exposes its five economic actions
    as the declared context ``omega``, and evaluates one frozen response model.
    The exact trace is retained for correctness and diagnostics; the current
    E1 actor is intentionally conditioned on ``omega``, not on a lossless
    encoding of every field in the trace.
    """

    def __init__(
            self,
            checkpoint,
            *,
            follower_role,
            model_factory=None,
            device="cpu",
    ):
        if follower_role not in ROLES:
            raise ValueError(f"unknown follower role: {follower_role!r}")
        if model_factory is None:
            if checkpoint is None:
                raise ValueError("response checkpoint is required")
            from stable_baselines3 import PPO

            path = Path(checkpoint).expanduser()
            if not path.is_file() and Path(f"{path}.zip").is_file():
                path = Path(f"{path}.zip")
            if not path.is_file():
                raise FileNotFoundError(
                    f"meta-response checkpoint does not exist: {path}"
                )
            self.model = PPO.load(str(path), device=device)
        else:
            self.model = model_factory(checkpoint, device=device)

        self.follower_role = follower_role
        policy = getattr(self.model, "policy", None)
        response_role = getattr(policy, "economic_role", None)
        if response_role != follower_role:
            raise ValueError(
                f"frozen response must be a {follower_role} E1 policy, "
                f"got {response_role!r}"
            )
        response_input = getattr(policy, "economic_input_mode", None)
        if response_input != "full":
            raise ValueError(
                "frozen E1 meta-response must use the full actor state, "
                f"got {response_input!r}"
            )
        if policy is not None:
            policy.set_training_mode(False)
            for parameter in policy.parameters():
                parameter.requires_grad = False
        self.reset_episode()

    def reset_episode(self):
        self._query_observations = []
        self._query_actions = []
        self.query_trace = None
        self.completed_iterations = 0
        self.iteration_type_profile_counts = {}

    @property
    def query_actions(self):
        return tuple(np.array(value, copy=True) for value in self._query_actions)

    def observe_query(self, values, action):
        if self.query_trace is not None:
            raise RuntimeError("cannot append to a finalized leader query trace")
        if len(self._query_actions) >= NUM_TRADE_EVENTS:
            raise RuntimeError("received more than five leader queries")
        self._query_observations.append(actor_observation(values))
        self._query_actions.append(np.array(action, dtype=np.float32, copy=True))
        # Finalize immediately after the fifth exact query.  Newer generic
        # phase wrappers also call ``request_completion`` at the boundary;
        # the method is idempotent.  Immediate finalization keeps this response
        # adapter compatible with older StackPOMDPWrapper checkpoints/runners
        # that switch phases solely from the fixed response-step count.
        if len(self._query_actions) == NUM_TRADE_EVENTS:
            self.request_completion()

    def request_completion(self):
        if self.query_trace is not None:
            return True
        if len(self._query_actions) != NUM_TRADE_EVENTS:
            return False
        self.query_trace = LeaderQueryTrace.capture(
            self._query_observations, self._query_actions
        )
        self.completed_iterations = 1
        return True

    def response_ready(self):
        return self.query_trace is not None

    @property
    def economic_commitment(self):
        if self.query_trace is None:
            raise RuntimeError("meta-response requested before trace completion")
        return self.query_trace.economic_commitment

    def response_strategy(self):
        # A conditional neural policy is not representable as the finite
        # strategy lists used by MW/Q-learning diagnostics.  The leader's
        # commitment is reported separately and with its correct name below.
        return []

    def phase_info(self):
        if not self.response_ready():
            return {}
        return {
            "query_trace_sha256": self.query_trace.sha256,
            "leader_commitment": tuple(
                float(value) for value in self.economic_commitment
            ),
            "response_algorithm": "frozen_meta_policy",
        }

    def predict(self, values):
        if not self.response_ready():
            raise RuntimeError("frozen response queried before five leader queries")
        action, _ = self.model.predict(values, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (2,):
            raise RuntimeError("meta-follower must return two action coordinates")
        return action


class AtariMetaFollowerWrapper(FollowerWrapper):
    """Reduce the bilateral Atari game to one leader-facing Gym environment."""

    follower_state_kind = "frozen_meta_policy"

    def __init__(
            self,
            env,
            *,
            response_checkpoint=None,
            response_model_factory=None,
            device="cpu",
            seed=None,
    ):
        if not isinstance(env, BilateralAtariRewardEnv):
            raise TypeError("AtariMetaFollowerWrapper requires BilateralAtariRewardEnv")
        super().__init__(env)
        self.leader_role = env.leader_role
        self.follower_role = env.follower_role
        self.config = env.config
        self.response = FrozenMetaPolicyResponse(
            response_checkpoint,
            follower_role=self.follower_role,
            model_factory=response_model_factory,
            device=device,
        )
        self.rng = np.random.default_rng(
            int(self.config.seed if seed is None else seed) + 193_939
        )
        self.action_space = action_space(self.env.core.game_action_count)
        self.observation_space = observation_space(
            self.env.core.image_space, self.env.core.game_action_count
        )
        self.dummy_image = np.zeros(
            self.env.core.image_space.shape, dtype=self.env.core.image_space.dtype
        )
        self.canonical_action_mask = np.ones(
            self.env.core.game_action_count, dtype=np.float32
        )
        fire_indices = tuple(
            int(index)
            for index in self.env.core.side(self.leader_role)
            .ammo_wrapper.fire_action_indices
        )
        self.canonical_action_mask[list(fire_indices)] = 0.0
        self._last_observation = None
        self._reward_started = False
        self.query_index = 0
        self.query_transitions = 0
        self.cache_hits = 0
        self.follower_full_actions = np.zeros(
            (NUM_TRADE_EVENTS, 2), dtype=np.float32
        )

    @property
    def core(self):
        return self.env.core

    @property
    def leader_query_trace(self):
        return self.response.query_trace

    def _critic_state(self, *, phase, event_index=0):
        values = np.zeros(CRITIC_STATE_DIM, dtype=np.float32)
        values[:8] = (
            float(phase),
            float(self.query_index) / NUM_TRADE_EVENTS,
            float(event_index) / (NUM_TRADE_EVENTS - 1),
            float(self.response.response_ready()),
            float(self.query_transitions) / NUM_TRADE_EVENTS,
            float(self.env.gameplay_transitions) / self.config.gameplay_horizon,
            float(self.env.trade_transitions) / NUM_TRADE_EVENTS,
            float(self._reward_started),
        )
        for index, action in enumerate(self.response.query_actions):
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
        result = self._canonical_observation(event_index, CACHED_TRADE_REPLAY)
        if self.response.query_trace is None:
            raise RuntimeError("trade replay reached before response completion")
        query_observation, _ = self.response.query_trace.reconstruct()[event_index]
        assert_actor_observations_equal(result, query_observation)
        return result

    def _live_observation(self, role, *, follower, decision_kind):
        commitment = (
            self.response.economic_commitment
            if follower and self.response.response_ready()
            else np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        )
        return self.env.role_observation(
            role,
            opponent_commitment=commitment,
            decision_kind=decision_kind,
            critic_state=self._critic_state(
                phase=1.0,
                event_index=min(self.core.next_event, NUM_TRADE_EVENTS - 1),
            ),
        )

    def _leader_gameplay_observation(self):
        return self._live_observation(
            self.leader_role,
            follower=False,
            decision_kind=TERMINAL if self.env._done else GAMEPLAY,
        )

    def _follower_observation(self, *, trade):
        return self._live_observation(
            self.follower_role,
            follower=True,
            decision_kind=FOLLOWER_TRADE if trade else GAMEPLAY,
        )

    def _next_leader_observation(self):
        if not self._reward_started:
            if self.query_index >= NUM_TRADE_EVENTS:
                return self._query_observation(NUM_TRADE_EVENTS - 1)
            return self._query_observation(self.query_index)
        if self.env._done:
            return self._leader_gameplay_observation()
        if self.core.at_event:
            return self._trade_observation(self.core.next_event)
        return self._leader_gameplay_observation()

    def _emit(self, values):
        self._last_observation = _copy_observation(values)
        return _copy_observation(values)

    def current_leader_observation(self):
        return self._emit(self._next_leader_observation())

    def reset(self):
        self.response.reset_episode()
        self.query_index = 0
        self.query_transitions = 0
        self.cache_hits = 0
        self.follower_full_actions.fill(0.0)
        self._reward_started = False
        return self._emit(self._query_observation(0))

    def on_step_mode_changed(self, mode):
        if mode == "reward":
            if not self.response.response_ready():
                raise RuntimeError("reward phase started before response completion")
            self.env.reset(seed=int(self.rng.integers(0, 2 ** 31 - 1)))
            self._reward_started = True
        elif mode == "response":
            self._reward_started = False

    def request_response_completion(self):
        return self.response.request_completion()

    def response_ready(self):
        return self.response.response_ready()

    def response_strategy(self):
        return self.response.response_strategy()

    def response_phase_info(self):
        return OrderedDict(self.response.phase_info())

    def reward_phase_length(self, default_length):
        return self.env.reward_phase_length(default_length)

    def max_reward_phase_length(self, default_length):
        return self.env.max_reward_phase_length(default_length)

    def max_subepisode_transitions(self):
        return self.env.max_subepisode_transitions()

    @staticmethod
    def _validated_leader_action(action, game_action_count):
        return BilateralAtariRewardEnv._validated_action(
            action, game_action_count
        )

    def _query_step(self, action):
        if self.query_index >= NUM_TRADE_EVENTS:
            raise RuntimeError("received more than five response queries")
        values = self._validated_leader_action(
            action, self.core.game_action_count
        )
        completed = self.query_index
        self.response.observe_query(self._last_observation, values)
        self.query_index += 1
        self.query_transitions += 1
        info = {
            "reward_generated": True,
            "is_query": True,
            "substep_type": LEADER_QUERY,
            "query_index": int(completed),
            "query_count": int(self.query_index),
            "emulator_advanced": False,
        }
        return self._emit(self._next_leader_observation()), 0.0, False, info

    def _cached_leader_action(self, event_index, action):
        if self.response.query_trace is None:
            raise RuntimeError("reward trade reached before trace finalization")
        query_observation, query_action = (
            self.response.query_trace.reconstruct()[event_index]
        )
        assert_actor_observations_equal(self._last_observation, query_observation)
        if not np.allclose(action, query_action, rtol=0.0, atol=1.0e-7):
            raise RuntimeError(
                f"leader cache miss at event {event_index}: "
                f"{action.tolist()} != {query_action.tolist()}"
            )
        self.cache_hits += 1
        return np.array(query_action, copy=True)

    def _episode_info(self):
        if self.response.query_trace is None:
            raise RuntimeError("terminal episode is missing its query trace")
        return {
            **self.env.episode_info(),
            "query_actions": tuple(
                tuple(float(value) for value in action)
                for action in self.response.query_trace.full_action_matrix()
            ),
            "query_trace_sha256": self.response.query_trace.sha256,
            "leader_commitment": tuple(
                float(value)
                for value in self.response.query_trace.economic_commitment
            ),
            "follower_actions": tuple(
                tuple(float(value) for value in action)
                for action in self.follower_full_actions
            ),
            "query_transitions": int(self.query_transitions),
            "outer_transition_count": int(
                self.query_transitions
                + self.env.gameplay_transitions
                + self.env.trade_transitions
            ),
            "cache_hits": int(self.cache_hits),
            "response_algorithm": "frozen_meta_policy",
        }

    def _reward_step(self, action):
        leader_action = self._validated_leader_action(
            action, self.core.game_action_count
        )
        trade = bool(self.core.at_event)
        follower_action = self.response.predict(
            self._follower_observation(trade=trade)
        )
        follower_action = BilateralAtariRewardEnv._validated_action(
            follower_action, self.core.game_action_count
        )
        event_index = int(self.core.next_event) if trade else None
        if trade:
            leader_action = self._cached_leader_action(event_index, leader_action)
            self.follower_full_actions[event_index] = follower_action

        joint_actions = {
            self.leader_role: leader_action,
            self.follower_role: follower_action,
        }
        _, rewards, inner_done, info = self.env.step(joint_actions)
        reward = float(rewards[self.leader_role])
        substep = CACHED_TRADE_REPLAY if trade else GAMEPLAY
        info.update({
            "substep_type": substep,
            "is_query": False,
            "leader_reward_delta": reward,
            "query_transitions": int(self.query_transitions),
            "gameplay_transitions": int(self.env.gameplay_transitions),
            "trade_transitions": int(self.env.trade_transitions),
            "leader_executed_action": tuple(
                float(value) for value in leader_action
            ),
            "follower_action": tuple(float(value) for value in follower_action),
        })
        if trade:
            info.update({"event_index": event_index, "cache_hit": True})
        if inner_done:
            episode_info = self._episode_info()
            if episode_info["outer_transition_count"] != (
                    NUM_TRADE_EVENTS
                    + self.config.gameplay_horizon
                    + NUM_TRADE_EVENTS
            ):
                raise RuntimeError("E2 outer transition accounting is inconsistent")
            if episode_info["cache_hits"] != NUM_TRADE_EVENTS:
                raise RuntimeError("E2 terminated without five cache hits")
            info.update(episode_info)
            info["episode"] = {
                "r": self.env.leader_reward(),
                "l": episode_info["outer_transition_count"],
                **episode_info,
            }
        return self._emit(self._next_leader_observation()), reward, inner_done, info

    def step(self, action):
        if self.this_step_mode == "response":
            return self._query_step(action)
        if self.this_step_mode == "reward":
            return self._reward_step(action)
        raise RuntimeError(
            f"unknown Atari follower-wrapper mode {self.this_step_mode!r}"
        )


def make_stackpomdp_atari_leader_env(
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
    """Compose Atari E2 with the shared game-agnostic phase wrapper."""

    resolved = (config or BilateralAtariConfig()).resolved()
    reward_env = BilateralAtariRewardEnv(
        leader_role=leader_role,
        config=resolved,
        core_factory=core_factory,
        side_factory=side_factory,
        env_factory=env_factory,
    )
    follower_env = AtariMetaFollowerWrapper(
        reward_env,
        response_checkpoint=response_checkpoint,
        response_model_factory=response_model_factory,
        device=device,
        seed=resolved.seed,
    )
    return StackPOMDPWrapper(
        follower_env,
        tot_num_response_episodes=NUM_TRADE_EVENTS,
        tot_num_reward_episodes=(
            resolved.gameplay_horizon + NUM_TRADE_EVENTS
        ),
        critic_obs=None,
        response_variant="stackelberg",
    )


__all__ = [
    "AtariMetaFollowerWrapper",
    "BilateralAtariRewardEnv",
    "FrozenMetaPolicyResponse",
    "make_stackpomdp_atari_leader_env",
]
