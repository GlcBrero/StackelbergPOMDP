"""Stepwise Stackelberg-POMDP leader environment for bilateral Atari.

This module is the correctness-first counterpart to the event-compressed
leader environment.  An outer episode contains five response queries followed
by a fresh, fixed-horizon reward game.  Crucially, every leader transition in
the reward game is returned to the RL algorithm: ``H`` gameplay transitions
and five paused trade transitions.

The first five observations are canonical trade observations.  Their complete
actor-visible observations and complete two-component actions are recorded as
the leader commitment.  A frozen meta-follower receives that entire trace at
each real trade event.  The policy-side action cache must return the same full
action when the corresponding canonical observation recurs; this environment
checks that invariant but does not choose an action on the policy's behalf.

Actor observations contain no response/reward phase flag.  Query observation
``j`` and reward-game trade observation ``j`` are bit-identical after removing
``critic:*`` fields.  The critic receives the complete outer-protocol state in
``critic:state``.
"""

from collections import OrderedDict
import hashlib
from pathlib import Path

import gym
from gym import spaces
import numpy as np

from stackelberg_pomdp.atari.stackpomdp_env import (
    BUYER,
    CRITIC_STATE_DIM,
    NUM_TRADE_EVENTS,
    ROLES,
    SELLER,
    BilateralAtariConfig,
    DualAtariTradeCore,
)
from stackelberg_pomdp.atari.query_trace import (
    FixedCanonicalQueryFeatureEncoder,
    LeaderQueryTrace,
    legacy_economic_action_projection,
)

# phase, query count/index, game/event state, complete economic accounting,
# five exogenous event times, ten cached full-action coordinates, response-ready
FULL_LEADER_CRITIC_STATE_DIM = 31


def _copy_mapping(mapping):
    return OrderedDict(
        (key, np.array(value, copy=True)) for key, value in mapping.items()
    )


def actor_observation(observation):
    """Copy the part of an observation that is available to the actor."""
    return OrderedDict(
        (key, np.array(value, copy=True))
        for key, value in observation.items()
        if not key.startswith("critic:")
    )


class StepwiseDualAtariTradeCore(DualAtariTradeCore):
    """Expose one frozen-controller gameplay transition at a time.

    The submitted leader game action is checked against the deterministic E0
    controller.  The follower game action is generated directly by its frozen
    E0 controller.  This makes it impossible for the economic leader trainer
    to silently fine-tune or override gameplay.
    """

    def side_for(self, role):
        if role == SELLER:
            return self.seller
        if role == BUYER:
            return self.buyer
        raise ValueError(f"unknown Atari role: {role!r}")

    def observation_for(self, role):
        side = self.side_for(role)
        return _copy_mapping(side._refreshed_observation())

    def _frozen_actions(self):
        seller_observation = self.seller._refreshed_observation()
        buyer_observation = self.buyer._refreshed_observation()
        shared_controller = (
            self.seller.controller
            if self.seller.controller is self.buyer.controller
            else None
        )
        if shared_controller is not None and hasattr(
                shared_controller, "actions"
        ):
            values = shared_controller.actions(
                [seller_observation, buyer_observation]
            )
            if len(values) != 2:
                raise RuntimeError("shared E0 controller returned wrong batch size")
            return int(values[0]), int(values[1])
        return (
            int(self.seller.controller(seller_observation)),
            int(self.buyer.controller(buyer_observation)),
        )

    def step_gameplay(self, *, leader_role, leader_game_action):
        """Advance both Atari emulators by exactly one gameplay decision."""
        if self.done:
            raise RuntimeError("gameplay step requested after the fixed horizon")
        if self.at_event:
            raise RuntimeError("trade must resolve before gameplay can advance")

        seller_action, buyer_action = self._frozen_actions()
        supplied = int(np.rint(leader_game_action))
        expected = seller_action if leader_role == SELLER else buyer_action
        if supplied != expected:
            raise RuntimeError(
                "leader gameplay action differs from frozen deterministic E0: "
                f"received {supplied}, expected {expected}"
            )

        seller_reward, _, seller_info = self.seller.step(seller_action)
        buyer_reward, _, buyer_info = self.buyer.step(buyer_action)
        seller_delta = self.config.seller_game_reward_scale * seller_reward
        buyer_delta = self.config.buyer_game_reward_scale * buyer_reward
        self.seller_payoff += seller_delta
        self.buyer_payoff += buyer_delta
        self.game_step += 1
        if self.at_event:
            self.prepare_event()
        return {
            "seller_game_action": int(seller_action),
            "buyer_game_action": int(buyer_action),
            "seller_game_reward": float(seller_reward),
            "buyer_game_reward": float(buyer_reward),
            "seller_reward_delta": float(seller_delta),
            "buyer_reward_delta": float(buyer_delta),
            "seller_info": dict(seller_info),
            "buyer_info": dict(buyer_info),
        }


class FullTraceStackPOMDPAtariLeaderEnv(gym.Env):
    """Expose the complete StackPOMDP leader trajectory to PPO.

    The action is ``[game_action, economic_action]``.  On gameplay states the
    first component must equal the frozen deterministic E0 action and the
    second component is ignored.  On trade states the complete action must be
    the cache hit from the matching response query; the game component is
    ignored and the economic component is interpreted as a price or threshold.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
            self,
            *,
            leader_role,
            response_checkpoint,
            game_checkpoint,
            config=None,
            core_factory=StepwiseDualAtariTradeCore,
            controller_factory=None,
            response_model_factory=None,
            device="cpu",
    ):
        super().__init__()
        if leader_role not in ROLES:
            raise ValueError(f"leader_role must be one of {sorted(ROLES)}")
        self.leader_role = leader_role
        self.follower_role = BUYER if leader_role == SELLER else SELLER
        self.config = (config or BilateralAtariConfig()).resolved()
        self.rng = np.random.default_rng(self.config.seed + 193_939)
        self.core = core_factory(
            self.config,
            game_checkpoint=game_checkpoint,
            controller_factory=controller_factory,
        )
        for method in ("observation_for", "step_gameplay"):
            if not hasattr(self.core, method):
                raise TypeError(
                    f"stepwise Atari core must implement {method}()"
                )

        self.game_action_count = int(self.core.game_action_count)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array(
                [float(self.game_action_count - 1), 1.0], dtype=np.float32
            ),
            dtype=np.float32,
        )
        observation_spaces = OrderedDict(
            self.core.observation_space.spaces.items()
        )
        observation_spaces.update([
            (
                "event_active",
                spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            ),
            (
                "event_one_hot",
                spaces.Box(
                    0.0,
                    1.0,
                    shape=(NUM_TRADE_EVENTS,),
                    dtype=np.float32,
                ),
            ),
            (
                "opponent_context",
                spaces.Box(
                    0.0,
                    1.0,
                    shape=(NUM_TRADE_EVENTS,),
                    dtype=np.float32,
                ),
            ),
            (
                "critic:state",
                spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(FULL_LEADER_CRITIC_STATE_DIM,),
                    dtype=np.float32,
                ),
            ),
        ])
        self.observation_space = spaces.Dict(observation_spaces)
        image_space = self.core.observation_space.spaces["image"]
        self.dummy_image = np.zeros(image_space.shape, dtype=image_space.dtype)

        loaded_real_response = response_model_factory is None
        if loaded_real_response:
            from stable_baselines3 import PPO

            response_path = Path(response_checkpoint).expanduser().resolve()
            self.response_model = PPO.load(str(response_path), device=device)
            self.response_model.policy.set_training_mode(False)
            for parameter in self.response_model.policy.parameters():
                parameter.requires_grad = False
        else:
            self.response_model = response_model_factory(
                response_checkpoint, device=device
            )
        response_role = getattr(
            getattr(self.response_model, "policy", None),
            "economic_role",
            None,
        )
        if response_role is None:
            raise TypeError(
                "response checkpoint lacks required economic_role metadata"
            )
        if response_role != self.follower_role:
            raise ValueError(
                f"response checkpoint role {response_role!r} does not match "
                f"required follower {self.follower_role!r}"
            )
        if loaded_real_response:
            self._check_gameplay_fingerprint(game_checkpoint)

        self._canonical_actor_observations = tuple(
            self._make_canonical_actor_trade_observation(event_index)
            for event_index in range(NUM_TRADE_EVENTS)
        )
        canonical_reference = LeaderQueryTrace.capture(
            self._canonical_actor_observations,
            [
                np.zeros(2, dtype=np.float32)
                for _ in range(NUM_TRADE_EVENTS)
            ],
        )
        # Bound once to an independently constructed template.  Every episode
        # must pass its byte-exact observation check before any projection is
        # supplied to a legacy five-economic-input follower.
        self._trace_encoder = FixedCanonicalQueryFeatureEncoder.bind(
            canonical_reference
        )
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

    def _check_gameplay_fingerprint(self, game_checkpoint):
        response_fingerprint = getattr(
            self.response_model.policy, "gameplay_fingerprint", None
        )
        game_path = Path(game_checkpoint).expanduser()
        if not game_path.is_file():
            zip_path = Path(f"{game_path}.zip")
            if zip_path.is_file():
                game_path = zip_path
        if not game_path.is_file():
            raise FileNotFoundError(
                f"game checkpoint does not exist: {game_checkpoint}"
            )
        expected = hashlib.sha256(game_path.read_bytes()).hexdigest()
        if response_fingerprint != expected:
            raise ValueError(
                "response/gameplay checkpoint fingerprint mismatch: "
                f"{response_fingerprint!r} != {expected!r}"
            )

    def seed(self, seed=None):
        seed = self.config.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(seed + 193_939)
        return [seed]

    @property
    def leader_query_trace(self):
        """Return the immutable exact trace, or ``None`` before query five."""
        return self.query_trace

    def _make_canonical_actor_trade_observation(self, event_index):
        one_hot = np.zeros(NUM_TRADE_EVENTS, dtype=np.float32)
        one_hot[int(event_index)] = 1.0
        result = OrderedDict()
        for key, space in self.core.observation_space.spaces.items():
            if key.startswith("critic:"):
                continue
            if key == "image":
                value = np.array(self.dummy_image, copy=True)
            elif key == "action_mask":
                value = np.ones(space.shape, dtype=space.dtype)
            elif key == "opportunities_remaining":
                value = np.ones(space.shape, dtype=space.dtype)
            else:
                value = np.zeros(space.shape, dtype=space.dtype)
            result[key] = value
        result["event_active"] = np.array([1.0], dtype=np.float32)
        result["event_one_hot"] = one_hot
        result["opponent_context"] = np.zeros(
            NUM_TRADE_EVENTS, dtype=np.float32
        )
        return result

    def _critic_state(self, *, phase, event_index=0):
        values = np.zeros(FULL_LEADER_CRITIC_STATE_DIM, dtype=np.float32)
        values[0] = float(phase)
        values[1] = float(len(self._query_actions)) / NUM_TRADE_EVENTS
        values[2] = float(event_index) / float(NUM_TRADE_EVENTS - 1)
        values[30] = float(self.query_trace is not None)
        for index, action in enumerate(self._query_actions):
            start = 20 + 2 * index
            values[start:start + 2] = action

        if phase < 0.5:
            return values

        core = self.core
        horizon = float(self.config.gameplay_horizon)
        values[3] = float(core.game_step) / horizon
        values[4] = float(core.next_event) / NUM_TRADE_EVENTS
        values[5] = float(core.at_event)
        values[6] = float(core.bullets_arrived) / NUM_TRADE_EVENTS
        values[7] = float(core.seller.ammo) / NUM_TRADE_EVENTS
        values[8] = float(core.buyer.ammo) / NUM_TRADE_EVENTS
        values[9] = float(core.transfers) / NUM_TRADE_EVENTS
        values[10] = float(core.payments)
        values[11] = float(core.seller.game_reward)
        values[12] = float(core.buyer.game_reward)
        values[13] = float(core.seller_payoff)
        values[14] = float(core.buyer_payoff)
        values[15:20] = np.asarray(core.event_steps, dtype=np.float32) / horizon
        return values

    def _with_critic_state(self, actor_values, *, phase, event_index):
        result = _copy_mapping(actor_values)
        # Base Atari environments may already define critic-only fields.  They
        # remain neutral; only critic:state carries the outer protocol phase.
        for key, space in self.core.observation_space.spaces.items():
            if key.startswith("critic:") and key not in result:
                result[key] = np.zeros(space.shape, dtype=space.dtype)
        result["critic:state"] = self._critic_state(
            phase=phase, event_index=event_index
        )
        return result

    def _query_observation(self, event_index):
        return self._with_critic_state(
            self._canonical_actor_observations[event_index],
            phase=0.0,
            event_index=event_index,
        )

    def canonical_reward_observation(self, event_index):
        """Diagnostic copy of the actor-visible deployment observation."""
        return _copy_mapping(self._canonical_actor_observations[event_index])

    def _trade_observation(self, event_index):
        return self._with_critic_state(
            self._canonical_actor_observations[event_index],
            phase=1.0,
            event_index=event_index,
        )

    def _gameplay_observation(self):
        base = self.core.observation_for(self.leader_role)
        result = _copy_mapping(base)
        result["event_active"] = np.array([0.0], dtype=np.float32)
        result["event_one_hot"] = np.zeros(
            NUM_TRADE_EVENTS, dtype=np.float32
        )
        result["opponent_context"] = np.zeros(
            NUM_TRADE_EVENTS, dtype=np.float32
        )
        return self._with_critic_state(
            result,
            phase=1.0,
            event_index=min(self.core.next_event, NUM_TRADE_EVENTS - 1),
        )

    def _emit(self, observation):
        self._last_observation = _copy_mapping(observation)
        return _copy_mapping(observation)

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
            raise ValueError("leader action must be [game_action, economic_action]")
        return np.array([
            np.clip(values[0], 0.0, self.game_action_count - 1),
            np.clip(values[1], 0.0, 1.0),
        ], dtype=np.float32)

    def _start_reward_game(self):
        self.core.reset(seed=int(self.rng.integers(0, 2 ** 31 - 1)))
        self._reward_started = True
        if self.core.at_event:
            self.core.prepare_event()
        return self._next_reward_observation()

    def _next_reward_observation(self):
        if self.core.done:
            return self._gameplay_observation()
        if self.core.at_event:
            return self._trade_observation(self.core.next_event)
        return self._gameplay_observation()

    def _query_step(self, action):
        current_actor_observation = actor_observation(self._last_observation)
        self._query_observations.append(current_actor_observation)
        self._query_actions.append(np.array(action, copy=True))
        completed_index = self.query_index
        self.query_index += 1
        self.query_transitions += 1
        if self.query_index < NUM_TRADE_EVENTS:
            next_observation = self._query_observation(self.query_index)
            reward_game_started = False
        else:
            self.query_trace = LeaderQueryTrace.capture(
                self._query_observations, self._query_actions
            )
            # The exact trace is authoritative.  The fixed-template encoder
            # proves that the legacy five-scalar projection is lossless for
            # this restricted canonical-query experiment.
            self._trace_encoder.encode(self.query_trace)
            next_observation = self._start_reward_game()
            reward_game_started = True
        info = {
            "is_query": True,
            "is_reward_phase": False,
            "substep_type": "query",
            "query_index": int(completed_index),
            "query_count": int(self.query_index),
            "reward_game_started": reward_game_started,
            "exclude_from_buffer": False,
        }
        return self._emit(next_observation), 0.0, False, info

    def _follower_trade_observation(self, event_index):
        if self.query_trace is None:
            raise RuntimeError("meta-follower requested before all five queries")
        side = (
            self.core.buyer
            if self.follower_role == BUYER
            else self.core.seller
        )
        actor_values = _copy_mapping(
            self._canonical_actor_observations[event_index]
        )
        if "ammo_fraction" in actor_values:
            actor_values["ammo_fraction"] = np.array(
                [float(side.ammo) / NUM_TRADE_EVENTS], dtype=np.float32
            )
        self._trace_encoder.encode(self.query_trace)
        actor_values["opponent_context"] = (
            legacy_economic_action_projection(self.query_trace)
            .astype(np.float32, copy=False)
        )
        result = _copy_mapping(actor_values)
        for key, space in self.core.observation_space.spaces.items():
            if key.startswith("critic:") and key not in result:
                result[key] = np.zeros(space.shape, dtype=space.dtype)
        # Preserve the exact observation schema of the already-trained meta
        # follower.  Its critic is irrelevant at deterministic inference, but
        # SB3 still validates this 12-entry field's shape.
        follower_critic_state = np.zeros(CRITIC_STATE_DIM, dtype=np.float32)
        follower_critic_state[:9] = (
            float(self.core.game_step) / self.config.gameplay_horizon,
            float(NUM_TRADE_EVENTS - event_index) / NUM_TRADE_EVENTS,
            float(self.core.seller.ammo) / NUM_TRADE_EVENTS,
            float(self.core.buyer.ammo) / NUM_TRADE_EVENTS,
            float(self.core.transfers) / NUM_TRADE_EVENTS,
            float(self.core.payments) / NUM_TRADE_EVENTS,
            float(self.core.seller.game_reward) / NUM_TRADE_EVENTS,
            float(self.core.buyer.game_reward) / NUM_TRADE_EVENTS,
            float(event_index) / float(NUM_TRADE_EVENTS - 1),
        )
        result["critic:state"] = follower_critic_state
        return result

    def _predict_follower(self, observation):
        """Invoke a full-trace adapter or the verified legacy projection path."""
        if self.query_trace is None:
            raise RuntimeError("meta-follower requested before all five queries")
        predict_from_trace = getattr(
            self.response_model, "predict_from_query_trace", None
        )
        if predict_from_trace is not None:
            return predict_from_trace(
                observation,
                self.query_trace,
                deterministic=True,
            )
        # Existing frozen followers consume opponent_context.  That field was
        # derived only after validating the complete exact trace above.
        return self.response_model.predict(observation, deterministic=True)

    @staticmethod
    def _actor_observations_equal(left, right):
        if tuple(left.keys()) != tuple(right.keys()):
            return False
        return all(np.array_equal(left[key], right[key]) for key in left)

    def _validate_trade_cache_hit(self, event_index, action):
        if self.query_trace is None:
            raise RuntimeError("reward trade reached before trace finalization")
        trace_observation, trace_action = self.query_trace.reconstruct()[
            event_index
        ]
        observed = actor_observation(self._last_observation)
        if not self._actor_observations_equal(
                observed, trace_observation
        ):
            raise RuntimeError(
                "reward trade observation differs from canonical query "
                f"observation for event {event_index}"
            )
        if not np.allclose(
                action, trace_action, rtol=0.0, atol=1.0e-7
        ):
            raise RuntimeError(
                "leader policy cache miss at reward trade event "
                f"{event_index}: received {action.tolist()}, expected "
                f"{trace_action.tolist()}"
            )
        self.cache_hits += 1
        return np.array(trace_action, copy=True)

    def _leader_reward(self):
        if self.leader_role == SELLER:
            return float(self.core.seller_payoff)
        return float(self.core.buyer_payoff)

    def _trade_step(self, action):
        event_index = int(self.core.next_event)
        cached_action = self._validate_trade_cache_hit(event_index, action)
        follower_observation = self._follower_trade_observation(event_index)
        response_action, _ = self._predict_follower(follower_observation)
        follower_values = np.asarray(
            response_action, dtype=np.float32
        ).reshape(-1)
        if follower_values.shape != (2,):
            raise RuntimeError(
                "meta-follower must return [game_action, economic_action]"
            )
        follower_values = np.array([
            np.clip(follower_values[0], 0.0, self.game_action_count - 1),
            np.clip(follower_values[1], 0.0, 1.0),
        ], dtype=np.float32)
        self.follower_full_actions[event_index] = follower_values

        leader_economic_action = float(cached_action[1])
        follower_economic_action = float(follower_values[1])
        if self.leader_role == SELLER:
            price = leader_economic_action
            threshold = follower_economic_action
        else:
            price = follower_economic_action
            threshold = leader_economic_action

        previous_payoff = self._leader_reward()
        seller_steps_before = getattr(self.core.seller, "step_calls", None)
        buyer_steps_before = getattr(self.core.buyer, "step_calls", None)
        event = self.core.trade(price=price, threshold=threshold)
        seller_steps_after = getattr(self.core.seller, "step_calls", None)
        buyer_steps_after = getattr(self.core.buyer, "step_calls", None)
        if (
                seller_steps_before is not None
                and seller_steps_before != seller_steps_after
        ) or (
                buyer_steps_before is not None
                and buyer_steps_before != buyer_steps_after
        ):
            raise RuntimeError("an Atari emulator advanced on a trade substep")
        reward = self._leader_reward() - previous_payoff
        self.trade_transitions += 1
        info = self._transition_info("trade", reward)
        info.update({
            "event_index": event_index,
            "trade_event": dict(event),
            "cache_hit": True,
            "leader_executed_action": tuple(float(x) for x in cached_action),
            "follower_action": tuple(float(x) for x in follower_values),
            "emulator_advanced": False,
        })
        return self._emit(self._next_reward_observation()), reward, False, info

    def _gameplay_step(self, action):
        previous_payoff = self._leader_reward()
        transition = self.core.step_gameplay(
            leader_role=self.leader_role,
            leader_game_action=int(np.rint(action[0])),
        )
        reward = self._leader_reward() - previous_payoff
        self.gameplay_transitions += 1
        self._done = bool(self.core.done)
        if self._done:
            if self.core.next_event != NUM_TRADE_EVENTS:
                raise RuntimeError("fixed horizon ended before all five events")
            self.core.assert_accounting()
        info = self._transition_info("gameplay", reward)
        info.update({
            "gameplay": dict(transition),
            "leader_executed_game_action": int(np.rint(action[0])),
            "emulator_advanced": True,
        })
        if self._done:
            info.update(self._episode_info())
        return (
            self._emit(self._next_reward_observation()),
            reward,
            self._done,
            info,
        )

    def _transition_info(self, substep_type, reward):
        return {
            "is_query": False,
            "is_reward_phase": True,
            "substep_type": substep_type,
            "exclude_from_buffer": False,
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
            raise RuntimeError("terminal episode is missing its leader query trace")
        query_actions = self.query_trace.full_action_matrix()
        return {
            "leader_role": self.leader_role,
            "follower_role": self.follower_role,
            "query_actions": tuple(
                tuple(float(x) for x in action)
                for action in query_actions
            ),
            "query_trace_sha256": self.query_trace.sha256,
            "follower_actions": tuple(
                tuple(float(x) for x in action)
                for action in self.follower_full_actions
            ),
            "event_steps": tuple(int(x) for x in core.event_steps),
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
            "trade_opportunities": int(core.next_event),
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
            raise RuntimeError("step called after outer episode termination")
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


# A descriptive alias for call sites that prefer the word "stepwise".
StepwiseStackPOMDPAtariLeaderEnv = FullTraceStackPOMDPAtariLeaderEnv


__all__ = [
    "FULL_LEADER_CRITIC_STATE_DIM",
    "FullTraceStackPOMDPAtariLeaderEnv",
    "StepwiseDualAtariTradeCore",
    "StepwiseStackPOMDPAtariLeaderEnv",
    "actor_observation",
]
