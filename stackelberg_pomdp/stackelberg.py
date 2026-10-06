"""Execute experiment-supplied phases against one episode commitment.

Phases yield LeaderQuery or Reward events and receive actions through ``send``.
Lazy execution reuses the first action at each observation; standard execution
exposes every query through Gym. Neither phase holds a neural-policy reference.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import gym
import numpy as np

from .episode import EpisodeLogger, RewardAccumulator
from .observations import CriticObservation


class _ObservationKey:
    """Sentinel requesting the canonical actor-observation key."""


OBSERVATION_KEY = _ObservationKey()


def observation_key(observation):
    """Hash actor inputs only; privileged critic state never changes a key."""
    if isinstance(observation, dict):
        return tuple((key, observation_key(value))
                     for key, value in observation.items()
                     if not key.startswith("critic:"))
    array = np.asarray(observation)
    if array.dtype.kind in "fc":
        if not np.isfinite(array).all():
            raise ValueError("Leader observations must be finite")
        values = tuple(array.ravel().tolist())
    else:
        values = tuple(array.ravel().tolist())
    return array.shape, values


@dataclass(frozen=True)
class LeaderQuery:
    observation: Any
    # None denotes an ordinary fresh policy draw (e.g. Atari gameplay).
    # Otherwise an explicit domain key, or the actor observation by default.
    key: Any = OBSERVATION_KEY


@dataclass(frozen=True)
class Reward:
    value: float
    elapsed_steps: int = 1
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseReady:
    """Selected follower profile, with optional experiment-specific evaluation state.

    The default reward executor accepts a PolicyProfile and initializes a fresh
    game. Custom evaluators may use an exact representation of that profile
    (for example a joint distribution) and an explicit evaluation_context.
    Certificates and diagnostics are separate from the selected strategies.
    """

    followers: Any
    certificate: Optional[dict] = None
    info: dict = field(default_factory=dict)
    evaluation_context: Any = None


@dataclass(frozen=True)
class RewardDone:
    observation: Any = None
    info: dict = field(default_factory=dict)


class ResponsePhase(ABC):
    """An experiment's complete follower-response procedure."""

    allow_rewards = False  # overridden only by the paper reward-timing ablation

    @abstractmethod
    def run(self, game):
        """Yield leader queries and return ResponseReady for this episode."""
        raise NotImplementedError


class RewardPhase(ABC):
    """An experiment's post-response evaluation procedure."""

    @abstractmethod
    def run(self, game, response_result):
        """Yield leader queries/rewards and return RewardDone."""
        raise NotImplementedError


class POMG(ABC):
    """Multi-agent game; it contains domain rules, not follower policies.

    Define leader, horizon, action_spaces, observation_spaces, and
    leader_observation_space. reset() draws a fresh game initial state;
    step(actions) takes a joint action and returns observations, rewards,
    done, info. Observations and rewards are dictionaries keyed by agent.
    leader_observation(history) encodes only the leader's own game history
    for the neural policy. Use the same encoding in response queries.
    """

    leader = "leader"

    @property
    def action_space(self):
        return self.action_spaces[self.leader]

    @property
    def observation_space(self):
        return self.leader_observation_space

    @abstractmethod
    def reset(self):
        """Return initial observations; discard any response simulation state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, actions):
        """Apply joint actions and return observations, rewards, done, info."""
        raise NotImplementedError

    @abstractmethod
    def leader_observation(self, history):
        """Encode a tuple of leader observations, excluding response history."""
        raise NotImplementedError


class PolicyProfile:
    """A fixed collection of policies acting on their own observation histories.

    Each policy implements act(history). Learning has finished; action sampling
    may remain stochastic. A correlated profile can override act(histories) to
    preserve its common randomization instead of sampling independent marginals.
    Returned actions must use only each agent's information and shared randomness.
    """

    def __init__(self, policies):
        self.policies = dict(policies)

    def act(self, histories):
        return {agent: policy.act(histories[agent])
                for agent, policy in self.policies.items()}


class GameplayReward(RewardPhase):
    """Fresh finite-horizon play of a POMG against the returned follower profile."""

    def run(self, game, response_result):
        if response_result.evaluation_context is not None:
            raise ValueError("Response-dependent evaluation state needs a custom RewardPhase")
        if not isinstance(game, POMG):
            raise TypeError("Default gameplay requires a POMG with joint actions")
        if not isinstance(game.horizon, int) or game.horizon < 1:
            raise ValueError("The game horizon must be a positive integer")
        followers = response_result.followers
        if not callable(getattr(followers, "act", None)):
            raise TypeError("The returned follower profile must implement act(histories)")
        observations = game.reset()
        histories = {agent: (deepcopy(observations[agent]),)
                     for agent in game.action_spaces}
        follower_ids = set(game.action_spaces) - {game.leader}
        for _ in range(game.horizon):
            # The response's game state and histories never enter fresh gameplay.
            actions = followers.act(deepcopy({agent: histories[agent]
                                             for agent in follower_ids}))
            if set(actions) != follower_ids:
                raise ValueError("The follower profile must return one action per follower")
            actions = dict(actions)
            observation = game.leader_observation(histories[game.leader])
            query = (game.leader_query(observation)
                     if hasattr(game, "leader_query") else LeaderQuery(observation))
            actions[game.leader] = yield query
            observations, rewards, done, info = game.step(actions)
            for agent in histories:
                histories[agent] += (deepcopy(observations[agent]),)
            yield Reward(float(rewards[game.leader]), info=info)
            if done:
                break
        return RewardDone(game.leader_observation(histories[game.leader]), info)


class Commitment:
    """Private, write-once actions. A new instance is made at outer reset."""

    def __init__(self):
        self._actions = {}
        self._observations = {}

    def key(self, query):
        return (observation_key(query.observation)
                if isinstance(query.key, _ObservationKey) else query.key)

    def lookup(self, query):
        key = self.key(query)
        if key is None or key not in self._actions:
            return False, None
        if self._observations[key] != observation_key(query.observation):
            raise ValueError("One commitment key was used for different actor observations")
        return True, deepcopy(self._actions[key])

    def record(self, query, action):
        key = self.key(query)
        if key is None:
            return
        if key in self._actions:
            raise RuntimeError("A committed action cannot be overwritten")
        self._observations[key] = observation_key(query.observation)
        self._actions[key] = deepcopy(action)

    def __len__(self):
        return len(self._actions)

    def snapshot(self):
        """Copy for diagnostics, never an actor input or a mutable cache view."""
        return deepcopy(self._actions)


class StackelbergPOMDPEnv(gym.Env):
    """Run a PI response, then evaluate its selected follower strategies.

    Both phases ask through LeaderQuery. With lazy=True, the episode commitment
    answers repeated queries; otherwise every query requests a fresh action.
    _episode defines the two phases; _run_until_action_needed executes them.
    Logging, critic observations and reward accumulation have separate helpers.
    """

    phase_execution = True

    def __init__(self, game, response, reward=None, *, lazy=True, gamma=1.0,
                 add_phase_observation=True):
        if not isinstance(lazy, bool):
            raise TypeError("lazy must be True (cached actions) or False (fresh actions)")
        if not isinstance(response, ResponsePhase):
            raise TypeError("response must implement ResponsePhase")
        self.game, self.response, self.lazy = game, response, lazy
        self.reward_phase = GameplayReward() if reward is None else reward
        if not isinstance(self.reward_phase, RewardPhase):
            raise TypeError("reward must implement RewardPhase")
        self.gamma = float(gamma)
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        self.add_phase_observation = bool(add_phase_observation)
        self.observations = CriticObservation(game, response, add_phase=self.add_phase_observation)
        self.action_space = game.action_space
        self.observation_space = self.observations.space
        self.logger = EpisodeLogger()
        self.commitment = Commitment()
        self.phase = "done"
        self._program = None
        self._pending = None
        self._last_observation = None
        self.response_result = None

    def _episode(self):
        """Select follower strategies, then evaluate the leader against them."""
        self.phase = "response"
        self.response_result = yield from self.response.run(self.game)
        if not isinstance(self.response_result, ResponseReady):
            raise TypeError("ResponsePhase must return ResponseReady")
        self.logger.response(self.response_result)

        self.phase = "reward"
        result = yield from self.reward_phase.run(self.game, self.response_result)
        if not isinstance(result, RewardDone):
            raise TypeError("RewardPhase must return RewardDone")
        return result

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.seed(seed)
        if self._program is not None:
            self._program.close()
        self.commitment = Commitment()
        self._pending = None
        self.response_result = None
        self.logger.reset()
        self._program = self._episode()
        observation, reward, done, _ = self._run_until_action_needed()
        if done or reward != 0:
            raise RuntimeError("An outer episode must request a leader action before returning reward")
        return observation

    def step(self, action):
        """Answer the pending query and continue the response or reward phase."""
        if self._pending is None or self.phase == "done":
            raise RuntimeError("Call reset() before stepping a completed episode")
        action_phase = self.phase
        if self.lazy:
            self.commitment.record(self._pending, action)
        self._pending = None
        observation, reward, done, info = self._run_until_action_needed(deepcopy(action))
        info["action_phase"] = action_phase
        return observation, reward, done, info

    def _run_until_action_needed(self, action=None):
        """Execute internal work until an unanswered query or the episode end."""
        rewards = RewardAccumulator(self.gamma)
        while True:
            try:
                event = self._program.send(action)
            except StopIteration as completed:
                return self._finish(completed.value, rewards)
            action = None

            if isinstance(event, LeaderQuery):
                found, cached = self.commitment.lookup(event) if self.lazy else (False, None)
                self.logger.query(self.phase, cached=found)
                if found:
                    action = cached
                    continue
                self._pending = event
                self._last_observation = self.observations.encode(
                    event.observation, self.commitment, self.phase)
                return self._last_observation, rewards.total, False, {**rewards.info, "phase": self.phase}

            if not isinstance(event, Reward):
                raise TypeError("A phase must yield LeaderQuery or Reward events")
            if self.phase != "reward" and event.value != 0 and not self.response.allow_rewards:
                raise ValueError("Response interactions cannot generate leader reward")
            rewards.add(event)
            self.logger.reward(event)

    def _finish(self, result, rewards):
        terminal = (self._last_observation if result.observation is None else
                    self.observations.encode(result.observation, self.commitment, self.phase))
        self.phase = "done"
        info = self.logger.finish(result.info, commitment_entries=len(self.commitment),
                                  transition_info=rewards.info)
        return terminal, rewards.total, True, info

    @property
    def episode_count(self):
        return self.logger.episode_count

    @property
    def last_episode(self):
        return self.logger.last_episode

    def seed(self, seed=None):
        if hasattr(self.game, "seed"):
            self.game.seed(seed)
        if hasattr(self.response, "seed"):
            self.response.seed(seed)
        return [seed]

    def close(self):
        if self._program is not None:
            self._program.close()
        if hasattr(self.game, "close"):
            self.game.close()
