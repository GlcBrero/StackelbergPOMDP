from collections import OrderedDict
import gym
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np

from stackelberg_pomdp.follower_responses import (
    MultiplicativeWeightsResponse,
    QLearningResponse,
    RoundRobinResponse,
)
from stackelberg_pomdp.gym_envs.envs.base_envs import BertrandCompetitionEnv
from stackelberg_pomdp.utils import (
    check_empirical_bcce_gap,
    get_all_wrappers,
)


# Follower-response wrappers reduce the multi-agent base game to a leader POMDP.
class FollowerWrapper(gym.Wrapper):
    """Base class for wrappers that operate follower responses."""

    follower_state_kind = None

    def __init__(self, env):
        super().__init__(env)
        self.this_step_mode = "response"

    def set_step_mode(self, mode):
        old_mode = self.this_step_mode
        self.this_step_mode = mode
        if old_mode != mode:
            self.on_step_mode_changed(mode)

    def on_step_mode_changed(self, mode):
        return

    def reward_phase_length(self, default_length):
        return self.env.unwrapped.reward_phase_length(default_length)

    def critic_observation_spaces(self):
        return OrderedDict()

    def critic_observation(self):
        return OrderedDict()

    def response_strategy(self):
        return []


class MWFollowersWrapper(FollowerWrapper):
    """Gym glue for multiplicative-weights follower responses."""

    follower_state_kind = "mw"
    DEFAULT_EPS = MultiplicativeWeightsResponse.DEFAULT_EPS

    def __init__(
            self,
            env,
            epsilon=DEFAULT_EPS,
            reset_weights_each_episode=True,
    ):

        super().__init__(env)

        self.epsilon = epsilon
        self.reset_weights_each_episode = reset_weights_each_episode

        self.step_counter = 0
        self._rng = self.env.unwrapped._rng
        self.response = MultiplicativeWeightsResponse(
            followers_list=self.followers_list,
            followers_observation_space=self.followers_observation_space,
            followers_action_space=self.followers_action_space,
            rng=self._rng,
            epsilon=self.epsilon,
            reset_weights_each_episode=self.reset_weights_each_episode,
        )

    @property
    def weights(self):
        return self.response.weights

    def _to_leader_obs(self):
        return self.env.leader_observation()

    def current_leader_observation(self):
        return OrderedDict({
            "base_environment": self.env.leader_observation(getattr(self, "current_actions", None))
        })

    def current_follower_actions(self):
        return dict(getattr(self, "current_actions", {}))

    def on_step_mode_changed(self, mode):
        if mode == "reward" and hasattr(self, "followers_obs"):
            self.env.unwrapped.start_reward_phase()
            self._prepare_next_sub_env()

    def reset(self):

        self.response.reset_episode()

        obs = self.env.reset()
        self.followers_obs = {a: obs[a] for a in self.followers_list}

        self.env.freeze_types = self.this_step_mode == "response"
        self.current_actions = self._next_follower_actions(self.followers_obs)

        return OrderedDict({"base_environment": self._to_leader_obs()})

    def _prepare_next_sub_env(self):
        obs = self.env.reset()
        self.followers_obs = {a: obs[a] for a in self.followers_list}

        self.env.freeze_types = self.this_step_mode == "response"
        self.current_actions = self._next_follower_actions(self.followers_obs)

    def step(self, action):
        self.step_counter += 1
        follower_actions = dict(self.current_actions)

        _, reward, done, info = self.env.step(
            self._actions_for_base_env(action, follower_actions)
        )

        if done:
            info["followers_actions"] = follower_actions
            self._finish_subepisode(reward, info)

        info["reward_generated"] = done
        reward = info.get('surplus', reward)
        return OrderedDict({"base_environment": self._to_leader_obs()}), reward, False, info

    def _actions_for_base_env(self, leader_action, follower_actions):
        actions = dict(follower_actions)
        actions[self.env.leader] = leader_action
        return actions

    def _finish_subepisode(self, reward, info):
        if self.this_step_mode == "response":
            self._finish_response_subepisode(info)
            self._prepare_next_response_subepisode()
            return

        self._prepare_next_sampled_reward_subepisode()

    def _finish_response_subepisode(self, info):
        info["response_record_generated"] = self.response.observe_response_result(
            self.followers_obs,
            info,
        )

    def _prepare_next_response_subepisode(self):
        # Freeze the same type profile until the current MW iteration finishes.
        if self.response.iteration_complete:
            self.env.freeze_types = False
        self._prepare_next_sub_env()

    def _prepare_next_sampled_reward_subepisode(self):
        self.env.unwrapped.advance_reward_phase_profile()
        self.env.freeze_types = False
        self._prepare_next_sub_env()

    def _next_follower_actions(self, followers_observations):

        if self.this_step_mode == "reward":
            # Reward phase uses the deterministic projection of final MW weights.
            self.current_actions = self.response.reward_actions(followers_observations)
            return self.current_actions

        return self.response.response_actions(followers_observations)

    def weights_to_norm_vec(self):
        return self.response.weights_to_norm_vec()

    def critic_observation_spaces(self):
        num_weights = sum(len(weight.flatten()) for weight in self.weights)
        return OrderedDict({
            "critic:weights": Box(low=-1.0, high=1.0, shape=(num_weights,)),
        })

    def critic_observation(self):
        return OrderedDict({
            "critic:weights": self.weights_to_norm_vec(),
        })

    def response_strategy(self):
        return self.response.response_strategy()

    def log_info(self, info):
        self.logger.record("weights", str(self.response.weights))
        self.env.log_info(info)


"""Q-learning follower wrapper."""
class QLearningFollowersWrapper(FollowerWrapper):
    follower_state_kind = "qlearning"

    def __init__(
            self,
            env,
            alpha=0.15,
            delta=0.95,
            beta=0.00001,
            leader_action_space=None,
            warm_start_q=False,
            q_tables_path=None,
    ):
        super().__init__(env)

        self.alpha = alpha
        self.delta = delta
        self.beta = beta
        self.warm_start_q = warm_start_q
        self._q_tables_path = q_tables_path
        self._rng = self.env.unwrapped._rng  # Share base env's RNG for reproducibility

        self.num_followers = len(self.env.followers_list)
        self.n_follower_actions = self.env.followers_action_space[self.env.followers_list[0]].n
        self.response = QLearningResponse(
            followers_list=self.env.followers_list,
            followers_action_space=self.env.followers_action_space,
            rng=self._rng,
            make_q_entry=self._make_q_entry,
            alpha=self.alpha,
            delta=self.delta,
            beta=self.beta,
            warm_start_q=self.warm_start_q,
            q_tables_path=self._q_tables_path,
        )

        # Leader observation: null (no_observation). Subclass overrides for reactive.
        self.observation_space = Dict({'base_environment': MultiDiscrete([1])})

        # Leader action space
        if leader_action_space is not None:
            self.action_space = leader_action_space
        elif hasattr(self.env, 'platform_intervention') and self.env.platform_intervention == 'learn_binary_threshold':
            self.action_space = Discrete(2)
        elif hasattr(self.env, 'platform_intervention') and self.env.platform_intervention == 'learn_threshold':
            self.action_space = Discrete(self.n_follower_actions)
        else:
            self.action_space = Discrete(self.num_followers)

    @property
    def q_tables(self):
        return self.response.q_tables

    @property
    def step_counter(self):
        return self.response.step_counter

    def load_q_tables(self, path):
        self.response.load_q_tables(path)

    def reset(self):
        self.response.reset_episode()

        obs_sub_env = self.env.reset()
        # Handle both array and scalar observations from base env
        raw_obs = obs_sub_env[self.env.followers_list[0]]
        self.current_obs = np.atleast_1d(np.array(raw_obs, dtype=int))
        self.current_obs_key = tuple(self.current_obs)
        self.current_actions = self._get_follower_actions(self.current_obs_key, "standard")

        return OrderedDict({"base_environment": self._to_leader_obs()})

    def _to_leader_obs(self):
        return np.array([0])

    def current_leader_observation(self):
        return OrderedDict({
            "base_environment": self.env.leader_observation(getattr(self, "current_actions", None))
        })

    def current_follower_actions(self):
        return dict(getattr(self, "current_actions", {}))

    def on_step_mode_changed(self, mode):
        if hasattr(self, "current_obs_key"):
            action_mode = "standard" if mode == "response" else (
                "argmax" if mode == "reward" else mode
            )
            self.current_actions = self._get_follower_actions(self.current_obs_key, action_mode)

    def _execute_step(self, action):
        """Execute one step: apply leader action, step base env with all actions, update Q-tables."""
        self.response.begin_step()

        # Build full actions dict: leader + all followers
        executed_actions = dict(self.current_actions)
        all_actions = dict(executed_actions)
        all_actions[self.env.leader] = action
        obs_sub_env, rewards, sub_done, info = self.env.step(all_actions)
        # rewards is a dict {agent: reward} for Bertrand, or scalar for old games
        if not isinstance(rewards, dict):
            rewards = info.get('utilities', {})

        next_obs_sub_env = self.env.reset() if sub_done else obs_sub_env
        raw_next_obs = next_obs_sub_env[self.env.followers_list[0]]
        next_obs = np.atleast_1d(np.array(raw_next_obs, dtype=int))[:self.num_followers]
        next_obs_key = tuple(next_obs)

        self.response.observe_transition(
            current_observation=self.current_obs_key,
            executed_actions=executed_actions,
            rewards=rewards,
            next_observation=next_obs_key,
            learning_enabled=(
                info.get("reward_generated", sub_done)
                and self.this_step_mode in ('response', 'standard')
            ),
        )

        self.current_obs = next_obs
        self.current_obs_key = next_obs_key

        # Pick next follower actions
        mode = "standard" if self.this_step_mode == 'response' else (
            "argmax" if self.this_step_mode == 'reward' else self.this_step_mode
        )
        self.current_actions = self._get_follower_actions(self.current_obs_key, mode)

        reward = info.get('surplus', 0)
        if "reward_generated" not in info:
            info["reward_generated"] = sub_done
        info["utilities"] = {self.env.leader: reward, **rewards}
        info["followers_actions"] = executed_actions
        info["reward_pricing_agents"] = rewards

        return reward, info

    def step(self, action):
        """Basic (no_observation): leader acts simultaneously with followers."""
        reward, info = self._execute_step(action)
        return OrderedDict({"base_environment": self._to_leader_obs()}), reward, False, info

    def _get_follower_actions(self, observation, action_type):
        return self.response.actions(observation, action_type)

    def _make_q_entry(self):
        base = self.env.unwrapped
        if hasattr(base, 'compute_q_init_entry'):
            return base.compute_q_init_entry(self.delta)
        return np.zeros(self.n_follower_actions)

    def q_matrices_to_norm_vec(self):
        return self.response.q_matrices_to_norm_vec()

    def critic_observation_spaces(self):
        num_q_entries = sum(
            len(np.array(list(self.q_tables[follower].values())).flatten())
            for follower in self.q_tables.keys()
        )
        return OrderedDict({
            "critic:exploration_rates": Box(
                low=0,
                high=1.0,
                shape=(len(self.env.followers_list),),
            ),
            "critic:Q_matrices": Box(low=-1.0, high=1.0, shape=(num_q_entries,)),
        })

    def critic_observation(self):
        exp_rate = np.exp(-1 * self.beta * self.step_counter)
        return OrderedDict({
            "critic:Q_matrices": self.q_matrices_to_norm_vec(),
            "critic:exploration_rates": np.full(len(self.env.followers_list), exp_rate),
        })

"""Round-robin (monopolist) follower wrapper.

Tries each price in sequence during the response phase, picks the most profitable
for the reward phase. All followers play the same price at each step.

Response phase = m steps (one per price). Reward phase = as configured.
Game-agnostic interface: step(leader_action) → (obs, reward, done, info).
"""
class RoundRobinFollowersWrapper(FollowerWrapper):
    follower_state_kind = "roundrobin"

    def __init__(self, env):
        super().__init__(env)
        self.num_followers = len(env.followers_list)
        self.n_actions = env.followers_action_space[env.followers_list[0]].n
        self.response = RoundRobinResponse(env.followers_list, self.n_actions)

        # Leader obs = null (no_observation by default; use ReactiveLeaderWrapper for price_profile)
        self.observation_space = Dict({'base_environment': MultiDiscrete([1])})
        if hasattr(env, 'platform_intervention') and env.platform_intervention == 'learn_binary_threshold':
            self.action_space = Discrete(2)
        else:
            self.action_space = Discrete(self.n_actions)

    @property
    def price_idx(self):
        return self.response.action_idx

    @property
    def profits(self):
        return self.response.profits

    @property
    def best_profit(self):
        return self.response.best_profit

    @property
    def best_price(self):
        return self.response.best_action

    def _to_leader_obs(self):
        return np.array([0])

    def current_leader_observation(self):
        return OrderedDict({
            "base_environment": self.env.leader_observation(getattr(self, "current_actions", None))
        })

    def current_follower_actions(self):
        return dict(getattr(self, "current_actions", {}))

    def on_step_mode_changed(self, mode):
        if mode == "reward":
            self.current_actions = self.response.reward_actions()

    def reset(self):
        self.env.reset()
        self.response.reset_episode()
        self.current_actions = self.response.response_actions()
        return OrderedDict({"base_environment": self._to_leader_obs()})

    def critic_observation_spaces(self):
        return OrderedDict({
            "critic:strategy_idx": Discrete(self.n_actions + 1),
            "critic:best_profit": Box(low=-10.0, high=10.0, shape=(1,)),
        })

    def critic_observation(self):
        return OrderedDict({
            "critic:strategy_idx": min(self.price_idx, self.n_actions),
            "critic:best_profit": np.array([self.best_profit], dtype=np.float32),
        })

    def step(self, action):
        in_response = (
            self.this_step_mode in ('response', 'standard')
            and not self.response.response_complete()
        )
        follower_actions = (
            self.response.response_actions()
            if in_response
            else self.response.reward_actions()
        )
        all_actions = dict(follower_actions)
        all_actions[self.env.leader] = action
        obs, rewards, done, info = self.env.step(all_actions)

        if not isinstance(rewards, dict):
            rewards = info.get('utilities', {})

        if in_response:
            self.response.observe_response_result(rewards)
            self.current_actions = self.response.next_actions()
        else:
            self.current_actions = self.response.reward_actions()

        reward = info.get('surplus', 0)
        info["reward_generated"] = info.get("reward_generated", True)
        info["utilities"] = {self.env.leader: reward, **{a: rewards.get(a, 0) for a in self.env.followers_list}}
        info["followers_actions"] = dict(follower_actions)

        return OrderedDict({"base_environment": self._to_leader_obs()}), reward, False, info


class ReactiveLeaderWrapper(gym.Wrapper):
    """Expose reactive leader observations.

    The follower wrapper below this layer chooses follower actions/messages/prices.
    The base environment supplies the leader-state component as
    ``base_environment`` and converts current follower actions into
    ``base:follower_actions``. Both keys are actor-visible; StackPOMDPWrapper
    appends critic-only keys separately.

    Examples:
    - simple allocation: null state, reported message
    - MSPM: mechanism state, bids
    - Bertrand price profile: previous price profiles, current price profile

    Stack: StackPOMDPWrapper -> ReactiveLeaderWrapper -> FollowerWrapper -> BaseEnv.
    """

    def __init__(self, env):
        super().__init__(env)
        self._base_env = env.env
        self.observation_space = self._base_env.reactive_observation_space()

    def current_leader_observation(self):
        # StackPOMDPWrapper uses this at the response->reward boundary, before
        # taking the first reward-phase step.
        return self._leader_observation()

    def _leader_observation(self):
        if hasattr(self.env, "current_follower_actions"):
            follower_actions = self.env.current_follower_actions()
        else:
            follower_actions = getattr(self.env, "current_actions", None)
        return self._base_env.reactive_leader_observation(follower_actions)

    def reset(self):
        self.env.reset()
        return self._leader_observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._leader_observation(), reward, done, info


# StackPOMDP phase wrapper: response phase, then reward phase.
"""Wrapper for Stackelberg POMDP"""
class StackPOMDPWrapper(gym.Wrapper):

    def __init__(
            self,
            env,
            tot_num_response_episodes=1000,
            tot_num_reward_episodes=10,
            critic_obs="full",
            response_variant="stackelberg",
            response_bcce_threshold=None,
            response_bcce_min_records=1,
            response_bcce_check_freq=1,
    ):

        super(StackPOMDPWrapper, self).__init__(env)

        # This sets the total number of response and reward steps in StackPOMDP.
        self.tot_num_response_episodes = tot_num_response_episodes
        self.tot_num_reward_episodes = tot_num_reward_episodes
        self.critic_obs = critic_obs
        self.response_variant = response_variant
        self.response_bcce_threshold = response_bcce_threshold
        self.response_bcce_min_records = int(response_bcce_min_records)
        self.response_bcce_check_freq = max(1, int(response_bcce_check_freq))
        self.response_bcce_gap = None
        self.last_response_strategy = None
        self.last_response_bcce_gap = None
        self.last_response_stop_reason = None

        self.tot_num_steps = 0
        self.follower_wrapper = self._find_follower_wrapper()
        self.response_leader_policy = None

        self.observation_space = Dict(self._observation_spaces())

    def _observation_spaces(self):
        spaces = OrderedDict(self.env.observation_space.spaces)
        if self.critic_obs in ("flag", "full"):
            spaces["critic:is_reward_step"] = Discrete(2)
        if self.critic_obs == "full":
            spaces.update(self.follower_wrapper.critic_observation_spaces())
        return spaces

    def _find_follower_wrapper(self):
        for wrapper in get_all_wrappers(self.env):
            if isinstance(wrapper, FollowerWrapper):
                return wrapper
        raise ValueError("StackPOMDPWrapper requires a FollowerWrapper below it.")

    def _set_follower_mode(self, mode):
        """Switch follower wrappers between response learning and reward evaluation."""
        self.follower_wrapper.set_step_mode(mode)

    def _enter_response_mode(self):
        self._set_follower_mode("response")

    def _enter_reward_mode(self):
        if not self.reward_phase_started:
            self.reward_phase_started = True
            self.reward_phase_start_counter = self.phase_episode_counter
        self._set_follower_mode("reward")
        self.current_reward_phase_episodes = self.reward_phase_length()

    def reward_phase_length(self):
        return self.follower_wrapper.reward_phase_length(self.tot_num_reward_episodes)

    def rollout_buffer_episode_length(self):
        if self.response_variant == "hidden_queries":
            return self.reward_phase_length()
        return int(self.tot_num_response_episodes) + int(self.reward_phase_length())

    def _response_phase_threshold(self):
        return self.tot_num_response_episodes

    def _episode_done_threshold(self):
        return self.reward_phase_start_counter + self.current_reward_phase_episodes

    def _in_response_phase(self):
        return not self.reward_phase_started

    def _in_reward_phase(self):
        return self.reward_phase_started and self.phase_episode_counter < self._episode_done_threshold()

    def _hide_response_transition_from_buffer(self):
        return self.response_variant == "hidden_queries"

    def _response_reward(self, response_reward):
        if self.response_variant == "reward_during_response":
            return response_reward
        return 0

    def _leader_observation(self, obs):
        full_observation = OrderedDict(
            (key, obs[key])
            for key in self.env.observation_space.spaces.keys()
        )
        self.augment_observation(full_observation)
        return full_observation

    def _response_phase_done(self):
        return self.phase_episode_counter >= self._response_phase_threshold()

    def set_response_leader_policy(self, leader_policy):
        self.response_leader_policy = leader_policy

    def _response_strategy_for_diagnostic(self):
        return self.follower_wrapper.response_strategy()

    def _compute_response_bcce_gap(self):
        if self.response_bcce_threshold is None:
            return None
        if self.response_leader_policy is None:
            return None

        response_strategy = self._response_strategy_for_diagnostic()
        if len(response_strategy) < self.response_bcce_min_records:
            return None

        gap = check_empirical_bcce_gap(self, self.response_leader_policy, response_strategy)
        self.response_bcce_gap = gap
        return gap

    def _reward_phase_done(self):
        return self.phase_episode_counter >= self._episode_done_threshold()

    def _count_generated_reward(self, info):
        if info.get("reward_generated"):
            self.phase_episode_counter += 1

    def _step_response_phase(self, action):
        self._enter_response_mode()
        obs, response_reward, done, info = self.env.step(action)

        self._count_generated_reward(info)
        response_phase_done = self._response_phase_done()
        if response_phase_done:
            self.last_response_strategy = list(self._response_strategy_for_diagnostic())
            self.last_response_bcce_gap = self._compute_response_bcce_gap()
            self.last_response_stop_reason = "fixed_response_phase"
            if self.last_response_bcce_gap is not None:
                info["response_bcce_gap"] = self.last_response_bcce_gap
                info["response_bcce_records"] = len(self.last_response_strategy)
            self._enter_reward_mode()
            obs = self._current_leader_observation()

        reward = self._response_reward(response_reward)
        info["exclude_from_buffer"] = self._hide_response_transition_from_buffer()
        info["is_reward_phase"] = False
        info["response_phase_done"] = response_phase_done
        info["response_phase_stop_reason"] = self.last_response_stop_reason if response_phase_done else None
        info["reward"] = reward

        return self._leader_observation(obs), reward, False, info

    def _step_reward_phase(self, action):
        self._enter_reward_mode()
        obs, reward, done, info = self.env.step(action)

        self._count_generated_reward(info)
        done = self._reward_phase_done()
        full_observation = OrderedDict(
            (key, obs[key])
            for key in self.env.observation_space.spaces.keys()
        )
        self.augment_observation(full_observation, is_reward_step=1)

        info["exclude_from_buffer"] = False
        info["leader_action"] = action
        info["is_reward_phase"] = True
        info["tot_num_reward_steps"] = self.current_reward_phase_episodes
        info["count_steps"] = self.tot_num_steps
        info["reward"] = reward

        return full_observation, reward, done, info

    def _current_leader_observation(self):
        if hasattr(self.env, "current_leader_observation"):
            return self.env.current_leader_observation()
        raise ValueError("StackPOMDPWrapper requires current_leader_observation() on the wrapped leader env.")


    def reset(self):

        self._enter_response_mode()
        self.phase_episode_counter = 0
        self.current_reward_phase_episodes = self.tot_num_reward_episodes
        self.reward_phase_started = False
        self.reward_phase_start_counter = 0

        obs_sub_env = self.env.reset() # Restart sub_env

        full_observation = OrderedDict(
            (key, obs_sub_env[key])
            for key in self.env.observation_space.spaces.keys()
        )
        self.augment_observation(full_observation)
        return full_observation


    def augment_observation(self, observation, is_reward_step=0):
        if self.critic_obs == "flag" or self.critic_obs == "full":
            observation["critic:is_reward_step"] = is_reward_step
        if self.critic_obs == "full":
            observation.update(self.follower_wrapper.critic_observation())


    def step(self, action):

        if not hasattr(self, "is_eval"):
            self.tot_num_steps += 1

        if self._in_response_phase():
            return self._step_response_phase(action)

        elif self._in_reward_phase():
            return self._step_reward_phase(action)

        raise RuntimeError("StackPOMDPWrapper.step called after the episode is done.")


# Top-level wrappers operate above StackPOMDPWrapper.
"""Detects cycles in reward-phase states and normalizes reward.

During the reward phase, tracks base_environment observations. When a state
repeats (cycle detected), delivers the average per-step reward over one
complete cycle and zeros out all other reward steps. Keeps episode length
fixed for RL buffer alignment.

StackPOMDPWrapper divides each reward step by tot_num_reward_episodes. This
wrapper undoes that normalization when accumulating, computes the cycle
average on the raw rewards, then delivers it as the total episode reward on
the cycle-detection step. All other reward steps (before and after the cycle)
are zeroed out.

Works for any StackPOMDP setting with discrete/hashable observations. For
continuous observations, pass a custom state_key_fn that rounds appropriately.

If no cycle is detected within the reward phase, falls back to delivering the
average reward over all observed reward steps on the last step.
"""
class StationaryCycleRewardWrapper(gym.Wrapper):

    def __init__(self, env, state_key_fn=None):
        super().__init__(env)
        self._state_key_fn = state_key_fn or self._default_state_key

    @staticmethod
    def _default_state_key(obs):
        if isinstance(obs, dict):
            obs = obs['base_environment']
        return tuple(np.atleast_1d(obs).flatten())

    def reset(self):
        self._reward_states = []
        self._reward_accumulator = []  # Raw (un-normalized) rewards
        self._cycle_detected = False
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if info.get("is_reward_phase", False):
            state_key = self._state_key_fn(obs)

            if self._cycle_detected:
                reward = 0.0
            elif state_key in self._reward_states:
                # Cycle detected — deliver average reward over one cycle
                cycle_start = self._reward_states.index(state_key)
                cycle_rewards = self._reward_accumulator[cycle_start:]
                reward = sum(cycle_rewards) / len(cycle_rewards)
                info['consumer_surplus'] = reward
                self._cycle_detected = True
            else:
                # Accumulate, zero out
                self._reward_states.append(state_key)
                self._reward_accumulator.append(reward)
                reward = 0.0

            # Fallback: last step, no cycle detected
            if done and not self._cycle_detected and self._reward_accumulator:
                reward = sum(self._reward_accumulator) / len(self._reward_accumulator)
                info['consumer_surplus'] = reward

        return obs, reward, done, info


"""Openness evaluation wrapper.

After the normal episode ends, enters evaluation mode: cycles through all m^2
follower price pairs, presents each as an observation to the leader, records
the leader's threshold response, and computes the average exclusion rate.

The penalty (lambda * avg_exclusion) is subtracted from the episode reward.
This measures how restrictive the leader's POLICY is across all possible
price profiles, not just at the realized prices.

Sits above StationaryCycleRewardWrapper:
  OpennessEvaluationWrapper → StationaryCycleRewardWrapper → StackPOMDP → ...
"""
class OpennessEvaluationWrapper(gym.Wrapper):

    def __init__(self, env, intervention_lambda=0.0, m=4):
        super().__init__(env)
        self.intervention_lambda = intervention_lambda
        self.m = m
        self._eval_mode = False
        self._eval_idx = 0
        # Check if leader can see price profiles (obs size > 1)
        base_shape = env.observation_space['base_environment'].shape
        self._is_state_based = base_shape and base_shape[0] > 1
        # Detect leader_k from observation size: 2 entries = k=1, 4 entries = k=2
        self._leader_k = base_shape[0] // 2 if self._is_state_based else 1
        # Total eval pairs: m^2 for k=1, m^4 for k=2
        self._n_eval = m ** (2 * self._leader_k)
        # Detect binary action space
        self._binary_actions = (env.action_space.n == 2)

    def _make_eval_obs(self, idx):
        """Build evaluation observation for eval index."""
        obs = OrderedDict(self._held_obs)
        if self._is_state_based:
            if self._leader_k == 1:
                i = idx // self.m
                j = idx % self.m
                obs["base_environment"] = np.array([i, j])
            else:
                # k=2: decode idx into (p0_prev, p1_prev, p0_curr, p1_curr)
                p1c = idx % self.m
                p0c = (idx // self.m) % self.m
                p1p = (idx // (self.m ** 2)) % self.m
                p0p = (idx // (self.m ** 3)) % self.m
                obs["base_environment"] = np.array([p0p, p1p, p0c, p1c])
        return obs

    def reset(self):
        self._eval_mode = False
        self._eval_idx = 0
        self._held_reward = 0
        self._held_obs = None
        self._held_info = {}
        self._exclusions = []
        self._episode_reward = 0  # Accumulate total reward across episode
        return self.env.reset()

    def _current_prices_from_idx(self, idx):
        """Extract current price indices (i, j) from eval index."""
        if self._leader_k == 1:
            return idx // self.m, idx % self.m
        else:
            # k=2: current prices are the last two components
            p0c = (idx // self.m) % self.m
            p1c = idx % self.m
            return p0c, p1c

    def step(self, action):
        if self._eval_mode:
            # Evaluation step: action is the leader's threshold for this test observation
            i, j = self._current_prices_from_idx(self._eval_idx)
            if self._binary_actions:
                # Binary: action 0 = open (threshold=m-1), action 1 = close (threshold=0)
                effective_threshold = 0 if action == 1 else (self.m - 1)
            else:
                effective_threshold = action
            n_excluded = (1 if i > effective_threshold else 0) + (1 if j > effective_threshold else 0)
            self._exclusions.append(n_excluded / 2.0)
            self._eval_idx += 1

            if self._eval_idx >= self._n_eval:
                # All pairs evaluated — deliver penalized reward
                avg_exclusion = np.mean(self._exclusions)
                penalty = self.intervention_lambda * avg_exclusion
                self._eval_mode = False
                info = dict(self._held_info)
                info['avg_exclusion'] = avg_exclusion
                info['consumer_surplus'] = self._held_reward
                info['penalized_surplus'] = self._held_reward - penalty
                return self._held_obs, self._held_reward - penalty, True, info
            else:
                # Return next test observation
                return self._make_eval_obs(self._eval_idx), 0, False, {}

        # Normal step
        obs, reward, done, info = self.env.step(action)
        self._episode_reward += reward

        if done and self.intervention_lambda > 0:
            # Enter round-robin evaluation mode for both obs types
            self._eval_mode = True
            self._eval_idx = 0
            self._held_reward = self._episode_reward  # Total episode reward (pure CS)
            self._held_obs = obs
            self._held_info = info
            self._exclusions = []
            return self._make_eval_obs(0), 0, False, {}

        return obs, reward, done, info


"""Top-level logging wrapper.

Sits at the top of the wrapper stack. At done=True, reads the info dict
(populated by all layers below) and writes everything to the CSV logger.
No other layer should log — they only write to info.
"""
class LoggingWrapper(gym.Wrapper):

    def __init__(
            self,
            env,
            logger=None,
            log_regular_done=True,
    ):
        super().__init__(env)
        self.logger = logger
        self.log_regular_done = log_regular_done

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if done and self.logger:
            if not self.log_regular_done:
                return obs, reward, done, info

            # Base env logging (domain-specific: prices, c_i, etc.)
            self.env.unwrapped.log_info(info)

            # Pure consumer surplus (set by StationaryCycleRewardWrapper or OpennessWrapper)
            cs = info.get("consumer_surplus", info.get("surplus", 0))
            self.logger.record("consumer_surplus", cs)

            # Leader reward = penalized surplus if available, else pure CS
            lr = info.get("penalized_surplus", cs)
            self.logger.record("leader_reward", lr)

            # Step count
            self.logger.record("count_steps", info.get("count_steps", 0))

            # Pure consumer surplus (from OpennessEvaluationWrapper or base env)
            cs = info.get("consumer_surplus", info.get("surplus", 0))
            self.logger.record("consumer_surplus", cs)

            # Penalized surplus and exclusion (from OpennessEvaluationWrapper)
            if "penalized_surplus" in info:
                self.logger.record("penalized_surplus", info["penalized_surplus"])
            if "avg_exclusion" in info:
                self.logger.record("avg_exclusion", info["avg_exclusion"])

            # Follower info
            if "utilities" in info:
                for agent, util in info["utilities"].items():
                    if agent != self.env.unwrapped.leader:
                        self.logger.record(f"{agent}_reward", util)
            if "followers_actions" in info:
                for agent, act in info["followers_actions"].items():
                    self.logger.record(f"{agent}_action", act)

            self.logger.dump(info.get("count_steps", 0))

        return obs, reward, done, info
