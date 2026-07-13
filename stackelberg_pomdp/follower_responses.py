import numpy as np
from itertools import product
import pickle


class MultiplicativeWeightsResponse:
    """Follower response process based on multiplicative weights.

    A response episode is a sequence of MW iterations. Each iteration samples a
    reference action profile for the current type profile, tests every
    one-follower deviation, and updates the corresponding type/action weights.
    The paper experiments use the deterministic projection of the final weights:
    each follower type sends the message with highest final weight. This keeps
    the learned follower response interpretable and avoids reporting randomized
    follower strategies.
    """

    DEFAULT_EPS = 0.1

    def __init__(
            self,
            followers_list,
            followers_observation_space,
            followers_action_space,
            rng,
            epsilon=DEFAULT_EPS,
            reset_weights_each_episode=True,
    ):
        self.followers_list = list(followers_list)
        self.followers_observation_space = followers_observation_space
        self.followers_action_space = followers_action_space
        self._rng = rng
        self.epsilon = epsilon
        self.reset_weights_each_episode = reset_weights_each_episode

        self.weights = self._initial_weights()
        self.deviation_utilities = self._empty_deviation_utilities()
        self._reset_iteration()

    def reset_episode(self):
        if self.reset_weights_each_episode:
            self.weights = self._initial_weights()
        self.deviation_utilities = self._empty_deviation_utilities()
        self.completed_iterations = 0
        self.iteration_type_profile_counts = {}
        self._reset_iteration()

    def response_actions(self, observations):
        if self.iteration_complete:
            self.reference_actions = self._sample_actions_from_current_weights(observations)
            self.iteration_complete = False

        follower = self.followers_list[self.deviation_follower_idx]
        actions = self.reference_actions.copy()
        actions[follower] = self.deviation_action_idx
        return actions

    def reward_actions(self, observations):
        actions = {}
        for follower_idx, follower in enumerate(self.followers_list):
            obs = observations[follower]
            actions[follower] = int(np.argmax(self.weights[follower_idx][obs]))
        return actions

    def observe_response_result(self, observations, info):
        follower = self.followers_list[self.deviation_follower_idx]
        self.deviation_utilities[
            self.deviation_follower_idx
        ][self.deviation_action_idx] = info["utilities"][follower]
        self._advance_deviation_action()

        if not self.iteration_complete:
            return False

        type_profile = tuple(observations[follower] for follower in self.followers_list)
        self.iteration_type_profile_counts[type_profile] = (
            self.iteration_type_profile_counts.get(type_profile, 0) + 1
        )
        self._update_weights(observations)
        self.completed_iterations += 1
        self.deviation_utilities = self._empty_deviation_utilities()
        return True

    def response_strategy(self):
        return [self.current_deterministic_strategy()]

    def current_deterministic_strategy(self):
        strategy = {}
        for follower, weight in zip(self.followers_list, self.weights):
            follower_strategy = {}
            for obs in range(self.followers_observation_space[follower].n):
                probs = np.zeros(self.followers_action_space[follower].n, dtype=np.float64)
                probs[int(np.argmax(weight[obs]))] = 1.0
                follower_strategy[obs] = probs
            strategy[follower] = follower_strategy
        return strategy

    def weights_to_norm_vec(self):
        weights = np.empty((0))
        for follower_idx in range(len(self.followers_list)):
            weights_vec = self.weights[follower_idx].flatten()
            max_abs = max(abs(weights_vec))
            if max_abs > 0:
                weights_vec = weights_vec / max_abs
            weights = np.append(weights, weights_vec)
        return weights

    def _initial_weights(self):
        return [
            np.ones(
                (
                    self.followers_observation_space[follower].n,
                    self.followers_action_space[follower].n,
                ),
                dtype=np.float64,
            )
            for follower in self.followers_list
        ]

    def _empty_deviation_utilities(self):
        return [
            [0 for _ in range(self.followers_action_space[follower].n)]
            for follower in self.followers_list
        ]

    def _reset_iteration(self):
        self.deviation_follower_idx = 0
        self.deviation_action_idx = 0
        self.iteration_complete = True
        self.reference_actions = {}

    def _advance_deviation_action(self):
        follower = self.followers_list[self.deviation_follower_idx]
        self.deviation_action_idx += 1
        if self.deviation_action_idx < self.followers_action_space[follower].n:
            return

        if self.deviation_follower_idx < len(self.followers_list) - 1:
            self.deviation_follower_idx += 1
            self.deviation_action_idx = 0
            return

        self._reset_iteration()

    def _sample_actions_from_current_weights(self, observations):
        actions = {}
        for follower_idx, follower in enumerate(self.followers_list):
            actions[follower] = self._rng.choices(
                range(self.followers_action_space[follower].n),
                self.weights[follower_idx][observations[follower]],
            )[0]
        return actions

    def _update_weights(self, observations):
        for follower_idx, follower in enumerate(self.followers_list):
            obs = observations[follower]
            for action in range(self.followers_action_space[follower].n):
                self.weights[follower_idx][obs][action] *= (
                    (1 + self.epsilon) ** self.deviation_utilities[follower_idx][action]
                )
            self.weights[follower_idx][obs] = self._normalized_row(
                self.weights[follower_idx][obs]
            )

    @staticmethod
    def _normalized_row(row):
        row = row.astype(np.float64)
        row_sum = row.sum()
        if not np.isfinite(row_sum) or row_sum <= 0:
            return np.ones_like(row) / len(row)
        return row / row_sum


class QLearningResponse:
    """Independent follower Q-learning response process."""

    def __init__(
            self,
            followers_list,
            followers_action_space,
            rng,
            make_q_entry,
            alpha=0.15,
            delta=0.95,
            beta=0.00001,
            warm_start_q=False,
            q_tables_path=None,
    ):
        self.followers_list = list(followers_list)
        self.followers_action_space = followers_action_space
        self._rng = rng
        self._make_q_entry = make_q_entry
        self.alpha = alpha
        self.delta = delta
        self.beta = beta
        self.warm_start_q = warm_start_q
        self.q_tables_path = q_tables_path
        self.num_followers = len(self.followers_list)
        self.n_actions = self.followers_action_space[self.followers_list[0]].n

        self.q_tables = {}
        self._q_init()
        self.step_counter = 0
        self.steps_since_restart = [0] * self.num_followers

    def reset_episode(self):
        if self.warm_start_q and self.q_tables:
            pass
        elif self.q_tables_path:
            self._q_init()
            self.load_q_tables(self.q_tables_path)
        else:
            self._q_init()
        self.step_counter = 0
        self.steps_since_restart = [0] * self.num_followers

    def begin_step(self):
        self.step_counter += 1
        for idx in range(self.num_followers):
            self.steps_since_restart[idx] += 1

    def actions(self, observation, mode):
        actions = {}
        for idx, agent in enumerate(self.followers_list):
            self._ensure_observation(agent, observation)
            q_vals = self.q_tables[agent][observation]
            if mode == "random":
                actions[agent] = self._rng.randint(0, self.n_actions - 1)
            elif mode == "argmax":
                actions[agent] = int(np.argmax(q_vals))
            else:
                eps = np.exp(-self.beta * self.steps_since_restart[idx])
                if self._rng.random() < eps:
                    actions[agent] = self._rng.randint(0, self.n_actions - 1)
                else:
                    actions[agent] = int(np.argmax(q_vals))
        return actions

    def observe_transition(
            self,
            current_observation,
            executed_actions,
            rewards,
            next_observation,
            learning_enabled,
    ):
        if not learning_enabled:
            return

        for agent_idx, agent in enumerate(self.followers_list):
            self._ensure_observation(agent, next_observation)
            action = executed_actions[agent]
            prev_q = self.q_tables[agent][current_observation]
            new_val = (
                (1 - self.alpha) * prev_q[action]
                + self.alpha * (
                    rewards[agent]
                    + self.delta * np.max(self.q_tables[agent][next_observation])
                )
            )
            prev_q[action] = new_val
            self._update_flat_entry(agent_idx, current_observation, action, new_val)
        self._q_cache_dirty = True

    def load_q_tables(self, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        for agent in self.followers_list:
            if agent in state["q_tables"]:
                self.q_tables[agent] = state["q_tables"][agent]
        self._rebuild_flat_cache()

    def q_matrices_to_norm_vec(self):
        if not self._q_cache_dirty and self._q_norm_cache is not None:
            return self._q_norm_cache
        entries_per_agent = self._q_flat_entries_per_agent
        result = np.empty_like(self._q_flat)
        for idx in range(len(self.followers_list)):
            start = idx * entries_per_agent
            end = start + entries_per_agent
            chunk = self._q_flat[start:end]
            max_abs = np.max(np.abs(chunk))
            result[start:end] = chunk / max_abs if max_abs > 0 else chunk
        self._q_norm_cache = result
        self._q_cache_dirty = False
        return result

    def _q_init(self):
        self.q_tables = {agent: {} for agent in self.followers_list}
        action_ranges = [
            range(self.followers_action_space[agent].n)
            for agent in self.followers_list
        ]
        for combo in product(*action_ranges):
            key = tuple(combo)
            for agent in self.followers_list:
                self.q_tables[agent][key] = self._make_q_entry()
        self._rebuild_flat_cache()

    def _rebuild_flat_cache(self):
        self._q_keys_ordered = list(self.q_tables[self.followers_list[0]].keys())
        self._key_to_idx = {
            key: idx for idx, key in enumerate(self._q_keys_ordered)
        }
        n_keys = len(self._q_keys_ordered)
        self._q_flat_entries_per_agent = n_keys * self.n_actions
        self._q_flat = np.empty(
            len(self.followers_list) * self._q_flat_entries_per_agent
        )
        offset = 0
        for agent in self.followers_list:
            for key in self._q_keys_ordered:
                self._q_flat[offset:offset + self.n_actions] = self.q_tables[agent][key]
                offset += self.n_actions
        self._q_cache_dirty = True
        self._q_norm_cache = None

    def _update_flat_entry(self, agent_idx, observation, action, value):
        key_idx = self._key_to_idx.get(observation)
        if key_idx is None:
            return
        offset = (
            agent_idx * self._q_flat_entries_per_agent
            + key_idx * self.n_actions
            + action
        )
        self._q_flat[offset] = value

    def _ensure_observation(self, agent, observation):
        if observation in self.q_tables[agent]:
            return
        self.q_tables[agent][observation] = self._make_q_entry()


class RoundRobinResponse:
    """Try each common follower action once, then exploit the best one."""

    def __init__(self, followers_list, n_actions):
        self.followers_list = list(followers_list)
        self.n_actions = n_actions
        self.reset_episode()

    def reset_episode(self):
        self.action_idx = 0
        self.profits = np.zeros(self.n_actions)
        self.best_profit = 0.0
        self.best_action = 0

    def response_actions(self):
        return {agent: self.action_idx for agent in self.followers_list}

    def reward_actions(self):
        return {agent: self.best_action for agent in self.followers_list}

    def observe_response_result(self, rewards):
        self.profits[self.action_idx] = sum(rewards.get(agent, 0) for agent in self.followers_list)
        self.action_idx += 1
        self.best_profit = float(np.max(self.profits[:self.action_idx]))
        self.best_action = int(np.argmax(self.profits[:self.action_idx]))

    def response_complete(self):
        return self.action_idx >= self.n_actions

    def next_actions(self):
        return self.reward_actions() if self.response_complete() else self.response_actions()
