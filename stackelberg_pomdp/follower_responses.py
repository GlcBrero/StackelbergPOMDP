import numpy as np
from itertools import product
import pickle

from stackelberg_pomdp.utils import check_empirical_bcce_gap


def round_down_mw_response_games(requested, games_per_update):
    """Keep only complete joint-action sweeps in a fixed MW response prefix."""
    requested, games_per_update = int(requested), int(games_per_update)
    if games_per_update < 1:
        raise ValueError("An MW update must contain at least one follower game")
    aligned = requested - requested % games_per_update
    if aligned < games_per_update:
        raise ValueError(
            f"tot_num_response_episodes={requested} is shorter than one MW "
            f"update period ({games_per_update})."
        )
    return aligned


class MultiplicativeWeightsResponse:
    """Follower response process based on multiplicative weights.

    For each sampled private-type profile, one MW iteration enumerates every
    joint follower-action profile.  The resulting payoffs are integrated
    exactly against the opponents' current mixed strategies before updating
    the corresponding type/action rows.  Thus the update has no sampled
    reference-action noise; only the private-type sequence is sampled.
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
        self.action_profiles = list(product(*[
            range(self.followers_action_space[follower].n)
            for follower in self.followers_list
        ]))

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
            self.iteration_complete = False
            self.action_profile_idx = 0

        action_profile = self.action_profiles[self.action_profile_idx]
        self.reference_actions = {
            follower: int(action)
            for follower, action in zip(self.followers_list, action_profile)
        }
        # Preserve the fixed-size critic interface. These fields now identify
        # a deterministic joint-profile query rather than a sampled deviation.
        self.deviation_follower_idx = (
            self.action_profile_idx % len(self.followers_list)
        )
        query_follower = self.followers_list[self.deviation_follower_idx]
        self.deviation_action_idx = self.reference_actions[query_follower]
        return dict(self.reference_actions)

    def reward_actions(self, observations):
        actions = {}
        for follower_idx, follower in enumerate(self.followers_list):
            obs = observations[follower]
            actions[follower] = int(np.argmax(self.weights[follower_idx][obs]))
        return actions

    def observe_response_result(self, observations, info):
        self._accumulate_expected_utilities(observations, info["utilities"])
        self.action_profile_idx += 1
        self.iteration_complete = self.action_profile_idx >= len(self.action_profiles)

        if not self.iteration_complete:
            return False

        type_profile = tuple(observations[follower] for follower in self.followers_list)
        self.iteration_type_profile_counts[type_profile] = (
            self.iteration_type_profile_counts.get(type_profile, 0) + 1
        )
        self._update_weights(observations)
        self.completed_iterations += 1
        self.deviation_utilities = self._empty_deviation_utilities()
        self._reset_iteration()
        return True

    @property
    def response_games_per_update(self):
        return len(self.action_profiles)

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

    def current_mixed_strategy(self):
        """Return the product strategy represented by the current MW weights."""
        strategy = {}
        for follower, weight in zip(self.followers_list, self.weights):
            strategy[follower] = {
                obs: np.array(weight[obs], dtype=np.float64, copy=True)
                for obs in range(self.followers_observation_space[follower].n)
            }
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
            np.full(
                (
                    self.followers_observation_space[follower].n,
                    self.followers_action_space[follower].n,
                ),
                1.0 / self.followers_action_space[follower].n,
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
        self.action_profile_idx = 0
        self.deviation_follower_idx = 0
        self.deviation_action_idx = 0
        self.iteration_complete = True
        self.reference_actions = {}

    def _accumulate_expected_utilities(self, observations, utilities):
        for follower_idx, follower in enumerate(self.followers_list):
            own_action = self.reference_actions[follower]
            opponents_probability = 1.0
            for opponent_idx, opponent in enumerate(self.followers_list):
                if opponent == follower:
                    continue
                opponents_probability *= self.weights[opponent_idx][
                    observations[opponent]
                ][self.reference_actions[opponent]]
            self.deviation_utilities[follower_idx][own_action] += (
                utilities[follower] * opponents_probability
            )

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


class FiniteFollowerResponse:
    """A finite correlated mixture of product-strategy snapshots.

    The common snapshot index is sampled first, then followers independently
    sample from that snapshot.  Keeping the common index is important: simply
    averaging every follower's marginal strategy would discard the correlation
    generated by the empirical MW trajectory.
    """

    def __init__(
            self,
            name,
            gap,
            strategy_snapshots,
            followers_list,
            followers_action_space,
            completed_iterations,
    ):
        if not strategy_snapshots:
            raise ValueError("A finite follower response needs at least one snapshot.")
        self.name = name
        self.gap = gap
        self.strategy_snapshots = [
            self._copy_strategy(snapshot) for snapshot in strategy_snapshots
        ]
        self.followers_list = list(followers_list)
        self.followers_action_space = followers_action_space
        self.completed_iterations = int(completed_iterations)

    @staticmethod
    def _copy_strategy(strategy):
        return {
            follower: {
                observation: np.array(probabilities, dtype=np.float64, copy=True)
                for observation, probabilities in follower_strategy.items()
            }
            for follower, follower_strategy in strategy.items()
        }

    def action_distribution(self, observations):
        """Return the exact joint-action distribution conditional on types."""
        probabilities = {}
        snapshot_probability = 1.0 / len(self.strategy_snapshots)
        action_ranges = [
            range(self.followers_action_space[follower].n)
            for follower in self.followers_list
        ]

        for snapshot in self.strategy_snapshots:
            for action_profile in product(*action_ranges):
                probability = snapshot_probability
                for follower, action in zip(self.followers_list, action_profile):
                    probability *= snapshot[follower][observations[follower]][action]
                if probability > 0:
                    probabilities[action_profile] = (
                        probabilities.get(action_profile, 0.0) + float(probability)
                    )

        total_probability = sum(probabilities.values())
        if total_probability <= 0:
            raise RuntimeError("The selected follower response has empty support.")
        return [
            (
                {
                    follower: int(action)
                    for follower, action in zip(self.followers_list, action_profile)
                },
                probability / total_probability,
            )
            for action_profile, probability in sorted(probabilities.items())
        ]


class CertifiedMWResponse(MultiplicativeWeightsResponse):
    """Multiplicative weights with adaptive, exact response certification.

    All MW-specific policy choices live here: the three response candidates,
    their priority, empirical-history semantics, exact certification, and the
    response distribution used for reward evaluation.  A Gym wrapper only has
    to feed completed response games into this object and ask whether it is
    ready.
    """

    CANDIDATE_PRIORITY = (
        "last_mixed",
        "last_deterministic",
        "empirical_average",
    )

    def __init__(
            self,
            followers_list,
            followers_observation_space,
            followers_action_space,
            rng,
            epsilon=MultiplicativeWeightsResponse.DEFAULT_EPS,
            reset_weights_each_episode=True,
            certification_threshold=None,
            certification_min_records=1,
            certification_check_freq=1,
            certification_max_extra_updates=10000,
            fixed_seed=None,
    ):
        super().__init__(
            followers_list=followers_list,
            followers_observation_space=followers_observation_space,
            followers_action_space=followers_action_space,
            rng=rng,
            epsilon=epsilon,
            reset_weights_each_episode=reset_weights_each_episode,
        )
        self.certification_threshold = certification_threshold
        self.certification_min_records = max(1, int(certification_min_records))
        self.certification_check_freq = max(1, int(certification_check_freq))
        self.certification_max_extra_updates = (
            None
            if certification_max_extra_updates is None
            else max(0, int(certification_max_extra_updates))
        )
        self.fixed_seed = fixed_seed
        self._evaluation_env = None
        self._leader_policy = None
        self._clear_certification_state()

    def reset_episode(self):
        if self.fixed_seed is not None:
            self._rng.seed(self.fixed_seed)
        super().reset_episode()
        self._clear_certification_state()

    def _clear_certification_state(self):
        self.strategy_history = []
        self.last_game_completed_update = False
        self.selected_response = None
        self.certification_requested = False
        self.certification_start_iteration = None
        self.last_checked_iteration = None
        self.last_candidate_gaps = {}
        self.last_response_bcce_gap = None

    def set_evaluation_context(self, env, leader_policy):
        self._evaluation_env = env
        self._leader_policy = leader_policy

    def observe_response_result(self, observations, info):
        completed = super().observe_response_result(observations, info)
        self.last_game_completed_update = completed
        if not completed:
            return False

        self.strategy_history.append(self.current_mixed_strategy())
        if self.certification_requested and self.selected_response is None:
            self._try_certify()
        return True

    def request_completion(self):
        """Start certification at the end of the fixed visible prefix."""
        if self.selected_response is not None:
            return True
        if not self.last_game_completed_update:
            return False

        if not self.certification_requested:
            self.certification_requested = True
            self.certification_start_iteration = self.completed_iterations

        if self.certification_threshold is None:
            self.selected_response = self._make_response(
                "last_deterministic",
                None,
                [self.current_deterministic_strategy()],
            )
            return True

        self._try_certify(force=True)
        return self.selected_response is not None

    def response_ready(self):
        return self.selected_response is not None

    def _candidate_strategies(self):
        candidates = [
            ("last_mixed", [self.current_mixed_strategy()]),
            ("last_deterministic", [self.current_deterministic_strategy()]),
        ]
        if len(self.strategy_history) >= self.certification_min_records:
            candidates.append(("empirical_average", self.strategy_history))
        return candidates

    def _try_certify(self, force=False):
        if self.selected_response is not None:
            return True
        if self.certification_threshold is None:
            return self.request_completion()
        if self._evaluation_env is None or self._leader_policy is None:
            raise RuntimeError(
                "Certified MW needs a leader policy before the response prefix ends."
            )

        extra_updates = self.extra_updates
        at_cap = (
            self.certification_max_extra_updates is not None
            and extra_updates >= self.certification_max_extra_updates
        )
        should_check = (
            force
            or at_cap
            or extra_updates % self.certification_check_freq == 0
        )
        if (
                not should_check
                or self.last_checked_iteration == self.completed_iterations
        ):
            return False

        self.last_checked_iteration = self.completed_iterations
        self.last_candidate_gaps = {}
        for name, strategy in self._candidate_strategies():
            gap = self.compute_bcce_gap(strategy)
            self.last_candidate_gaps[name] = gap
            if gap <= self.certification_threshold:
                self.selected_response = self._make_response(name, gap, strategy)
                self.last_response_bcce_gap = gap
                return True

        self.last_response_bcce_gap = min(self.last_candidate_gaps.values())
        if at_cap:
            raise RuntimeError(
                "MW did not produce a certified response before the safety cap: "
                f"threshold={self.certification_threshold}, "
                f"prefix_updates={self.certification_start_iteration}, "
                f"extra_updates={extra_updates}, "
                f"candidate_gaps={self.last_candidate_gaps}."
            )
        return False

    def compute_bcce_gap(self, response_strategy=None, leader_policy=None):
        policy = leader_policy or self._leader_policy
        if self._evaluation_env is None or policy is None:
            return None
        strategy = response_strategy or self.response_strategy()
        if not strategy:
            return None
        return check_empirical_bcce_gap(
            self._evaluation_env,
            policy,
            strategy,
        )

    @property
    def extra_updates(self):
        if self.certification_start_iteration is None:
            return 0
        return self.completed_iterations - self.certification_start_iteration

    @property
    def certified(self):
        if self.certification_threshold is None:
            return self.selected_response is not None
        return (
            self.selected_response is not None
            and self.selected_response.gap is not None
            and self.selected_response.gap <= self.certification_threshold
        )

    def _make_response(self, name, gap, strategy):
        return FiniteFollowerResponse(
            name=name,
            gap=gap,
            strategy_snapshots=strategy,
            followers_list=self.followers_list,
            followers_action_space=self.followers_action_space,
            completed_iterations=self.completed_iterations,
        )

    def response_strategy(self):
        if self.selected_response is not None:
            return self.selected_response.strategy_snapshots
        return [self.current_deterministic_strategy()]

    def reward_scenarios(self, type_profiles):
        """Expand type profiles into exact type-by-action reward scenarios."""
        if self.selected_response is None:
            raise RuntimeError("Cannot evaluate reward before selecting a response.")

        scenarios = []
        for profile in type_profiles:
            for actions, probability in self.selected_response.action_distribution(
                    profile["types"]
            ):
                scenario = dict(profile)
                scenario["types"] = dict(profile["types"])
                scenario["follower_actions"] = actions
                scenario["response_action_weight"] = probability
                scenario["response_candidate"] = self.selected_response.name
                scenarios.append(scenario)
        return scenarios

    def max_action_profiles(self):
        count = 1
        for follower in self.followers_list:
            count *= self.followers_action_space[follower].n
        return count

    def phase_info(self):
        diagnostics = {
            "response_bcce_certified": self.certified,
            "response_bcce_gap": self.last_response_bcce_gap,
            "response_bcce_threshold": self.certification_threshold,
            "response_candidate": (
                self.selected_response.name
                if self.selected_response is not None
                else None
            ),
            "response_candidate_gaps": dict(self.last_candidate_gaps),
            "response_bcce_records": len(self.response_strategy()),
            "response_prefix_updates": self.certification_start_iteration,
            "response_extra_updates": self.extra_updates,
        }
        return {
            "response_assessment": {
                "certified": self.certified,
                "diagnostics": diagnostics,
            },
            **diagnostics,
        }


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
