import inspect
import random
import unittest
from unittest.mock import Mock, patch

import gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from stackelberg_pomdp.env_setups import get_mspm_env, get_spm_env
from stackelberg_pomdp.policies.generic import CustomPolicy
from stackelberg_pomdp.callbacks import (
    BackgroundEvalCallback,
    ExactSPMEvaluationCallback,
    _wandb_log,
)
from stackelberg_pomdp.experiments.common import finalized_config
from stackelberg_pomdp.experiments.mspm import build_parser
from stackelberg_pomdp.follower_responses import (
    CertifiedMWResponse,
    FiniteFollowerResponse,
    MultiplicativeWeightsResponse,
)
from stackelberg_pomdp.games import PISetting
from stackelberg_pomdp.envs.spm import BaseMessageSPM
from stackelberg_pomdp.wrappers.core import (
    ExpectedResponseRewardWrapper,
    MWFollowersWrapper,
    StackPOMDPWrapper,
)
from stackelberg_pomdp.evaluation import BaselinePolicyWrapper
from stackelberg_pomdp.rl_trainer_setup import (
    get_custom_training_algorithm,
    get_observation_split,
)
from stackelberg_pomdp.run_setups import _experiment_name
from stackelberg_pomdp.utils import get_all_wrappers


class _RecordingLogger:
    def __init__(self):
        self.values = {}

    def record(self, key, value):
        self.values[key] = value


class _FixedActionPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.deterministic_flags = []

    def predict(self, observation, deterministic=False):
        self.deterministic_flags.append(deterministic)
        return self.action.copy(), None


class _FixedActionModel(_FixedActionPolicy):
    def __init__(self, action):
        super().__init__(action)
        self.logger = _RecordingLogger()


class _EfficientPIPolicy:
    """Efficient two-message PI mechanism used for response smoke tests."""

    def get_action(self, observation):
        state = np.asarray(observation["base_environment"]).reshape(-1)
        messages = np.asarray(
            observation["base:follower_actions"]
        ).reshape(-1).astype(int)
        remaining = state[:2] > 0.5
        reported_values = np.asarray([
            (0.2, 1.0)[messages[0]],
            (0.0, 0.4)[messages[1]],
        ])
        chosen = int(np.argmax(np.where(remaining, reported_values, -np.inf)))
        scores = np.zeros(2, dtype=np.float32)
        scores[chosen] = 1.0
        price = (
            max(0.0, reported_values[1 - chosen] - 1e-3)
            if remaining.sum() == 2
            else 0.0
        )
        return np.asarray([scores[0], scores[1], price], dtype=np.float32)


class MSPMPIRegressionTests(unittest.TestCase):
    def test_wandb_logs_only_paper_evaluation_metrics(self):
        fake_wandb = Mock()
        fake_wandb.run = object()
        with patch("stackelberg_pomdp.callbacks.wandb", fake_wandb):
            _wandb_log(
                {
                    "reward": -0.04,
                    "best_reward": 0.0,
                    "response_candidate": "last_mixed",
                    "response_extra_updates": 3.0,
                    "raw_reward": -0.04,
                    "unwanted_sb3_metric": 99.0,
                },
                step=1234,
            )

        fake_wandb.log.assert_called_once_with({
            "reward": -0.04,
            "best_reward": 0.0,
            "response_candidate": "last_mixed",
            "response_extra_updates": 3.0,
            "global_step": 1234,
        })

    def test_bcce_assessment_is_owned_by_mw_not_stackpomdp(self):
        config = {
            "logger": None,
            "seed": 1,
            "setting": "PI",
            "num_types": 2,
            "num_messages": 2,
            "followers_algorithm": "MW",
            "tot_num_response_episodes": 4,
            "tot_num_reward_episodes": 4,
            "critic_obs": "full",
            "pomdp_mode": "stackelberg",
            "response_bcce_threshold": 0.05,
        }
        env = get_mspm_env(config)
        wrappers = get_all_wrappers(env)
        stack_env = next(
            wrapper for wrapper in wrappers
            if type(wrapper) == StackPOMDPWrapper
        )
        mw_env = next(
            wrapper for wrapper in wrappers
            if type(wrapper) == MWFollowersWrapper
        )

        self.assertNotIn("bcce", inspect.getsource(StackPOMDPWrapper).lower())
        self.assertIn("bcce", inspect.getsource(CertifiedMWResponse).lower())
        self.assertNotIn(
            "check_empirical_bcce_gap",
            inspect.getsource(MWFollowersWrapper),
        )
        self.assertNotIn(
            "empirical_average",
            inspect.getsource(MWFollowersWrapper),
        )
        self.assertFalse(any(
            "bcce" in attribute.lower()
            for attribute in stack_env.__dict__
        ))
        self.assertEqual(mw_env.response_bcce_threshold, 0.05)
        self.assertEqual(mw_env.fixed_seed, 0)
        self.assertTrue(any(
            type(wrapper) == ExpectedResponseRewardWrapper
            for wrapper in wrappers
        ))

    def test_certified_mspm_has_fixed_mw_seed_and_identifies_the_run(self):
        base_args = [
            "--setting", "PI",
            "--num_types", "2",
            "--num_messages", "2",
        ]
        default_config = finalized_config(
            build_parser().parse_args(base_args),
            "mspm:PI:2:2",
        )
        fixed_config = finalized_config(
            build_parser().parse_args(base_args + ["--mw_fixed_seed", "17"]),
            "mspm:PI:2:2",
        )

        self.assertEqual(default_config["mw_fixed_seed"], 0)
        self.assertEqual(fixed_config["mw_fixed_seed"], 17)
        self.assertIn("mwseed0", _experiment_name(default_config))
        self.assertIn("mwseed17", _experiment_name(fixed_config))
        self.assertEqual(default_config["response_bcce_threshold"], 0.05)
        self.assertNotEqual(
            _experiment_name(default_config),
            _experiment_name(dict(default_config, response_bcce_threshold=None)),
        )

    def test_experiment_name_distinguishes_entropy_coefficients(self):
        args = build_parser().parse_args([
            "--setting", "PI",
            "--num_types", "2",
            "--num_messages", "2",
        ])
        config = finalized_config(args, "mspm:PI:2:2")

        ent0_name = _experiment_name(dict(config, ent_coef=0.0))
        ent001_name = _experiment_name(dict(config, ent_coef=0.01))

        self.assertNotEqual(ent0_name, ent001_name)
        self.assertIn("ent0", ent0_name)
        self.assertIn("ent0.01", ent001_name)

    def test_fixed_mw_seed_repeats_the_complete_response_and_reward(self):
        config = {
            "logger": None,
            "seed": 1,
            "setting": "PI",
            "num_types": 2,
            "num_messages": 2,
            "followers_algorithm": "MW",
            "mw_fixed_seed": 17,
            "tot_num_response_episodes": 40,
            "tot_num_reward_episodes": 4,
            "critic_obs": "full",
            "pomdp_mode": "stackelberg",
            "response_bcce_threshold": 0.05,
        }
        action = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        env = get_mspm_env(config)
        env.set_response_leader_policy(
            BaselinePolicyWrapper(
                _FixedActionPolicy(action),
                env,
                deterministic=True,
            )
        )

        snapshots = []
        with patch(
            "stackelberg_pomdp.follower_responses.check_empirical_bcce_gap",
            return_value=0.04,
        ):
            for _ in range(2):
                env.reset()
                done = False
                episode_return = 0.0
                while not done:
                    _, reward, done, _ = env.step(action)
                    episode_return += float(reward)

                strategy = [
                    {
                        follower: {
                            observation: np.array(probabilities, copy=True)
                            for observation, probabilities in follower_strategy.items()
                        }
                        for follower, follower_strategy in strategy_snapshot.items()
                    }
                    for strategy_snapshot in env.last_response_strategy
                ]
                snapshots.append({
                    "return": episode_return,
                    "gap": env.last_response_bcce_gap,
                    "certified": env.last_response_bcce_certified,
                    "counts": dict(env.last_response_type_profile_counts),
                    "weights": [
                        np.array(weight, copy=True)
                        for weight in env.last_response_weights
                    ],
                    "strategy": strategy,
                })

        first, second = snapshots
        self.assertEqual(first["return"], second["return"])
        self.assertEqual(first["gap"], second["gap"])
        self.assertEqual(first["certified"], second["certified"])
        self.assertEqual(first["counts"], second["counts"])
        for first_weights, second_weights in zip(first["weights"], second["weights"]):
            np.testing.assert_array_equal(first_weights, second_weights)
        for first_snapshot, second_snapshot in zip(first["strategy"], second["strategy"]):
            self.assertEqual(first_snapshot.keys(), second_snapshot.keys())
            for follower in first_snapshot:
                self.assertEqual(
                    first_snapshot[follower].keys(),
                    second_snapshot[follower].keys(),
                )
                for observation in first_snapshot[follower]:
                    np.testing.assert_array_equal(
                        first_snapshot[follower][observation],
                        second_snapshot[follower][observation],
                    )

    def test_default_ppo_uses_conservative_max_episode_rollout(self):
        args = build_parser().parse_args([
            "--setting", "PI",
            "--num_types", "2",
            "--num_messages", "2",
            "--tot_num_response_episodes", "100",
            "--tot_num_reward_episodes", "4",
            "--eval_freq", "0",
        ])
        config = finalized_config(args, "mspm:PI:2:2")
        config["logger"] = None
        env = get_mspm_env(config)
        model = get_custom_training_algorithm(config, env)

        self.assertEqual(env.max_episode_transitions(), 232)
        self.assertEqual(model.n_steps, 232)
        self.assertEqual(model.batch_size, 64)
        self.assertEqual(model.n_epochs, 10)
        self.assertEqual(model.learning_rate, 3e-4)
        self.assertEqual(model.ent_coef, 0.0)
        self.assertEqual(model.gamma, 1.0)

    def test_exact_profile_order_weights_and_callback_aggregation(self):
        game = PISetting(num_messages=2)
        env = BaseMessageSPM(game=game, seed=1)

        profiles = env._build_reward_phase_profiles()
        observed = [
            (
                tuple(profile["types"][follower] for follower in game.followers_list),
                profile["weight"],
            )
            for profile in profiles
        ]
        self.assertEqual(
            observed,
            [((0, 0), 0.4), ((0, 1), 0.4), ((1, 0), 0.1), ((1, 1), 0.1)],
        )

        # Always visit A0 first at price zero. The exact callback average must
        # equal the probability-weighted raw welfare loss.
        env.start_reward_phase()
        rows = []
        action = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        for _ in profiles:
            env.reset()
            done = False
            while not done:
                _, scaled_reward, done, info = env.step({
                    game.leader: action,
                    game.followers_list[0]: 0,
                    game.followers_list[1]: 0,
                })
            rows.append((
                info["exact_profile_weight"],
                info["unweighted_reward"],
                scaled_reward,
            ))
            env.advance_reward_phase_profile()

        expected_reward = sum(weight * raw for weight, raw, _ in rows)
        exact_phase_sum = sum(scaled for _, _, scaled in rows)
        self.assertAlmostEqual(exact_phase_sum, expected_reward)
        self.assertAlmostEqual(expected_reward, -0.08)

    def test_early_sale_ends_subepisode_and_max_horizon_is_conservative(self):
        config = {
            "logger": None,
            "seed": 1,
            "setting": "PI",
            "num_types": 2,
            "num_messages": 2,
            "followers_algorithm": "MW",
            "tot_num_response_episodes": 4,
            "tot_num_reward_episodes": 4,
            "critic_obs": "full",
            "pomdp_mode": "stackelberg",
        }
        env = get_mspm_env(config)
        self.assertEqual(env.max_episode_transitions(), 16)

        expected_full_critic_keys = {
            "critic:is_reward_step",
            "critic:weights",
            "critic:mw_types",
            "critic:mw_reference_actions",
            "critic:mw_deviation_position",
            "critic:mw_deviation_utilities",
            "critic:mw_iteration_complete",
        }
        self.assertTrue(expected_full_critic_keys.issubset(env.observation_space.spaces))

        initial_observation = env.reset()
        self.assertTrue(expected_full_critic_keys.issubset(initial_observation))
        action = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        done = False
        transitions = 0
        response_boundary = None
        while not done:
            _, _, done, info = env.step(action)
            transitions += 1
            if info.get("response_phase_done"):
                response_boundary = (
                    transitions,
                    info.get("response_updates"),
                )

        # A0 always has positive value and buys immediately at price zero, so
        # all eight generated games end after one transition. The conservative
        # maximum remains sixteen transitions (eight games times two buyers).
        self.assertEqual(transitions, 8)
        self.assertEqual(response_boundary, (4, 1))

        max_horizon_env = get_mspm_env(dict(config, seed=2))
        max_horizon_env.reset()
        done = False
        transitions = 0
        reject_action = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
        while not done:
            _, _, done, _ = max_horizon_env.step(reject_action)
            transitions += 1
        self.assertEqual(transitions, 16)

    def test_predict_path_caches_and_clears_episode_actions(self):
        config = {
            "logger": None,
            "seed": 1,
            "setting": "PI",
            "num_types": 2,
            "num_messages": 2,
            "followers_algorithm": "MW",
            "tot_num_response_episodes": 4,
            "tot_num_reward_episodes": 4,
            "critic_obs": "full",
            "pomdp_mode": "stackelberg",
        }
        env = get_mspm_env(config)
        cutoff_entry, actor_obs_keys = get_observation_split(env)
        policy = CustomPolicy(
            env.observation_space,
            env.action_space,
            lr_schedule=lambda _: 0.0,
            cutoff_entry=cutoff_entry,
            actor_obs_keys=actor_obs_keys,
        )
        policy.fix_policy_actions()
        observation = env.reset()

        first_action, _ = policy.predict(observation, deterministic=False)
        second_action, _ = policy.predict(observation, deterministic=False)
        np.testing.assert_array_equal(first_action, second_action)
        self.assertEqual(len(policy.obs_action_map), 1)

        policy.clear_obs_action_map()
        self.assertEqual(policy.obs_action_map, {})

    def test_spm_exact_evaluation_logs_argmax_expected_reward(self):
        config = {
            "logger": None,
            "seed": 1,
            "setting": "PI",
            "num_types": 2,
            "num_messages": 1,
            "discrete_prices": False,
        }
        eval_env = gym.wrappers.FlattenObservation(get_spm_env(config))
        callback = ExactSPMEvaluationCallback(eval_env, deterministic=True)
        callback.model = _FixedActionModel([1.0, 0.0, 0.0])
        callback.logger = callback.model.logger

        with patch("stackelberg_pomdp.callbacks._wandb_log") as wandb_log:
            callback._log_evaluation(step=123)

        metrics = wandb_log.call_args.args[0]
        self.assertAlmostEqual(metrics["reward"], -0.08)
        self.assertAlmostEqual(metrics["best_reward"], -0.08)
        self.assertEqual(set(metrics), {"reward", "best_reward"})
        self.assertEqual(wandb_log.call_args.kwargs["step"], 123)
        self.assertTrue(callback.model.deterministic_flags)
        self.assertTrue(all(callback.model.deterministic_flags))

    def test_mspm_evaluation_logs_exact_reward_at_captured_step(self):
        config = {
            "logger": None,
            "seed": 1,
            "setting": "PI",
            "num_types": 2,
            "num_messages": 2,
            "followers_algorithm": "MW",
            "tot_num_response_episodes": 4,
            "tot_num_reward_episodes": 4,
            "critic_obs": "full",
            "pomdp_mode": "stackelberg",
        }
        callback = BackgroundEvalCallback(
            Monitor(get_mspm_env(config)),
            eval_freq=1,
            n_eval_episodes=1,
            reward_steps=4,
        )
        callback._eval_policy = _FixedActionPolicy([1.0, 0.0, 0.0])

        with patch("stackelberg_pomdp.callbacks._wandb_log") as wandb_log:
            callback._run_eval(step=456)

        metrics = wandb_log.call_args.args[0]
        self.assertAlmostEqual(metrics["reward"], -0.08)
        self.assertAlmostEqual(metrics["best_reward"], -0.08)
        self.assertEqual(metrics["bcce_certified"], 1.0)
        self.assertEqual(metrics["response_candidate"], "last_deterministic")
        self.assertEqual(metrics["response_candidate_last_mixed"], 0.0)
        self.assertEqual(metrics["response_candidate_last_deterministic"], 1.0)
        self.assertEqual(metrics["response_candidate_empirical_average"], 0.0)
        self.assertEqual(metrics["response_prefix_updates"], 1.0)
        self.assertEqual(metrics["response_extra_updates"], 0.0)
        self.assertEqual(metrics["response_total_updates"], 1.0)
        self.assertEqual(metrics["response_snapshot_count"], 1.0)
        self.assertEqual(wandb_log.call_args.kwargs["step"], 456)
        self.assertTrue(callback._eval_policy.deterministic_flags)
        self.assertTrue(all(callback._eval_policy.deterministic_flags))

    def test_certified_mw_extends_response_then_evaluates_exactly(self):
        config = {
            "logger": None,
            "seed": 1,
            "setting": "PI",
            "num_types": 2,
            "num_messages": 2,
            "followers_algorithm": "MW",
            "tot_num_response_episodes": 7,
            "tot_num_reward_episodes": 4,
            "critic_obs": "full",
            "pomdp_mode": "stackelberg",
            "response_bcce_threshold": 0.05,
            "response_bcce_max_extra_updates": 2,
            "mw_fixed_seed": 0,
        }
        action = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        env = get_mspm_env(config)
        env.set_response_leader_policy(object())
        response_flags = []
        action_phase_flags = []
        reward_weights = []
        response_boundary = None
        episode_return = 0.0

        # Three candidates fail at the fixed prefix. The next complete MW
        # update certifies its last mixed strategy.
        gaps = [0.10, 0.10, 0.10, 0.04]
        with patch(
            "stackelberg_pomdp.follower_responses.check_empirical_bcce_gap",
            side_effect=gaps,
        ):
            observation = env.reset()
            done = False
            while not done:
                action_phase_flags.append(observation["critic:is_reward_step"])
                observation, reward, done, info = env.step(action)
                episode_return += float(reward)
                if not info.get("is_reward_phase", False):
                    response_flags.append(info["exclude_from_buffer"])
                elif info.get("reward_generated", False):
                    reward_weights.append(info["exact_profile_weight"])
                if info.get("response_phase_done", False):
                    response_boundary = dict(info)

        self.assertEqual(response_flags, [False] * 4 + [True] * 4)
        self.assertEqual(action_phase_flags[:8], [0] * 8)
        self.assertEqual(set(action_phase_flags[8:]), {1})
        self.assertEqual(response_boundary["response_updates"], 2)
        self.assertEqual(response_boundary["requested_response_games"], 7)
        self.assertEqual(response_boundary["response_prefix_games"], 4)
        self.assertEqual(response_boundary["response_extra_updates"], 1)
        self.assertEqual(response_boundary["response_candidate"], "last_mixed")
        self.assertTrue(response_boundary["response_bcce_certified"])
        self.assertAlmostEqual(response_boundary["response_bcce_gap"], 0.04)
        self.assertEqual(len(reward_weights), 16)
        self.assertAlmostEqual(sum(reward_weights), 1.0)
        self.assertAlmostEqual(episode_return, -0.08)
        self.assertNotEqual(episode_return, -1.0)
        self.assertEqual(env.max_episode_transitions(), 40)

    def test_certified_mw_safety_cap_fails_loudly_without_penalty(self):
        config = {
            "logger": None,
            "seed": 1,
            "setting": "PI",
            "num_types": 2,
            "num_messages": 2,
            "followers_algorithm": "MW",
            "tot_num_response_episodes": 4,
            "tot_num_reward_episodes": 4,
            "critic_obs": "full",
            "pomdp_mode": "stackelberg",
            "response_bcce_threshold": 0.05,
            "response_bcce_max_extra_updates": 1,
        }
        env = get_mspm_env(config)
        env.set_response_leader_policy(object())
        action = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        with patch(
            "stackelberg_pomdp.follower_responses.check_empirical_bcce_gap",
            return_value=0.10,
        ):
            env.reset()
            with self.assertRaisesRegex(RuntimeError, "safety cap"):
                while True:
                    env.step(action)

    def test_real_certification_queries_preserve_hidden_tail_state(self):
        config = {
            "logger": None,
            "seed": 1,
            "setting": "PI",
            "num_types": 2,
            "num_messages": 2,
            "followers_algorithm": "MW",
            "mw_fixed_seed": 1,
            "tot_num_response_episodes": 20,
            "tot_num_reward_episodes": 4,
            "critic_obs": "full",
            "pomdp_mode": "stackelberg",
            "response_bcce_threshold": 0.05,
            "response_bcce_max_extra_updates": 50,
        }
        policy = _EfficientPIPolicy()
        env = get_mspm_env(config)
        env.set_response_leader_policy(policy)
        observation = env.reset()
        done = False
        episode_return = 0.0
        response_boundary = None
        while not done:
            observation, reward, done, info = env.step(
                policy.get_action(observation)
            )
            episode_return += float(reward)
            if info.get("response_phase_done", False):
                response_boundary = dict(info)

        # Seed 1 needs a genuine hidden extension for this mechanism. Exact
        # payoff queries must not overwrite the live subepisode being extended.
        self.assertGreater(response_boundary["response_extra_updates"], 0)
        self.assertTrue(response_boundary["response_bcce_certified"])
        self.assertAlmostEqual(response_boundary["response_bcce_gap"], 0.0)
        self.assertAlmostEqual(episode_return, 0.0)

    def test_empirical_response_preserves_common_snapshot_correlation(self):
        game = PISetting(num_messages=2)
        base_env = BaseMessageSPM(game=game, seed=1)
        snapshots = []
        for selected_action in (0, 1):
            snapshots.append({
                follower: {
                    observation: np.eye(2)[selected_action]
                    for observation in range(game.num_types)
                }
                for follower in game.followers_list
            })
        response = FiniteFollowerResponse(
            name="empirical_average",
            gap=0.0,
            strategy_snapshots=snapshots,
            followers_list=game.followers_list,
            followers_action_space=base_env.followers_action_space,
            completed_iterations=2,
        )

        distribution = {
            tuple(actions[follower] for follower in game.followers_list): probability
            for actions, probability in response.action_distribution({
                follower: 0 for follower in game.followers_list
            })
        }
        self.assertEqual(set(distribution), {(0, 0), (1, 1)})
        self.assertAlmostEqual(distribution[(0, 0)], 0.5)
        self.assertAlmostEqual(distribution[(1, 1)], 0.5)

    def test_mw_update_enumerates_joint_messages_and_uses_expectations(self):
        followers = ["A0", "A1"]
        observation_spaces = {follower: gym.spaces.Discrete(2) for follower in followers}
        action_spaces = {follower: gym.spaces.Discrete(2) for follower in followers}
        response = MultiplicativeWeightsResponse(
            followers_list=followers,
            followers_observation_space=observation_spaces,
            followers_action_space=action_spaces,
            rng=random.Random(123),
            epsilon=0.1,
        )
        response.reset_episode()
        observations = {follower: 0 for follower in followers}

        queried_profiles = []
        for _ in range(4):
            actions = response.response_actions(observations)
            queried_profiles.append(tuple(actions[follower] for follower in followers))
            completed = response.observe_response_result(
                observations,
                {
                    "utilities": {
                        follower: float(actions[follower])
                        for follower in followers
                    }
                },
            )

        self.assertEqual(
            queried_profiles,
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        self.assertTrue(completed)
        self.assertEqual(response.completed_iterations, 1)
        expected_row = np.asarray([1.0, 1.1]) / 2.1
        for weights in response.weights:
            np.testing.assert_allclose(weights[0], expected_row)
            np.testing.assert_allclose(weights[1], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
