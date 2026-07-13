import unittest

import numpy as np

from stackelberg_pomdp.env_setups import get_mspm_env
from stackelberg_pomdp.baselines_utils import CustomPolicy
from stackelberg_pomdp.experiments.common import finalized_config
from stackelberg_pomdp.experiments.mspm import build_parser
from stackelberg_pomdp.games import PISetting
from stackelberg_pomdp.gym_envs.envs.base_envs import BaseMessageSPM
from stackelberg_pomdp.rl_trainer_setup import (
    get_custom_training_algorithm,
    get_observation_split,
)


class MSPMPIRegressionTests(unittest.TestCase):
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

        self.assertEqual(env.max_episode_transitions(), 208)
        self.assertEqual(model.n_steps, 208)
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


if __name__ == "__main__":
    unittest.main()
