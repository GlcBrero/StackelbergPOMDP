import unittest

from stackelberg_pomdp.rl_trainer_setup import ppo_rollout_geometry


class PPORolloutGeometryTests(unittest.TestCase):
    def geometry(self, response_games, reward_games, pomdp_mode):
        return ppo_rollout_geometry(
            {
                "ppo_rollout_geometry": "historical_ratio_scaled",
                "ppo_episodes_per_batch": 16,
                "tot_num_response_episodes": response_games,
                "tot_num_reward_episodes": reward_games,
                "pomdp_mode": pomdp_mode,
            },
            max_episode_transitions=response_games + reward_games,
        )

    def test_simple_allocation_historical_geometry(self):
        stack = self.geometry(100, 30, "stackelberg")
        basic = self.geometry(100, 30, "hidden_queries")
        self.assertEqual(stack["n_steps"], 131_072)
        self.assertEqual(stack["batch_size"], 64)
        self.assertEqual(basic["n_steps"], 32_768)
        self.assertEqual(basic["batch_size"], 64)
        self.assertEqual(stack["minimum_completed_episodes"], 0)

    def test_matrix_design_historical_geometry(self):
        stack = self.geometry(30, 1, "stackelberg")
        basic = self.geometry(30, 1, "hidden_queries")
        self.assertEqual(stack["n_steps"], 1_015_808)
        self.assertEqual(stack["batch_size"], 64)
        self.assertEqual(basic["n_steps"], 32_768)
        self.assertEqual(basic["batch_size"], 64)

    def test_complete_episode_geometry_remains_default(self):
        geometry = ppo_rollout_geometry(
            {},
            max_episode_transitions=130,
        )
        self.assertEqual(geometry["mode"], "complete_episodes")
        self.assertEqual(geometry["n_steps"], 2_080)
        self.assertEqual(geometry["batch_size"], 64)
        self.assertEqual(geometry["minimum_completed_episodes"], 16)

    def test_short_episodes_do_not_change_the_minibatch_or_break_ppo(self):
        geometry = ppo_rollout_geometry({}, max_episode_transitions=1)
        self.assertEqual(geometry["n_steps"], 2048)
        self.assertEqual(geometry["batch_size"], 64)

    def test_explicit_geometry_and_minibatch_are_independent(self):
        geometry = ppo_rollout_geometry(
            {"ppo_episodes_per_batch": 2, "ppo_batch_size": 32}, 130,
        )
        self.assertEqual(geometry["n_steps"], 260)
        self.assertEqual(geometry["batch_size"], 32)
        for batch_size in (0, 1, -1):
            with self.assertRaises(ValueError):
                ppo_rollout_geometry({"ppo_batch_size": batch_size}, 130)


if __name__ == "__main__":
    unittest.main()
