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
        self.assertEqual(stack["batch_size"], 8_192)
        self.assertEqual(basic["n_steps"], 32_768)
        self.assertEqual(basic["batch_size"], 2_048)
        self.assertEqual(stack["minimum_completed_episodes"], 0)

    def test_matrix_design_historical_geometry(self):
        stack = self.geometry(30, 1, "stackelberg")
        basic = self.geometry(30, 1, "hidden_queries")
        self.assertEqual(stack["n_steps"], 1_015_808)
        self.assertEqual(stack["batch_size"], 63_488)
        self.assertEqual(basic["n_steps"], 32_768)
        self.assertEqual(basic["batch_size"], 2_048)

    def test_complete_episode_geometry_remains_default(self):
        geometry = ppo_rollout_geometry(
            {"ppo_episodes_per_batch": 16},
            max_episode_transitions=130,
        )
        self.assertEqual(geometry["mode"], "complete_episodes")
        self.assertEqual(geometry["n_steps"], 2_080)
        self.assertEqual(geometry["batch_size"], 130)
        self.assertEqual(geometry["minimum_completed_episodes"], 16)


if __name__ == "__main__":
    unittest.main()
