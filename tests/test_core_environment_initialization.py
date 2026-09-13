import unittest

from stackelberg_pomdp.env_setups import wrap_env
from stackelberg_pomdp.games import (
    MatrixDesignGame,
    SimpleAllocationGame,
    get_mspm_setting,
    get_normal_form_game,
)
from stackelberg_pomdp.envs.matrix_design import BaseEnvMatrixDesignGame
from stackelberg_pomdp.envs.normal_form import BaseEnvSimpleMatrixGame
from stackelberg_pomdp.envs.simple_allocation import BaseSimpleAllocation
from stackelberg_pomdp.envs.spm import BaseMessageSPM, BaseSPM


class CoreEnvironmentInitializationTests(unittest.TestCase):
    def test_game_environments_forward_logger_and_seed_by_name(self):
        logger = object()
        cases = (
            BaseEnvSimpleMatrixGame(
                get_normal_form_game("game_2"), logger=logger, seed=17
            ),
            BaseEnvMatrixDesignGame(
                MatrixDesignGame(), logger=logger, seed=17
            ),
            BaseSimpleAllocation(
                SimpleAllocationGame(num_messages=3), logger=logger, seed=17
            ),
            BaseSPM(
                get_mspm_setting("MSGSpace", 3, 2), logger=logger, seed=17
            ),
            BaseMessageSPM(
                get_mspm_setting("MSGSpace", 3, 2), logger=logger, seed=17
            ),
        )

        for env in cases:
            with self.subTest(environment=type(env).__name__):
                self.assertIs(env.logger, logger)

    def test_unknown_follower_algorithm_fails_before_training(self):
        for algorithm in ("RoundRobin", "misspelled_algorithm"):
            with self.subTest(algorithm=algorithm):
                base = BaseEnvSimpleMatrixGame(get_normal_form_game("game_3"), seed=0)
                try:
                    with self.assertRaisesRegex(ValueError, "Unsupported followers_algorithm"):
                        wrap_env(base, {"followers_algorithm": algorithm})
                finally:
                    base.close()

    def test_equal_seeds_reproduce_private_type_sequence(self):
        def sequence(seed):
            env = BaseSimpleAllocation(
                SimpleAllocationGame(num_messages=3), seed=seed
            )
            return [env.game.sample_types() for _ in range(20)]

        self.assertEqual(sequence(123), sequence(123))
        self.assertNotEqual(sequence(123), sequence(124))


if __name__ == "__main__":
    unittest.main()
