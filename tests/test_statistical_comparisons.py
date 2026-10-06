"""Numerical checks for inference with ties and crossed evaluation seeds."""
import itertools
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduction/plots"))

from statistical_comparisons import (crossed_mean, exact_bootstrap_variance,
                                    holm, seller_means, sign_comparison)


class StatisticalChecks(unittest.TestCase):
    def test_exact_sign_with_constant_differences_and_ties(self):
        constant = sign_comparison("test", "test", "test", [1.] * 25, 1e-10)
        self.assertEqual(constant["p_raw"], 2 / 2**25)
        self.assertEqual(constant["se"], 0.)
        tied = sign_comparison("test", "test", "test", [1.] * 20 + [0.] * 5, 1e-10)
        self.assertEqual(tied["p_raw"], 2 / 2**20)
        self.assertEqual(tied["ties"], 5)
        self.assertEqual(tied["effect"], .8)
        opposite = sign_comparison("test", "test", "test", [-1.] * 20 + [0.] * 5, 1e-10)
        self.assertEqual(opposite["p_raw"], tied["p_raw"])
        self.assertEqual(opposite["effect"], -tied["effect"])
        null = sign_comparison("test", "test", "test", [0.] * 25, 1e-10)
        self.assertEqual(null["p_raw"], 1.)

    def test_holm_preserves_order_and_handles_ties(self):
        np.testing.assert_allclose(holm([.04, .01, .03, .8]), [.09, .04, .09, .8])
        np.testing.assert_allclose(holm([.01, .01, 1.]), [.03, .03, 1.])

    def test_bootstrap_matches_exhaustive_index_resampling(self):
        matrix = np.array([[1., 3.], [4., 8.]])
        samples, weights = [], []
        for rows, columns in itertools.product(itertools.product(range(2), repeat=2), repeat=2):
            samples.append(matrix[np.ix_(rows, columns)].mean())
            weights.append((np.bincount(rows, minlength=2), np.bincount(columns, minlength=2)))
        actual = crossed_mean(matrix, np.array([w[0] for w in weights]),
                              np.array([w[1] for w in weights]))
        np.testing.assert_allclose(actual, samples)
        self.assertAlmostEqual(np.var(samples), exact_bootstrap_variance(matrix))

    def test_shared_seller_seed_variation_cancels(self):
        baseline = np.array([1., 5., 2.])
        matrix = np.tile(baseline, (2, 1))
        rows, columns = np.array([[0, 2], [1, 1]]), np.array([[3, 0, 0], [0, 1, 2]])
        difference = crossed_mean(matrix, rows, columns) - columns @ baseline / 3
        np.testing.assert_allclose(difference, 0, atol=1e-15)
        self.assertEqual(exact_bootstrap_variance(matrix-baseline), 0.)

    def test_costs_stay_with_seller_seed_and_missing_cells_fail(self):
        responses = [dict(cost=c, seed=s, frozen_cs=c+s) for s in (10, 11) for c in (1., 1.5)]
        np.testing.assert_allclose(seller_means(responses[::-1], [11, 10]), [12.25, 11.25])
        with self.assertRaises(AssertionError):
            seller_means(responses[:-1], [10, 11])


if __name__ == "__main__":
    unittest.main()
