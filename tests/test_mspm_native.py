"""Native MW preserves the Python recurrence and exact BCCE certificates."""
import numpy as np
import pytest

from experiments.mspm._support._native import exact_bcce_gap_two_player


@pytest.mark.parametrize("seed", [1, 9, 73])
@pytest.mark.parametrize("snapshots", [1, 5])
def test_cpp_gap_matches_joint_distribution_reference(seed, snapshots):
    rng = np.random.RandomState(seed)
    # Unequal type/action counts catch accidental axis swaps. Nonuniform
    # priors distinguish interim certification from an ex-ante average.
    first = rng.dirichlet(np.ones(2), size=(snapshots, 3))
    second = rng.dirichlet(np.ones(3), size=(snapshots, 2))
    probabilities = (np.asarray([.1, .3, .6]), np.asarray([.8, .2]))
    utilities = rng.normal(size=(3, 2, 2, 3, 2))
    joint = np.einsum("sac,sbd->abcd", first, second) / snapshots
    gains = [0.0]
    for player in range(2):
        for own_type in range((3, 2)[player]):
            current = 0.0
            deviations = np.zeros((2, 3)[player])
            for other_type in range((2, 3)[player]):
                t0, t1 = ((own_type, other_type) if player == 0 else (other_type, own_type))
                weight = probabilities[1 - player][other_type]
                for a0 in range(2):
                    for a1 in range(3):
                        mass = weight * joint[t0, t1, a0, a1]
                        current += mass * utilities[t0, t1, a0, a1, player]
                        for action in range(len(deviations)):
                            u = (utilities[t0, t1, action, a1, player] if player == 0
                                 else utilities[t0, t1, a0, action, player])
                            deviations[action] += mass * u
            gains.extend(deviations - current)
    observed = exact_bcce_gap_two_player(first, second, *probabilities, utilities)
    assert observed == pytest.approx(max(gains), abs=1e-12)


@pytest.mark.parametrize("types,messages", [(2, 2), (3, 3), (6, 4)])
def test_native_updates_preserve_python_weights_history_and_certificates(types, messages):
    from experiments.mspm._support._native import NativeMW, bcce_gap

    rng = np.random.RandomState(91)
    weights = rng.dirichlet(np.ones(messages), size=(2, types))
    probabilities = rng.dirichlet(np.ones(types), size=2)
    utility = np.zeros((types, types, messages, messages, 2))
    # Matching pennies has no pure equilibrium; unequal type probabilities
    # keep the deterministic candidate from accidentally mixing 50/50.
    first, second = np.indices((messages, messages))
    utility[..., 0] = (first % 2 == second % 2)
    utility[..., 1] = 1 - utility[..., 0]
    joint = lambda w: np.einsum("ac,bd->abcd", w[0], w[1])
    empirical = joint(weights)
    history = [weights.copy()]
    deviations = np.zeros((2, messages))
    with NativeMW(weights, utility, probabilities, empirical, deviations,
                  epsilon=.1, tolerance=0.) as learner:
        learner.certify()
        for update in range(2, 10):
            assert not learner.certified
            sampled = rng.randint(types, size=2)
            deviations[:] = 0
            for a in range(messages):
                for b in range(messages):
                    deviations[0, a] += utility[sampled[0], sampled[1], a, b, 0] * weights[1, sampled[1], b]
                    deviations[1, b] += utility[sampled[0], sampled[1], a, b, 1] * weights[0, sampled[0], a]
            for player in range(2):
                row = weights[player, sampled[player]]
                row *= 1.1 ** deviations[player]
                row /= row.sum()
            history.append(weights.copy())
            empirical += (joint(weights) - empirical) / update
            learner.advance(sampled)
            result = learner.result()
            assert result["updates"] == update
            np.testing.assert_allclose(result["weights"], weights, rtol=0, atol=1e-12)
            np.testing.assert_allclose(result["deviations"], deviations, rtol=0, atol=1e-12)
            np.testing.assert_allclose(result["empirical_joint"], empirical, rtol=0, atol=1e-12)
            pure = np.eye(messages)[weights.argmax(axis=-1)]
            candidates = ([weights], [pure], history)
            for name, snapshots in zip(NativeMW.CANDIDATES, candidates):
                assert result["gaps"][name] == pytest.approx(
                    bcce_gap(snapshots, utility, probabilities), abs=1e-12)
    with pytest.raises(RuntimeError, match="closed"):
        learner.result()


def test_native_response_respects_the_uncertified_update_budget():
    from experiments.mspm.experiment import build

    env = build(num_types=3, response_bcce_max_extra_updates=0)
    rules = np.random.RandomState(8).uniform(0, 1, (4, 3))
    observation = env.reset()
    try:
        with pytest.raises(RuntimeError, match="exhausted its certification budget"):
            while True:
                first, second = map(int, observation["base:follower_actions"])
                action = rules[2 * first + second].astype(np.float32).copy()
                if observation["base_environment"][:2].sum() == 1:
                    action[2] *= .5
                observation, _, _, _ = env.step(action)
        assert env.response.state.progress[0] == 1
    finally:
        env.close()
