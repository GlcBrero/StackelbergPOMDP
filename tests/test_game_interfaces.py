"""Paper applications expose the same joint-action game contract as the example."""
import numpy as np
import pytest
from stackelberg_pomdp import POMG


@pytest.mark.parametrize("setting", ["mspm", "platform", "atari", "commitment_consistency", "reward_timing"])
def test_game_has_private_observations_and_per_agent_rewards(setting):
    if setting == "mspm":
        from experiments.mspm.game import Game
        game = Game()
    elif setting == "platform":
        from experiments.platform.game import Game
        game = Game(np.linspace(1., 2.1, 4), (1., 1.5))
    elif setting == "atari":
        from experiments.atari.game import Game
        from atari_fixtures import _FakeSide, _bilateral_config
        game = Game(leader_role="seller", config=_bilateral_config(), side_factory=_FakeSide)
    elif setting == "commitment_consistency":
        from experiments.ablations.commitment_consistency.game import Game
        game = Game()
    else:
        from experiments.ablations.reward_timing.game import Game
        game = Game()
    assert isinstance(game, POMG)
    observations = game.reset()
    agents = set(game.action_spaces)
    done = False
    for _ in range(game.horizon):
        assert set(observations) == set(game.observation_spaces) == agents
        for agent, value in observations.items():
            assert game.observation_spaces[agent].contains(value), (setting, agent)
        assert isinstance(game.leader_observation((observations[game.leader],)), dict)
        # Fixed valid actions keep the check deterministic and independent of learning.
        actions = {agent: 0 for agent in agents}
        if setting == "mspm":
            actions[game.leader] = np.asarray([1., 0., 0.], dtype=np.float32)
        elif setting == "atari":
            actions = {agent: np.asarray([0., .5], dtype=np.float32) for agent in agents}
        observations, rewards, done, info = game.step(actions)
        assert set(rewards) == agents and all(np.isfinite(value) for value in rewards.values())
        if done:
            break
    assert done
    if hasattr(game, "close"):
        game.close()


def test_mspm_private_types_and_strict_acceptance_determine_both_payoffs():
    from experiments.mspm.game import Game
    game = Game()
    observations = game.reset(types=(1, 1))
    assert observations["buyer_0"] == observations["buyer_1"] == 1
    assert "critic:types" not in observations[game.leader]
    game.step({game.leader: np.zeros(3), "buyer_0": 0, "buyer_1": 1})
    # Offer to buyer 1 at exactly its value: strict acceptance rejects it.
    _, rewards, done, _ = game.step({game.leader: [0., 1., .4], "buyer_0": 0, "buyer_1": 0})
    assert not done and all(reward == 0 for reward in rewards.values())
    _, rewards, done, _ = game.step({game.leader: [1., 0., .5], "buyer_0": 0, "buyer_1": 0})
    assert done and rewards == {game.leader: 0., "buyer_0": .5, "buyer_1": 0.}
