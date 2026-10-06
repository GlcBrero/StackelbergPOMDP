"""Economic and PPO equivalence of direct admissions and the archived encoding."""
import itertools
import json

import numpy as np
import pytest
import torch

from experiments.platform._support.train import make_env, make_model
from experiments.platform._support import q_learning as bertrand_finite
from experiments.platform.game import (
    admission_game, admission_payoff_tables, admissions_from_rule, payoff_tables,
)
from experiments.platform._support.q_learning import (
    AdmissionRuleQLearning, SellerQLearning, admission_table_from_actions,
    table_from_actions,
)
def expected_game(tables, policy):
    rule = np.asarray(policy["rule"], dtype=int)
    first, second = np.indices(rule.shape)
    return tables[0][rule, first, second], tables[1][rule, first, second]


def archived_threshold_rule(actions, m, ordered=False):
    pairs = (list(itertools.product(range(m), repeat=2)) if ordered else
             list(itertools.combinations_with_replacement(range(m), 2)))
    indices = []
    for action, pair in zip(actions, pairs):
        low, high = sorted(pair)
        indices.append((max(low - 1, 0), low, high)[int(action)])
    return table_from_actions(indices, m, ordered)


@pytest.mark.parametrize('ordered', [False, True])
def test_every_price_pair_and_action_preserves_display_decisions(ordered):
    m = 15
    count = m * m if ordered else m * (m + 1) // 2
    first, second = np.indices((m, m))
    for action in range(3):
        actions = [action] * count
        old = archived_threshold_rule(actions, m, ordered)
        expected = np.stack((first <= old, second <= old), axis=-1)
        actual = admission_table_from_actions(actions, m, ordered)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == bool
        np.testing.assert_array_equal(actual, actual.transpose(1, 0, 2)[..., ::-1])
        assert actual[0, :, 0].all() and actual[:, 0, 1].all()
    assert admission_table_from_actions([0] * count, m, ordered)[0, 0].all()
    assert admission_table_from_actions([1] * count, m, ordered)[7, 7].all()


@pytest.mark.parametrize('cost', [1., 1.5])
@pytest.mark.parametrize('serialized', [False, True])
def test_all_threshold_outcomes_have_identical_direct_payoffs(cost, serialized):
    prices = np.linspace(1., 2.1, 15)
    if serialized:
        # Archived JSON price lists may differ from the base grid by one ULP.
        prices[1:-1] = np.nextafter(prices[1:-1], np.inf)
    old_tables = payoff_tables(cost, prices)
    direct_tables = admission_payoff_tables(cost, prices)
    for threshold in range(15):
        rule = np.full((15, 15), threshold)
        old = expected_game(old_tables, {'rule': rule})
        direct = admission_game(direct_tables, admissions_from_rule(rule))
        for actual, expected in zip(direct, old):
            np.testing.assert_array_equal(actual, expected)


def test_paired_normalized_ppo_updates_are_unchanged(tmp_path, monkeypatch):
    p = dict(m=15, prices=np.linspace(1., 2.1, 15).tolist(), costs=[1., 1.5],
             response_steps=2000, alpha=.15, delta=.95, beta=.004,
             q_initialization='no_intervention',
             leader_action_representation='admission_classes',
             actor_reward='mean_training_costs', episodes=8,
             rl_methods={'ppo': dict(algorithm='ppo', episodes_per_rollout=2,
                 learning_rate=.0003, ent_coef=0., n_epochs=2, batch_size=121,
                 clip_range=.2, normalize_advantage=True)})

    def run(name):
        env = make_env(tmp_path / name, p, 801)
        model = make_model(env, p, 801, 'ppo')
        _, callback = model._setup_learn(2 * model.n_steps)
        callback.on_training_start({}, {})
        for _ in range(2):
            assert model.collect_rollouts(model.env, callback, model.rollout_buffer, model.n_steps)
            model.train()
        callback.on_training_end()
        return model

    def forbidden(*args, **kwargs):
        raise AssertionError('Direct admission training entered a threshold path')

    with monkeypatch.context() as patch:
        patch.setattr(SellerQLearning, 'table_from_actions', forbidden)
        patch.setattr(bertrand_finite, 'payoff_tables', forbidden)
        direct = run('direct')

    class ArchivedEncoding(AdmissionRuleQLearning):
        def table_from_actions(self, actions):
            return archived_threshold_rule(actions, self.m, self.ordered)

    monkeypatch.setattr(bertrand_finite, 'AdmissionRuleQLearning', ArchivedEncoding)
    archived = run('archived')
    for name, value in direct.policy.state_dict().items():
        torch.testing.assert_close(value, archived.policy.state_dict()[name], rtol=0, atol=0)
    for field in ('actions', 'rewards', 'returns', 'advantages', 'actor_credit'):
        np.testing.assert_array_equal(getattr(direct.rollout_buffer, field),
                                      getattr(archived.rollout_buffer, field))


@pytest.mark.parametrize('actions', [[0] * 119, [3] * 120, [.5] * 120])
def test_invalid_admission_actions_are_rejected(actions):
    with pytest.raises(ValueError):
        admission_table_from_actions(actions, 15)


def test_invalid_saved_admission_rules_are_rejected():
    with pytest.raises(ValueError):
        admissions_from_rule(np.full((15, 15, 2), .5))
    with pytest.raises(ValueError):
        admissions_from_rule(np.ones((15, 15, 3)))


def test_fresh_confirmation_accepts_direct_and_archived_rules(tmp_path):
    from experiments.platform._support.cli import evaluate_rule

    m = 4
    p = dict(prices=np.linspace(1.05, 1.7, m).tolist(), costs=[1.],
             response_steps=80, stability_required=0, alpha=.25,
             delta=.95, beta=.004, leader_action_representation='threshold')
    actions = [0, 1, 2, 0, 1, 0, 1, 2, 0, 1]
    old = archived_threshold_rule(actions, m)
    direct = admission_table_from_actions(actions, m)
    outputs = []
    for name, rule in [('old', old), ('direct', direct)]:
        record = evaluate_rule(tmp_path / name, p, name, rule, [601])
        outputs.append(record['responses'])
    assert outputs[0] == outputs[1]
