"""Pause/resume equivalence and variable-query training budgets."""
import json
import numpy as np
import pytest

from experiments.platform._support.q_learning import SellerQLearning, AdmissionRuleQLearning
from experiments.platform.experiment import build


@pytest.mark.parametrize('ordered', [False, True])
@pytest.mark.parametrize('direct', [False, True])
def test_paused_seller_updates_equal_complete_rule(tmp_path, ordered, direct):
    learner = SellerQLearning(tmp_path, prices=np.linspace(1., 2.1, 4),
        costs=(1., 1.5), steps=2000, beta=.01, alpha=.15,
        stability_required=80, q_initialization='no_intervention', ordered=ordered)
    if direct:
        learner = AdmissionRuleQLearning(learner)
    actions = np.asarray([(3*a+b) % 3 for a,b in learner.pairs])
    for cost_index in range(2):
        expected = learner.run_seller_q_learning(actions, cost_index, 601)
        commitment = np.full(len(actions), -1, dtype=int)
        asked = []
        with learner.start_response(commitment, cost_index, 601) as response:
            index = response.next_query()
            while index is not None:
                assert commitment[index] == -1 and index not in asked
                asked.append(index)
                # Waiting for an answer cannot consume another random draw or round.
                waiting = response.session.result()
                assert response.next_query() == index
                repeated = response.session.result()
                for key in ('q', 'visits', 'joint', 'stats'):
                    np.testing.assert_array_equal(waiting[key], repeated[key])
                commitment[index] = actions[index]
                index = response.next_query()
            actual = response.result()
        for key in ('q', 'seller_policy', 'cycle_actions', 'training_joint',
                    'training_tail_cs', 'stable_policy_rounds', 'response_steps'):
            np.testing.assert_array_equal(actual[key], expected[key])


def test_gameplay_extends_partial_response_commitment(tmp_path):
    learner = AdmissionRuleQLearning(SellerQLearning(tmp_path,
        prices=np.linspace(1., 2.1, 4), costs=(1., 1.5), steps=1,
        beta=.004, q_initialization='no_intervention'))
    env = build(learner, [601, 602], 801)
    try:
        observation = env.reset()
        observed = []
        while True:
            observed.append(tuple(observation['base_environment']))
            observation, reward, done, info = env.step(2)
            if done:
                break
        assert info['response_policy_queries'] == 1
        assert info['reward_policy_queries'] > 0
        assert len(observed) == len(set(observed)) < env.n_queries
        assert info['commitment_entries'] == len(observed)
        assert sum(a < 0 for a in info['query_actions']) > 0
        assert info['response_protocol'] == 'on_demand_queries'
        expected = [learner.run_seller_q_learning([2]*env.n_queries, cost, 601)
                    for cost in range(2)]
        assert reward == pytest.approx(np.mean([r['frozen_cs'] for r in expected]), abs=1e-12)
        env.reset()
        assert len(env.commitment) == 0
    finally:
        env.close()


def test_training_counts_complete_episodes_with_partial_tables(tmp_path):
    from experiments.platform._support.train import train
    from stable_baselines3.common.save_util import load_from_zip_file
    import torch
    parameters = dict(m=4, prices=np.linspace(1.,2.1,4).tolist(), costs=[1.,1.5],
        response_steps=1, alpha=.15, delta=.95, beta=.004, stability_required=0,
        q_initialization='no_intervention', leader_action_representation='admission_classes',
        episodes=8, evaluation_every=4, report_budgets=[8], validation_seeds=[701],
        rl_methods={'ppo':dict(algorithm='ppo', episodes_per_rollout=2, batch_size=4,
            n_epochs=1, learning_rate=.0003, learning_rate_schedule='hold25_then_linear_to_zero',
            ent_coef=0., clip_range=.2, normalize_advantage=True)})
    result = train(tmp_path, parameters, 'ppo', 801)
    assert result['responses'] == 8 and result['updates'] == 2
    folder = tmp_path/'runs/ppo_seed801'
    rows = [json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines()]
    assert len(rows) == 8
    assert all(row['policy_queries'] < 10 for row in rows)
    assert all(row['archive_completed_entries'] > 0 for row in rows)
    _, before, _ = load_from_zip_file(folder/'initial_model.zip', device='cpu', load_data=False)
    _, after, _ = load_from_zip_file(folder/'final_model.zip', device='cpu', load_data=False)
    assert not torch.equal(before['policy']['action_net.weight'], after['policy']['action_net.weight'])
