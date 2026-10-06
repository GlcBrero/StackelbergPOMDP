"""Read the compact figure and table CSVs; no training-code or log dependency."""
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
INPUTS = set()
INTEGERS = {'num_types', 'seed', 'episodes', 'bcce_certified', 'curve_points', 'responses',
            'leader_seed', 'seller_seed', 'seller1_index', 'seller2_index', 'admit_seller1',
            'admit_seller2', 'low_cost_outcome', 'high_cost_outcome', 'search_candidate',
            'evaluation_seed', 'transfers', 'controlled_shots', 'training_total_timesteps',
            'evaluation_episodes', 'wall_seconds', 'allocated_cores', 'allocated_core_seconds',
            'memory_gb', 'global_step', 'policy_updates', 'update', 'transitions',
            'n00', 'n01', 'n10', 'n11'}
STRINGS = {'status', 'series', 'method', 'cycle_actions', 'search_final_sha256',
           'controlled_role', 'leader_role', 'job_id', 'condition', 'matrix', 'experiment',
           'response_checkpoint_sha256'}


def csv_rows(name):
    path = DATA / (name + '.csv')
    INPUTS.add(path)
    with path.open(newline='') as handle:
        return [{key: (value if key in STRINGS or value == '' else
                       int(value) if key in INTEGERS else float(value))
                 for key, value in row.items()} for row in csv.DictReader(handle)]


def mspm_curves():
    return csv_rows('pi_spm_vs_mspm_exact_eval_allseeds') + csv_rows('mspm_multitype_training')


def platform_learning():
    curves, baselines = defaultdict(list), {}
    for row in csv_rows('hidden_cost_learning'):
        if row['series'] == 'PPO':
            curves[row['seed']].append(row)
        else:
            assert row['series'] not in baselines
            baselines[row['series']] = row['consumer_surplus']
    assert set(curves) == set(range(801, 826))
    for history in curves.values():
        assert [r['responses'] for r in history] == list(range(400, 20001, 400))
    return curves, baselines


def platform_results():
    """Keep individual seller seeds for the crossed statistical comparison."""
    groups = {}
    for row in csv_rows('table_platform_results'):
        key = (row['method'], row['leader_seed'])
        if key not in groups:
            groups[key] = dict(seed=row['leader_seed'], cs=row['policy_mean_cs'], responses=[])
        group = groups[key]
        assert group['cs'] == row['policy_mean_cs']
        group['responses'].append(dict(cost=row['cost'], seed=row['seller_seed'],
            frozen_cs=row['consumer_surplus'],
            cycle_actions=[[int(x) for x in pair.split(':')] for pair in row['cycle_actions'].split(';')]))
    for group in groups.values():
        assert len(group['responses']) == 50
        assert len({(r['cost'], r['seed']) for r in group['responses']}) == 50
        assert np.isclose(np.mean([r['frozen_cs'] for r in group['responses']]), group['cs'], atol=1e-12)
    return groups


def platform_policy():
    records = csv_rows('hidden_cost_policy')
    assert len(records) == 225
    assert len({(r['seller1_index'], r['seller2_index']) for r in records}) == 225
    assert len({r['leader_seed'] for r in records}) == 1
    prices = np.empty(15)
    admissions = np.empty((15, 15, 2), dtype=bool)
    outcomes = {}
    for row in records:
        i, j = row['seller1_index'], row['seller2_index']
        prices[i] = row['seller1_price']
        admissions[i, j] = row['admit_seller1'], row['admit_seller2']
        for cost, key in [('1.0', 'low_cost_outcome'), ('1.5', 'high_cost_outcome')]:
            if row[key]:
                assert cost not in outcomes
                outcomes[cost] = {'price_indices': [i, j]}
    assert set(outcomes) == {'1.0', '1.5'}
    return records[0]['leader_seed'], prices, admissions, outcomes


def atari_curves():
    records = csv_rows('atari_meta_stackpomdp')
    assert len(records) == 300
    assert {(r['leader_role'], r['seed'], r['training_total_timesteps']) for r in records} == {
        (role, seed, step) for role in ('buyer', 'seller') for seed in range(1, 26)
        for step in (0, 400680, 800520, 1200360, 1600200, 2000040)}
    assert all(r['evaluation_episodes'] == 20 for r in records)
    return records


def atari_fresh():
    records = csv_rows('table_atari_fresh_final')
    assert len(records) == 50 and {(r['leader_role'], r['seed']) for r in records} == {
        (role, seed) for role in ('buyer', 'seller') for seed in range(1, 26)}
    assert all(r['evaluation_episodes'] == 100 for r in records)
    return records


def matrix_runs(experiment, expected):
    name = {'commitment_consistency': 'fig_memory_pg', 'reward_timing': 'fig_bots_leaderreward'}[experiment]
    def key(row):
        return tuple(row[k] for k in ('condition', 'matrix', 'learning_rate', 'seed'))
    curves = defaultdict(list)
    for row in csv_rows(name):
        curves[key(row)].append(row)
    finals = [r for r in csv_rows('table_ablation_endpoints') if r['experiment'] == experiment]
    assert len(finals) == len(curves) == expected
    assert {key(r) for r in finals} == set(curves)
    runs = []
    for row in finals:
        history = curves[key(row)]
        assert row['evaluation_episodes'] == 100
        assert history[-1]['global_step'] >= (200000 if experiment == 'commitment_consistency' else 22000)
        assert len({r['global_step'] for r in history}) == len(history)
        runs.append((row, history, {'per_stage_summary': {'mean': row['mean_reward'], 'n': 100}}))
    return sorted(runs, key=lambda run: run[0]['seed'])
