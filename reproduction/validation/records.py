"""Cross-check retained observations against the CSVs used by the paper plots."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def check_records(bundle):
    data = Path(bundle) / 'data'

    def rows(name):
        with (data / name).open(newline='') as handle:
            return list(csv.DictReader(handle))

    def close(actual, expected, label):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-11):
            raise ValueError(f'{label}: {actual} != {expected}')

    def require(condition, label):
        if not condition:
            raise ValueError(label)

    groups = defaultdict(list)
    for row in rows('raw_atari_evaluations.csv'):
        groups[row['phase'], row['leader_role'], row['seed'], row['training_total_timesteps']].append(row)
    expected = set()
    for phase, filename, count, start in (
        ('fixed_endpoints', 'atari_meta_stackpomdp.csv', 20, 4000001),
        ('fresh_final', 'table_atari_fresh_final.csv', 100, 5000001),
    ):
        for summary in rows(filename):
            key = phase, summary['leader_role'], summary['seed'], summary.get('training_total_timesteps', '2000040')
            expected.add(key)
            records = groups[key]
            require(len(records) == count, f'Atari episode count: {key}')
            require({int(r['evaluation_seed']) for r in records} == set(range(start, start + count)),
                    f'Atari evaluation seeds: {key}')
            require(len({r['checkpoint_sha256'] for r in records}) == 1, f'Atari checkpoint identity: {key}')
            for role in ('seller', 'buyer'):
                close(statistics.mean(float(r[role + '_payoff']) for r in records),
                      summary['mean_' + role + '_payoff'], f'Atari {role}: {key}')
            price = sum(float(r['payments']) for r in records) / sum(int(r['purchases']) for r in records)
            close(price, summary['mean_accepted_price'], f'Atari price: {key}')
    require(set(groups) == expected, 'Unexpected Atari episode groups')

    fields = ('experiment', 'condition', 'matrix', 'learning_rate', 'seed')
    groups = defaultdict(list)
    for row in rows('raw_ablation_final_evaluations.csv'):
        groups[tuple(row[k] for k in fields)].append(row)
    expected = set()
    for summary in rows('table_ablation_endpoints.csv'):
        key = tuple(summary[k] for k in fields)
        expected.add(key)
        records = groups[key]
        require(len(records) == int(summary['evaluation_episodes']) == 100, f'Ablation episode count: {key}')
        require(len({r['evaluation_seed'] for r in records}) == 100, f'Ablation evaluation seeds: {key}')
        require({r['response_checkpoint_sha256'] for r in records} == {summary['response_checkpoint_sha256']},
                f'Ablation response identity: {key}')
        close(statistics.mean(float(r['leader_reward_per_stage']) for r in records),
              summary['mean_reward'], f'Ablation endpoint: {key}')
    require(set(groups) == expected, 'Unexpected ablation episode groups')
    for name in ('fig_memory_pg.csv', 'fig_bots_leaderreward.csv'):
        for row in rows(name):
            require(int(row['evaluation_episodes']) == 20, f'Intermediate evaluation count: {name}')
            close(float(row['evaluation_std']) / math.sqrt(20), row['evaluation_sem'],
                  f'Intermediate evaluation standard error: {name}')

    groups = defaultdict(list)
    for row in rows('raw_platform_validation.csv'):
        groups[row['series'], row['leader_seed'], row['responses']].append(row)
    expected = set()
    for summary in rows('hidden_cost_learning.csv'):
        key = summary['series'], summary['seed'], summary['responses']
        expected.add(key)
        records = groups[key]
        require(len(records) == 10, f'Platform response count: {key}')
        require(len({(r['cost'], r['seller_seed']) for r in records}) == 10,
                f'Platform response identities: {key}')
        close(statistics.mean(float(r['consumer_surplus']) for r in records),
              summary['consumer_surplus'], f'Platform validation score: {key}')
    require(set(groups) == expected, 'Unexpected platform validation groups')

    candidates = rows('platform_random_search_candidates.csv')
    require(len(candidates) == 10000, 'Random search must contain 10000 candidates')
    require({int(r['candidate']) for r in candidates} == set(range(1, 10001)), 'Random-search candidate identities')
    require(len({r['thresholds'] for r in candidates}) == 10000, 'Random-search rules must be distinct')
    best = None
    for row in candidates:
        thresholds = list(map(int, row['thresholds'].split(';')))
        pairs = [(i, j) for i in range(15) for j in range(i, 15)]
        require(len(thresholds) == len(pairs), 'Incomplete random-search rule')
        require(all(value in {max(i-1, 0), i, j} for (i, j), value in zip(pairs, thresholds)),
                'Invalid random-search threshold encoding')
        close((float(row['low_consumer_surplus']) + float(row['high_consumer_surplus'])) / 2,
              row['score'], 'Random-search selection score')
        if best is None or float(row['score']) > float(best['score']) + 1e-12:
            best = row
    for row in rows('table_platform_results.csv'):
        if row['method'] == 'Random search':
            require(row['search_candidate'] == best['candidate'], 'Random-search selected candidate')
            close(row['search_selection_score'], best['score'], 'Random-search selected score')

    updates = defaultdict(list)
    for row in rows('mspm_training_updates.csv'):
        updates[int(row['num_types']), int(row['seed'])].append(int(row['episodes']))
    require(set(updates) == {(types, seed) for types in range(2, 7) for seed in range(96, 121)},
            'MSPM update cohort')
    require(all(points == list(range(40, 10001, 40)) for points in updates.values()),
            'MSPM update history is incomplete')

    certificates = defaultdict(dict)
    for row in rows('recorded_run_configurations.csv'):
        if row['experiment'] == 'commitment_follower':
            certificates[row['run']][row['parameter']] = json.loads(row['value'])
    require(set(certificates) == {f'seed{seed}' for seed in range(25)}, 'Certified follower cohort')
    for run, certificate in certificates.items():
        require(certificate['certificate.max_regret'] <= 0.25, f'Follower regret: {run}')
        require(certificate['certificate.commitments'] == 8, f'Follower commitment coverage: {run}')
        require(500 <= certificate['certificate.completed_updates'] <= 10000, f'Follower stopping budget: {run}')
    for row in rows('table_ablation_endpoints.csv'):
        if row['experiment'] == 'commitment_consistency':
            require(row['response_checkpoint_sha256'] == certificates['seed' + row['seed']]['certificate.checkpoint_sha256'],
                    'Leader must use its seed-matched certified follower')

    for row in rows('recorded_source_hashes.csv'):
        relative = row['file']
        path = (Path(bundle) / relative[len('reproduction/'):]
                if relative.startswith('reproduction/') else Path(bundle).parent / relative)
        # Standalone figure bundles do not contain the experiment source tree.
        if path.is_file():
            require(hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'],
                    f'Recorded response-credit source differs: {relative}')
    print('Retained episode records, training histories, and random-search selection reproduce the plotted summaries.')
