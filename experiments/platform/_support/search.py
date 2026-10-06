"""Single-pass uniform policy search, followed by fresh winner-only evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
from support.common import fresh_output, record_run, verify_run

ROOT = None
PAIRS = tuple((i, j) for i in range(15) for j in range(i, 15))
OPTIONS = tuple(tuple(sorted({max(i - 1, 0), i, j})) for i, j in PAIRS)


def verify():
    return verify_run(ROOT)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sample(rng):
    return bytes(options[int(rng.randint(len(options)))] for options in OPTIONS)


def rule_from_thresholds(thresholds):
    assert len(thresholds) == len(PAIRS)
    table = np.empty((15, 15), dtype=np.int32)
    for (i, j), value, choices in zip(PAIRS, thresholds, OPTIONS):
        assert value in choices
        table[i, j] = table[j, i] = value
    return table


def make_seller_q_learning(p):
    from experiments.platform.response import make_seller_q_learning as original
    return original(ROOT, p)


def response_row(result):
    return {key: result[key] for key in ('cost', 'seed', 'frozen_cs',
            'cycle_actions', 'seller_policy', 'training_joint',
            'response_steps', 'stable_policy_rounds')}


def scalar_cs(p, table, cycle):
    prices, qualities, mu = p['prices'], p['qualities'], p['mu']
    outside = math.exp(p['outside_quality'] / mu)
    return statistics.mean(mu * math.log(outside + sum(
        math.exp((qualities[k] - prices[price]) / mu)
        for k, price in enumerate((i, j)) if price <= table[i, j]))
        for i, j in cycle)


def audit_response(p, table, row):
    m = p['m']
    policy = np.asarray(row['seller_policy'])
    assert policy.shape == (2, m * m)
    assert np.all((policy >= 0) & (policy < m))
    current = divmod(int(row['training_joint']), m)
    visited, trajectory = {}, []
    while current not in visited:
        visited[current] = len(trajectory)
        trajectory.append(current)
        current = tuple(int(x) for x in policy[:, current[0] * m + current[1]])
    cycle = trajectory[visited[current]:]
    reported = [tuple(x) for x in row['cycle_actions']]
    assert len(cycle) == len(reported)
    assert any(cycle == reported[k:] + reported[:k] for k in range(len(reported)))
    value = scalar_cs(p, table, reported)
    assert abs(value - row['frozen_cs']) < 2e-12
    assert 0 < row['response_steps'] <= p['response_steps']
    if row['stable_policy_rounds'] < p['stability_required']:
        assert row['response_steps'] == p['response_steps']
    return value


def response_pair(seller_learning, p, table, seller_seed):
    rows = [response_row(seller_learning.evaluate_rule(table, cost, seller_seed))
            for cost in p['costs']]
    values = [audit_response(p, table, row) for row in rows]
    return rows, statistics.mean(values)


def train():
    p = verify()
    assert p['total_seller_responses'] == 2 * p['target_unique_policies']
    run = ROOT / 'run'
    assert not (run / 'candidates.jsonl').exists()
    rng = np.random.RandomState(p['proposal_rng_seed'])
    schedule = read(ROOT / 'training_seeds/schedule.json')
    assert sha(ROOT / 'training_seeds/schedule.json') == p['training_seed_schedule_sha256']
    assert len(schedule) == p['target_unique_policies']
    assert len(set(schedule)) == len(schedule)
    assert set(schedule).isdisjoint(p['confirmation_seeds'])
    seller_learning = make_seller_q_learning(p)
    seen, candidates, history = set(), [], []
    draws, calls, unstable = 0, 0, 0
    best_index, best_score = None, -float('inf')
    start = time.perf_counter()
    with (run / 'candidates.jsonl').open('x', buffering=1) as log:
        for index in range(1, p['target_unique_policies'] + 1):
            while True:
                thresholds = sample(rng)
                draws += 1
                if thresholds not in seen:
                    seen.add(thresholds)
                    break
            table = rule_from_thresholds(thresholds)
            rows, score = response_pair(seller_learning, p, table, schedule[index - 1])
            calls += 2
            unstable += sum(row['stable_policy_rounds'] < p['stability_required'] for row in rows)
            candidates.append({'index': index, 'thresholds': list(thresholds), 'score': score})
            log.write(json.dumps({'index': index, 'proposal_draw': draws,
                                  'thresholds': list(thresholds), 'score': score,
                                  'responses': rows}) + '\n')
            if score > best_score + 1e-12:
                best_index, best_score = index, score
            if index % p['progress_every_unique_policies'] == 0:
                entry = {'unique_policies': index, 'physical_responses': calls,
                         'proposal_draws': draws, 'best_candidate': best_index,
                         'best_training_score': best_score,
                         'fresh_evaluation_responses': 0,
                         'unstable_training_responses': unstable,
                         'elapsed_seconds': time.perf_counter() - start}
                history.append(entry)
                write(run / 'history.json', history)
                write(run / 'latest.json', entry)
                print(json.dumps(entry), flush=True)
    assert len(seen) == p['target_unique_policies'] and calls == 2 * p['target_unique_policies']
    assert calls == p['total_seller_responses']
    winner = candidates[best_index - 1]
    assert all(row['score'] <= winner['score'] + 1e-12 for row in candidates)
    write(run / 'final.json', {'method': p['method'], 'candidate': winner['index'],
                              'thresholds': winner['thresholds'],
                              'rule': rule_from_thresholds(winner['thresholds']).tolist(),
                              'initial_training_score': winner['score'],
                              'selection': p['selection'],
                              'fresh_evaluation_responses_at_selection': 0,
                              'distinct_policies': len(seen),
                              'proposal_draws': draws,
                              'physical_responses': calls})
    write(run / 'completion.json', {'status': 'complete', 'method': p['method'],
                                   'unique_policies': len(seen), 'proposal_draws': draws,
                                   'physical_responses': calls,
                                   'unstable_training_responses': unstable,
                                   'fresh_evaluation_responses': 0,
                                   'elapsed_seconds': time.perf_counter() - start,
                                   'protocol_sha256': sha(ROOT / 'protocol.json'),
                                   'candidate_log_sha256': sha(run / 'candidates.jsonl'),
                                   'final_sha256': sha(run / 'final.json')})
    print(json.dumps({'training': 'complete', 'winner': winner['index'],
                      'distinct_policies': len(seen), 'physical_responses': calls}), flush=True)


def confirm():
    p = verify()
    final = read(ROOT / 'run/final.json')
    done = read(ROOT / 'run/completion.json')
    assert done['status'] == 'complete' and sha(ROOT / 'run/final.json') == done['final_sha256']
    table = rule_from_thresholds(final['thresholds'])
    np.testing.assert_array_equal(table, final['rule'])
    seller_learning = make_seller_q_learning(p)
    directory = ROOT / 'confirmation'
    assert not (directory / 'summary.json').exists()
    per_cost, stable = {}, 0
    for cost in p['costs']:
        scores = []
        for seller_seed in p['confirmation_seeds']:
            result = seller_learning.evaluate_rule(table, cost, seller_seed)
            row = response_row(result)
            score = audit_response(p, table, row)
            stable += row['stable_policy_rounds'] >= p['stability_required']
            q = np.asarray(result['q'])
            np.testing.assert_array_equal(q.argmax(axis=2), row['seller_policy'])
            stem = f'c{cost:g}_s{seller_seed}'
            np.savez_compressed(directory / (stem + '.npz'), q=q)
            row.update(q_sha256=sha(directory / (stem + '.npz')),
                       protocol_sha256=sha(ROOT / 'protocol.json'))
            write(directory / (stem + '.json'), row)
            scores.append(score)
        per_cost[str(cost)] = statistics.mean(scores)
    write(directory / 'summary.json', {'status': 'complete',
                                      'cs': statistics.mean(per_cost.values()),
                                      'per_cost': per_cost,
                                      'fresh_responses': len(p['costs']) * len(p['confirmation_seeds']),
                                      'converged_responses': stable,
                                      'final_sha256': sha(ROOT / 'run/final.json'),
                                      'protocol_sha256': sha(ROOT / 'protocol.json')})
    print(json.dumps({'confirmation': 'complete',
                      'cs': statistics.mean(per_cost.values())}), flush=True)


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'confirm', 'plot-validation'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Tiny integration check; not a paper result.')
    args = parser.parse_args()
    ROOT = args.output.resolve()
    if args.stage == 'train':
        p = read(Path(__file__).with_name('search.json'))
        if args.smoke:
            p.update(target_unique_policies=4, first_pass_seller_seeds=4,
                     total_seller_responses=8, progress_every_unique_policies=2,
                     proposal_rng_seed=17, seller_schedule_seed=18,
                     confirmation_seeds=[6261005901], fresh_seller_responses=2,
                     purpose='technical smoke only; disjoint seeds and full seller stopping rule')
        from experiments.platform._support.train import response_seeds
        schedule = response_seeds(p['seller_schedule_seed'], p['target_unique_policies'])
        p['training_seed_schedule_sha256'] = hashlib.sha256(
            (json.dumps(schedule, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
        fresh_output(ROOT)
        record_run(ROOT, 'platform-search-single-pass', p)
        write(ROOT / 'training_seeds/schedule.json', schedule)
        (ROOT / 'run').mkdir()
        train()
    elif args.stage == 'plot-validation':
        if args.smoke:
            parser.error('--smoke applies to train; evaluation uses the saved protocol')
        from experiments.platform._support.search_validation import evaluate
        print(evaluate(ROOT))
    else:
        if args.smoke:
            parser.error('--smoke applies to train; confirmation uses the saved protocol')
        verify_run(ROOT)
        (ROOT / 'confirmation').mkdir(exist_ok=False)
        confirm()


if __name__ == '__main__':
    main()
