"""Johnson-grid PPO and validation-selected baselines on the shared library."""
from support.checkpoint_compat import load_checkpoint
import argparse
import copy
import json
from pathlib import Path
import statistics

from support.common import fresh_output, record_run, verify_run, write_json

METHOD = 'ppo_class_mean40'
CONFIG = Path(__file__).with_name('paper.json')


def initial_state(output, protocol, seed):
    """Preserve the paper's shared threshold-network initialization."""
    from experiments.platform._support.train import make_env, make_model
    old = dict(protocol, leader_action_representation='threshold')
    threshold = make_model(make_env(output, old, seed), old, seed, METHOD)
    model = make_model(make_env(output, protocol, seed), protocol, seed, METHOD)
    state = model.policy.state_dict()
    copied = []
    for name, value in threshold.policy.state_dict().items():
        if state[name].shape == value.shape:
            state[name] = value.detach().cpu().clone()
            copied.append(name)
    assert len(copied) == 10
    assert set(state) - set(copied) == {'action_net.weight', 'action_net.bias'}
    assert model.normalize_advantage and not model.policy.optimizer.state
    assert model.n_steps == 4800 and model.batch_size == 64
    assert model.n_epochs == 10 and model.gamma == model.gae_lambda == 1.
    return state


def evaluate_rule(output, protocol, name, rule, seeds):
    import numpy as np
    from experiments.platform.response import make_seller_q_learning
    from experiments.platform._support.evaluation import response_record
    seller_learning = make_seller_q_learning(output, protocol)
    rule = np.asarray(rule)
    rows = []
    for cost in protocol['costs']:
        for seed in seeds:
            result = seller_learning.evaluate_rule(rule, cost, seed)
            row = response_record(result)
            # Independently compute the reported frozen-cycle objective.
            prices = protocol['prices']
            values = []
            for a, b in row['cycle_actions']:
                admitted = rule[a, b] if rule.ndim == 3 else np.asarray([a, b]) <= rule[a, b]
                values.append(.25 * np.log1p(sum(np.exp((2. - prices[k]) / .25)
                              for seller, k in enumerate((a, b)) if admitted[seller])))
            assert abs(float(np.mean(values)) - row['frozen_cs']) < 1e-12
            rows.append(row)
    result = {'name': name, 'rule': np.asarray(rule).tolist(), 'responses': rows,
              'cs': statistics.mean(row['frozen_cs'] for row in rows),
              'converged_responses': sum(row['stable_policy_rounds'] >= protocol['stability_required'] for row in rows)}
    write_json(output / (name + '.json'), result)
    return result


def confirm(output, p, seed):
    import numpy as np
    from experiments.platform._support.train import make_env, model_class, verify_checkpoint_policy
    from experiments.platform._support.evaluation import actor_table
    run = output / 'runs' / f'{METHOD}_seed{seed}'
    completed = json.loads((run / 'completion.json').read_text())
    if completed['status'] != 'complete':
        raise ValueError('Training did not complete')
    env = make_env(output, p, seed)
    env.reset()
    model = load_checkpoint(model_class(p, METHOD), run / 'final_model.zip', env=env, device='cpu')
    rule, probabilities = actor_table(model, env)
    mode = json.loads((run / 'mode.json').read_text())
    latest = json.loads((run / 'latest.json').read_text())
    np.testing.assert_array_equal(rule, mode['rule'])
    reload_check = verify_checkpoint_policy(model, run / 'final_model.zip', probabilities, latest['probabilities'])
    target = output / 'confirmation' / f'seed{seed}'
    target.mkdir(parents=True, exist_ok=False)
    result = evaluate_rule(target, p, f'ppo_seed{seed}', rule, p['confirmation_seeds'])
    result.update(seed=seed, reload_check=reload_check)
    write_json(target / 'summary.json', result)
    print(json.dumps({'seed': seed, 'fresh_cs': result['cs']}))


def baselines(output, p):
    import numpy as np
    folder = output / 'baselines'
    folder.mkdir(exist_ok=False)
    m = p['m']
    low = np.minimum.outer(np.arange(m), np.arange(m))
    families = {
        'fixed': [(f'fixed_{k}', np.full((m, m), k)) for k in range(m)],
        'clipped': [(f'clip_{a}_{b}', np.clip(low, a, b)) for a in range(m) for b in range(a, m)],
    }
    selected = []
    for family, policies in families.items():
        rows = [evaluate_rule(folder / 'validation', p, name, rule, p['validation_seeds']) for name, rule in policies]
        winner = max(rows, key=lambda row: row['cs'])  # first candidate wins a tie
        fresh = evaluate_rule(folder / 'confirmation', p, winner['name'], winner['rule'], p['confirmation_seeds'])
        selected.append({'family': family, 'name': winner['name'], 'validation_cs': winner['cs'], 'fresh_cs': fresh['cs']})
    write_json(folder / 'selection.json', {'selected': selected})
    print(json.dumps(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'confirm', 'baselines'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=801)
    parser.add_argument('--smoke', action='store_true', help='One PPO rollout with short seller responses; never a paper result.')
    args = parser.parse_args()
    if args.stage == 'train':
        p = copy.deepcopy(json.loads(CONFIG.read_text()))
        if args.seed not in p['leader_seeds']:
            parser.error('The paper cohort uses seeds 801–825')
        if args.smoke:
            p.update(episodes=80, evaluation_every=80, response_steps=200, stability_required=10,
                     validation_seeds=p['validation_seeds'][:1], confirmation_seeds=p['confirmation_seeds'][:1],
                     purpose='technical smoke only')
        output = fresh_output(args.output)
        record_run(output, 'platform', p)
        from experiments.platform._support.train import train
        train(output, p, METHOD, args.seed, initial_policy_state=initial_state(output, p, args.seed))
    else:
        if args.smoke:
            parser.error('--smoke applies only to train; evaluation uses the saved protocol')
        output = args.output.resolve()
        p = verify_run(output)
        if args.stage == 'confirm':
            confirm(output, p, args.seed)
        else:
            baselines(output, p)


if __name__ == '__main__':
    main()
