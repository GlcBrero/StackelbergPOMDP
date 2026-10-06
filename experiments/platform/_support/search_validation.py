"""Evaluate a frozen random-search winner on the figure's fixed seller cohort."""
import statistics
from pathlib import Path

from experiments.platform._support import search
from support.common import fresh_output, record_run, verify_run


def evaluate(source):
    source = Path(source).resolve()
    protocol = verify_run(source)
    final = search.read(source / 'run/final.json')
    frozen_hash = search.sha(source / 'run/final.json')
    completion = search.read(source / 'run/completion.json')
    if completion['status'] != 'complete' or frozen_hash != completion['final_sha256']:
        raise ValueError('Complete and freeze the search before plotting evaluation')
    seeds = search.read(Path(__file__).with_name('paper.json'))['validation_seeds']
    schedule = search.read(source / 'training_seeds/schedule.json')
    assert set(seeds).isdisjoint(schedule)
    assert set(seeds).isdisjoint(protocol['confirmation_seeds'])
    protocol = dict(protocol, validation_seeds=seeds,
                    purpose='Fixed validation cohort for the figure only; no selection',
                    selected_candidate=final['candidate'], frozen_winner_sha256=frozen_hash)
    output = fresh_output(source / 'plot_validation')
    record_run(output, 'platform-frozen-random-winner-plot-validation', protocol)
    previous_root = search.ROOT
    search.ROOT = output
    try:
        learner = search.make_seller_q_learning(protocol)
        rule = search.rule_from_thresholds(final['thresholds'])
        rows = []
        for cost in protocol['costs']:
            for seed in seeds:
                row = search.response_row(learner.evaluate_rule(rule, cost, seed))
                search.audit_response(protocol, rule, row)
                rows.append(row)
        verify_run(output)
        assert frozen_hash == search.sha(source / 'run/final.json')
        per_cost = {str(c): statistics.mean(r['frozen_cs'] for r in rows if r['cost'] == c)
                    for c in protocol['costs']}
        summary = dict(status='complete', selected_candidate=final['candidate'],
                       final_sha256=frozen_hash, validation_seeds=seeds,
                       validation_responses=len(rows),
                       converged_responses=sum(r['stable_policy_rounds'] >= protocol['stability_required']
                                               for r in rows),
                       cs=statistics.mean(per_cost.values()), per_cost=per_cost,
                       responses=rows, selection_changed=False, purpose=protocol['purpose'])
        search.write(output / 'summary.json', summary)
    finally:
        search.ROOT = previous_root
    return output
