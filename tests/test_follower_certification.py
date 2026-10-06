"""The paper stopping rule cannot release an early or uncertified follower."""
from types import SimpleNamespace
import json

import pytest

from experiments.ablations._support import meta_training as training
from experiments.ablations._support.cli import build_parser


class Learner:
    _n_updates = 0

    def __init__(self):
        self.calls = []

    def learn(self, total_timesteps, reset_num_timesteps):
        self.calls.append((total_timesteps, reset_num_timesteps))
        self._n_updates += total_timesteps // 10

    def save(self, path):
        path.write_bytes(b'test checkpoint')


@pytest.mark.parametrize('qualifies_at, expected', [(100, 500), (700, 700), (1100, None)])
def test_first_eligible_certificate_or_failure(tmp_path, monkeypatch, qualifies_at, expected):
    model = Learner()
    monkeypatch.setattr(training, 'evaluate_meta_follower', lambda model, spec:
                        {'summary': {'max_regret': .25 if model._n_updates >= qualifies_at else 1.}})
    def write_contract(spec, checkpoint, algorithm, path, metadata):
        path.write_text(json.dumps(metadata))
    monkeypatch.setattr(training, 'write_response_checkpoint_contract', write_contract)
    config = {'certification': dict(minimum_updates=500, maximum_updates=1000,
                                    check_every_updates=100, max_regret=.25)}
    args = SimpleNamespace(n_steps=10, seed=0)
    if expected is None:
        with pytest.raises(RuntimeError, match='uncertified'):
            training._train_certified_follower(model, args, None, config, tmp_path)
        assert (tmp_path / 'uncertified_final.zip').is_file()
        assert not (tmp_path / 'model.zip').exists()
        assert not (tmp_path / 'response_contract.json').exists()
    else:
        evaluation, artifacts = training._train_certified_follower(model, args, None, config, tmp_path)
        assert model._n_updates == expected
        assert evaluation['certification']['completed_updates'] == expected
        assert json.loads(artifacts['response_contract'].read_text())['completed_updates'] == expected
    assert model.calls[0] == (1000, True)
    assert all(call == (1000, False) for call in model.calls[1:])
    history = [json.loads(line) for line in (tmp_path / 'certification.jsonl').read_text().splitlines()]
    assert [row['updates'] for row in history] == list(range(100, (expected or 1000) + 1, 100))


def test_paper_certification_budget_is_explicit():
    args = build_parser().parse_args(['meta-follower', '--memory-mode', 'opponent', '--certify'])
    training.validate_meta_args(args)
    config = training.meta_config(args, training._meta_spec(args))
    assert config['certification'] == dict(minimum_updates=500, maximum_updates=10000,
                                         check_every_updates=100, max_regret=.25)
    assert args.timesteps == 10000 * args.n_steps
    args = build_parser().parse_args(['meta-follower', '--certify', '--timesteps', '10'])
    with pytest.raises(ValueError):
        training.validate_meta_args(args)
