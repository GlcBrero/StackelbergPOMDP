"""Evaluate all paper endpoints or the final policy, without checkpoint selection."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.atari._support.checkpoints.atari_evaluation import load_e1_response, load_e2_checkpoint
from experiments.atari._support.evaluation.contracts import environment_config, environment_config_sha256
from experiments.atari._support.evaluation.rollouts import evaluate_checkpoint


PAPER_ENDPOINTS = (0, 391140, 781460, 1171780, 1562100)


def evaluate(checkpoints, output, *, role, response, rom, episodes=20, seed_start=4000001):
    """The fixed-endpoint evaluation used by the October Unity reruns."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(leader_role=role, response_checkpoint=str(response),
                           device='cpu', gameplay_horizon=200, event_tail_steps=0,
                           fixed_event_steps=None, noop_max=30, max_frames=100000,
                           rom_path=str(rom))
    config = environment_config(args)
    config_hash = environment_config_sha256(config)
    response_hash = hashlib.sha256(Path(response).read_bytes()).hexdigest()
    frozen = load_e1_response(response, leader_role=role, device='cpu', expected_sha256=response_hash)
    rows = []
    for checkpoint in checkpoints:
        checkpoint = Path(checkpoint)
        model = load_e2_checkpoint(checkpoint, leader_role=role, device='cpu')
        result = evaluate_checkpoint(
            model, checkpoint, args=args, response_model=frozen,
            response_hash=response_hash, config=config, config_hash=config_hash,
            episodes=episodes, seed_start=seed_start, phase='fixed_endpoint')
        # Preserve failed protocol evidence before refusing the result.
        with gzip.open(output / (checkpoint.stem + '.json.gz'), 'wt') as handle:
            json.dump(result, handle, allow_nan=False)
        if not result['protocol']['passed']:
            raise ValueError(f"Atari protocol failed: {result['protocol']}")
        decisions = result['training_total_timesteps']
        if decisions % 820:
            raise ValueError('A paper endpoint must contain complete four-episode updates')
        rows.append(dict(checkpoint=checkpoint.name, checkpoint_sha256=result['checkpoint_sha256'],
                         policy_decisions=decisions, physical_transitions=decisions // 205 * 210,
                         optimizer_epochs=int(model._n_updates), episodes=episodes,
                         seed_start=seed_start, seed_end=seed_start + episodes - 1,
                         protocol_passed=True, summary=result['summary']))
        (output / 'summary.json').write_text(json.dumps(
            dict(role=role, response_sha256=response_hash,
                 selection='none; all prespecified endpoints', rows=rows),
            indent=2, allow_nan=False) + '\n')
        del model
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leader-role', choices=('buyer', 'seller'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Full-budget final leader ZIP.')
    parser.add_argument('--response-checkpoint', type=Path, required=True)
    parser.add_argument('--rom-path', default=os.environ.get('STACKPOMDP_SPACE_INVADERS_ROM'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh-final', action='store_true',
                        help='Final checkpoint only: 100 fresh episodes starting at seed 5000001.')
    args = parser.parse_args(argv)
    if not args.rom_path or not Path(args.rom_path).is_file():
        parser.error('Supply the Space Invaders ROM via --rom-path or STACKPOMDP_SPACE_INVADERS_ROM')
    checkpoints = ([] if args.fresh_final else [args.checkpoint.with_name(
        args.checkpoint.stem + f'_step{step}.zip') for step in PAPER_ENDPOINTS]) + [args.checkpoint]
    for path in [args.response_checkpoint, *checkpoints]:
        if not path.is_file():
            parser.error(f'Missing required checkpoint: {path}')
    rows = evaluate(checkpoints, args.output, role=args.leader_role,
                    response=args.response_checkpoint, rom=args.rom_path,
                    episodes=100 if args.fresh_final else 20,
                    seed_start=5000001 if args.fresh_final else 4000001)
    expected = [1952420] if args.fresh_final else [*PAPER_ENDPOINTS, 1952420]
    if [row['policy_decisions'] for row in rows] != expected:
        raise ValueError('The supplied models do not match the full paper endpoints')
    print(f'Evaluated {len(rows)} prespecified endpoints in {args.output}')


if __name__ == '__main__':
    main()
