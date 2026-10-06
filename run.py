#!/usr/bin/env python3
"""Select a paper command or a custom experiment; each runner owns its options."""
import os
from pathlib import Path
import runpy
import sys

CODE = Path(__file__).resolve().parent
TARGETS = {
    'mspm': ('experiments.mspm._support._runner', 'MSPM: types 2–6, seeds 96–120'),
    'platform': ('experiments.platform._support.cli', 'Johnson-grid PPO, confirmation and baselines'),
    'platform-search': ('experiments.platform._support.search', 'Uniform search over valid platform rules'),
    'atari-gameplay': ('pretrained.atari.gameplay', 'Atari E0 gameplay curriculum'),
    'atari-response': ('pretrained.atari.meta_response', 'Atari E1 meta-responses'),
    'atari-leader': ('experiments.atari._support.runners.train_atari_stackpomdp_leader_sb3', 'Atari E2 StackPOMDP leaders'),
    'atari-evaluate-fixed': ('experiments.atari._support.runners.evaluate_fixed', 'Paper endpoints and independent fresh-final evaluation'),
    'atari-evaluate-response': ('experiments.atari._support.runners.evaluate_atari_meta_response_sb3', 'Atari E1 response evaluation'),
    'atari-evaluate-leader': ('experiments.atari._support.runners.evaluate_atari_stackpomdp_leader_sb3', 'Atari E2 checkpoint selection/evaluation'),
    'normal-form': ('experiments.ablations._support.cli', 'Commitment consistency and reward timing'),
    'response-credit': ('experiments.ablations.response_credit.experiment', 'Independent scalar response-credit ablation'),
}

# Shortcuts expand to the same commands and arguments shown in the README.
ALIASES = {
    'atari': ('atari-leader', ()),
    'commitment-consistency': ('normal-form', ('leader', '--figure', 'fig_memory_pg')),
    'reward-timing': ('normal-form', ('leader', '--figure', 'fig_bots_leaderreward')),
}


def custom_experiments():
    """Discover copyable experiment folders and standalone user modules."""
    names = set()
    for path in (CODE / 'experiments').iterdir():
        if path.is_dir() and (path / 'experiment.py').is_file():
            names.add(path.name)
        elif path.is_file() and path.suffix == '.py':
            names.add(path.stem)
    return sorted(name for name in names if name.isidentifier() and not name.startswith('_'))


def show_commands():
    print('Usage: python run.py EXPERIMENT [arguments]\n')
    for name, (_, description) in TARGETS.items():
        print(f'  {name:26} {description}')
    for name, (target, arguments) in ALIASES.items():
        print(f'  {name:26} Alias for {" ".join((target, *arguments))}')
    for module_name in custom_experiments():
        name = module_name.replace('_', '-')
        if name not in TARGETS and name not in ALIASES:
            print(f'  {name:26} Custom experiment; shared default PPO trainer')
    print('\nUse EXPERIMENT --help for its arguments. See README.md for paper commands.')


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {'--list', '-h', '--help'}:
        show_commands()
        return

    command, *arguments = arguments
    command = command.replace('_', '-')
    target, prefix = ALIASES.get(command, (command, ()))
    arguments = [*prefix, *arguments]
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.path.insert(0, str(CODE))

    if target not in TARGETS:
        name = target.replace('-', '_')
        if name not in custom_experiments():
            raise SystemExit(f'Unknown experiment: {target}; use --list')
        from support.custom import main as run_custom
        return run_custom(name, arguments)

    # Existing paper runners read sys.argv. Restore it when they finish or fail.
    module, _ = TARGETS[target]
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *arguments]
        runpy.run_module(module, run_name='__main__')
    finally:
        sys.argv = previous_argv


if __name__ == '__main__':
    main()
