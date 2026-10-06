"""Train and certify the frozen contextual follower for commitment consistency."""
import hashlib
import math
import numpy as np
from itertools import product
from pathlib import Path
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from experiments.ablations._support.game import MatrixFixedCommitmentResponseEnv, RepeatedMatrixGame, get_matrix_game, write_response_checkpoint_contract
from experiments.ablations._support.profiles import profile_for_spec
from experiments.ablations._support.reinforce import LEGACY_REINFORCE_LEARNING_RATE, LEGACY_REINFORCE_PRETRAIN_ITERATIONS, Reinforce, legacy_pg_rollout_geometry, reinforce_checkpoint_geometry, save_evaluated_response_checkpoint
from experiments.ablations._support.artifacts import _run_dir, append_jsonl, canonical_json, file_sha256, finish_manifest, mean_summary, start_manifest, validate_logging_args, write_json, SCHEMA_VERSION

# The stopping rule used for all 25 followers in the current paper cohort.
# Only follower regret enters this rule; leader outcomes never select a follower.
CERTIFICATION = dict(minimum_updates=500, maximum_updates=10000,
                     check_every_updates=100, max_regret=0.25)

def _best_stationary_follower_return(spec, commitment):
    best_return = -np.inf
    best_policy = None
    for policy in product(range(spec.num_follower_actions), repeat=spec.num_follower_states):
        game = RepeatedMatrixGame(spec)
        game.reset()
        total = 0.0
        done = False
        while not done:
            leader_action = commitment[game.leader_state]
            follower_action = policy[game.follower_state]
            _, rewards, done, _ = game.step(leader_action, follower_action)
            total += rewards['follower_0']
        if total > best_return:
            best_return = float(total)
            best_policy = tuple((int(value) for value in policy))
    return (best_return, best_policy)

def evaluate_meta_follower(model, spec):
    rows = []
    for commitment in MatrixFixedCommitmentResponseEnv.commitments(spec):
        env = MatrixFixedCommitmentResponseEnv(spec, seed=0, fixed_commitment=commitment)
        observation = env.reset()
        total = 0.0
        actions = []
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            actions.append(action)
            observation, reward, done, _ = env.step(action)
            total += float(reward)
        optimal, best_policy = _best_stationary_follower_return(spec, commitment)
        rows.append({'leader_commitment': list(commitment), 'follower_actions': actions, 'follower_return': total, 'best_stationary_return': optimal, 'best_stationary_policy': list(best_policy), 'regret': float(optimal - total)})
    regrets = [row['regret'] for row in rows]
    returns = [row['follower_return'] for row in rows]
    return {'summary': {**mean_summary(returns), 'commitments': len(rows), 'mean_regret': float(np.mean(regrets)), 'max_regret': float(np.max(regrets)), 'optimal_commitments': int(sum((value <= 1e-09 for value in regrets)))}, 'commitment_rows': rows}

def _resolve_meta_defaults(args):
    if args.algorithm != "REINFORCE":
        raise ValueError("The paper contextual follower uses REINFORCE")
    geometry = legacy_pg_rollout_geometry(_meta_spec(args).episode_length, _meta_spec(args).num_leader_states)
    updates = (CERTIFICATION["maximum_updates"] if getattr(args, "certify", False)
               else LEGACY_REINFORCE_PRETRAIN_ITERATIONS)
    defaults = dict(timesteps=geometry["follower_gradient_samples_per_update"] * updates,
        learning_rate=LEGACY_REINFORCE_LEARNING_RATE, ent_coef=0.,
        episodes_per_batch=geometry["episodes_per_update"], net_arch=(), n_epochs=1,
        n_steps=geometry["follower_gradient_samples_per_update"], batch_size=geometry["follower_gradient_samples_per_update"])
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)

def _meta_model(args, env, episode_length=None):
    _resolve_meta_defaults(args)
    return Reinforce(env=env, learning_rate=args.learning_rate, n_steps=args.n_steps,
                     seed=args.seed, device=args.device, policy_kwargs={"net_arch": []})

def _meta_spec(args):
    return get_matrix_game(args.matrix, memory_mode=args.memory_mode, profile_id=args.profile_id)


def validate_meta_args(args):
    _resolve_meta_defaults(args)
    validate_logging_args(args)
    if args.timesteps < 1:
        raise ValueError('timesteps must be positive')
    spec = _meta_spec(args)
    geometry = legacy_pg_rollout_geometry(spec.episode_length, spec.num_leader_states)
    expected_n_steps = geometry['follower_gradient_samples_per_update']
    if args.n_steps != expected_n_steps:
        raise ValueError('legacy-compatible REINFORCE requires n_steps={} for {} query states'.format(expected_n_steps, spec.num_leader_states))
    if args.episodes_per_batch != geometry['episodes_per_update']:
        raise ValueError('legacy-compatible REINFORCE requires episodes_per_batch={}'.format(geometry['episodes_per_update']))
    if args.learning_rate <= 0:
        raise ValueError('REINFORCE learning_rate must be positive')
    if args.ent_coef != 0.0:
        raise ValueError('legacy REINFORCE has no entropy bonus')
    if tuple(args.net_arch):
        raise ValueError('legacy REINFORCE requires --net-arch linear')
    if args.timesteps % expected_n_steps:
        raise ValueError('REINFORCE timesteps must contain complete optimizer updates')
    if getattr(args, 'certify', False):
        if args.timesteps != expected_n_steps * CERTIFICATION['maximum_updates']:
            raise ValueError('--certify uses its fixed stopping rule; omit --timesteps')
    if args.checkpoint_every:
        raise ValueError('REINFORCE uses post-update checkpoints automatically; do not set --checkpoint-every')
    return

def meta_config(args, spec):
    _resolve_meta_defaults(args)
    sweep_plan = Path(args.sweep_plan).resolve() if args.sweep_plan else None
    profile = profile_for_spec(spec)
    config = {'schema_version': SCHEMA_VERSION, 'stage': 'meta_follower', 'profile_id': profile.profile_id, 'profile': profile.to_dict(), 'matrix': spec.name, 'payoffs': spec.payoffs.tolist(), 'reward_offset': spec.reward_offset, 'reward_profile': profile.profile_id, 'episode_length': spec.episode_length, 'memory_mode': spec.memory_mode, 'query_states': list(range(spec.num_leader_states)), 'algorithm': args.algorithm, 'seed': args.seed, 'timesteps': args.timesteps, 'learning_rate': args.learning_rate, 'ent_coef': args.ent_coef, 'episodes_per_batch': args.episodes_per_batch, 'n_steps': args.n_steps, 'batch_size': args.batch_size, 'n_epochs': args.n_epochs, 'net_arch': list(args.net_arch), 'device': args.device, 'checkpoint_every': args.checkpoint_every, 'attempt': args.attempt, 'sweep_id': args.sweep_id, 'record_key': args.record_key, 'sweep_plan': str(sweep_plan) if sweep_plan else None, 'sweep_plan_sha256': file_sha256(sweep_plan) if sweep_plan else None}
    rollout_geometry = legacy_pg_rollout_geometry(spec.episode_length, spec.num_leader_states)
    checkpoint_geometry = reinforce_checkpoint_geometry(spec.episode_length, spec.num_leader_states)
    config['reinforce_protocol'] = {'loss': 'negative_mean_logp_times_undiscounted_reward_to_go', 'gamma': 1.0, 'baseline': 'none', 'advantage_normalization': False, 'entropy_coefficient': 0.0, 'gradient_clipping': False, 'optimizer': 'Adam', 'optimizer_epsilon': 1e-08, 'learning_rate': float(args.learning_rate), 'batch_steps': int(args.n_steps), 'batch_mode': 'complete_episodes', 'rollout_geometry': rollout_geometry, 'policy': 'bias_free_linear_categorical', 'policy_initialization': 'normc_0.01', 'default_pretraining_iterations': LEGACY_REINFORCE_PRETRAIN_ITERATIONS, 'checkpoint_geometry': checkpoint_geometry, 'checkpoint_semantics': 'post_optimizer_update', 'commitment_distribution': 'uniform_deterministic_tables', 'target': profile.interpretation, 'state_count': int(spec.num_follower_states), 'training_reward_offset': float(spec.reward_offset), 'canonical_legacy_learning_rate': LEGACY_REINFORCE_LEARNING_RATE, 'uses_canonical_legacy_learning_rate': math.isclose(args.learning_rate, LEGACY_REINFORCE_LEARNING_RATE), 'legacy_final_run_state_count': 3, 'legacy_training_reward_offset': -2.5, 'finite_sample_reward_protocol_differs_from_legacy': profile.profile_id != 'legacy_opponent_v1', 'provenance': ['stackerlberg/train/experiments/configurations.py:smipd_leadermemory_pg_pg', 'stackerlberg/models/linear_torch_model.py:LinearTorchModel', 'stackerlberg/wrappers/dict_to_discrete_obs_wrapper.py:DictToDiscreteObsWrapper', 'ray/rllib/algorithms/pg/pg_torch_policy.py:pg_torch_loss', 'ray/rllib/algorithms/pg/utils.py:post_process_advantages']}
    if getattr(args, 'certify', False):
        config['certification'] = dict(CERTIFICATION)
    return config


def _train_certified_follower(model, args, spec, config, run_dir):
    """Continue the same learner until its first qualifying post-update check.

    Matches the October rerun procedure, including checks before the minimum.
    A failed capped run retains its evaluation and an explicitly uncertified
    checkpoint; it never produces model.zip or a usable response contract.
    """
    stopping = config['certification']
    interval = stopping['check_every_updates']
    log = run_dir / 'certification.jsonl'
    evaluation_path = run_dir / 'evaluation.json'
    for updates in range(interval, stopping['maximum_updates'] + 1, interval):
        model.learn(total_timesteps=interval * args.n_steps,
                    reset_num_timesteps=(updates == interval))
        if model._n_updates != updates:
            raise RuntimeError('Follower update accounting mismatch')
        evaluation = evaluate_meta_follower(model, spec)
        summary = evaluation['summary']
        append_jsonl(log, dict(updates=updates, **summary))
        if (updates >= stopping['minimum_updates']
                and np.isfinite(summary['max_regret'])
                and summary['max_regret'] <= stopping['max_regret']):
            checkpoint = run_dir / 'model.zip'
            model.save(checkpoint)
            contract = run_dir / 'response_contract.json'
            write_response_checkpoint_contract(
                spec, checkpoint, 'REINFORCE', path=contract,
                metadata=dict(evaluation=summary, seed=int(args.seed),
                              completed_updates=updates,
                              certification=stopping,
                              checkpoint_semantics='post_optimizer_update'))
            evaluation['certification'] = dict(stopping, completed_updates=updates)
            return evaluation, dict(model=checkpoint, response_contract=contract,
                                    certification=log)
    write_json(evaluation_path, evaluation)
    model.save(run_dir / 'uncertified_final.zip')
    raise RuntimeError('Follower remains uncertified after {} updates'.format(
        stopping['maximum_updates']))

def _train_reinforce_with_post_update_checkpoints(model, args, spec, config, run_dir):
    """Train in bounded chunks and certify only fully applied updates."""
    rollout = legacy_pg_rollout_geometry(spec.episode_length, spec.num_leader_states)
    cadence = reinforce_checkpoint_geometry(spec.episode_length, spec.num_leader_states)
    interval = cadence['follower_gradient_samples_interval']
    config_digest = hashlib.sha256(canonical_json(config).encode('utf-8')).hexdigest()
    checkpoint_log = run_dir / 'checkpoint_evaluations.jsonl'
    artifacts = {'checkpoint_evaluations': checkpoint_log}
    final_evaluation = None
    first_chunk = True
    while int(model.num_timesteps) < int(args.timesteps):
        current = int(model.num_timesteps)
        next_regular = (current // interval + 1) * interval
        target = min(next_regular, int(args.timesteps))
        chunk = target - current
        if chunk < 1 or chunk % int(args.n_steps):
            raise RuntimeError('post-update checkpoint chunk does not contain whole updates')
        model.learn(total_timesteps=chunk, reset_num_timesteps=first_chunk)
        first_chunk = False
        follower_steps = int(model.num_timesteps)
        completed_updates = int(model._n_updates)
        expected_updates = follower_steps // int(args.n_steps)
        if completed_updates != expected_updates:
            raise RuntimeError('REINFORCE update accounting mismatch: observed {}, expected {}'.format(completed_updates, expected_updates))
        executed_steps = completed_updates * rollout['executed_env_steps_per_update']
        is_final = follower_steps == int(args.timesteps)
        checkpoint = run_dir / 'model.zip' if is_final else run_dir / 'meta_follower_update{:04d}_follower{}.zip'.format(completed_updates, follower_steps)
        evaluation, record = save_evaluated_response_checkpoint(model, checkpoint, lambda current_model: evaluate_meta_follower(current_model, spec))
        record.update({'schema_version': SCHEMA_VERSION, 'profile_id': config['profile_id'], 'seed': int(args.seed), 'learning_rate': float(args.learning_rate), 'completed_updates': completed_updates, 'follower_gradient_samples': follower_steps, 'executed_equivalent_steps': int(executed_steps), 'training_config_sha256': config_digest, 'checkpoint_semantics': 'post_optimizer_update'})
        contract_path = run_dir / 'response_contract.json' if is_final else checkpoint.with_suffix('.response_contract.json')
        write_response_checkpoint_contract(spec, checkpoint, 'REINFORCE', path=contract_path, metadata={'evaluation': evaluation['summary'], 'seed': int(args.seed), 'training_config_sha256': config_digest, 'response_training': config['reinforce_protocol'], 'completed_updates': completed_updates, 'follower_gradient_samples': follower_steps, 'executed_equivalent_steps': int(executed_steps)})
        record.update({'contract_filename': contract_path.name, 'contract_sha256': file_sha256(contract_path)})
        append_jsonl(checkpoint_log, record)
        suffix = 'final' if is_final else 'update{:04d}'.format(completed_updates)
        artifacts['checkpoint_{}'.format(suffix)] = checkpoint
        artifacts['checkpoint_contract_{}'.format(suffix)] = contract_path
        if is_final:
            artifacts['model'] = checkpoint
            artifacts['response_contract'] = contract_path
            final_evaluation = evaluation
    if final_evaluation is None:
        raise RuntimeError('REINFORCE training produced no final checkpoint')
    return (final_evaluation, artifacts)

def train_meta_follower(args):
    validate_meta_args(args)
    spec = _meta_spec(args)
    config = meta_config(args, spec)
    run_dir = _run_dir(args.output_root, config)
    config['run_dir'] = str(run_dir)
    write_json(run_dir / 'config.json', config)
    manifest = start_manifest(run_dir, config)
    vec_env = DummyVecEnv([lambda: Monitor(MatrixFixedCommitmentResponseEnv(spec, seed=args.seed), filename=str(run_dir / 'train_monitor'))])
    try:
        model = _meta_model(args, vec_env, spec.episode_length)
        train = (_train_certified_follower if getattr(args, 'certify', False)
                 else _train_reinforce_with_post_update_checkpoints)
        evaluation, artifacts = train(model, args, spec, config, run_dir)
        contract_path = artifacts['response_contract']
        evaluation.update({'schema_version': SCHEMA_VERSION, 'config': config})
        write_json(run_dir / 'evaluation.json', evaluation)
        artifacts['evaluation'] = run_dir / 'evaluation.json'
        finish_manifest(run_dir, manifest, 'completed', artifacts=artifacts)
    except Exception as exc:
        finish_manifest(run_dir, manifest, 'failed', error=exc)
        raise
    finally:
        vec_env.close()
    return run_dir
