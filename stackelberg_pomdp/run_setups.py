# Python standard library imports
import hashlib
import json
import os

# Third-party library imports
import gym
try:
    import wandb
    HAS_WANDB = True
except ImportError as wandb_import_error:
    HAS_WANDB = False
    WANDB_IMPORT_ERROR = wandb_import_error
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

# Local application/library specific imports
try:
    from .callbacks import EVALUATION_WANDB_METRICS, FixPolicyActionsCallback, CustomCheckpointCallback, BackgroundEvalCallback, ExactSPMEvaluationCallback, TrainingProgressCallback, ResponsePhaseDiagnosticsCallback, TrainingRewardCallback, ResponsePhasePolicyCallback, RewardEpisodeTraceCallback
    from .env_setups import get_standard_matrix_env, get_simple_allocation_env, get_mspm_env, get_spm_env, get_matrix_design_env, get_bertrand_env
    from .wrappers.core import MWFollowersWrapper
    from .rl_trainer_setup import get_custom_training_algorithm
except ImportError:
    from callbacks import EVALUATION_WANDB_METRICS, FixPolicyActionsCallback, CustomCheckpointCallback, BackgroundEvalCallback, ExactSPMEvaluationCallback, TrainingProgressCallback, ResponsePhaseDiagnosticsCallback, TrainingRewardCallback, ResponsePhasePolicyCallback, RewardEpisodeTraceCallback
    from env_setups import get_standard_matrix_env, get_simple_allocation_env, get_mspm_env, get_spm_env, get_matrix_design_env, get_bertrand_env
    from wrappers.core import MWFollowersWrapper
    from rl_trainer_setup import get_custom_training_algorithm
from stable_baselines3.common import logger
from stackelberg_pomdp.rl_trainer_setup import ppo_rollout_geometry
from stackelberg_pomdp.training_defaults import resolve_common_optimizer_defaults
from stackelberg_pomdp.follower_responses import round_down_mw_response_games
from stackelberg_pomdp.experiments.common import mw_update_period_from_config


def _format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _requested_response_episodes(config_dict):
    return config_dict['tot_num_response_episodes']


def _effective_tot_num_response_episodes(config_dict):
    requested = _requested_response_episodes(config_dict)
    if config_dict.get('followers_algorithm') != 'MW':
        return requested

    period = mw_update_period_from_config(config_dict)
    if period is None:
        return requested
    return round_down_mw_response_games(requested, period)


def _experiment_name(config_dict):
    config_dict = resolve_common_optimizer_defaults(dict(config_dict))
    experiment_family = config_dict["experiment_type"].split(":")[0]
    if experiment_family == "spm":
        _, setting, num_types = config_dict["experiment_type"].split(":")
        parts = [
            "exp",
            experiment_family,
            setting,
            f"types{num_types}",
            config_dict["algorithm"],
            f"steps{config_dict['max_steps']}",
            f"seed{config_dict['seed']}",
            f"lr{_format_value(config_dict['learning_rate'])}",
            f"ent{_format_value(config_dict.get('ent_coef', 0.01))}",
        ]
        return ".".join(_format_value(part).replace("/", "-") for part in parts)

    parts = [
        "exp",
        experiment_family,
        config_dict["learning_method"],
        config_dict["experiment_type"],
        f"steps{config_dict['max_steps']}",
        config_dict["algorithm"],
        f"seed{config_dict['seed']}",
        f"lr{_format_value(config_dict['learning_rate'])}",
        f"ent{_format_value(config_dict.get('ent_coef', 0.01))}",
        f"ppobatch{config_dict.get('ppo_batch_size', 'na')}",
        f"ppoepochs{config_dict.get('ppo_n_epochs', 'na')}",
        f"pporollout{config_dict.get('ppo_episodes_per_batch') or 'auto'}ep",
        f"reward{config_dict['tot_num_reward_episodes']}",
        f"response{config_dict.get('effective_tot_num_response_episodes', _effective_tot_num_response_episodes(config_dict))}",
        f"critic{config_dict['critic_obs']}",
        config_dict["followers_algorithm"],
        f"mweps{_format_value(config_dict.get('mw_epsilon', MWFollowersWrapper.DEFAULT_EPS))}",
    ]
    if config_dict.get("ppo_rollout_geometry", "complete_episodes") != "complete_episodes":
        parts.append(f"ppogeom{config_dict['ppo_rollout_geometry']}")
    if config_dict.get("mw_fixed_seed") is not None:
        parts.append(f"mwseed{config_dict['mw_fixed_seed']}")
    parts.extend([
        "mwexactact",
        "mwreset" if config_dict.get('mw_reset_weights_each_episode', True) else "mwpersist",
    ])
    parts.append(config_dict.get("pomdp_mode", "stackelberg"))
    if config_dict.get("response_bcce_threshold") is not None:
        parts.extend([
            f"certbcce{_format_value(config_dict['response_bcce_threshold'])}",
            f"minrec{config_dict.get('response_bcce_min_records', 1)}",
            f"maxextra{config_dict.get('response_bcce_max_extra_updates', 10000)}",
        ])
        if config_dict.get("response_bcce_check_freq", 1) != 1:
            parts.append(f"bccefreq{config_dict['response_bcce_check_freq']}")
    if experiment_family == "bertrand":
        parts.extend([
            config_dict.get("platform_observation_space", "no_observation"),
            config_dict.get("platform_intervention", "learn_threshold"),
            f"m{config_dict.get('price_grid_length', 4)}",
            f"p{_format_value(config_dict.get('price_min', 1.05))}-{_format_value(config_dict.get('price_max', 1.7))}",
            f"alpha{_format_value(config_dict.get('follower_alpha', 0.25))}",
            f"beta{_format_value(config_dict.get('follower_beta', 1e-4))}",
            f"lambda{_format_value(config_dict.get('intervention_lambda', 0.0))}",
            f"leaderk{config_dict.get('leader_k', 1)}",
            "warmq" if config_dict.get("warm_start_q", False) else "freshq",
            "sorted" if config_dict.get("sort_obs", False) else "unsorted",
        ])
    elif experiment_family == "simple_allocation":
        parts.append(f"messages{config_dict['experiment_type'].split(':')[1]}")
    elif experiment_family == "mspm":
        _, setting, num_types, num_messages = config_dict["experiment_type"].split(":")
        parts.extend([
            setting,
            f"types{num_types}",
            f"messages{num_messages}",
            "exactexpectedscale",
            "fullmwstate",
            "earlytermination",
            "maxhorizonrollout",
        ])
    elif experiment_family == "normal_form":
        _, game_name, randomized = config_dict["experiment_type"].split(":")
        parts.extend([game_name, f"randomized{randomized}"])

    name = ".".join(_format_value(part).replace("/", "-") for part in parts)
    if len(name) > 180:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
        name = f"{name[:169]}.h{digest}"
    return name


def train_run(config_dict):
    resolve_common_optimizer_defaults(config_dict)
    # First, we use config_dict to name our experiment and set up the folder where we log our results
    exp_name = _experiment_name(config_dict)

    log_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs", exp_name)
    log = logger.configure(folder=log_folder, format_strings=["csv", "stdout"])
    config_dict["logger"] = log

    # Then, we set up weights and biases, which we use to monitor training
    if config_dict['use_wandb'] and not HAS_WANDB:
        raise ImportError(
            "--use_wandb was requested, but the wandb package is not available "
            "in this Python environment."
        ) from WANDB_IMPORT_ERROR

    if config_dict['use_wandb']:
        wandb.init(project="StackPOMDP", name=exp_name)
        wandb.config.setdefaults(config_dict)
        wandb.define_metric("global_step")
        for metric_name in EVALUATION_WANDB_METRICS:
            wandb.define_metric(metric_name, step_metric="global_step")

    # We now create the environments used for training and evaluation depending on which experiment we want to run
    experiment_type_to_env_function = {
        "normal_form": get_standard_matrix_env,
        "matrix_design": get_matrix_design_env,
        "simple_allocation": get_simple_allocation_env,
        "spm": get_spm_env,
        "mspm": get_mspm_env,
        "bertrand": get_bertrand_env,
    }

    experiment_type = config_dict["experiment_type"].split(":")[0]
    if experiment_type in experiment_type_to_env_function:
        env = experiment_type_to_env_function[experiment_type](config_dict)
    else:
        raise ValueError(
            "Error: Experiment type not supported. Supported values are: 'normal_form', 'matrix_design', 'simple_allocation', 'spm', 'mspm', 'bertrand'")

    if config_dict['learning_method'].split(":")[0] != 'RL':
        raise ValueError(f"Unsupported learning_method: {config_dict['learning_method']}")
    if config_dict['learning_method'].split(":")[1] != 'Standard':
        raise ValueError(f"Unsupported RL learning method: {config_dict['learning_method']}")

    callback_list = [
        CustomCheckpointCallback(save_freq=10000000, save_path=log_folder),
        TrainingProgressCallback(
            total_timesteps=config_dict['max_steps'],
            print_freq=config_dict.get('progress_freq', 10000),
        ),
        TrainingRewardCallback(
            print_freq=config_dict.get('reward_print_freq', 1),
        ),
    ]

    response_diagnostic_freq = int(config_dict.get('response_diagnostic_freq', 0))
    if response_diagnostic_freq > 0:
        callback_list.append(ResponsePhaseDiagnosticsCallback(
            print_freq=response_diagnostic_freq,
        ))

    reward_trace_targets = [
        value.strip()
        for value in config_dict.get('reward_trace_targets', '').split(',')
        if value.strip()
    ]
    if reward_trace_targets:
        callback_list.append(RewardEpisodeTraceCallback(
            reward_trace_targets,
            tolerance=config_dict.get('reward_trace_tol', 1e-6),
        ))

    if config_dict.get('response_bcce_threshold') is not None:
        callback_list.append(ResponsePhasePolicyCallback())

    if experiment_type != "spm" and config_dict['fix_episode_actions']:
        callback_list.append(FixPolicyActionsCallback())

    if experiment_type == "spm":
        if config_dict["algorithm"] != "PPO":
            raise ValueError("The standard SPM baseline is defined for PPO.")
        if gym.__version__.split(".")[0] == "0":
            env = gym.wrappers.FlattenObservation(env)
        else:
            env = gym.wrappers.FlattenObservation(env)
        env = Monitor(env)
        spm_eval_config = dict(config_dict)
        spm_eval_env = gym.wrappers.FlattenObservation(
            get_spm_env(spm_eval_config)
        )
        callback_list.append(ExactSPMEvaluationCallback(
            spm_eval_env,
            print_freq=config_dict.get('spm_exact_eval_freq', 10000),
            deterministic=config_dict.get('spm_eval_deterministic', True),
            action_samples=config_dict.get('spm_eval_action_samples', 1),
        ))
        max_episode_transitions = env.unwrapped.max_episode_transitions()
        geometry = ppo_rollout_geometry(config_dict, max_episode_transitions)
        mod = PPO(
            policy="MlpPolicy",
            env=env,
            gamma=1.0,
            learning_rate=config_dict['learning_rate'],
            seed=config_dict["training_seed"],
            n_steps=geometry['n_steps'],
            batch_size=geometry['batch_size'],
            n_epochs=config_dict['ppo_n_epochs'],
            ent_coef=config_dict.get('ent_coef', 0.01),
        )
        mod.set_logger(log)
        _write_optimizer_parameters(mod, log_folder)
        print(f"[train] standard_spm_ppo=true log_folder={log_folder}", flush=True)
        mod.learn(
            total_timesteps=config_dict['max_steps'],
            callback=CallbackList(callback_list),
            log_interval=config_dict.get('ppo_log_interval', 100),
        )
        if config_dict['use_wandb']:
            wandb.finish()
        return

    eval_log = logger.configure(folder=log_folder + "/eval", format_strings=["csv"])
    eval_config = dict(config_dict, logger=eval_log)
    eval_config['is_eval'] = True
    eval_config['tot_num_reward_episodes'] = config_dict.get(
        'eval_reward_episodes',
        config_dict['tot_num_reward_episodes'],
    )
    eval_env = experiment_type_to_env_function[experiment_type](eval_config)
    eval_env = Monitor(eval_env)
    eval_freq = int(config_dict.get('eval_freq', 10000))
    if eval_freq > 0:
        callback_list.append(BackgroundEvalCallback(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=int(config_dict.get('eval_episodes', 1)),
            reward_steps=eval_config['tot_num_reward_episodes'],
        ))

    # We are now ready to train our policy
    mod = get_custom_training_algorithm(config_dict, env, tensorboard_folder=None)
    _write_optimizer_parameters(mod, log_folder)
    print(f"[train] log_folder={log_folder}", flush=True)
    mod.learn(
        total_timesteps=config_dict['max_steps'],
        callback=CallbackList(callback_list),
    )

    if config_dict['use_wandb']:
        wandb.finish()


def _write_optimizer_parameters(model, log_folder):
    """Persist actual constructor values, including automatic rollout sizing."""
    import stable_baselines3

    parameters = {
        "stable_baselines3_version": stable_baselines3.__version__,
        "algorithm": type(model).__name__,
    }
    for key in ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
                "gae_lambda", "ent_coef", "clip_range", "vf_coef", "max_grad_norm"):
        if hasattr(model, key):
            value = getattr(model, key)
            parameters[key] = value(1.0) if callable(value) else value
    with open(os.path.join(log_folder, "optimizer_parameters.json"), "w") as stream:
        json.dump(parameters, stream, indent=2)
