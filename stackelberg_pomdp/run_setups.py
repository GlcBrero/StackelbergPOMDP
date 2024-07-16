# Python standard library imports
import os

# Third-party library imports
import wandb
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback

# Local application/library specific imports
from callbacks import GiveModelToEnvCallback, FixPolicyActionsCallback, CustomCheckpointCallback, CustomEvalCallback
from policy_enumeration import policy_enumeration
from env_setups import get_standard_matrix_env, get_simple_allocation_env, get_mspm_env, get_matrix_design_env
from rl_trainer_setup import get_custom_training_algorithm
from stable_baselines3.common import logger

def train_run(config_dict):
    # First, we use config_dict to name our experiment and set up the folder where we log our results
    exp_name = f"exp.{config_dict['learning_method']}." \
               f"{config_dict['experiment_type']}." \
               f"{config_dict['max_steps']}." \
               f"{config_dict['algorithm']}." \
               f"{config_dict['seed']}." \
               f"{config_dict['training_seed']}." \
               f"{config_dict['tot_num_reward_episodes']}." \
               f"{config_dict['tot_num_eq_episodes']}." \
               f"{config_dict['critic_obs']}." \
               f"{config_dict['fix_episode_actions']}." \
               f"{config_dict['followers_algorithm']}." \
               f"{int(config_dict['response_phase_prob'] * 100)}"

    log_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs", exp_name)
    log = logger.configure(folder=log_folder, format_strings=["csv", "stdout", "tensorboard"])
    config_dict["logger"] = log

    # Then, we set up weights and biases, which we use to monitor training
    if config_dict['use_wandb']:
        wandb.tensorboard.patch(root_logdir=log_folder, pytorch=True)
        wandb.init(project="StackPOMDP")
        wandb.config.setdefaults(config_dict)

    # We now create the environments used for training and evaluation depending on which experiment we want to run
    experiment_type_to_env_function = {
        "normal_form": get_standard_matrix_env,
        "matrix_design": get_matrix_design_env,
        "simple_allocation": get_simple_allocation_env,
        "mspm": get_mspm_env
    }

    experiment_type = config_dict["experiment_type"].split(":")[0]
    if experiment_type in experiment_type_to_env_function:
        env = experiment_type_to_env_function[experiment_type](config_dict)
    else:
        raise ValueError(
            "Error: Experiment type not supported. Supported values are: 'normal_form', 'matrix_design', 'simple_allocation', 'mspm'")

    if config_dict['learning_method'].split(":")[0] == 'RL':
        # Common callbacks for both 'Standard' and 'StopOnThreshold'
        callbacks = {
            'giveModelToEnv': GiveModelToEnvCallback(),
            'checkpoint': CustomCheckpointCallback(save_freq=10000000, save_path=log_folder),
            'fixPolicyActions': FixPolicyActionsCallback()
        }

        callback_list = [callbacks['giveModelToEnv'], callbacks['checkpoint']]

        if config_dict['use_wandb']:
            callback_list.append(WandbCallback())

        if config_dict['fix_episode_actions'] == "True":
            callback_list.append(callbacks['fixPolicyActions'])

        if config_dict['learning_method'].split(":")[1] == 'Standard':
            eval_env = experiment_type_to_env_function[experiment_type](config_dict)
            eval_env.is_eval = True
            callbacks['customEval'] = CustomEvalCallback(eval_env, eval_freq=int(config_dict['max_steps'] / 1000))
            callback_list.append(callbacks['customEval'])

        # We are now ready to train our policy
        mod = get_custom_training_algorithm(config_dict, env, tensorboard_folder=os.path.join(log_folder))
        mod.learn(
            total_timesteps=config_dict['max_steps'],
            callback=CallbackList(callback_list),
        )



    elif config_dict["learning_method"].split(":")[0] == 'PolicyEnumeration':

        number_of_policies = int(config_dict["learning_method"].split(":")[1])
        policy_enumeration(env, number_of_policies, log)

    wandb.finish()