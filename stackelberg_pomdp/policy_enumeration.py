from leader_policies import RandomPolicy
from utils import run_episode

def policy_enumeration(env, number_of_policies, logger):
    # Create a pool of policies
    policy_pool = {i: RandomPolicy(env, i) for i in range(number_of_policies)}

    # Loop over the policies
    for i, policy in policy_pool.items():
        env.policy = policy
        run_episode(env, policy)