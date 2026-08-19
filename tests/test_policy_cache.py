from stackelberg_pomdp.policy_cache import FixedActionPolicyMixin


class _ScalarPolicy(FixedActionPolicyMixin):
    def __init__(self):
        self._initialize_fixed_action_cache()


class _VectorPolicy(_ScalarPolicy):
    def _action_cache_row(self, key):
        return key[0]


def test_fixed_action_cache_lifecycle():
    policy = _ScalarPolicy()
    assert not policy.fix_actions
    assert policy.obs_action_map == {}

    policy.fix_policy_actions()
    policy.obs_action_map[(1,)] = "action"
    policy.clear_obs_action_map()

    assert policy.fix_actions
    assert policy.obs_action_map == {}


def test_scalar_policy_treats_row_completion_as_full_episode_completion():
    policy = _ScalarPolicy()
    policy.obs_action_map[(1,)] = "action"

    policy.clear_obs_action_map(rows=[0])

    assert policy.obs_action_map == {}


def test_vector_policy_clears_only_completed_rows():
    policy = _VectorPolicy()
    policy.obs_action_map[(0, b"a")] = "first"
    policy.obs_action_map[(1, b"b")] = "second"

    policy.clear_obs_action_map(rows=[1])

    assert policy.obs_action_map == {(0, b"a"): "first"}
