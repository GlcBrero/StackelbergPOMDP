"""Shared fixed-action cache contract for StackPOMDP leader policies.

The StackPOMDP construction requires one sampled leader action to be reused
whenever the same actor-visible observation recurs in an outer episode.
Concrete policies remain responsible for constructing cache keys and applying
cached actions because their observation and action structures differ.
"""


class FixedActionPolicyMixin:
    """Manage cache lifecycle shared by ordinary and Atari policies."""

    def _initialize_fixed_action_cache(self):
        self.fix_actions = False
        self.obs_action_map = {}

    def fix_policy_actions(self):
        """Enable fixed actions for subsequent policy calls."""

        self.fix_actions = True

    def _action_cache_row(self, key):
        """Return a vector-environment row for ``key``, if one is encoded."""

        del key
        return None

    def clear_obs_action_map(self, rows=None):
        """Clear all cached actions or caches belonging to completed rows.

        Ordinary policies use one environment and therefore clear the entire
        map.  Vector-aware policies override :meth:`_action_cache_row` so only
        completed rows are removed.
        """

        if rows is None:
            self.obs_action_map = {}
            return
        completed = {int(row) for row in rows}
        if not self.obs_action_map:
            return
        row_values = {
            self._action_cache_row(key) for key in self.obs_action_map
        }
        if None in row_values:
            self.obs_action_map = {}
            return
        self.obs_action_map = {
            key: value
            for key, value in self.obs_action_map.items()
            if self._action_cache_row(key) not in completed
        }


__all__ = ["FixedActionPolicyMixin"]
