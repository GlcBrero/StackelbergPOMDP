"""Deterministic Atari evaluation, protocol auditing, and checkpoint selection.

Start with workflow.run_selection for the full screen/confirm pipeline;
rollouts collects trajectories, protocol_audit checks them, selection ranks
candidates, and reporting writes the resulting artifacts.
"""
