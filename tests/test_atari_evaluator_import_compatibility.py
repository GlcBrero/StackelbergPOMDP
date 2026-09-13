"""Historical entry-point globals still resolve after extracting their owners."""

import importlib
import pickle

import pytest


@pytest.mark.parametrize("legacy_module,name,owner", [
    (
        "replication.atari.evaluate_atari_stackpomdp_leader_sb3",
        "evaluate_e2_model",
        "stackelberg_pomdp.evaluation.atari.rollouts",
    ),
    (
        "replication.atari.evaluate_atari_stackpomdp_leader_sb3",
        "audit_e2_protocol",
        "stackelberg_pomdp.evaluation.atari.protocol_audit",
    ),
    (
        "replication.atari.evaluate_atari_stackpomdp_leader_sb3",
        "load_e2_checkpoint",
        "stackelberg_pomdp.checkpoints.atari_evaluation",
    ),
    (
        "replication.atari.evaluate_atari_stackpomdp_leader_sb3",
        "run_selection",
        "stackelberg_pomdp.evaluation.atari.workflow",
    ),
    (
        "replication.atari.train_atari_meta_response_sb3",
        "gameplay_actor_sha256",
        "stackelberg_pomdp.checkpoints.atari",
    ),
    (
        "replication.atari.train_atari_meta_response_sb3",
        "module_parameter_sha256",
        "stackelberg_pomdp.checkpoints.atari",
    ),
    (
        "replication.atari.train_atari_stackpomdp_leader_sb3",
        "validate_e2_provenance_manifest",
        "stackelberg_pomdp.checkpoints.atari_provenance",
    ),
])
def test_historical_helper_import_and_pickle_global(legacy_module, name, owner):
    expected = getattr(importlib.import_module(owner), name)
    assert getattr(importlib.import_module(legacy_module), name) is expected
    # Protocol 0 GLOBAL represents an object saved under its old module path.
    stored_global = f"c{legacy_module}\n{name}\n.".encode("ascii")
    assert pickle.loads(stored_global) is expected
