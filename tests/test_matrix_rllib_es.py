import os
from pathlib import Path

import numpy as np
import pytest

from stackelberg_pomdp.experiments.matrix_ablations import (
    build_parser,
    leader_config,
    validate_leader_args,
)
from stackelberg_pomdp.matrix_ablations.envs import get_matrix_game
from stackelberg_pomdp.matrix_ablations.rllib_es import (
    EXPECTED_RAY_VERSION,
    RLLIB_ES_IMPLEMENTATION,
    FiniteLookupResponse,
    RllibESSettings,
    _ray_env_creator,
    train_rllib_es,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_native_es_defaults_match_the_audited_historical_runtime():
    spec = get_matrix_game("modified_pd", profile_id="paper_joint_v1")
    settings = RllibESSettings()
    protocol = settings.protocol(spec)

    assert EXPECTED_RAY_VERSION == "2.0.1"
    assert RLLIB_ES_IMPLEMENTATION == "ray_rllib_es_2_0_1"
    assert settings.iterations == 300
    assert settings.num_workers == 1
    assert settings.episodes_per_batch == 1_000
    assert settings.train_batch_size == 1_000
    assert settings.noise_stdev == pytest.approx(0.02)
    assert settings.stepsize == pytest.approx(0.01)
    assert settings.l2_coeff == pytest.approx(0.005)
    assert settings.eval_prob == pytest.approx(0.03)
    assert settings.report_length == 10
    assert settings.noise_size == 250_000_000
    assert protocol["implementation"] == RLLIB_ES_IMPLEMENTATION
    assert protocol["model"] == {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "vf_share_layers": True,
    }
    assert protocol["observation_filter"] == "MeanStdFilter"


def test_finite_response_lookup_is_serializable_and_builds_both_conditions():
    spec = get_matrix_game("modified_pd", profile_id="paper_joint_v1")
    shape = (
        spec.num_leader_actions ** spec.num_leader_states,
        spec.num_follower_states,
    )
    actions = np.zeros(shape, dtype=np.int64)
    response = FiniteLookupResponse(spec, actions.tolist())
    action, state = response.predict(0, deterministic=True)
    assert int(action) == 0
    assert state is None

    for condition in ("observed", "hidden"):
        env = _ray_env_creator({
            "matrix": "modified_pd",
            "profile_id": "paper_joint_v1",
            "condition": condition,
            "seed": 7,
            "response_actions": actions.tolist(),
        })
        try:
            observation = env.reset()
            assert env.observation_space.contains(observation)
            assert observation in range(spec.num_leader_states)
        finally:
            env.close()


def test_cli_has_one_es_treatment_and_no_backend_selector_or_ars():
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            for subparser in choices.values():
                option_strings.update(
                    option
                    for sub_action in subparser._actions
                    for option in sub_action.option_strings
                )
    assert "--es-backend" not in option_strings
    with pytest.raises(SystemExit):
        parser.parse_args([
            "leader", "--experiment", "hidden_queries",
            "--condition", "observed", "--algorithm", "ARS",
        ])


def test_es_config_records_iterations_not_a_requested_timestep_budget():
    checkpoint_dir = (
        REPO_ROOT / "replication/matrix_ablations/checkpoints/"
        "certified_response_seed2"
    )
    args = build_parser().parse_args([
        "leader", "--experiment", "hidden_queries",
        "--condition", "observed", "--algorithm", "ES",
        "--profile-id", "paper_joint_v1",
        "--response-checkpoint", str(checkpoint_dir / "model.zip"),
        "--response-contract-path", str(checkpoint_dir / "response_contract.json"),
        "--response-algorithm", "REINFORCE",
    ])
    validate_leader_args(args)
    spec = get_matrix_game(args.matrix, profile_id=args.profile_id)
    config = leader_config(args, spec)
    assert config["timesteps"] is None
    assert config["es_protocol"]["iterations"] == 300
    assert config["es_protocol"]["step_accounting"] == (
        "measured RLlib timesteps; batch overshoot retained"
    )


@pytest.mark.skipif(
    os.environ.get("RUN_RLLIB_ES_SMOKE") != "1",
    reason="set RUN_RLLIB_ES_SMOKE=1 in the dedicated Ray environment",
)
def test_real_rllib_es_checkpoint_smoke(tmp_path):
    """Opt-in native smoke; the regular unit suite never imports Ray."""

    spec = get_matrix_game("modified_pd", profile_id="paper_joint_v1")
    checkpoint = (
        REPO_ROOT / "replication/matrix_ablations/checkpoints/"
        "certified_response_seed2/model.zip"
    )
    result = train_rllib_es(
        spec=spec,
        condition="hidden",
        seed=999,
        response_checkpoint=checkpoint,
        response_algorithm="REINFORCE",
        run_dir=tmp_path,
        settings=RllibESSettings(
            iterations=1,
            num_workers=1,
            episodes_per_batch=2,
            train_batch_size=20,
            noise_size=100_000,
            eval_prob=0.5,
            explicit_eval_every=1,
            explicit_eval_episodes=2,
            final_eval_episodes=2,
        ),
    )
    assert result["progress"][0]["measured_timesteps_total"] > 0
    assert result["evaluation"]["implementation"] == RLLIB_ES_IMPLEMENTATION
    checkpoint_path = result["artifacts"]["ray_checkpoint"]
    assert checkpoint_path.is_dir()
    assert result["artifacts"]["ray_checkpoint_state"].is_file()
