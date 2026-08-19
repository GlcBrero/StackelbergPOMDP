import json
from collections import OrderedDict

import numpy as np
import torch

from stackelberg_pomdp.experiments.matrix_ablations import (
    build_parser,
    train_leader,
    validate_leader_args,
)
from stackelberg_pomdp.envs.matrix import (
    MatrixFixedCommitmentResponseEnv,
    get_matrix_game,
    make_meta_leader_env,
    write_response_checkpoint_contract,
)
from stackelberg_pomdp.matrix_ablations.pg import (
    LeaderPolicyGradient,
    leader_pg_policy_loss,
    leader_pg_rollout_geometry,
)
from stackelberg_pomdp.matrix_ablations.reinforce import Reinforce


class _FixedResponse:
    def predict(self, observation, deterministic=True):
        return np.asarray(0), None


def _leader_env(memory_mode="joint", visibility="hidden"):
    spec = get_matrix_game("modified_pd", memory_mode=memory_mode)
    env = make_meta_leader_env(
        spec,
        response_model=_FixedResponse(),
        query_visibility=visibility,
        seed=17,
    )
    return spec, env


def test_leader_pg_geometry_matches_historical_complete_episode_collection():
    paper_observed = leader_pg_rollout_geometry(5, 5, False)
    paper_hidden = leader_pg_rollout_geometry(5, 5, True)
    legacy_observed = leader_pg_rollout_geometry(5, 3, False)
    legacy_hidden = leader_pg_rollout_geometry(5, 3, True)

    assert paper_observed["episodes_per_update"] == 10
    assert paper_observed["executed_env_steps_per_update"] == 100
    assert paper_observed["stored_gradient_samples_per_update"] == 100
    assert paper_hidden["episodes_per_update"] == 10
    assert paper_hidden["executed_env_steps_per_update"] == 100
    assert paper_hidden["stored_gradient_samples_per_update"] == 50

    assert legacy_observed["episodes_per_update"] == 13
    assert legacy_observed["executed_env_steps_per_update"] == 104
    assert legacy_observed["stored_gradient_samples_per_update"] == 104
    assert legacy_hidden["episodes_per_update"] == 13
    assert legacy_hidden["executed_env_steps_per_update"] == 104
    assert legacy_hidden["stored_gradient_samples_per_update"] == 65


def test_leader_pg_has_exact_linear_policy_and_independent_training_samples():
    _, env = _leader_env()
    try:
        model = LeaderPolicyGradient(
            env=env, n_steps=50, learning_rate=0.008, seed=7, device="cpu"
        )
        trainable = [
            name for name, parameter in model.policy.named_parameters()
            if parameter.requires_grad
        ]
        assert trainable == ["action_net.weight"]
        assert model.policy.action_net.bias is None
        assert torch.allclose(
            torch.linalg.vector_norm(model.policy.action_net.weight, dim=1),
            torch.full((2,), 0.01),
            atol=1e-7,
        )
        assert model.policy.optimizer.defaults["eps"] == 1e-8
        assert model.gamma == 1.0
        assert model.gae_lambda == 1.0
        assert model.ent_coef == 0.0
        assert model.vf_coef == 0.0
        assert np.isinf(model.max_grad_norm)
        assert not hasattr(model.policy, "obs_action_map")

        # Uniform logits expose the distinction from Atari's exact-action
        # replay: repeated visits independently sample both actions.
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
        observation = OrderedDict({"base_environment": 0})
        stochastic = [
            int(model.predict(observation, deterministic=False)[0])
            for _ in range(256)
        ]
        deterministic = [
            int(model.predict(observation, deterministic=True)[0])
            for _ in range(32)
        ]
        assert set(stochastic) == {0, 1}
        assert set(deterministic) == {0}
    finally:
        env.close()


def test_leader_pg_loss_is_exact_logp_times_undiscounted_reward_to_go():
    probabilities = torch.tensor([0.25, 0.75], requires_grad=True)
    log_prob = torch.log(probabilities)
    reward_to_go = torch.tensor([2.0, -1.0])
    loss = leader_pg_policy_loss(log_prob, reward_to_go)
    expected = -(2.0 * np.log(0.25) - np.log(0.75)) / 2.0
    assert np.isclose(float(loss.detach()), expected)
    loss.backward()
    assert torch.allclose(probabilities.grad, torch.tensor([-4.0, 2.0 / 3.0]))


def test_leader_pg_observed_and_hidden_rollouts_execute_full_episodes():
    for memory_mode in ("joint", "opponent"):
        for visibility in ("observed", "hidden"):
            spec, env = _leader_env(memory_mode, visibility)
            geometry = leader_pg_rollout_geometry(
                spec.episode_length,
                spec.num_leader_states,
                visibility == "hidden",
            )
            try:
                model = LeaderPolicyGradient(
                    env=env,
                    n_steps=geometry["stored_gradient_samples_per_update"],
                    learning_rate=0.008,
                    seed=13,
                    device="cpu",
                )
                model.min_completed_episodes_per_rollout = geometry[
                    "episodes_per_update"
                ]
                model.learn(
                    total_timesteps=geometry["executed_env_steps_per_update"]
                )
                assert model._n_updates == 1
                assert model.num_timesteps == geometry[
                    "executed_env_steps_per_update"
                ]
                assert model.last_rollout_stored_steps == geometry[
                    "stored_gradient_samples_per_update"
                ]
                assert model.last_rollout_executed_steps == geometry[
                    "executed_env_steps_per_update"
                ]
                assert model.last_rollout_completed_episodes == geometry[
                    "episodes_per_update"
                ]
            finally:
                env.close()


def test_leader_pg_deterministic_evaluation_is_phase_consistent_and_loadable(
        tmp_path
):
    spec, env = _leader_env("joint", "hidden")
    try:
        model = LeaderPolicyGradient(
            env=env, n_steps=50, learning_rate=0.008, seed=5, device="cpu"
        )
        observation = env.reset()
        done = False
        consistency = []
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, info = env.step(action)
            if info.get("is_reward_phase"):
                consistency.append(info["commitment_consistent"])
        assert consistency == [True] * spec.episode_length

        checkpoint = tmp_path / "leader_pg"
        model.save(str(checkpoint))
        loaded = LeaderPolicyGradient.load(
            str(checkpoint), env=env, device="cpu"
        )
        observation = OrderedDict({"base_environment": 0})
        assert int(loaded.predict(observation, deterministic=True)[0]) in (0, 1)
        assert [
            name for name, parameter in loaded.policy.named_parameters()
            if parameter.requires_grad
        ] == ["action_net.weight"]
    finally:
        env.close()


def test_pg_cli_records_protocol_and_all_three_step_counters(tmp_path):
    spec = get_matrix_game("modified_pd", memory_mode="joint")
    response = Reinforce(
        env=MatrixFixedCommitmentResponseEnv(spec, seed=99),
        n_steps=50,
        seed=99,
        device="cpu",
    )
    response_path = tmp_path / "response"
    response.save(str(response_path))
    response_checkpoint = response_path.with_suffix(".zip")
    write_response_checkpoint_contract(
        spec,
        response_checkpoint,
        "REINFORCE",
        metadata={"purpose": "leader_pg_cli_test"},
    )

    args = build_parser().parse_args([
        "leader",
        "--experiment", "hidden_queries",
        "--condition", "hidden",
        "--algorithm", "PG",
        "--response-checkpoint", str(response_checkpoint),
        "--response-algorithm", "REINFORCE",
        "--allow-uncertified-response",
        "--timesteps", "1",
        "--eval-freq", "50",
        "--eval-episodes", "1",
        "--final-eval-episodes", "1",
        "--output-root", str(tmp_path / "runs"),
    ])
    run_dir = train_leader(args)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    progress = [
        json.loads(line)
        for line in (run_dir / "progress.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert config["algorithm"] == "PG"
    assert config["ent_coef"] == 0.0
    assert config["pg_protocol"]["training_action_sampling"] == (
        "independent_per_state_visit"
    )
    assert config["pg_protocol"]["requested_executed_env_steps"] == 1
    assert config["pg_protocol"]["planned_updates"] == 1
    assert config["pg_protocol"]["planned_executed_env_steps"] == 100
    assert manifest["status"] == "completed"
    assert manifest["training_accounting"]["completed_updates"] == 1
    assert manifest["training_accounting"]["stored_gradient_samples"] == 50
    assert manifest["training_accounting"]["executed_env_steps"] == 100
    assert any(row["row_type"] == "training_update" for row in progress)
    for row in progress:
        assert "completed_updates" in row
        assert "stored_gradient_samples" in row
        assert "executed_env_steps" in row


def test_pg_rejects_unaudited_non_hidden_query_diagnostics():
    args = build_parser().parse_args([
        "leader",
        "--experiment", "q_reset",
        "--condition", "reset",
        "--algorithm", "PG",
    ])
    try:
        validate_leader_args(args)
    except ValueError as exc:
        assert "currently defined only for hidden_queries" in str(exc)
    else:
        raise AssertionError("PG silently generalized to an unaudited diagnostic")
