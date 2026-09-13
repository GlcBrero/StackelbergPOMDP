import json

import numpy as np
import pytest

from stackelberg_pomdp.envs.matrix import (
    LegacyMatrixQLeaderEnv,
    MatrixFixedCommitmentResponseEnv,
    MatrixMetaLeaderEnv,
    RepeatedMatrixGame,
    decode_meta_follower_observation,
    encode_meta_follower_observation,
    get_matrix_game,
    validate_response_checkpoint_contract,
    write_response_checkpoint_contract,
)


class ConstantFollower:
    def __init__(self, action=0):
        self.action = int(action)

    def predict(self, observation, deterministic=True):
        assert deterministic
        return np.asarray(self.action), None


def test_joint_memory_preserves_raw_centered_payoffs_and_five_states():
    spec = get_matrix_game("prisoners_dilemma", memory_mode="joint")
    assert spec.num_leader_states == 5
    assert spec.num_follower_states == 5
    np.testing.assert_allclose(
        spec.centered_payoffs[:, :, 0],
        [[-1.0, -3.0], [0.0, -2.0]],
    )
    np.testing.assert_allclose(
        spec.centered_payoffs[:, :, 1],
        [[-1.0, 0.0], [-3.0, -2.0]],
    )

    game = RepeatedMatrixGame(spec)
    game.reset()
    observations, rewards, done, _ = game.step(1, 0)
    assert observations == {"leader": 3, "follower_0": 3}
    assert rewards == {"leader": 0.0, "follower_0": -3.0}
    assert not done
    for _ in range(4):
        _, _, done, _ = game.step(0, 0)
    assert done


def test_opponent_memory_remains_available_for_legacy_comparison():
    spec = get_matrix_game("prisoners_dilemma", memory_mode="opponent")
    game = RepeatedMatrixGame(spec)
    game.reset()
    observations, _, _, _ = game.step(1, 0)
    assert observations == {"leader": 1, "follower_0": 2}
    assert spec.num_leader_states == 3
    assert spec.num_follower_states == 3


def test_e1_commitments_cover_all_tables_and_stay_fixed_within_episode():
    spec = get_matrix_game("prisoners_dilemma")
    commitments = MatrixFixedCommitmentResponseEnv.commitments(spec)
    assert len(commitments) == 32
    assert commitments[0] == (0, 0, 0, 0, 0)
    assert commitments[-1] == (1, 1, 1, 1, 1)

    env = MatrixFixedCommitmentResponseEnv(
        spec, fixed_commitment=(0, 1, 0, 1, 0)
    )
    observation = env.reset()
    assert env.observation_space.n == 160
    assert observation == 10
    for _ in range(spec.episode_length):
        observation, _, done, info = env.step(0)
        assert info["leader_commitment"] == (0, 1, 0, 1, 0)
    assert done


def test_meta_follower_encoding_matches_legacy_mixed_radix_layout():
    joint = get_matrix_game("prisoners_dilemma", memory_mode="joint")
    assert encode_meta_follower_observation(
        joint, 0, (0, 1, 0, 1, 0)
    ) == 10
    assert encode_meta_follower_observation(
        joint, 4, (1, 1, 1, 1, 1)
    ) == 159

    opponent = get_matrix_game("prisoners_dilemma", memory_mode="opponent")
    assert encode_meta_follower_observation(
        opponent, 2, (1, 0, 1)
    ) == 21

    seen = set()
    for state in range(joint.num_follower_states):
        for commitment in MatrixFixedCommitmentResponseEnv.commitments(joint):
            encoded = encode_meta_follower_observation(
                joint, state, commitment
            )
            assert decode_meta_follower_observation(joint, encoded) == (
                state, commitment
            )
            seen.add(encoded)
    assert seen == set(range(160))


def test_response_contract_reuses_identical_follower_game_and_rejects_mismatch(
        tmp_path
):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"frozen response bytes")
    from dataclasses import replace

    canonical = get_matrix_game("prisoners_dilemma")
    payoffs = canonical.payoffs.copy()
    payoffs[:, :, 0] += 1.0  # Follower behavior is independent of leader payoffs.
    modified = replace(canonical, payoffs=payoffs)
    contract_path = write_response_checkpoint_contract(
        modified,
        checkpoint,
        "PPO",
        metadata={"evaluation": {"max_regret": 0.0}},
    )
    payload = validate_response_checkpoint_contract(
        canonical, checkpoint, algorithm="PPO"
    )
    assert payload["metadata"]["evaluation"]["max_regret"] == 0.0

    incompatible = canonical.with_memory_mode("opponent")
    with pytest.raises(ValueError, match="incompatible"):
        validate_response_checkpoint_contract(
            incompatible, checkpoint, algorithm="PPO"
        )
    with contract_path.open() as handle:
        contract = json.load(handle)
    assert contract["checkpoint_filename"] == "model.zip"
    assert contract["response_game"]["follower_observation_space_n"] == 160


def test_reinforce_response_mechanics_checkpoint_and_meta_leader_load(tmp_path):
    import torch

    from stackelberg_pomdp.experiments.matrix_ablations import (
        _meta_model,
        build_parser,
        evaluate_meta_follower,
        meta_config,
        validate_meta_args,
    )
    from stackelberg_pomdp.matrix_ablations.reinforce import (
        Reinforce,
        save_evaluated_response_checkpoint,
    )

    args = build_parser().parse_args([
        "meta-follower", "--algorithm", "REINFORCE", "--timesteps", "50",
    ])
    validate_meta_args(args)
    spec = get_matrix_game("prisoners_dilemma")
    response_env = MatrixFixedCommitmentResponseEnv(spec, seed=args.seed)
    model = _meta_model(args, response_env, spec.episode_length)
    assert isinstance(model, Reinforce)
    assert model.n_steps == 50
    assert model.policy.action_net.bias is None
    torch.testing.assert_close(
        torch.linalg.vector_norm(model.policy.action_net.weight, dim=1),
        torch.full((2,), 0.01),
    )
    assert torch.count_nonzero(model.policy.value_net.weight) == 0
    assert model.policy.value_net.weight.requires_grad is False
    assert model.policy.optimizer.param_groups[0]["eps"] == 1e-8
    assert [
        name for name, parameter in model.policy.named_parameters()
        if parameter.requires_grad
    ] == ["action_net.weight"]

    observation_tensor, _ = model.policy.obs_to_tensor(37)
    features = model.policy.extract_features(observation_tensor)
    latent_pi, _ = model.policy.mlp_extractor(features)
    logits = model.policy.action_net(latent_pi)[0]
    torch.testing.assert_close(logits, model.policy.action_net.weight[:, 37])

    protocol = meta_config(args, spec)["reinforce_protocol"]
    assert protocol["baseline"] == "none"
    assert protocol["advantage_normalization"] is False
    assert protocol["rollout_geometry"] == {
        "collection_target_env_steps": 100,
        "episodes_per_update": 10,
        "executed_env_steps_per_update": 100,
        "query_env_steps_per_update": 50,
        "follower_gradient_samples_per_update": 50,
    }
    assert protocol["training_reward_offset"] == -4.0
    assert protocol["legacy_training_reward_offset"] == -2.5

    model.learn(total_timesteps=args.timesteps)
    assert model._n_updates == 1
    checkpoint = tmp_path / "reinforce_response.zip"
    _, checkpoint_record = save_evaluated_response_checkpoint(
        model,
        checkpoint,
        lambda current: evaluate_meta_follower(current, spec),
    )
    assert checkpoint.exists()
    assert checkpoint_record["commitments"] == 32

    write_response_checkpoint_contract(
        spec,
        checkpoint,
        "REINFORCE",
        metadata={"evaluation": {"max_regret": 0.0}},
    )
    leader_env = MatrixMetaLeaderEnv(
        spec,
        response_checkpoint=checkpoint,
        response_algorithm="REINFORCE",
    )
    assert isinstance(leader_env.response_model, Reinforce)
    observation = leader_env.reset()
    for action in (0, 1, 0, 1, 0):
        observation, _, done, _ = leader_env.step(action)
        assert not done
    while not done:
        observation, _, done, _ = leader_env.step(0)
    model.get_env().close()


def test_reinforce_loss_gradient_and_adam_match_manual_calculation():
    import torch

    from stackelberg_pomdp.matrix_ablations.reinforce import (
        reinforce_policy_loss,
    )

    logits = torch.tensor([
        [0.3, -0.2],
        [-0.4, 0.7],
        [0.1, 0.2],
    ], requires_grad=True)
    actions = torch.tensor([0, 1, 0])
    returns = torch.tensor([2.0, -1.0, 0.5])
    log_prob = torch.log_softmax(logits, dim=1)[
        torch.arange(len(actions)), actions
    ]
    loss = reinforce_policy_loss(log_prob, returns)
    loss.backward()

    probabilities = torch.softmax(logits.detach(), dim=1)
    selected = torch.nn.functional.one_hot(actions, num_classes=2).float()
    expected_gradient = -(
        returns[:, None] * (selected - probabilities)
    ) / len(actions)
    torch.testing.assert_close(logits.grad, expected_gradient)

    first = torch.nn.Parameter(torch.tensor([[0.2, -0.1], [0.4, 0.3]]))
    second = torch.nn.Parameter(first.detach().clone())
    first_optimizer = torch.optim.Adam([first], lr=0.02, eps=1e-8)
    second_optimizer = torch.optim.Adam([second], lr=0.02, eps=1e-8)
    observations = torch.tensor([0, 1, 0])
    first_log_prob = torch.log_softmax(first[:, observations].T, dim=1)[
        torch.arange(3), actions
    ]
    reinforce_policy_loss(first_log_prob, returns).backward()
    first_optimizer.step()
    manual_log_prob = torch.log_softmax(second[:, observations].T, dim=1)[
        torch.arange(3), actions
    ]
    (-(manual_log_prob * returns).sum() / 3).backward()
    second_optimizer.step()
    torch.testing.assert_close(first, second)


def test_reinforce_post_update_checkpoint_accounting(tmp_path):
    from stackelberg_pomdp.experiments.matrix_ablations import (
        build_parser,
        train_meta_follower,
    )

    args = build_parser().parse_args([
        "meta-follower",
        "--algorithm", "REINFORCE",
        "--timesteps", "5000",
        "--output-root", str(tmp_path),
        "--seed", "3",
    ])
    run_dir = train_meta_follower(args)
    records = [
        json.loads(line)
        for line in (run_dir / "checkpoint_evaluations.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["completed_updates"] == 100
    assert records[0]["follower_gradient_samples"] == 5000
    assert records[0]["executed_equivalent_steps"] == 10_000
    assert records[0]["checkpoint_semantics"] == "post_optimizer_update"
    assert records[0]["checkpoint_filename"] == "model.zip"
    with (run_dir / "response_contract.json").open() as handle:
        contract = json.load(handle)
    assert contract["schema_version"] == 2
    assert contract["response_game"]["profile_id"] == "paper_joint_v1"
    assert contract["metadata"]["completed_updates"] == 100


def test_dqn_response_saves_loads_and_runs_inside_meta_leader(tmp_path):
    from stable_baselines3 import DQN

    from stackelberg_pomdp.experiments.matrix_ablations import (
        _meta_model,
        build_parser,
        meta_config,
        validate_meta_args,
    )

    args = build_parser().parse_args([
        "meta-follower",
        "--algorithm", "DQN",
        "--timesteps", "12",
        "--learning-rate", "0.001",
        "--batch-size", "4",
        "--net-arch", "8",
        "--dqn-exploration-steps", "6",
        "--dqn-target-update-interval", "5",
        "--dqn-learning-starts", "0",
        "--dqn-final-epsilon", "0.1",
    ])
    validate_meta_args(args)
    spec = get_matrix_game("prisoners_dilemma")
    response_env = MatrixFixedCommitmentResponseEnv(spec, seed=args.seed)
    model = _meta_model(args, response_env, spec.episode_length)
    assert isinstance(model, DQN)
    assert meta_config(args, spec)["dqn_protocol"] == {
        "buffer_size": 1_000_000,
        "learning_starts": 0,
        "batch_size": 4,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 5,
        "exploration_steps": 6,
        "exploration_fraction": 0.5,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.1,
    }
    model.learn(total_timesteps=args.timesteps)
    model.save(str(tmp_path / "dqn_response"))
    checkpoint = tmp_path / "dqn_response.zip"
    write_response_checkpoint_contract(
        spec,
        checkpoint,
        "DQN",
        metadata={"evaluation": {"max_regret": 0.0}},
    )

    leader_env = MatrixMetaLeaderEnv(
        spec,
        response_checkpoint=checkpoint,
        response_algorithm="DQN",
        query_visibility="observed",
    )
    assert isinstance(leader_env.response_model, DQN)
    observation = leader_env.reset()
    commitment = (0, 1, 0, 1, 0)
    for action in commitment:
        observation, reward, done, _ = leader_env.step(action)
        assert reward == 0.0 and not done
    reward_steps = 0
    while not done:
        action = commitment[observation["base_environment"]]
        observation, _, done, info = leader_env.step(action)
        reward_steps += 1
        assert info["commitment_consistent"]
    assert reward_steps == spec.episode_length
    model.get_env().close()


@pytest.mark.parametrize("visibility, expected_stored", [
    ("observed", 10),
    ("hidden", 5),
])
def test_meta_e2_query_trace_fresh_reward_game_and_geometry(
        visibility, expected_stored
):
    spec = get_matrix_game("prisoners_dilemma")
    env = MatrixMetaLeaderEnv(
        spec,
        response_model=ConstantFollower(0),
        query_visibility=visibility,
        phase_observable=True,
    )
    observation = env.reset()
    assert observation == {
        "base_environment": 0,
        "base:is_reward_phase": 0,
    }
    query_actions = (0, 1, 0, 1, 0)
    for index, action in enumerate(query_actions):
        observation, reward, done, info = env.step(action)
        assert reward == 0.0 and not done
        assert info["query_state"] == index
        assert info["exclude_from_buffer"] == (visibility == "hidden")
        assert env.game.current_step == 0
    assert info["response_phase_done"]
    assert observation == {
        "base_environment": 0,
        "base:is_reward_phase": 1,
    }
    assert tuple(env.commitment) == query_actions

    reward_steps = 0
    while True:
        state = observation["base_environment"]
        observation, _, done, info = env.step(query_actions[state])
        reward_steps += 1
        assert info["is_reward_phase"]
        assert info["commitment_consistent"]
        if done:
            break
    assert reward_steps == 5
    assert env.max_episode_transitions() == expected_stored
    assert env.max_executed_transitions() == 10


def test_phase_conditions_have_identical_spaces_but_distinct_visible_keys():
    spec = get_matrix_game("prisoners_dilemma")
    visible = MatrixMetaLeaderEnv(
        spec, response_model=ConstantFollower(), phase_observable=True
    )
    hidden = MatrixMetaLeaderEnv(
        spec, response_model=ConstantFollower(), phase_observable=False
    )
    assert visible.observation_space == hidden.observation_space
    visible.reset()
    hidden.reset()
    for action in (0, 0, 0, 0, 0):
        visible_obs, _, _, _ = visible.step(action)
        hidden_obs, _, _, _ = hidden.step(action)
    assert visible_obs["base_environment"] == hidden_obs["base_environment"] == 0
    assert visible_obs["base:is_reward_phase"] == 1
    assert hidden_obs["base:is_reward_phase"] == 0


def test_tabular_q_uses_terminal_target_and_reset_or_carry_semantics():
    spec = get_matrix_game("battle_of_the_sexes")
    reset_env = LegacyMatrixQLeaderEnv(
        spec,
        response_episodes=1,
        reset_between_episodes=True,
        q_alpha=0.5,
        q_epsilon=0.0,
        q_init="zero",
        seed=7,
    )
    reset_env.reset()
    _, _, _, info = reset_env.step(0)
    assert info["q_target"] == 1.0
    np.testing.assert_allclose(reset_env.q_values, [0.5, 0.0])
    reset_env.step(0)
    reset_env.reset()
    np.testing.assert_allclose(reset_env.q_values, [0.0, 0.0])

    ongoing_env = LegacyMatrixQLeaderEnv(
        spec,
        response_episodes=1,
        reset_between_episodes=False,
        q_alpha=0.5,
        q_epsilon=0.0,
        q_init="zero",
        seed=7,
    )
    ongoing_env.reset()
    ongoing_env.step(0)
    ongoing_env.step(0)
    ongoing_env.reset()
    np.testing.assert_allclose(ongoing_env.q_values, [0.5, 0.0])


def test_response_reward_changes_only_leader_training_reward():
    spec = get_matrix_game("coordination_penalized_miscoordination")
    common = dict(
        spec=spec,
        response_episodes=3,
        reset_between_episodes=True,
        q_alpha=0.2,
        q_epsilon=0.1,
        exploration="parameter_noise",
        q_init="zero",
        seed=13,
    )
    excluded = LegacyMatrixQLeaderEnv(
        **common, include_response_reward=False
    )
    included = LegacyMatrixQLeaderEnv(
        **common, include_response_reward=True
    )
    excluded.reset()
    included.reset()
    for _ in range(3):
        _, excluded_reward, _, excluded_info = excluded.step(0)
        _, included_reward, _, included_info = included.step(0)
        assert excluded_reward == 0.0
        assert included_reward == 2.0
        assert excluded_info["follower_action"] == included_info["follower_action"]
        np.testing.assert_allclose(excluded.q_values, included.q_values)
    _, excluded_reward, excluded_done, excluded_info = excluded.step(0)
    _, included_reward, included_done, included_info = included.step(0)
    assert excluded_done and included_done
    assert excluded_reward == included_reward == 2.0
    assert excluded_info["is_reward_phase"]
    assert included_info["is_reward_phase"]
