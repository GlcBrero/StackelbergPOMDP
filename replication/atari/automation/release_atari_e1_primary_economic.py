"""Fresh, rank-one-only release of the Atari E1 buyer economic policy.

The original temporal-v1 selector remains an immutable failed timing result.
This module implements a separately named, explicitly post-hoc primary-
economic release: the already selected screen winner is evaluated once on a
new preregistered holdout, with no checkpoint search and no fallback.
"""

from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np

from replication.atari import evaluate_atari_meta_response_sb3 as evaluator
from replication.atari.automation import (
    validate_atari_e1_temporal_contingency as temporal_validator,
)
from stackelberg_pomdp.atari.schedule import ExactFiveEventSchedule
from stackelberg_pomdp.atari.stackpomdp_env import BUYER


SCHEMA_VERSION = 1
PROTOCOL_KIND = "stackpomdp.atari.e1_buyer_primary_economic_protocol.v1"
REPORT_KIND = "stackpomdp.atari.e1_buyer_primary_economic_confirmation.v1"
GATE_KIND = "stackpomdp.atari.e1_buyer_primary_economic_gate.v1"
EVALUATOR_NAME = "clean_atari_e1_primary_economic_confirmation_v1"
SOURCE_KIND = "primary_economic_v1"
SAMPLER_MODE = "temporal-marginal-v1"

RANDOM_EPISODES = 100
RANDOM_SEED_START = 8_000_001
FIXED_EPISODES = 20
FIXED_SEED_START = 8_100_001
FIXED_VALUES = tuple(value / 10.0 for value in range(11))
FIXED_EVENT_STEPS = (20, 50, 80, 110, 140)
EXPECTED_CHECK_NAMES = (
    "random buyer net payoff",
    "low/mid-price mean purchases",
    "low/mid-price mean shots",
    "price 0.5 purchases",
    "price 0.5 shots",
    "price 0.5 purchased bullets used",
    "price 0.5 buyer net payoff",
    "price 1 demand",
    "low-to-high demand drop",
    "largest adjacent demand reversal",
    "price 0.0 buyer net payoff",
    "price 0.1 buyer net payoff",
    "price 0.2 buyer net payoff",
    "price 0.3 buyer net payoff",
    "price 0.4 buyer net payoff",
    "price 0.5 buyer net payoff",
)

EXPECTED_SHA256 = {
    "source_report": "74c363ff3bee2a70d37153813d09bb899eb7015718eb7f291811d6bdf5c18de4",
    "timing_diagnostic": "dec463cc7f3a13c038cc83869e22d8f4971db3a7ecb8ef74833079498c7a907b",
    "training_family": "79dc19e3590f9ae6b568b20f124e4b23b931053caa973e51276cf24eb9220700",
    "activation": "6d63d7ba37db93d73e0ae5005e8214be0375a360ab87ccde475c9f261bcbabea",
    "preflight": "0a34e2447ea07665170d6b2fff6ee1e95c384203e996074c80af26a044b4c3c7",
    "source_checkpoint": "5d19fbb579f04da191ab006c5b9e9dcef574fb840e10544338dcab1b019d9e79",
    "e0b": "3a9ded5c53e10bf0b2215f1223f7bde15ba23d197dd0c651590bcd980c375ca3",
    "rom": "7224b17462b992d67f4e06a3c85f269c9822b06df6015bf038b55f384ced0301",
}
TRAINING_CODE_REVISION = "0c0148a2232711f4fc661cd2c658a313a43d5348"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _absolute_nofollow(path):
    """Return an absolute lexical path without following its final symlink."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def sha256_file(path):
    path = _absolute_nofollow(path)
    if path.is_symlink():
        raise FileNotFoundError(f"expected a nonsymlink immutable file: {path}")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"expected a regular immutable file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {path}",
    )
    return digest.hexdigest()


def load_json(path):
    path = _absolute_nofollow(path)
    if path.is_symlink():
        raise FileNotFoundError(f"expected a nonsymlink JSON artifact: {path}")
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _same_file(first, second):
    try:
        return os.path.samefile(first, second)
    except (FileNotFoundError, OSError, TypeError):
        return Path(first).expanduser().resolve() == Path(second).expanduser().resolve()


def git_revision(code_root):
    root = Path(code_root).expanduser().resolve()
    value = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(len(value) == 40 and all(c in "0123456789abcdef" for c in value),
             "evaluator code revision is not a full lowercase Git SHA")
    scoped = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(not scoped, "refusing to preregister from dirty Atari evaluator code")
    return value


def require_commit(code_root, revision):
    _require(
        isinstance(revision, str) and len(revision) == 40
        and all(c in "0123456789abcdef" for c in revision),
        "recorded evaluator revision is not a full Git SHA",
    )
    result = subprocess.run(
        ["git", "-C", str(Path(code_root).expanduser().resolve()),
         "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    _require(result.returncode == 0, "recorded evaluator commit is unavailable")
    return revision


def require_execution_root(code_root):
    """Bind imports and the claimed revision to this module's checkout."""

    root = Path(code_root).expanduser().resolve()
    module_root = Path(__file__).resolve().parents[3]
    _require(
        os.path.samefile(root, module_root),
        "--code-root is not the checkout executing the release module",
    )
    expected_imports = (
        (evaluator, root / "replication/atari/evaluate_atari_meta_response_sb3.py"),
        (
            temporal_validator,
            root / "replication/atari/automation/validate_atari_e1_temporal_contingency.py",
        ),
        (
            ExactFiveEventSchedule,
            root / "stackelberg_pomdp/atari/schedule.py",
        ),
        (
            evaluator.trainer,
            root / "replication/atari/train_atari_meta_response_sb3.py",
        ),
        (
            evaluator.StackPOMDPAtariPolicy,
            root / "stackelberg_pomdp/atari/stackpomdp_policy.py",
        ),
    )
    for imported, expected in expected_imports:
        _require(
            Path(inspect.getfile(imported)).resolve() == expected.resolve(),
            f"release dependency was imported outside --code-root: {expected}",
        )
    return root


def require_active_revision(code_root, revision):
    """Require validation to execute from the exact clean preregistered tree."""

    code_root = require_execution_root(code_root)
    revision = require_commit(code_root, revision)
    current = git_revision(code_root)
    _require(
        current == revision,
        "active evaluator revision differs from the preregistered revision",
    )
    return revision


def atomic_write_new_json(path, value):
    """Publish a new JSON object atomically and never overwrite a path."""

    path = _absolute_nofollow(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _check(name, actual, relation, target):
    passed = {">": actual > target, ">=": actual >= target, "<=": actual <= target}[relation]
    return {
        "name": name,
        "actual": float(actual),
        "relation": relation,
        "target": float(target),
        "passed": bool(passed),
    }


def primary_economic_gate(*, random_result, fixed_results):
    """Independent copy of the unchanged sixteen original buyer checks."""

    by_value = {}
    for row in fixed_results:
        value = round(float(row["opponent_value"]), 6)
        _require(value not in by_value, f"duplicate fixed-grid value {value}")
        by_value[value] = row["summary"]
    _require(tuple(sorted(by_value)) == FIXED_VALUES,
             "primary gate requires exactly fixed prices 0.0,0.1,...,1.0")
    zero, mid, one = (by_value[value] for value in (0.0, 0.5, 1.0))
    curves = [by_value[value] for value in FIXED_VALUES]
    low = curves[:6]
    purchases = float(np.mean([row["mean_purchases"] for row in low]))
    shots = float(np.mean([row["mean_buyer_shots_fired"] for row in low]))
    max_up = max(
        later["mean_purchases"] - earlier["mean_purchases"]
        for earlier, later in zip(curves, curves[1:])
    )
    checks = [
        _check("random buyer net payoff", random_result["summary"]["mean_controlled_payoff"], ">", 0.0),
        _check("low/mid-price mean purchases", purchases, ">=", 4.5),
        _check("low/mid-price mean shots", shots, ">=", 4.25),
        _check("price 0.5 purchases", mid["mean_purchases"], ">=", 4.5),
        _check("price 0.5 shots", mid["mean_buyer_shots_fired"], ">=", 4.25),
        _check("price 0.5 purchased bullets used", mid["mean_purchases"] - mid["mean_buyer_shots_fired"], "<=", 0.5),
        _check("price 0.5 buyer net payoff", mid["mean_controlled_payoff"], ">", 0.0),
        _check("price 1 demand", one["mean_purchases"], "<=", 0.5),
        _check("low-to-high demand drop", zero["mean_purchases"] - one["mean_purchases"], ">=", 4.0),
        _check("largest adjacent demand reversal", max_up, "<=", 0.5),
    ]
    checks.extend(
        _check(f"price {value:.1f} buyer net payoff", by_value[value]["mean_controlled_payoff"], ">", 0.0)
        for value in FIXED_VALUES[:6]
    )
    _require(tuple(row["name"] for row in checks) == EXPECTED_CHECK_NAMES,
             "primary economic check set changed")
    mechanics = random_result["protocol"]["passed"] is True and all(
        row["protocol"]["passed"] is True for row in fixed_results
    )
    return {
        "passed": bool(mechanics and all(row["passed"] for row in checks)),
        "mechanics_passed": bool(mechanics),
        "checks": checks,
    }


def _record(path, label):
    path = _absolute_nofollow(path)
    digest = sha256_file(path)
    _require(digest == EXPECTED_SHA256[label], f"unexpected {label} SHA-256")
    return {"path": str(path), "sha256": digest}


def validate_source_bundle(paths):
    """Validate the observed temporal-v1 evidence and all byte relationships."""

    records = {
        name: _record(getattr(paths, name), name)
        for name in (
            "source_report", "timing_diagnostic", "training_family",
            "activation", "preflight", "source_checkpoint", "e0b", "rom",
        )
    }
    activation = temporal_validator.validate_activation(records["activation"]["path"])
    _require(activation["code_revision"] == TRAINING_CODE_REVISION,
             "temporal activation training revision changed")
    family = temporal_validator.validate_training_family(
        records["training_family"]["path"]
    )
    _require(_same_file(family["activation"]["path"], records["activation"]["path"]),
             "family names another activation")
    _require(_same_file(family["preflight"]["path"], records["preflight"]["path"]),
             "family names another preflight")
    preflight = load_json(records["preflight"]["path"])
    _require(preflight.get("passed") is True, "temporal preflight did not pass")
    _require(preflight.get("activation_sha256") == records["activation"]["sha256"],
             "preflight belongs to another activation")
    _require(preflight.get("code_revision") == TRAINING_CODE_REVISION,
             "preflight revision differs from activation")
    _require(activation["e0b_source"]["sha256"] == records["e0b"]["sha256"],
             "activation uses another E0b")
    _require(activation["rom"]["sha256"] == records["rom"]["sha256"],
             "activation uses another ROM")

    report = load_json(records["source_report"]["path"])
    _require(report.get("evaluator") == evaluator.EVALUATOR_NAME,
             "unknown source selector")
    _require(report.get("role") == BUYER and report.get("passed") is False,
             "source selector is not the immutable failed buyer result")
    screen = report.get("screen", {})
    results = screen.get("results", [])
    ranking = report.get("ranking", [])
    attempts = report.get("confirmation_attempts", [])
    family_hashes = list(family["candidate_sha256"])
    protocol = report.get("protocol", {})
    for key, expected in {
        "screen_episodes": 20,
        "screen_seed_start": 6_000_001,
        "confirmation_episodes": 100,
        "confirmation_seed_start": 6_100_001,
        "fixed_context_episodes": 20,
        "fixed_context_seed_start": 6_200_001,
        "paired_timing_episodes_per_condition": 20,
        "paired_timing_seed_start": 6_300_001,
        "confirmation_policy": "screen_winner_only_no_fallback",
        "outer_transitions": 205,
    }.items():
        _require(protocol.get(key) == expected,
                 f"source selector protocol changed in {key}")
    _require(report.get("selection", {}).get("fallback_allowed") is False,
             "source selector permits confirmation fallback")
    _require(screen.get("common_pairing", {}).get("passed") is True,
             "source all-six common screen failed")
    _require(
        [row.get("evaluation_seed") for row in
         screen.get("common_pairing", {}).get("seed_context_pairs", [])]
        == list(range(6_000_001, 6_000_021)),
        "source common screen uses another seed schedule",
    )
    _require(len(results) == len(ranking) == len(family_hashes) == 6,
             "source selector is not all-six")
    _require([row["metadata"]["sha256"] for row in results] == family_hashes,
             "source selector screened another family")
    _require(len(attempts) == 1 and attempts[0]["metadata"]["sha256"] == EXPECTED_SHA256["source_checkpoint"],
             "source selector did not confirm only the expected rank one")
    _require(ranking[0]["checkpoint_sha256"] == EXPECTED_SHA256["source_checkpoint"],
             "eligible checkpoint is not source rank one")
    _require(family_hashes[0] == EXPECTED_SHA256["source_checkpoint"]
             and _same_file(
                 family["candidate_metadata"][0]["path"],
                 records["source_checkpoint"]["path"],
             ), "eligible checkpoint is not the first preregistered family member")
    _require(_same_file(
        attempts[0]["metadata"]["path"], records["source_checkpoint"]["path"]
    ), "source confirmation metadata names another checkpoint")
    _require(report.get("selected_alias") is None,
             "failed timing selector unexpectedly retained an alias")
    selection = report.get("selection", {})
    _require(selection.get("screen_selected_checkpoint_sha256")
             == EXPECTED_SHA256["source_checkpoint"]
             and selection.get("selected_checkpoint_sha256") is None,
             "source failed selection metadata changed")
    attempt = attempts[0]
    _require(
        [row.get("evaluation_seed") for row in
         attempt.get("random", {}).get("episode_rows", [])]
        == list(range(6_100_001, 6_100_101)),
        "source confirmation uses another seed schedule",
    )
    _require(len(attempt.get("fixed_contexts", [])) == 11, "source fixed grid is incomplete")
    for expected_value, result in zip(FIXED_VALUES, attempt["fixed_contexts"]):
        _require(np.isclose(result.get("opponent_value"), expected_value, atol=0, rtol=0),
                 "source fixed grid changed")
        _require([row.get("evaluation_seed") for row in result.get("episode_rows", [])]
                 == list(range(6_200_001, 6_200_021)),
                 "source fixed grid uses another seed schedule")
    _require(len(attempt.get("paired_timing", [])) == 10,
             "source paired timing grid is incomplete")
    _require(all(
        [row.get("evaluation_seed") for row in result.get("episode_rows", [])]
        == list(range(6_300_001, 6_300_021))
        for result in attempt["paired_timing"]
    ), "source paired timing uses another seed schedule")
    old_gate = evaluator.behavioral_gate(
        role=BUYER,
        random_result=attempt["random"],
        fixed_results=attempt["fixed_contexts"],
        timing_results=attempt["paired_timing"],
    )
    _require(old_gate == attempts[0]["behavioral_gate"],
             "source selector gate is not reproducible from its raw rows")
    _require(old_gate["mechanics_passed"] is True
             and old_gate["data_calibration_passed"] is True
             and old_gate["timing_behavior_passed"] is False
             and old_gate["passed"] is False,
             "source selector does not have the recorded timing-only failure")
    _require(tuple(row["name"] for row in old_gate["checks"][:16]) == EXPECTED_CHECK_NAMES
             and all(row["passed"] for row in old_gate["checks"][:16]),
             "source rank one did not pass the original primary economic checks")

    diagnostic = load_json(records["timing_diagnostic"]["path"])
    _require(diagnostic.get("schema") == "stackpomdp.atari.e1_temporal_timing_diagnostic.v1",
             "unknown timing diagnostic")
    _require(diagnostic.get("diagnostic_only_not_selection_eligible") is True,
             "timing diagnostic is incorrectly selection eligible")
    _require(diagnostic.get("family_sha256") == records["training_family"]["sha256"]
             and _same_file(diagnostic.get("family"), records["training_family"]["path"]),
             "timing diagnostic belongs to another family")
    candidates = diagnostic.get("candidates", [])
    diagnostic_hashes = [row.get("metadata", {}).get("sha256") for row in candidates]
    _require(diagnostic_hashes == family_hashes and len(set(diagnostic_hashes)) == 6,
             "timing diagnostic does not cover the exact all-six family")
    _require(all(row.get("would_pass_existing_timing_gate") is False for row in candidates),
             "timing diagnostic contains a timing-passing candidate")
    rank_one = candidates[0]
    for key, expected in {
        "early_acceptance": 0.0,
        "late_acceptance": 0.0,
        "acceptance_drop": 0.0,
        "early_forced_buy_advantage": 0.25,
        "late_forced_reject_advantage": 0.75,
        "early_regret": 0.25,
        "late_regret": 0.0,
    }.items():
        _require(np.isclose(rank_one.get(key), expected, atol=1e-12, rtol=0),
                 f"rank-one timing diagnostic changed in {key}")
    records["timing_limitation"] = {
        "status": "failed",
        "release_blocking": False,
        "no_timing_optimality_claim": True,
        "data_calibration_passed": True,
        "timing_behavior_passed": False,
        "rank_one_metrics": {
            key: rank_one[key] for key in (
                "early_acceptance", "late_acceptance", "acceptance_drop",
                "early_forced_buy_advantage", "late_forced_reject_advantage",
                "early_regret", "late_regret",
            )
        },
    }
    records["eligible_checkpoint_metadata"] = attempt["metadata"]
    return records


def _protocol_value(*, sources, revision):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PROTOCOL_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": BUYER,
        "evaluator": EVALUATOR_NAME,
        "evaluator_code_revision": revision,
        "source_kind": SOURCE_KIND,
        "sampler_mode": SAMPLER_MODE,
        "decision_record": {
            "criterion_change_after_observing_timing_failure": True,
            "primary_criterion": "unchanged first 16 buyer economic checks",
            "secondary_timing_status": "failed",
            "secondary_timing_release_blocking": False,
            "no_timing_optimality_claim": True,
        },
        "evaluation_semantics": {
            "deterministic_action": "masked Atari argmax and Beta mean",
            "gameplay_transitions": 200,
            "trade_transitions": 5,
            "outer_transitions": 205,
            "gameplay_horizon": 200,
            "event_tail_steps": 0,
            "noop_max": 30,
            "frame_skip": 4,
            "frame_stack": 4,
            "episodic_life": True,
            "clip_game_rewards": True,
            "seller_game_reward_scale": 0.1,
            "buyer_game_reward_scale": 1.0,
            "max_frames": 100_000,
        },
        "execution": {"device": "cpu"},
        "candidate_policy": {
            "eligible_count": 1,
            "selection_basis": "immutable temporal-v1 common-screen rank one",
            "candidate_search": False,
            "fallback_allowed": False,
            "checkpoint": sources["source_checkpoint"],
            "checkpoint_metadata": sources["eligible_checkpoint_metadata"],
        },
        "holdout": {
            "random": {
                "episodes": RANDOM_EPISODES,
                "seed_start": RANDOM_SEED_START,
                "seed_end": RANDOM_SEED_START + RANDOM_EPISODES - 1,
                "context": "evaluator.random_context(seed)",
                "schedule": "canonical seeded ExactFiveEventSchedule",
            },
            "fixed_grid": {
                "episodes_per_value": FIXED_EPISODES,
                "seed_start": FIXED_SEED_START,
                "seed_end": FIXED_SEED_START + FIXED_EPISODES - 1,
                "values": list(FIXED_VALUES),
                "shared_seeds": True,
                "event_steps": list(FIXED_EVENT_STEPS),
            },
            "timing_evaluation_run": False,
        },
        "economic_check_names": list(EXPECTED_CHECK_NAMES),
        "sources": sources,
    }


def build_protocol(args):
    require_execution_root(args.code_root)
    return _protocol_value(
        sources=validate_source_bundle(args),
        revision=git_revision(args.code_root),
    )


def validate_protocol(path, *, code_root=None):
    path = _absolute_nofollow(path)
    value = load_json(path)
    _require(value.get("schema_version") == SCHEMA_VERSION
             and value.get("kind") == PROTOCOL_KIND,
             "unknown primary-economic protocol")
    _require(value.get("role") == BUYER and value.get("evaluator") == EVALUATOR_NAME,
             "primary protocol role/evaluator mismatch")
    _require(value.get("source_kind") == SOURCE_KIND
             and value.get("sampler_mode") == SAMPLER_MODE,
             "primary protocol source kind changed")
    _require(value.get("economic_check_names") == list(EXPECTED_CHECK_NAMES),
             "primary protocol economic checks changed")
    candidate = value.get("candidate_policy", {})
    _require(candidate.get("eligible_count") == 1
             and candidate.get("candidate_search") is False
             and candidate.get("fallback_allowed") is False,
             "primary protocol permits candidate search/fallback")
    holdout = value.get("holdout", {})
    root = code_root or Path(__file__).resolve().parents[3]
    revision = (
        require_active_revision(root, value.get("evaluator_code_revision"))
        if code_root is not None
        else require_commit(root, value.get("evaluator_code_revision"))
    )
    sources = validate_source_bundle(SimpleNamespace(
        **{name: value["sources"][name]["path"] for name in EXPECTED_SHA256}
    ))
    for name in (
        *EXPECTED_SHA256,
        "timing_limitation",
        "eligible_checkpoint_metadata",
    ):
        _require(value["sources"].get(name) == sources[name],
                 f"primary protocol source record changed: {name}")
    expected = _protocol_value(sources=sources, revision=revision)
    for key in (
        "role", "evaluator", "evaluator_code_revision", "source_kind",
        "sampler_mode", "decision_record", "evaluation_semantics",
        "execution", "candidate_policy", "holdout",
        "economic_check_names", "sources",
    ):
        _require(value.get(key) == expected.get(key),
                 f"primary protocol differs in {key}")
    return value


def expected_random_schedule(seed):
    """Reproduce the exact wrapper/core RNG chain used by one fresh episode."""

    wrapper_rng = np.random.default_rng(int(seed) + 74_711)
    inner_seed = int(wrapper_rng.integers(0, 2 ** 31 - 1))
    schedule = ExactFiveEventSchedule(
        gameplay_horizon=200, tail_steps=0, fixed_event_steps=None
    )
    return schedule.sample(np.random.default_rng(inner_seed))


def _evaluation_args(protocol, *, e0b, rom, selected, device):
    """Build the canonical evaluator namespace without invoking selection."""

    _require(device == protocol["execution"]["device"] == "cpu",
             "fresh confirmation must run on the preregistered CPU device")
    args = evaluator.parse_args([
        "--role", BUYER,
        "--checkpoint", protocol["candidate_policy"]["checkpoint"]["path"],
        "--e0b-checkpoint", str(e0b),
        "--selected-checkpoint", str(selected),
        "--screen-seed-start", "7900001",
        "--confirmation-seed-start", str(RANDOM_SEED_START),
        "--fixed-seed-start", str(FIXED_SEED_START),
        "--timing-seed-start", "8200001",
        "--rom-path", str(rom),
        "--device", str(device),
    ])
    args.fixed_event_steps = None
    return args


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _recomputed_result(result, *, expected_seeds, expected_contexts,
                       expected_steps, phase, schedule_stratum,
                       checkpoint_path):
    """Recompute mechanics and summary only from raw terminal rows."""

    rows = result.get("episode_rows")
    _require(isinstance(rows, list) and len(rows) == len(expected_seeds),
             f"{phase} has the wrong number of episode rows")
    observed_seeds = [row.get("evaluation_seed") for row in rows]
    _require(observed_seeds == list(expected_seeds),
             f"{phase} uses another seed schedule")
    violations = []
    for index, (row, context) in enumerate(zip(rows, expected_contexts)):
        _require(row.get("phase") == phase, f"{phase} row has wrong phase")
        _require(np.array_equal(
            np.asarray(row.get("opponent_commitment"), dtype=np.float32),
            np.asarray(context, dtype=np.float32),
        ), f"{phase} row {index} has another opponent context")
        steps = expected_steps[index] if isinstance(expected_steps, list) else expected_steps
        _require(tuple(row.get("event_steps", ())) == tuple(steps),
                 f"{phase} row {index} has another event schedule")
        _require(row.get("checkpoint_sha256") == EXPECTED_SHA256["source_checkpoint"]
                 and _same_file(row.get("checkpoint_path"), checkpoint_path)
                 and row.get("training_timesteps") == 2_400_960,
                 f"{phase} row {index} is not bound to the eligible policy")
        _require(row.get("fifth_economic_override") is None
                 and row.get("fifth_economic_override_applied") in (None, 0),
                 f"{phase} row {index} contains an economic override")
        _require(row.get("e1_sampler_mode") == "uniform"
                 and row.get("e1_context_stratum") == "external"
                 and row.get("e1_schedule_stratum") == schedule_stratum,
                 f"{phase} row {index} has another evaluation sampler")
        violations.extend(evaluator.audit_episode(row, role=BUYER))
    summary = evaluator._summary(rows, role=BUYER)
    recomputed = {
        "summary": summary,
        "protocol": {"passed": not violations, "violations": violations},
        "episode_rows": rows,
    }
    _require("event_rows" not in result,
             f"{phase} retains unaudited flattened event rows")
    _require(_canonical_json(result.get("summary")) == _canonical_json(summary),
             f"{phase} serialized summary differs from raw rows")
    _require(_canonical_json(result.get("protocol"))
             == _canonical_json(recomputed["protocol"]),
             f"{phase} serialized mechanics differ from raw rows")
    return recomputed


def _validate_fresh_results(report):
    """Validate exact holdout rows and recompute the release decision."""

    random_seeds = list(range(RANDOM_SEED_START, RANDOM_SEED_START + RANDOM_EPISODES))
    random_contexts = [evaluator.random_context(seed) for seed in random_seeds]
    random_steps = [expected_random_schedule(seed) for seed in random_seeds]
    random_result = _recomputed_result(
        report.get("random", {}),
        expected_seeds=random_seeds,
        expected_contexts=random_contexts,
        expected_steps=random_steps,
        phase="primary_confirmation_random",
        schedule_stratum="unconditional",
        checkpoint_path=report["checkpoint"]["path"],
    )
    fixed_raw = report.get("fixed_contexts")
    _require(isinstance(fixed_raw, list) and len(fixed_raw) == len(FIXED_VALUES),
             "fresh report must contain exactly eleven fixed-grid results")
    fixed_seeds = list(range(FIXED_SEED_START, FIXED_SEED_START + FIXED_EPISODES))
    fixed_results = []
    for expected_value, result in zip(FIXED_VALUES, fixed_raw):
        value = float(result.get("opponent_value", np.nan))
        _require(np.isclose(value, expected_value, atol=0, rtol=0),
                 "fresh fixed-grid values changed or are out of order")
        context = np.full(5, expected_value, dtype=np.float32)
        recomputed = _recomputed_result(
            result,
            expected_seeds=fixed_seeds,
            expected_contexts=[context] * FIXED_EPISODES,
            expected_steps=FIXED_EVENT_STEPS,
            phase=f"fixed_{expected_value:.2f}",
            schedule_stratum="fixed",
            checkpoint_path=report["checkpoint"]["path"],
        )
        fixed_results.append({"opponent_value": expected_value, **recomputed})
    gate = primary_economic_gate(
        random_result=random_result, fixed_results=fixed_results
    )
    _require(_canonical_json(report.get("primary_economic_gate"))
             == _canonical_json(gate),
             "fresh serialized primary gate differs from raw rows")
    _require(report.get("passed") is gate["passed"],
             "fresh report outcome differs from recomputed primary gate")
    return random_result, fixed_results, gate


def validate_report(*, protocol_path: Path, report_path: Path,
                    selected_checkpoint: Path | None = None,
                    code_root: Path | None = None,
                    require_published_alias: bool = True):
    protocol_path = _absolute_nofollow(protocol_path)
    report_path = _absolute_nofollow(report_path)
    protocol = validate_protocol(protocol_path, code_root=code_root)
    report = load_json(report_path)
    _require(report.get("schema_version") == SCHEMA_VERSION
             and report.get("kind") == REPORT_KIND,
             "unknown primary-economic confirmation report")
    _require(report.get("evaluator") == EVALUATOR_NAME
             and report.get("role") == BUYER,
             "primary confirmation role/evaluator mismatch")
    _require(report.get("source_kind") == SOURCE_KIND
             and report.get("sampler_mode") == SAMPLER_MODE,
             "primary confirmation source kind changed")
    for key, expected in {
        "actor_loss_mode": "balanced",
        "candidate_search": False,
        "fallback_allowed": False,
    }.items():
        _require(report.get(key) == expected,
                 f"primary confirmation differs in {key}")
    _require(report.get("evaluator_code_revision")
             == protocol["evaluator_code_revision"],
             "report evaluator revision differs from protocol")
    protocol_record = report.get("protocol", {})
    _require(_same_file(protocol_record.get("path"), protocol_path)
             and protocol_record.get("sha256") == sha256_file(protocol_path),
             "report does not bind the exact protocol")
    checkpoint = report.get("checkpoint", {})
    _require(checkpoint.get("sha256") == EXPECTED_SHA256["source_checkpoint"]
             and _same_file(
                 checkpoint.get("path"),
                 protocol["candidate_policy"]["checkpoint"]["path"],
             ), "report evaluates another checkpoint")
    _require(checkpoint.get("role") == BUYER
             and checkpoint.get("economic_input_mode") == "full"
             and checkpoint.get("training_timesteps") == 2_400_960
             and checkpoint.get("training_config", {}).get("actor_loss_mode")
             == "balanced",
             "report checkpoint metadata differs from the eligible buyer")
    _require(checkpoint == protocol["candidate_policy"]["checkpoint_metadata"],
             "report checkpoint metadata differs from the immutable source report")
    _require(sha256_file(checkpoint["path"]) == checkpoint["sha256"],
             "source checkpoint bytes changed after confirmation")
    for name in ("e0b", "rom"):
        _require(report.get(f"{name}_source") == protocol["sources"][name],
                 f"report does not bind the exact {name.upper()} used")
        _require(sha256_file(report[f"{name}_source"]["path"])
                 == report[f"{name}_source"]["sha256"],
                 f"report {name.upper()} bytes changed")
    _require(report.get("secondary_timing")
             == protocol["sources"]["timing_limitation"],
             "report obscures the recorded timing failure")
    environment = report.get("environment", {})
    _require(environment == protocol["evaluation_semantics"],
             "report environment differs from protocol")
    _require(report.get("execution") == protocol["execution"] == {"device": "cpu"},
             "report execution device differs from protocol")
    _random, _fixed, gate = _validate_fresh_results(report)
    artifacts = report.get("artifacts", {})
    _require(isinstance(artifacts, dict)
             and set(artifacts) == {"report", "selected_checkpoint", "gate"}
             and _same_file(artifacts.get("report"), report_path),
             "fresh report does not identify itself")
    expected_gate = report_path.with_name(f"{report_path.stem}.gate.json")
    if gate["passed"]:
        alias = report.get("selected_alias")
        _require(isinstance(alias, dict), "passing fresh report has no selected alias")
        selected = _absolute_nofollow(
            selected_checkpoint or alias.get("pinned_path", "")
        )
        _require(set(alias) == {"pinned_path", "sha256"}
                 and _same_file(alias.get("pinned_path"), selected)
                 and alias.get("sha256") == checkpoint["sha256"],
                 "fresh report identifies another selected alias")
        _require(_same_file(artifacts.get("selected_checkpoint"), selected)
                 and _same_file(artifacts.get("gate"), expected_gate),
                 "fresh report identifies other release artifacts")
        if require_published_alias or os.path.lexists(selected):
            _require(sha256_file(selected) == checkpoint["sha256"],
                     "fresh selected alias bytes changed")
    else:
        _require(report.get("selected_alias") is None,
                 "failed fresh report retained selected-alias metadata")
        _require(artifacts.get("selected_checkpoint") is None
                 and artifacts.get("gate") is None,
                 "failed fresh report names release artifacts")
        _require(not os.path.lexists(expected_gate),
                 "failed fresh report retained an orphan gate")
        if selected_checkpoint is not None:
            _require(not os.path.lexists(selected_checkpoint),
                     "failed fresh report retained a selected alias")
    return report


def build_gate(*, protocol_path, report_path, selected_checkpoint,
               code_root=None):
    report = validate_report(
        protocol_path=protocol_path,
        report_path=report_path,
        selected_checkpoint=selected_checkpoint,
        code_root=code_root,
    )
    _require(report["passed"] is True, "only a passing fresh report is a gate")
    protocol = validate_protocol(protocol_path, code_root=code_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": GATE_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "role": BUYER,
        "actor_loss_mode": "balanced",
        "sampler_mode": SAMPLER_MODE,
        "source_kind": SOURCE_KIND,
        "protocol": {
            "path": str(Path(protocol_path).expanduser().resolve()),
            "sha256": sha256_file(protocol_path),
        },
        "report": {
            "path": str(Path(report_path).expanduser().resolve()),
            "sha256": sha256_file(report_path),
            "evaluator": EVALUATOR_NAME,
        },
        "selected_checkpoint": {
            "path": str(Path(selected_checkpoint).expanduser().resolve()),
            "sha256": sha256_file(selected_checkpoint),
        },
        "evaluator_code_revision": protocol["evaluator_code_revision"],
        "candidate_search": False,
        "fallback_allowed": False,
        "secondary_timing_status": "failed",
        "secondary_timing_release_blocking": False,
        "no_timing_optimality_claim": True,
        "execution": protocol["execution"],
        "sources": protocol["sources"],
    }


def validate_gate(*, protocol_path: Path, report_path: Path, gate_path: Path,
                  selected_checkpoint: Path, code_root=None) -> dict:
    """Revalidate a primary release and every bound byte from raw rows."""

    gate_path = _absolute_nofollow(gate_path)
    value = load_json(gate_path)
    _require(value.get("schema_version") == SCHEMA_VERSION
             and value.get("kind") == GATE_KIND,
             "unknown primary-economic gate")
    for key, expected in {
        "passed": True,
        "role": BUYER,
        "actor_loss_mode": "balanced",
        "sampler_mode": SAMPLER_MODE,
        "source_kind": SOURCE_KIND,
        "candidate_search": False,
        "fallback_allowed": False,
        "secondary_timing_status": "failed",
        "secondary_timing_release_blocking": False,
        "no_timing_optimality_claim": True,
        "execution": {"device": "cpu"},
    }.items():
        _require(value.get(key) == expected, f"primary gate differs in {key}")
    expected = build_gate(
        protocol_path=protocol_path,
        report_path=report_path,
        selected_checkpoint=selected_checkpoint,
        code_root=code_root,
    )
    for key in (
        "passed", "role", "actor_loss_mode", "sampler_mode", "source_kind",
        "protocol", "report", "selected_checkpoint",
        "evaluator_code_revision", "candidate_search", "fallback_allowed",
        "secondary_timing_status", "secondary_timing_release_blocking",
        "no_timing_optimality_claim", "execution", "sources",
    ):
        _require(value.get(key) == expected.get(key),
                 f"primary gate differs in {key}")
    return value


def atomic_copy_new(source, destination):
    """Publish immutable checkpoint bytes with no overwrite or symlink race."""

    source = _absolute_nofollow(source)
    destination = _absolute_nofollow(destination)
    _require(destination.suffix == ".zip", "selected checkpoint must end in .zip")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite selected checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        _require(sha256_file(temporary) == sha256_file(source),
                 "selected checkpoint staging changed bytes")
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite selected checkpoint: {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return {"pinned_path": str(destination), "sha256": sha256_file(destination)}


def finalize_release(*, protocol_path, report_path, selected_checkpoint,
                     gate_path, code_root=None):
    """Recover or complete the report -> alias -> gate publication transaction.

    A report is the immutable outcome marker.  For a passing report, an
    interrupted process may legitimately leave either only the report or the
    report plus the exact alias.  This helper recreates only those missing
    downstream artifacts after revalidating every raw row.  It never reruns
    evaluation and never overwrites an existing path.
    """

    protocol_path = _absolute_nofollow(protocol_path)
    report_path = _absolute_nofollow(report_path)
    selected = _absolute_nofollow(selected_checkpoint)
    gate_path = _absolute_nofollow(gate_path)
    expected_gate = report_path.with_name(f"{report_path.stem}.gate.json")
    _require(gate_path == expected_gate,
             "primary gate must be adjacent to its report")
    report = validate_report(
        protocol_path=protocol_path,
        report_path=report_path,
        selected_checkpoint=selected,
        code_root=code_root,
        require_published_alias=False,
    )
    if report["passed"] is False:
        _require(not os.path.lexists(gate_path),
                 "failed fresh report retained a gate")
        return {"report": report, "gate": None}

    protocol = validate_protocol(protocol_path, code_root=code_root)
    source = Path(protocol["candidate_policy"]["checkpoint"]["path"])
    expected_digest = protocol["candidate_policy"]["checkpoint"]["sha256"]
    _require(sha256_file(source) == expected_digest,
             "eligible source checkpoint bytes changed")
    if os.path.lexists(gate_path) and not os.path.lexists(selected):
        raise ValueError("primary gate exists without its selected alias")
    if os.path.lexists(selected):
        _require(sha256_file(selected) == expected_digest,
                 "existing selected alias has other bytes")
    else:
        alias = atomic_copy_new(source, selected)
        _require(alias["sha256"] == expected_digest,
                 "recovered selected alias has other bytes")

    validate_report(
        protocol_path=protocol_path,
        report_path=report_path,
        selected_checkpoint=selected,
        code_root=code_root,
    )
    if os.path.lexists(gate_path):
        gate = validate_gate(
            protocol_path=protocol_path,
            report_path=report_path,
            gate_path=gate_path,
            selected_checkpoint=selected,
            code_root=code_root,
        )
    else:
        gate = build_gate(
            protocol_path=protocol_path,
            report_path=report_path,
            selected_checkpoint=selected,
            code_root=code_root,
        )
        atomic_write_new_json(gate_path, gate)
        gate = validate_gate(
            protocol_path=protocol_path,
            report_path=report_path,
            gate_path=gate_path,
            selected_checkpoint=selected,
            code_root=code_root,
        )
    return {"report": report, "gate": gate}


def _strip_event_rows(result):
    return {key: value for key, value in result.items() if key != "event_rows"}


def run_fresh_confirmation(args):
    """Evaluate the sole eligible policy and publish report, alias, then gate."""

    protocol_path = _absolute_nofollow(args.protocol)
    report_path = _absolute_nofollow(args.report)
    selected = _absolute_nofollow(args.selected_checkpoint)
    gate_path = _absolute_nofollow(args.gate)
    expected_gate = report_path.with_name(f"{report_path.stem}.gate.json")
    _require(args.device == "cpu", "fresh confirmation must run on CPU")
    _require(gate_path == expected_gate, "primary gate must be adjacent to its report")
    for path in (report_path, selected, gate_path):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite release artifact: {path}")
    protocol = validate_protocol(protocol_path, code_root=args.code_root)
    _require(git_revision(args.code_root) == protocol["evaluator_code_revision"],
             "fresh rollout code differs from the preregistered revision")
    source = Path(protocol["candidate_policy"]["checkpoint"]["path"])
    e0b = Path(protocol["sources"]["e0b"]["path"])
    rom = Path(protocol["sources"]["rom"]["path"])

    with tempfile.TemporaryDirectory(prefix="stackpomdp-e1-primary-") as raw:
        pin_root = Path(raw)
        source_pin = evaluator.pin_file(source, pin_root / "candidate.zip")
        e0b_pin = evaluator.pin_file(e0b, pin_root / "e0b.zip")
        rom_pin = evaluator.pin_file(rom, pin_root / "space_invaders.bin")
        local = _evaluation_args(
            protocol,
            e0b=e0b_pin["pinned_path"],
            rom=rom_pin["pinned_path"],
            selected=selected,
            device=args.device,
        )
        model, metadata = evaluator.load_candidate(
            source_pin["pinned_path"],
            role=BUYER,
            e0b_sha256=e0b_pin["sha256"],
            device=args.device,
            display_path=source_pin["source_path"],
        )
        _require(metadata["sha256"] == EXPECTED_SHA256["source_checkpoint"]
                 and metadata["training_timesteps"] == 2_400_960,
                 "loaded policy is not the sole eligible checkpoint")
        _require(metadata["training_config"]["actor_loss_mode"] == "balanced",
                 "eligible buyer does not use balanced actor loss")
        random_seeds = list(range(
            RANDOM_SEED_START, RANDOM_SEED_START + RANDOM_EPISODES
        ))
        random_result = evaluator.evaluate_rows(
            model,
            local,
            metadata,
            seeds=random_seeds,
            contexts=[evaluator.random_context(seed) for seed in random_seeds],
            phase="primary_confirmation_random",
        )
        fixed_seeds = list(range(
            FIXED_SEED_START, FIXED_SEED_START + FIXED_EPISODES
        ))
        fixed_results = evaluator.fixed_grid(
            model, local, metadata, seeds=fixed_seeds
        )
        del model
        random_result = _strip_event_rows(random_result)
        fixed_results = [
            {key: value for key, value in result.items() if key != "event_rows"}
            for result in fixed_results
        ]
        release_gate = primary_economic_gate(
            random_result=random_result, fixed_results=fixed_results
        )
        canonical_gate = evaluator.buyer_primary_economic_gate(
            random_result=random_result, fixed_results=fixed_results
        )
        _require(_canonical_json(release_gate) == _canonical_json(canonical_gate),
                 "independent and canonical primary gates disagree")
        for pin, expected in (
            (source_pin, EXPECTED_SHA256["source_checkpoint"]),
            (e0b_pin, EXPECTED_SHA256["e0b"]),
            (rom_pin, EXPECTED_SHA256["rom"]),
        ):
            _require(sha256_file(pin["pinned_path"]) == expected
                     and sha256_file(pin["source_path"]) == expected,
                     "source bytes changed during fresh confirmation")

    selected_alias = (
        {"pinned_path": str(selected), "sha256": EXPECTED_SHA256["source_checkpoint"]}
        if release_gate["passed"] else None
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": release_gate["passed"],
        "role": BUYER,
        "evaluator": EVALUATOR_NAME,
        "evaluator_code_revision": protocol["evaluator_code_revision"],
        "actor_loss_mode": "balanced",
        "sampler_mode": SAMPLER_MODE,
        "source_kind": SOURCE_KIND,
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "checkpoint": metadata,
        "e0b_source": protocol["sources"]["e0b"],
        "rom_source": protocol["sources"]["rom"],
        "environment": protocol["evaluation_semantics"],
        "execution": protocol["execution"],
        "random": random_result,
        "fixed_contexts": fixed_results,
        "primary_economic_gate": release_gate,
        "secondary_timing": protocol["sources"]["timing_limitation"],
        "candidate_search": False,
        "fallback_allowed": False,
        "selected_alias": selected_alias,
        "artifacts": {
            "report": str(report_path),
            "selected_checkpoint": str(selected) if release_gate["passed"] else None,
            "gate": str(gate_path) if release_gate["passed"] else None,
        },
    }
    atomic_write_new_json(report_path, report)
    if not release_gate["passed"]:
        validate_report(protocol_path=protocol_path, report_path=report_path,
                        selected_checkpoint=selected, code_root=args.code_root)
        return {"report": report, "gate": None}

    finalized = finalize_release(
        protocol_path=protocol_path,
        report_path=report_path,
        selected_checkpoint=selected,
        gate_path=gate_path,
        code_root=args.code_root,
    )
    return {"report": report, "gate": finalized["gate"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    protocol = subparsers.add_parser("write-protocol")
    for name in EXPECTED_SHA256:
        protocol.add_argument(f"--{name.replace('_', '-')}", required=True)
    protocol.add_argument("--code-root", required=True)
    protocol.add_argument("--output", required=True)
    validate_protocol_parser = subparsers.add_parser("validate-protocol")
    validate_protocol_parser.add_argument("--protocol", required=True)
    validate_protocol_parser.add_argument("--code-root")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--report", required=True)
    evaluate.add_argument("--selected-checkpoint", required=True)
    evaluate.add_argument("--gate", required=True)
    evaluate.add_argument("--code-root", required=True)
    evaluate.add_argument("--device", choices=("cpu",), default="cpu")
    report = subparsers.add_parser("validate-report")
    report.add_argument("--protocol", required=True)
    report.add_argument("--report", required=True)
    report.add_argument("--selected-checkpoint")
    report.add_argument("--code-root")
    finalize = subparsers.add_parser("finalize-release")
    finalize.add_argument("--protocol", required=True)
    finalize.add_argument("--report", required=True)
    finalize.add_argument("--selected-checkpoint", required=True)
    finalize.add_argument("--gate", required=True)
    finalize.add_argument("--code-root", required=True)
    gate = subparsers.add_parser("validate-gate")
    gate.add_argument("--protocol", required=True)
    gate.add_argument("--report", required=True)
    gate.add_argument("--gate", required=True)
    gate.add_argument("--selected-checkpoint", required=True)
    gate.add_argument("--code-root", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "write-protocol":
        output = _absolute_nofollow(args.output)
        if output.exists():
            result = validate_protocol(output, code_root=args.code_root)
        else:
            atomic_write_new_json(output, build_protocol(args))
            result = validate_protocol(output, code_root=args.code_root)
        output_value = {"kind": "primary_economic_protocol", "passed": True,
                        "path": str(output), "protocol": result}
    elif args.command == "validate-protocol":
        result = validate_protocol(args.protocol, code_root=args.code_root)
        output_value = {"kind": "primary_economic_protocol", "passed": True,
                        "path": str(Path(args.protocol).resolve()), "protocol": result}
    elif args.command == "evaluate":
        result = run_fresh_confirmation(args)
        output_value = {
            "kind": "primary_economic_confirmation",
            "passed": result["report"]["passed"],
            "report": str(Path(args.report).resolve()),
            "gate": None if result["gate"] is None else str(Path(args.gate).resolve()),
        }
    elif args.command == "validate-report":
        result = validate_report(
            protocol_path=Path(args.protocol), report_path=Path(args.report),
            selected_checkpoint=(
                None if args.selected_checkpoint is None
                else Path(args.selected_checkpoint)
            ),
            code_root=(None if args.code_root is None else Path(args.code_root)),
        )
        output_value = {"kind": "primary_economic_confirmation",
                        "passed": result["passed"],
                        "report": str(Path(args.report).resolve())}
    elif args.command == "finalize-release":
        result = finalize_release(
            protocol_path=Path(args.protocol),
            report_path=Path(args.report),
            selected_checkpoint=Path(args.selected_checkpoint),
            gate_path=Path(args.gate),
            code_root=Path(args.code_root),
        )
        output_value = {
            "kind": "primary_economic_release",
            "passed": result["report"]["passed"],
            "report": str(Path(args.report).resolve()),
            "gate": None if result["gate"] is None else str(Path(args.gate).resolve()),
        }
    else:
        result = validate_gate(
            protocol_path=Path(args.protocol), report_path=Path(args.report),
            gate_path=Path(args.gate),
            selected_checkpoint=Path(args.selected_checkpoint),
            code_root=Path(args.code_root),
        )
        output_value = {"kind": "primary_economic_gate", "passed": True,
                        "gate": str(Path(args.gate).resolve()), "value": result}
    print(json.dumps(output_value, sort_keys=True), flush=True)
    if output_value.get("passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
