"""Source identity, manifest validation, and paper-policy checks for Atari leaders."""

import hashlib
import json
import platform
from importlib import metadata as importlib_metadata
from pathlib import Path

from stackelberg_pomdp.checkpoints.files import checkpoint_sha256 as _sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


E2_PROVENANCE_SCHEMA = "stackelberg_pomdp.atari.e2_provenance"


E2_PROVENANCE_VERSION = 1


E2_PROVENANCE_ATTRIBUTE = "e2_provenance_manifest"


E2_PROTOCOL_IMPLEMENTATION = "clean_atari_stackpomdp_e2_v1"


E2_IMPLEMENTATION_FILES = (
    "stackelberg_pomdp/checkpoints/atari_provenance.py",
    "stackelberg_pomdp/checkpoints/files.py",
    "replication/atari/train_atari_stackpomdp_leader_sb3.py",
    "stackelberg_pomdp/atari/training.py",
    "stackelberg_pomdp/envs/atari/space_invaders.py",
    "stackelberg_pomdp/envs/atari/gameplay.py",
    "stackelberg_pomdp/envs/atari/bilateral.py",
    "stackelberg_pomdp/wrappers/atari/preprocessing.py",
    "stackelberg_pomdp/wrappers/atari/meta_follower.py",
    "stackelberg_pomdp/policies/atari/policy.py",
    "stackelberg_pomdp/policies/atari/components.py",
    "stackelberg_pomdp/checkpoints/atari.py",
    "stackelberg_pomdp/atari/protocol.py",
    "stackelberg_pomdp/atari/query_trace.py",
    "stackelberg_pomdp/atari/sampling.py",
    "stackelberg_pomdp/policies/cache.py",
    "stackelberg_pomdp/envs/base.py",
    "stackelberg_pomdp/wrappers/core.py",
    "stackelberg_pomdp/callbacks.py",
)


E2_PACKAGE_DISTRIBUTIONS = (
    "stable-baselines3",
    "torch",
    "gym",
    "numpy",
    "multi-agent-ale-py",
    "opencv-python",
)


LEGACY_LAYOUT_E2_IMPLEMENTATION_SHA256 = frozenset({
    # Exact local implementation before this behavior-preserving extraction.
    'e3ac13c113e925ff9c5a76856f119fc92a38c930562825015effe8f6bcc726df',
    # Public ``jair-2026-v1.0.0`` checkpoints.  The following release only
    # reorganizes modules; accepting this exact fingerprint preserves
    # evaluation/resume compatibility without weakening provenance checks.
    "4c187999e6bf073d35c7caa71033d7e1790306d2dac45dce25e3e090862d037a",
})


def _canonical_json_copy(value):
    """Return a JSON-only deep copy and reject non-finite numbers."""

    return json.loads(json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ))


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def e2_implementation_provenance():
    """Hash the implementation and runtime packages that define E2 behavior."""

    source_hashes = {}
    for relative in E2_IMPLEMENTATION_FILES:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"E2 implementation file is missing: {path}")
        source_hashes[relative] = _sha256_file(path)
    packages = {}
    for distribution in E2_PACKAGE_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = "unavailable"
    return _canonical_json_copy({
        "protocol_implementation": E2_PROTOCOL_IMPLEMENTATION,
        "python": platform.python_version(),
        "packages": packages,
        "source_sha256": source_hashes,
    })


def e2_implementation_provenance_compatible(recorded, current=None):
    """Accept current code or the exact pre-reorganization release layout."""

    if not isinstance(recorded, dict):
        return False
    resolved_current = (
        e2_implementation_provenance() if current is None else current
    )
    if recorded == resolved_current:
        return True
    return _canonical_sha256(recorded) in (
        LEGACY_LAYOUT_E2_IMPLEMENTATION_SHA256
    )


def validate_e2_provenance_manifest(manifest):
    """Validate the schema and its self-consistent canonical checksums."""

    if not isinstance(manifest, dict):
        raise ValueError("E2 checkpoint has no valid provenance manifest")
    result = _canonical_json_copy(manifest)
    if result.get("schema") != E2_PROVENANCE_SCHEMA:
        raise ValueError("E2 checkpoint provenance schema is unsupported")
    if result.get("version") != E2_PROVENANCE_VERSION:
        raise ValueError("E2 checkpoint provenance version is unsupported")
    fingerprint = result.pop("fingerprint_sha256", None)
    if fingerprint != _canonical_sha256(result):
        raise ValueError("E2 checkpoint provenance fingerprint is invalid")
    result["fingerprint_sha256"] = fingerprint
    lineage = result.get("run_lineage_id")
    if not isinstance(lineage, str) or len(lineage) != 32:
        raise ValueError("E2 checkpoint provenance lineage ID is invalid")
    try:
        int(lineage, 16)
    except ValueError as error:
        raise ValueError(
            "E2 checkpoint provenance lineage ID is invalid"
        ) from error
    identity = {
        "scientific_config": result.get("scientific_config"),
        "artifacts": result.get("artifacts"),
    }
    if result.get("scientific_identity_sha256") != _canonical_sha256(identity):
        raise ValueError("E2 scientific identity checksum is invalid")
    return result


def validate_e2_gameplay_actor(model):
    """The paper's E2 leaders train both gameplay and economic actor branches."""
    if getattr(model.policy, "gameplay_actor_frozen", False):
        raise ValueError("The paper E2 workflow requires a trainable gameplay actor.")
    if getattr(model, "e2_frozen_gameplay_actor_sha256", None) is not None:
        raise ValueError("The E2 checkpoint carries an incompatible frozen-actor hash.")
