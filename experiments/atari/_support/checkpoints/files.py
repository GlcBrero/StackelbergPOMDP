"""Resolve, hash, and publish SB3 checkpoint files without replacing existing artifacts."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def _checkpoint_path(raw, *, label="checkpoint"):
    path = Path(raw).expanduser()
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy_no_overwrite(source, destination):
    """Copy exact bytes to a new alias without ever replacing a prior file."""

    source = Path(source).resolve()
    destination = Path(destination).expanduser()
    if destination.suffix != ".zip":
        destination = destination.with_suffix(".zip")
    destination = destination.parent.resolve() / destination.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite selected checkpoint alias: {destination}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    linked_identity = None
    copied_hash = None
    try:
        source_hash = checkpoint_sha256(source)
        shutil.copy2(source, temporary)
        copied_hash = checkpoint_sha256(temporary)
        if copied_hash != source_hash or checkpoint_sha256(source) != source_hash:
            raise RuntimeError(
                "source checkpoint changed while the selected alias was copied"
            )
        os.link(temporary, destination)
        linked = destination.stat()
        linked_identity = (
            linked.st_dev,
            linked.st_ino,
            linked.st_size,
            linked.st_mtime_ns,
        )
        destination_hash = checkpoint_sha256(destination)
        if destination_hash != source_hash:
            raise RuntimeError(
                "selected checkpoint copy failed its SHA-256 check"
            )
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite selected checkpoint alias: {destination}"
        ) from error
    except BaseException:
        if linked_identity is not None and copied_hash is not None:
            try:
                before = destination.stat()
                before_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                )
                current_hash = checkpoint_sha256(destination)
                after = destination.stat()
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if (
                        before_identity == linked_identity == after_identity
                        and current_hash == copied_hash
                ):
                    destination.unlink()
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source_checkpoint_path": str(source),
        "selected_checkpoint_path": str(destination),
        "checkpoint_sha256": destination_hash,
        "copy_verified": True,
        "created_by_selection_run": True,
    }


def rollback_new_selected_alias(report, *, expected_path):
    """Remove only the exact alias created by this failed artifact write."""

    alias = (report or {}).get("selected_alias")
    if not isinstance(alias, dict) or not (
            alias.get("created_by_selection_run")
            and alias.get("copy_verified")
    ):
        return False
    path = Path(alias.get("selected_checkpoint_path", "")).resolve()
    if path != Path(expected_path).expanduser().resolve() or path.is_symlink():
        return False
    expected_hash = alias.get("checkpoint_sha256")
    if not isinstance(expected_hash, str):
        return False
    try:
        before = path.stat()
        if checkpoint_sha256(path) != expected_hash:
            return False
        after = path.stat()
        if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
        ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
        ):
            return False
        path.unlink()
    except (FileNotFoundError, OSError):
        return False
    return True
