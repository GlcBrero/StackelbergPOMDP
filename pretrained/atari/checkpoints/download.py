#!/usr/bin/env python3
"""Download and cryptographically verify the representative Atari policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHECKPOINT_ROOT = HERE
MANIFEST = CHECKPOINT_ROOT / "manifest.csv"



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"artifact_id", "filename", "sha256", "size_bytes"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("checkpoint manifest has an unexpected schema")
    filenames: set[str] = set()
    artifact_ids: set[str] = set()
    for row in rows:
        filename = row["filename"]
        artifact_id = row["artifact_id"]
        if Path(filename).name != filename or not filename.endswith(".zip"):
            raise ValueError(f"unsafe checkpoint filename: {filename!r}")
        if filename in filenames or artifact_id in artifact_ids:
            raise ValueError("checkpoint manifest contains duplicate identifiers")
        if len(row["sha256"]) != 64 or any(
            character not in "0123456789abcdef" for character in row["sha256"]
        ):
            raise ValueError(f"invalid SHA-256 for {filename}")
        if int(row["size_bytes"]) <= 0:
            raise ValueError(f"invalid byte count for {filename}")
        filenames.add(filename)
        artifact_ids.add(artifact_id)
    return rows


def verify(path: Path, row: dict[str, str]) -> None:
    expected_size = int(row["size_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{path.name}: expected {expected_size} bytes, "
            f"found {path.stat().st_size}"
        )
    observed_hash = sha256(path)
    if observed_hash != row["sha256"]:
        raise ValueError(
            f"{path.name}: SHA-256 {observed_hash} does not match manifest"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Explicit checkpoint mirror; not needed for the full package.")
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in records():
        destination = output_dir / row["filename"]
        if not destination.is_file() and args.verify_only:
            raise FileNotFoundError(destination)
        if not destination.is_file():
            if not args.base_url:
                raise FileNotFoundError(f"{destination}: use the full submission archive or provide --base-url")
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.unlink(missing_ok=True)
            url = args.base_url.rstrip("/") + "/" + row["filename"]
            print(f"Downloading {url}")
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "StackelbergPOMDP-reproducibility/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
                    "wb"
                ) as out:
                    shutil.copyfileobj(response, out)
                # Never install unverified bytes as the canonical checkpoint.
                verify(temporary, row)
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        verify(destination, row)
        print(f"verified {destination.name}")


if __name__ == "__main__":
    main()
