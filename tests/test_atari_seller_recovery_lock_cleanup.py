"""Runtime regressions for zsh seller-recovery lock cleanup."""

from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUTOMATION = ROOT / "replication/atari/automation"
LOCK_COMMON = AUTOMATION / "atari_e2_pipeline_common.zsh"


def _shell_function(path, name):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"function {name}() {{")
    for stop in range(start + 1, len(lines)):
        if lines[stop] == "}":
            return "\n".join(lines[start:stop + 1])
    raise AssertionError(f"unterminated shell function {name}")


def test_zsh_automation_never_shadows_readonly_status_parameter():
    offenders = []
    pattern = re.compile(r"\blocal\s+status(?:=|\s|$)")
    for path in sorted(AUTOMATION.glob("*.sh")):
        for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if pattern.search(line):
                offenders.append(f"{path.name}:{line_number}:{line.strip()}")
    assert offenders == []


@pytest.mark.parametrize(
    ("launcher", "function_name", "variable_setup"),
    (
        (
            "run_atari_clean_e1_seller_threshold_residual_recovery.sh",
            "release_e1r2_locks",
            r'''
typeset -g E1R2_LOCK="$PROFILE"
typeset -g E1R2_V1_GUARD_LOCK="$GUARD"
typeset -g STACKPOMDP_E1R2_LOCK_TOKEN=profile-token
typeset -g STACKPOMDP_E1R2_GUARD_TOKEN=guard-token
typeset -g E1R2_PROFILE_LABEL=test-profile
stackpomdp_claim_owned_lock "$GUARD" "$STACKPOMDP_E1R2_GUARD_TOKEN" guard
typeset -g E1R2_GUARD_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"
stackpomdp_claim_owned_lock "$PROFILE" "$STACKPOMDP_E1R2_LOCK_TOKEN" profile
typeset -g E1R2_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"
''',
        ),
        (
            "run_e1_seller_threshold_residual_recovery_selector.sh",
            "release_e1r2_selector_lock",
            r'''
typeset -g E1R2_LOCK="$PROFILE"
typeset -g E1R2_V1_GUARD_LOCK="$GUARD"
typeset -g STACKPOMDP_E1R2_SELECTOR_TOKEN=profile-token
typeset -g STACKPOMDP_E1R2_SELECTOR_GUARD_TOKEN=guard-token
typeset -g E1R2_PROFILE_LABEL=test-profile
stackpomdp_claim_owned_lock "$GUARD" "$STACKPOMDP_E1R2_SELECTOR_GUARD_TOKEN" guard
typeset -g E1R2_SELECTOR_GUARD_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"
stackpomdp_claim_owned_lock "$PROFILE" "$STACKPOMDP_E1R2_SELECTOR_TOKEN" profile
typeset -g E1R2_SELECTOR_LOCK_OWNED="$STACKPOMDP_LOCK_RESULT_OWNED"
''',
        ),
    ),
)
def test_scientific_gate_exit_preserves_code_and_releases_both_locks(
        tmp_path, launcher, function_name, variable_setup,
):
    function = _shell_function(AUTOMATION / launcher, function_name)
    guard = tmp_path / "guard.lock"
    profile = tmp_path / "profile.lock"
    program = "\n".join((
        "set -euo pipefail",
        'source "$COMMON"',
        variable_setup,
        function,
        f"trap '{function_name}' EXIT",
        "exit 2",
    ))
    result = subprocess.run(
        ["zsh", "-c", program],
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
            "COMMON": str(LOCK_COMMON),
            "GUARD": str(guard),
            "PROFILE": str(profile),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert not guard.exists()
    assert not profile.exists()
