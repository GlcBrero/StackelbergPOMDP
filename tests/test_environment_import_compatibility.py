"""Compatibility tests for extracted non-Atari environment modules."""

import importlib
import pickle

import pytest


_EXTRACTED_ENVIRONMENTS = {
    "BaseEnvSimpleMatrixGame": "normal_form",
    "BaseEnvMatrixDesignGame": "matrix_design",
    "BaseSimpleAllocation": "simple_allocation",
    "BaseSPM": "spm",
    "BaseMessageSPM": "spm",
    "BertrandCompetitionEnv": "bertrand",
}


@pytest.mark.parametrize("class_name,module_name", _EXTRACTED_ENVIRONMENTS.items())
def test_legacy_environment_import_and_pickle_global_resolve_to_canonical_class(
        class_name,
        module_name,
):
    canonical_module = importlib.import_module(
        "stackelberg_pomdp.envs.{}".format(module_name)
    )
    canonical_class = getattr(canonical_module, class_name)

    legacy_module = importlib.import_module("stackelberg_pomdp.envs.base")
    assert getattr(legacy_module, class_name) is canonical_class

    legacy_global = (
        "cstackelberg_pomdp.envs.base\n{}\n.".format(class_name).encode("ascii")
    )
    assert pickle.loads(legacy_global) is canonical_class
