"""Bindings for native MW updates and the original exact-BCCE arithmetic."""

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


_LIBRARY = None


class NativeBCCEUnavailable(RuntimeError):
    """Raised when the native MSPM response cannot be built or loaded."""


def _load_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY

    sources = [Path(__file__).with_name(name) for name in ("bcce.cpp", "mw.cpp")]
    digest = hashlib.sha256(b"".join(source.read_bytes() for source in sources)).hexdigest()[:16]
    build_dir = Path(
        os.environ.get(
            "STACKELBERG_NATIVE_CACHE",
            str(Path(tempfile.gettempdir()) / "stackelberg-pomdp-native"),
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    target = build_dir / f"mspm_{digest}.so"
    if not target.exists():
        compiler_names = [
            name for name in (os.environ.get("CXX"), "g++", "c++", "clang++")
            if name
        ]
        compilers = []
        for name in compiler_names:
            compiler = shutil.which(name)
            if compiler is not None and compiler not in compilers:
                compilers.append(compiler)
        if not compilers:
            raise NativeBCCEUnavailable(
                "The MSPM response requires a C++17 compiler (clang++ or g++)"
            )
        # Separate temporary files keep simultaneous PPO workers from racing.
        descriptor, filename = tempfile.mkstemp(prefix="mspm_", suffix=".so", dir=build_dir)
        os.close(descriptor)
        temporary = Path(filename)
        errors = []
        for compiler in compilers:
            try:
                temporary.unlink(missing_ok=True)
                subprocess.run(
                    [compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
                     *(str(source) for source in sources), "-o", str(temporary)],
                    check=True, capture_output=True, text=True,
                )
                temporary.replace(target)
                break
            except (OSError, subprocess.CalledProcessError) as error:
                errors.append(f"{compiler}: {error}")
        else:
            temporary.unlink(missing_ok=True)
            raise NativeBCCEUnavailable("; ".join(errors))

    try:
        library = ctypes.CDLL(str(target))
    except OSError as error:
        raise NativeBCCEUnavailable(str(error)) from error
    double_array = np.ctypeslib.ndpointer(
        dtype=np.float64, flags="C_CONTIGUOUS"
    )
    function = library.exact_bcce_gap_two_player
    function.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, double_array, double_array, double_array,
        double_array, double_array,
    ]
    function.restype = ctypes.c_double
    handle = ctypes.c_void_p
    integer = ctypes.c_int
    library.mspm_mw_create.argtypes = [integer, integer] + [double_array]*6 + [ctypes.c_double]*2
    library.mspm_mw_create.restype = handle
    library.mspm_mw_certify.argtypes = [handle]
    library.mspm_mw_certify.restype = integer
    library.mspm_mw_advance.argtypes = [handle, integer, integer]
    library.mspm_mw_advance.restype = integer
    library.mspm_mw_export.argtypes = [handle] + [double_array]*5
    library.mspm_mw_export.restype = integer
    library.mspm_mw_destroy.argtypes = [handle]
    library.mspm_mw_destroy.restype = None
    _LIBRARY = library
    return _LIBRARY


def exact_bcce_gap_two_player(
        strategy0, strategy1, type_probability0, type_probability1,
        utilities):
    """Return the exact gap for a uniform mixture of product strategies."""
    strategy0 = np.ascontiguousarray(strategy0, dtype=np.float64)
    strategy1 = np.ascontiguousarray(strategy1, dtype=np.float64)
    type_probability0 = np.ascontiguousarray(
        type_probability0, dtype=np.float64
    )
    type_probability1 = np.ascontiguousarray(
        type_probability1, dtype=np.float64
    )
    utilities = np.ascontiguousarray(utilities, dtype=np.float64)

    if strategy0.ndim != 3 or strategy1.ndim != 3:
        raise ValueError("Strategies must have snapshot, type, and action axes")
    snapshots, t0_count, a0_count = strategy0.shape
    if strategy1.shape[0] != snapshots or snapshots < 1:
        raise ValueError("Follower strategies must share a nonempty snapshot axis")
    _, t1_count, a1_count = strategy1.shape
    if type_probability0.shape != (t0_count,):
        raise ValueError("Follower-0 type probabilities have the wrong shape")
    if type_probability1.shape != (t1_count,):
        raise ValueError("Follower-1 type probabilities have the wrong shape")
    if utilities.shape != (t0_count, t1_count, a0_count, a1_count, 2):
        raise ValueError("The utility table has the wrong shape")
    if not all(np.isfinite(array).all() for array in (
            strategy0, strategy1, type_probability0, type_probability1,
            utilities)):
        raise ValueError("Exact BCCE inputs must be finite")

    return float(_load_library().exact_bcce_gap_two_player(
        t0_count, t1_count, a0_count, a1_count, snapshots,
        strategy0, strategy1, type_probability0, type_probability1,
        utilities,
    ))


__all__ = ["NativeBCCEUnavailable", "exact_bcce_gap_two_player"]


def bcce_gap(snapshots, utilities, type_probabilities):
    """Certify a uniform mixture of product strategies using the paper kernel.

    snapshots has axes (update, buyer, private type, message). Keeping whole
    snapshots preserves correlation across buyers in the empirical mixture.
    The compiler and ctypes boundary stay out of the response procedure.
    """
    snapshots = np.asarray(snapshots, dtype=np.float64)
    if snapshots.ndim != 4 or snapshots.shape[1] != 2:
        raise ValueError("Expected strategy snapshots with update, buyer, type, message axes")
    return exact_bcce_gap_two_player(
        snapshots[:, 0], snapshots[:, 1],
        type_probabilities[0], type_probabilities[1], utilities,
    )


class NativeMW:
    """Own the numerical response after the complete payoff table is queried.

    Each advance consumes one type pair sampled by Python, updates both
    followers, and certifies mixed, deterministic and empirical candidates.
    The original certificate kernel and its summation order are retained.
    Use as a context manager to release the private native state reliably.
    """

    CANDIDATES = ("last_mixed", "last_deterministic", "empirical_average")

    def __init__(self, weights, utilities, type_probabilities, empirical_joint,
                 deviations, *, epsilon, tolerance):
        arrays = [np.ascontiguousarray(value, dtype=np.float64) for value in
                  (weights, utilities, type_probabilities, empirical_joint, deviations)]
        weights, utilities, probabilities, empirical, deviations = arrays
        if weights.ndim != 3 or weights.shape[0] != 2:
            raise ValueError("MW weights need buyer, type and message axes")
        self.types, self.messages = weights.shape[1:]
        t, m = self.types, self.messages
        if (t < 1 or m < 1 or utilities.shape != (t, t, m, m, 2)
                or probabilities.shape != (2, t) or empirical.shape != (t, t, m, m)
                or deviations.shape != (2, m)):
            raise ValueError("Native MW array shapes do not match")
        if (not all(np.isfinite(value).all() for value in arrays)
                or not np.isfinite(epsilon) or epsilon <= 0
                or not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Native MW inputs must be finite with valid update/certificate parameters")
        self._library = _load_library()
        self._handle = self._library.mspm_mw_create(
            t, m, weights, utilities, probabilities[0], probabilities[1],
            empirical, deviations, epsilon, tolerance,
        )
        if not self._handle:
            raise RuntimeError("Could not allocate the native MW response")
        self.updates = 1  # Python completed the query-producing initial sweep.
        self._candidate = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._handle is not None:
            self._library.mspm_mw_destroy(self._handle)
            self._handle = None

    def _open(self):
        if self._handle is None:
            raise RuntimeError("The native MW response is closed")
        return self._handle

    def _record_status(self, status):
        if status < -1:
            raise RuntimeError("Native MW failed to update or certify its response")
        self._candidate = status

    @property
    def certified(self):
        return self._candidate >= 0

    def certify(self):
        self._record_status(self._library.mspm_mw_certify(self._open()))

    def advance(self, types):
        if self.certified:
            raise RuntimeError("Stop at the first certified follower response")
        first, second = map(int, types)
        self._record_status(self._library.mspm_mw_advance(self._open(), first, second))
        self.updates += 1

    def result(self):
        t, m = self.types, self.messages
        weights = np.empty((2, t, m))
        empirical = np.empty((t, t, m, m))
        deviations = np.empty((2, m))
        joint = np.zeros_like(empirical)
        gaps = np.zeros(3)
        checked = self._library.mspm_mw_export(
            self._open(), weights, empirical, deviations, joint, gaps,
        )
        if checked < 0:
            raise RuntimeError("Could not export the native MW response")
        return {
            "weights": weights, "empirical_joint": empirical, "deviations": deviations,
            "joint": joint, "updates": self.updates,
            "candidate": self.CANDIDATES[self._candidate] if self.certified else None,
            "gaps": dict(zip(self.CANDIDATES[:checked], map(float, gaps[:checked]))),
        }
