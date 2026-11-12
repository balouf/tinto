"""Pytest configuration for the Tinto test suite."""

import os

# Disable numba JIT so that coverage can instrument the bodies of @jitclass /
# @njit functions. Must be set before numba is imported — conftest.py runs
# before any test or project module. CI mirrors this via the NUMBA_DISABLE_JIT
# env var in .github/workflows/build.yml.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Headless matplotlib backend for .plot() doctests. Must be set before any
# matplotlib import; MPLBACKEND is the official env var for that.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

# NumPy 2.0 changed scalar repr: np.int32(2) now prints as "np.int32(2)"
# instead of "2", which breaks doctests that were written against JIT output
# (plain Python ints). The legacy="1.25" flag restores the old repr.
np.set_printoptions(legacy="1.25")
