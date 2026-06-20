"""Shared fixtures for the HPFRACC test tiers.

Centralizes the two things the numerical tests repeatedly need: a JAX import
that skips cleanly when JAX is unavailable, and a scoped float64 toggle for the
convergence/gradient checks that are meaningless in single precision.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture
def jax_mod() -> Any:
    """Return the ``jax`` module, skipping the test if JAX is not installed."""

    return pytest.importorskip("jax")


@pytest.fixture
def enable_x64() -> Iterator[None]:
    """Enable JAX float64 for the duration of a test, then restore the flag.

    Grid-refinement and finite-difference checks need double precision; the
    default single precision masks the very errors they measure. This restores
    the previous setting so float32-default tests are unaffected.
    """

    jax = pytest.importorskip("jax")
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)
