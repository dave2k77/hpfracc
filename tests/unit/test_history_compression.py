"""Tests for short-memory and sum-of-exponentials history compression.

These strategies are opt-in approximations.  The tests do not demand exact
agreement with the full-history reference; instead they check that the methods
run, return the right shape, and stay within documented error bounds on simple
smooth signals.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from hpfracc.ops import caputo, grunwald_letnikov


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.9])
def test_short_memory_caputo_is_bounded_for_power_law(alpha: float) -> None:
    dt = 0.02
    n_steps = 101
    t = jnp.linspace(0.0, 1.0, n_steps)
    x = t**2

    full = caputo(x, dt=dt, order=alpha, history="full")
    short = caputo(
        x, dt=dt, order=alpha, history="short_memory", window_steps=80
    )

    # Relative error should be modest for a smooth power law when the window
    # covers a substantial fraction of the history.  Low alpha has a longer
    # memory kernel, so the tolerance is generous rather than tight.
    rel_err = jnp.max(jnp.abs(full - short) / jnp.maximum(jnp.abs(full), 1e-8))
    assert rel_err < 0.3


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.9])
def test_short_memory_grunwald_letnikov_is_bounded_for_power_law(alpha: float) -> None:
    dt = 0.02
    n_steps = 101
    t = jnp.linspace(0.0, 1.0, n_steps)
    x = t**2

    full = grunwald_letnikov(x, dt=dt, order=alpha, history="full")
    short = grunwald_letnikov(
        x, dt=dt, order=alpha, history="short_memory", window_steps=80
    )

    rel_err = jnp.max(jnp.abs(full - short) / jnp.maximum(jnp.abs(full), 1e-8))
    assert rel_err < 0.3


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_soe_caputo_is_bounded_for_power_law(alpha: float) -> None:
    dt = 0.02
    n_steps = 101
    t = jnp.linspace(0.0, 1.0, n_steps)
    x = t**2

    full = caputo(x, dt=dt, order=alpha, history="full")
    soe = caputo(x, dt=dt, order=alpha, history="soe", soe_poles=12)

    rel_err = jnp.max(jnp.abs(full - soe) / jnp.maximum(jnp.abs(full), 1e-8))
    assert rel_err < 0.2


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_soe_grunwald_letnikov_is_bounded_for_power_law(alpha: float) -> None:
    dt = 0.02
    n_steps = 101
    t = jnp.linspace(0.0, 1.0, n_steps)
    x = t**2

    full = grunwald_letnikov(x, dt=dt, order=alpha, history="full")
    soe = grunwald_letnikov(
        x, dt=dt, order=alpha, history="soe", soe_poles=12
    )

    rel_err = jnp.max(jnp.abs(full - soe) / jnp.maximum(jnp.abs(full), 1e-8))
    assert rel_err < 0.2


@pytest.mark.parametrize("history", ["short_memory", "soe"])
def test_compressed_history_methods_are_jit_compatible(history: str) -> None:
    dt = 0.05
    t = jnp.linspace(0.0, 1.0, 41)
    x = t**2

    kwargs = {"window_steps": 16} if history == "short_memory" else {"soe_poles": 8}
    fn = jax.jit(lambda v: caputo(v, dt=dt, order=0.5, history=history, **kwargs))
    result = fn(x)

    assert result.shape == x.shape
    assert jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize("history", ["short_memory", "soe"])
def test_compressed_history_operator_info_records_method(history: str) -> None:
    dt = 0.1
    t = jnp.linspace(0.0, 1.0, 21)
    kwargs = {"window_steps": 10} if history == "short_memory" else {"soe_poles": 6}
    result = caputo(
        t**2, dt=dt, order=0.5, history=history, return_info=True, **kwargs
    )

    assert result.operator_info.history.value == history
    assert result.operator_info.diagnostics.get("history") == history
    if history == "short_memory":
        assert result.operator_info.diagnostics.get("window_steps") == 10
    else:
        assert result.operator_info.diagnostics.get("soe_poles") == 6


def test_short_memory_window_steps_limits_cost() -> None:
    dt = 0.01
    n_steps = 201
    t = jnp.linspace(0.0, 1.0, n_steps)
    x = t**2

    # With a very small window the method should still run and return a finite
    # result, even though accuracy is poor.
    result = caputo(
        x, dt=dt, order=0.5, history="short_memory", window_steps=8
    )
    assert result.shape == x.shape
    assert jnp.all(jnp.isfinite(result))


def test_soe_rejects_invalid_parameters() -> None:
    dt = 0.1
    t = jnp.linspace(0.0, 1.0, 21)
    x = t**2

    with pytest.raises(ValueError, match="n_poles"):
        caputo(x, dt=dt, order=0.5, history="soe", soe_poles=0)

    with pytest.raises(ValueError, match="t_max"):
        caputo(x, dt=dt, order=0.5, history="soe", soe_t_max=-1.0)
