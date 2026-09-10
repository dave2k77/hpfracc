"""Comparison report checks for the experimental indicator."""

import importlib


def test_comparison_measures_cubic_improvement(enable_x64):
    audit = importlib.import_module("benchmarks.numerical.subdivision_comparison")
    legacy = audit.run_case("cubic", "step_doubling", 1e-3, "float64")
    guarded = audit.run_case("cubic", "subdivision", 1e-3, "float64")
    assert legacy.completed and guarded.completed
    assert guarded.max_abs_error < legacy.max_abs_error
    assert guarded.n_rejected > legacy.n_rejected


def test_comparison_omits_incomplete_errors():
    audit = importlib.import_module("benchmarks.numerical.subdivision_comparison")
    result = audit.run_case("half_order_decay", "subdivision", 1e-4,
                            "float32", max_steps=2)
    assert not result.completed
    assert result.max_abs_error is None and result.max_scaled_error is None
