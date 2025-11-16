import math
import numpy as np
import pytest

pytestmark = pytest.mark.week3


def test_error_functions_basic_correctness():
    from hpfracc.utils.error_analysis import (
        absolute_error,
        relative_error,
        l2_error,
        error_statistics,
        compute_error_metrics,
        validate_solution,
        analyze_convergence,
    )

    a = np.array([1.0, 2.0, 4.0])
    n = np.array([1.1, 1.8, 3.5])

    abs_err = absolute_error(a, n)
    rel_err = relative_error(a, n)
    l2 = l2_error(a, n)
    stats = error_statistics(a, n)
    metrics = compute_error_metrics(n, a)

    assert abs_err.shape == a.shape
    assert np.all(abs_err >= 0)
    assert np.all(rel_err >= 0)
    assert l2 >= 0 and math.isfinite(l2)

    for k in ["mean", "std", "min", "max", "rmse"]:
        assert k in stats
        assert math.isfinite(stats[k])

    for k in ["l1", "l2", "linf", "mse", "rmse", "max_error"]:
        assert k in metrics
        assert math.isfinite(float(metrics[k]))

    # validate_solution for arrays
    assert validate_solution(np.array([0.0, 1.0])) is True
    assert validate_solution(np.array([0.0, np.nan])) is False

    # analyze_convergence wrapper with grid_sizes + errors dict
    grid_sizes = [10, 20, 40, 80]
    errs = {"methodA": [0.5, 0.25, 0.125, 0.0625], "methodB": [0.4, 0.2, 0.1, 0.05]}
    conv = analyze_convergence(grid_sizes, errs)
    assert "convergence_rates" in conv
    assert set(conv["convergence_rates"].keys()) == {"methodA", "methodB"}