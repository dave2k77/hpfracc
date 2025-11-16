import os
import numpy as np
import pytest

pytestmark = pytest.mark.week3


@pytest.fixture(autouse=True)
def use_agg_backend():
    import matplotlib
    matplotlib.use("Agg")
    yield


def test_plot_manager_create_and_save(tmp_path):
    from hpfracc.utils.plotting import PlotManager

    pm = PlotManager(style="scientific", figsize=(4, 3))
    x = np.linspace(0, 1, 50)
    y = np.sin(2 * np.pi * x)

    fig, ax = pm.create_plot(x, y, title="Sine", xlabel="x", ylabel="y")
    assert fig is not None
    assert ax is not None

    out = tmp_path / "plots" / "plot.png"
    pm.save_plot(fig, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_top_level_plot_helpers(tmp_path):
    from hpfracc.utils import plotting as P

    x = np.linspace(0, 1, 20)
    y1 = np.sin(2 * np.pi * x)
    y2 = np.cos(2 * np.pi * x)

    # create_comparison_plot with dict
    data_dict = {"sin": (x, y1), "cos": (x, y2)}
    fig, axes = P.create_comparison_plot(data_dict)
    assert fig is not None

    # create_comparison_plot with x + dict of y
    fig2, axes2 = P.create_comparison_plot(x, {"sin": y1, "cos": y2})
    assert fig2 is not None

    # plot_convergence with grid_sizes + errors
    grid_sizes = [10, 20, 40]
    errors = {"L2": [0.2, 0.1, 0.05], "Linf": [0.3, 0.15, 0.075]}
    fig3, axes3 = P.plot_convergence(grid_sizes, errors)
    assert fig3 is not None

    # plot_error_analysis convenience
    fig4, axes4 = P.plot_error_analysis(y1, y1 + 0.1 * np.random.randn(*y1.shape))
    assert fig4 is not None

    # Saving via top-level save_plot
    out = tmp_path / "plots2" / "cmp.png"
    P.save_plot(fig, str(out))
    assert out.exists() and out.stat().st_size > 0