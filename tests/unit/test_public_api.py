from __future__ import annotations

import hpfracc as hp


def test_public_namespaces_are_exported() -> None:
    assert hp.__version__ == "0.1.0a0"
    assert hp.ops is not None
    assert hp.solvers is not None
    assert hp.solvers.PredictorCorrector is not None
    assert hp.solvers.simulate is not None
    assert hp.nn.NeuralFODE is not None
    assert hp.prob.grid_calibrate_scalar is not None
    assert hp.config is not None
    assert hp.metrics is not None
    assert hp.experimental is not None


def test_placeholder_namespaces_are_importable() -> None:
    import hpfracc.brain
    import hpfracc.data
    import hpfracc.nn
    import hpfracc.observe
    import hpfracc.prob
    import hpfracc.train
    import hpfracc.viz

    assert "NeuralFODE" in hpfracc.nn.__all__
    assert "grid_calibrate_scalar" in hpfracc.prob.__all__
    assert hpfracc.brain.__all__ == []
    assert hpfracc.observe.__all__ == []
    assert hpfracc.train.__all__ == []
    assert hpfracc.data.__all__ == []
    assert hpfracc.viz.__all__ == []
