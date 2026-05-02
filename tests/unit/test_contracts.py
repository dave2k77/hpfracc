from __future__ import annotations

import pytest

from hpfracc.config import ExperimentMetadata, Provenance, RuntimeTarget
from hpfracc.ops import FractionalOrder, validate_order
from hpfracc.solvers import SimulationResult, SolverInfo


def test_fractional_order_accepts_open_unit_interval() -> None:
    order = FractionalOrder(0.5)
    assert order.alpha == 0.5
    assert validate_order(0.75) == 0.75


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_fractional_order_rejects_values_outside_open_unit_interval(
    alpha: float,
) -> None:
    with pytest.raises(ValueError, match="open interval"):
        FractionalOrder(alpha)


def test_provenance_exports_runtime_target_value() -> None:
    provenance = Provenance(runtime_target=RuntimeTarget.CPU)
    payload = provenance.to_dict()
    assert payload["package_version"] == "0.1.0a0"
    assert payload["backend"] == "jax"
    assert payload["runtime_target"] == "cpu"


def test_experiment_metadata_exports_nested_provenance() -> None:
    metadata = ExperimentMetadata(name="smoke", tags=("contract",))
    payload = metadata.to_dict()
    assert payload["name"] == "smoke"
    assert payload["tags"] == ["contract"]
    assert payload["provenance"]["package_version"] == "0.1.0a0"


def test_simulation_result_contract_accepts_structured_metadata() -> None:
    solver_info = SolverInfo(
        name="placeholder",
        method="contract",
        fractional_order=0.5,
        step_size=0.01,
        n_steps=10,
        warnings=("example",),
    )
    result = SimulationResult(
        ts=[0.0, 0.01],
        latent_state=[[1.0], [0.9]],
        solver_info=solver_info,
    )
    assert result.observed is None
    assert result.solver_info is solver_info
    assert solver_info.to_dict()["warnings"] == ["example"]

