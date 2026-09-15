"""Shared typing aliases for public HPFRACC APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias

ArrayLike: TypeAlias = Any
PyTree: TypeAlias = Any
PRNGKey: TypeAlias = Any


class DynamicsFn(Protocol):
    """Callable protocol for explicit fractional dynamics."""

    def __call__(
        self,
        t: ArrayLike,
        state: PyTree,
        params: PyTree,
        *,
        rng_key: PRNGKey | None = None,
        inputs: PyTree | None = None,
    ) -> PyTree: ...


StateTransform: TypeAlias = Callable[[PyTree], PyTree]
