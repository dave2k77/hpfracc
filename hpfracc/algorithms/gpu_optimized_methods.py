"""
[DEPRECATED] This module is deprecated and will be removed in a future version.
Please use `hpfracc.algorithms.derivatives` with `backend='cuda'` or `backend='jax'`.

This module now serves as a compatibility shim around the Unified API.
"""

import warnings
import pytest # type: ignore
from typing import Union, Optional, Callable
import numpy as np

from ..core.definitions import FractionalOrder
from .derivatives import RiemannLiouville, Caputo, GrunwaldLetnikov
# Expose availability flags for tests
from .impls.jax_backend import JAX_AVAILABLE
from .impls.cuda_backend import CUPY_AVAILABLE

class GPUConfig:
    """
    [DEPRECATED] Configuration for legacy GPU dispatch.
    """
    def __init__(
        self,
        backend: str = "auto",
        memory_limit: float = 0.8,
        batch_size: Optional[int] = None,
        multi_gpu: bool = False,
        monitor_performance: bool = True,
        fallback_to_cpu: bool = True,
        device_id: Optional[int] = None,
        use_intelligent_selection: bool = True,
    ):
        warnings.warn(
            "GPUConfig is deprecated. Use arguments in Unified API classes directly.",
            DeprecationWarning,
            stacklevel=2
        )
        self.backend = backend
        self.performance_stats = {
            "gpu_time": 0.0,
            "cpu_time": 0.0,
            "memory_usage": [],
            "speedup": 1.0,
        }
        self.monitor_performance = monitor_performance
        self.fallback_to_cpu = fallback_to_cpu
        self.device_id = device_id
        self.memory_limit = memory_limit

    def select_backend_for_data(self, *args, **kwargs):
        return self.backend


class _LegacyShim:
    """Helper to map legacy config to Unified backend."""
    def _map_config(self, gpu_config):
        if gpu_config is None:
            return "auto"
        
        backend = gpu_config.backend
        if backend == "cupy":
            return "cuda"
        if backend == "jax":
            return "jax"
        if backend == "numpy":
            return "numpy"
        return "auto"


class GPUOptimizedRiemannLiouville(RiemannLiouville, _LegacyShim):
    """
    [DEPRECATED] Use `hpfracc.algorithms.RiemannLiouville(..., backend='cuda'/'jax')`.
    """
    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        gpu_config: Optional[GPUConfig] = None,
        *,
        config: Optional[GPUConfig] = None,
        batch_size: Optional[int] = None,
    ):
        warnings.warn(
            "GPUOptimizedRiemannLiouville is deprecated. Use RiemannLiouville(backend=...) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        legacy_conf = gpu_config or config or GPUConfig(backend="auto")
        backend = self._map_config(legacy_conf)
        
        super().__init__(alpha, backend=backend)
        self.gpu_config = legacy_conf 
        self.batch_size = batch_size

    # def compute(): Inherit from base
            
    def enable_monitoring(self): pass



class GPUOptimizedCaputo(Caputo, _LegacyShim):
    """
    [DEPRECATED] Use `hpfracc.algorithms.Caputo(..., backend='cuda'/'jax')`.
    """
    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        gpu_config: Optional[GPUConfig] = None,
        *,
        config: Optional[GPUConfig] = None,
        memory_efficient: Optional[bool] = None,
    ):
        warnings.warn(
            "GPUOptimizedCaputo is deprecated. Use Caputo(backend=...) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        legacy_conf = gpu_config or config or GPUConfig(backend="auto")
        backend = self._map_config(legacy_conf)
        
        super().__init__(alpha, backend=backend)
        self.gpu_config = legacy_conf
        self.memory_efficient = memory_efficient
        
    def compute(self, f, t, h=None, method="l1"):
        if method != "l1":
            warnings.warn("Method selection is deprecated/ignored in Unified API.")
        return super().compute(f, t, h)


class GPUOptimizedGrunwaldLetnikov(GrunwaldLetnikov, _LegacyShim):
    """
    [DEPRECATED] Use `hpfracc.algorithms.GrunwaldLetnikov(..., backend='cuda'/'jax')`.
    """
    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        gpu_config: Optional[GPUConfig] = None,
        *,
        config: Optional[GPUConfig] = None,
        use_shared_memory: Optional[bool] = None,
    ):
        warnings.warn(
            "GPUOptimizedGrunwaldLetnikov is deprecated. Use GrunwaldLetnikov(backend=...) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        legacy_conf = gpu_config or config or GPUConfig(backend="auto")
        backend = self._map_config(legacy_conf)
        
        super().__init__(alpha, backend=backend)
        self.gpu_config = legacy_conf
        self.use_shared_memory = use_shared_memory

# Stubs for other missing symbols
class MultiGPUManager:
    def __init__(self, *args, **kwargs):
        warnings.warn("MultiGPUManager is deprecated/stubbed.", DeprecationWarning)

def gpu_optimized_riemann_liouville(f, alpha, t, h=None, *args, **kwargs):
    warnings.warn("gpu_optimized_riemann_liouville is deprecated. Use Unified API.", DeprecationWarning)
    # Map args to compute
    return GPUOptimizedRiemannLiouville(alpha).compute(f, t, h)

def gpu_optimized_caputo(f, alpha, t, h=None, *args, **kwargs):
    warnings.warn("gpu_optimized_caputo is deprecated. Use Unified API.", DeprecationWarning)
    return GPUOptimizedCaputo(alpha).compute(f, t, h)

def gpu_optimized_grunwald_letnikov(f, alpha, t, h=None, *args, **kwargs):
    warnings.warn("gpu_optimized_grunwald_letnikov is deprecated. Use Unified API.", DeprecationWarning)
    return GPUOptimizedGrunwaldLetnikov(alpha).compute(f, t, h)

def benchmark_gpu_vs_cpu(*args, **kwargs):
    warnings.warn("benchmark_gpu_vs_cpu is deprecated.", DeprecationWarning)
    return {"gpu_time": 0.0, "cpu_time": 0.0}
