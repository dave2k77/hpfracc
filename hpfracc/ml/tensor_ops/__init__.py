from .base import TensorOps
from .torch_ops import TorchTensorOps
from .jax_ops import JaxTensorOps
from .numpy_ops import NumpyTensorOps
from hpfracc.ml.backends import get_backend_manager, BackendType
import warnings
from typing import Optional, Any

def get_tensor_ops(backend: Optional[BackendType] = None) -> TensorOps:
    """
    Factory function to get the appropriate TensorOps implementation.
    """
    if backend is None:
        manager = get_backend_manager()
        backend = manager.active_backend if manager is not None else BackendType.NUMBA
        
    if backend == BackendType.AUTO:
        manager = get_backend_manager()
        backend = manager.active_backend if manager is not None else BackendType.NUMBA

    if backend == BackendType.TORCH:
        return TorchTensorOps()
    elif backend == BackendType.JAX:
        return JaxTensorOps()
    elif backend == BackendType.NUMBA:
         # Numba typically uses NumPy arrays for storage
        return NumpyTensorOps()
    else:
        # Fallback or error
        raise ValueError(f"Unknown backend or backend not supported: {backend}")

def create_tensor(data: Any, *args, **kwargs) -> Any:
    """Wrapper to create tensor using active backend"""
    return get_tensor_ops().create_tensor(data, *args, **kwargs)

def switch_backend(backend: BackendType) -> bool:
    """Wrapper to switch backend"""
    return get_backend_manager().switch_backend(backend)



# For backward compatibility with wildcard imports
__all__ = ["get_tensor_ops", "TensorOps", "create_tensor", "switch_backend"]
