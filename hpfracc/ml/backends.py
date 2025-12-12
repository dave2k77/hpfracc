"""
Backend Management System for Multi-Framework Support

This module provides unified interfaces for PyTorch, JAX, and NUMBA backends,
enabling seamless switching between frameworks and automatic backend selection
based on data type, hardware availability, and performance requirements.
"""

from typing import Optional, Any, Dict, List, Callable
from enum import Enum
import warnings
import importlib

# Backend availability checking (lazy; do not import heavy libs at module import)


def _spec_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


TORCH_AVAILABLE = _spec_available("torch")
JAX_AVAILABLE = _spec_available("jax") and _spec_available("jax.numpy")
NUMPY_AVAILABLE = _spec_available("numpy")
# NUMBA lane uses NumPy arrays; consider lane available if NumPy exists
NUMBA_AVAILABLE = _spec_available("numba") or NUMPY_AVAILABLE


class BackendType(Enum):
    """Available computation backends"""
    TORCH = "torch"
    JAX = "jax"
    NUMBA = "numba"
    AUTO = "auto"


class BackendManager:
    """
    Manages backend selection and provides unified interfaces

    This class handles automatic backend selection based on:
    - Data type and size
    - Hardware availability (CPU/GPU)
    - Performance requirements
    - User preferences
    """

    def __init__(
        self,
        preferred_backend: BackendType = BackendType.AUTO,
        force_cpu: bool = False,
        enable_jit: bool = True,
        enable_gpu: bool = True
    ):
        self.preferred_backend = preferred_backend
        self.force_cpu = force_cpu
        self.enable_jit = enable_jit
        self.enable_gpu = enable_gpu

        # Detect available backends
        self.available_backends = self._detect_available_backends()

        # Current active backend
        self.active_backend = self._select_optimal_backend()

        # Backend-specific configurations
        self.backend_configs = self._initialize_backend_configs()

        try:
            print(
                f"Backend Manager initialized with {self.active_backend.value}")
            print(
                f"Available backends: {[b.value for b in self.available_backends]}")
        except UnicodeEncodeError:
            # Fallback for systems with encoding issues
            print(
                f"Backend Manager initialized with {self.active_backend.value}")
            print(
                f"Available backends: {[b.value for b in self.available_backends]}")

    def _detect_available_backends(self) -> List[BackendType]:
        """Detect which backends are available on the system"""
        available = []

        if TORCH_AVAILABLE:
            available.append(BackendType.TORCH)
            try:
                torch = importlib.import_module("torch")
                if hasattr(torch, "cuda") and torch.cuda.is_available() and not self.force_cpu:
                    print("🚀 PyTorch CUDA support detected")
            except Exception:
                pass

        if JAX_AVAILABLE:
            available.append(BackendType.JAX)
            try:
                jax = importlib.import_module("jax")
                devices = jax.devices()
                if any('gpu' in str(d).lower()
                       for d in devices) and not self.force_cpu:
                    print("🚀 JAX GPU support detected")
            except BaseException:
                pass

        if NUMBA_AVAILABLE:
            available.append(BackendType.NUMBA)
            try:
                numba = importlib.import_module("numba")
                if hasattr(numba, 'cuda') and numba.cuda.is_available() and not self.force_cpu:
                    print("🚀 NUMBA CUDA support detected")
            except BaseException:
                pass

        if not available:
            raise RuntimeError("No computation backends available!")

        return available

    def _select_optimal_backend(self) -> BackendType:
        """Select the optimal backend based on preferences and availability"""
        if self.preferred_backend == BackendType.AUTO:
            # Prefer PyTorch by default to match test expectations and widest
            # API coverage
            if BackendType.TORCH in self.available_backends:
                return BackendType.TORCH
            elif BackendType.JAX in self.available_backends and self.enable_gpu:
                return BackendType.JAX
            elif BackendType.NUMBA in self.available_backends:
                return BackendType.NUMBA
            else:
                return self.available_backends[0]
        else:
            if self.preferred_backend in self.available_backends:
                return self.preferred_backend
            else:
                warnings.warn(
                    f"Preferred backend {self.preferred_backend.value} not available, using {self.available_backends[0].value}")
                return self.available_backends[0]

    def _initialize_backend_configs(self) -> Dict[BackendType, Dict[str, Any]]:
        """Initialize backend-specific configurations"""
        configs = {}

        # PyTorch configuration (lazy import)
        if BackendType.TORCH in self.available_backends:
            try:
                torch = importlib.import_module("torch")
                configs[BackendType.TORCH] = {
                    'device': 'cuda' if hasattr(torch, 'cuda') and torch.cuda.is_available() and not self.force_cpu else 'cpu',
                    'dtype': getattr(torch, 'float32', None),
                    'enable_amp': True,
                    'enable_compile': hasattr(torch, 'compile'),
                }
            except Exception:
                configs[BackendType.TORCH] = {
                    'device': 'cpu',
                    'dtype': None,
                    'enable_amp': False,
                    'enable_compile': False,
                }

        # JAX configuration (lazy import)
        if BackendType.JAX in self.available_backends:
            try:
                jnp = importlib.import_module("jax.numpy")
                configs[BackendType.JAX] = {
                    'device': 'gpu' if self.enable_gpu and not self.force_cpu else 'cpu',
                    'dtype': getattr(jnp, 'float32', None),
                    'enable_jit': self.enable_jit,
                    'enable_x64': False,
                    'enable_amp': True,
                }
            except Exception:
                configs[BackendType.JAX] = {
                    'device': 'cpu',
                    'dtype': None,
                    'enable_jit': False,
                    'enable_x64': False,
                    'enable_amp': False,
                }

        # NUMBA configuration (lazy import)
        if BackendType.NUMBA in self.available_backends:
            # Prefer numpy-backed lane if numba not available
            np = importlib.import_module("numpy") if NUMPY_AVAILABLE else None
            gpu_available = False
            try:
                numba = importlib.import_module("numba")
                try:
                    gpu_available = hasattr(
                        numba, 'cuda') and numba.cuda.is_available()
                except BaseException:
                    gpu_available = False
                dtype_val = getattr(numba, 'float32', None) or (
                    getattr(np, 'float32', None) if np else None)
            except Exception:
                dtype_val = getattr(np, 'float32', None) if np else None

            configs[BackendType.NUMBA] = {
                'device': 'gpu' if gpu_available and not self.force_cpu else 'cpu',
                'dtype': dtype_val,
                'enable_jit': self.enable_jit and _spec_available("numba"),
                'enable_parallel': _spec_available("numba"),
                'enable_fastmath': _spec_available("numba"),
            }

        return configs

    def get_backend_config(
            self, backend: Optional[BackendType] = None) -> Dict[str, Any]:
        """Get configuration for a specific backend"""
        backend = backend or self.active_backend
        return self.backend_configs.get(backend, {})

    def switch_backend(self, backend: BackendType) -> bool:
        """Switch to a different backend"""
        if backend in self.available_backends:
            self.active_backend = backend
            print(f"🔄 Switched to {backend.value} backend")
            return True
        else:
            warnings.warn(f"Backend {backend.value} not available")
            return False

    def get_tensor_lib(self) -> Any:
        """Get the active tensor library"""
        if self.active_backend == BackendType.TORCH:
            return importlib.import_module("torch")
        elif self.active_backend == BackendType.JAX:
            return importlib.import_module("jax.numpy")
        elif self.active_backend == BackendType.NUMBA:
            # NUMBA lane arrays are NumPy
            return importlib.import_module("numpy")
        else:
            raise RuntimeError(f"Unknown backend: {self.active_backend}")

    def create_tensor(self, data: Any, **kwargs) -> Any:
        """Create a tensor in the active backend"""
        if self.active_backend == BackendType.TORCH:
            # Ensure consistent dtype for PyTorch
            if 'dtype' not in kwargs:
                # Preserve integer types for classification targets
                if hasattr(data, 'dtype') and 'int' in str(data.dtype):
                    kwargs['dtype'] = importlib.import_module("torch").long
                else:
                    kwargs['dtype'] = importlib.import_module("torch").float32
            return importlib.import_module("torch").tensor(data, **kwargs)
        elif self.active_backend == BackendType.JAX:
            # Ensure consistent dtype for JAX
            if 'dtype' not in kwargs:
                # Preserve integer types for classification targets
                if hasattr(data, 'dtype') and 'int' in str(data.dtype):
                    kwargs['dtype'] = importlib.import_module(
                        "jax.numpy").int32
                else:
                    kwargs['dtype'] = importlib.import_module(
                        "jax.numpy").float32
            return importlib.import_module("jax.numpy").array(data, **kwargs)
        elif self.active_backend == BackendType.NUMBA:
            # NUMBA works with numpy arrays
            import numpy as np
            if 'dtype' not in kwargs:
                # Preserve integer types for classification targets
                if hasattr(data, 'dtype') and 'int' in str(data.dtype):
                    kwargs['dtype'] = np.int32
                else:
                    kwargs['dtype'] = np.float32
            return np.array(data, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.active_backend}")

    def to_device(self, tensor: Any, device: Optional[str] = None) -> Any:
        """Move tensor to specified device"""
        if self.active_backend == BackendType.TORCH:
            return tensor.to(
                device or self.backend_configs[BackendType.TORCH]['device'])
        elif self.active_backend == BackendType.JAX:
            # JAX handles device placement differently
            return tensor
        elif self.active_backend == BackendType.NUMBA:
            # NUMBA handles device placement differently
            return tensor
        else:
            raise RuntimeError(f"Unknown backend: {self.active_backend}")

    def compile_function(self, func: Callable) -> Callable:
        """Compile a function using the active backend's compilation system"""
        if self.active_backend == BackendType.TORCH:
            torch = importlib.import_module("torch")
            if hasattr(torch, 'compile'):
                return torch.compile(func)
            else:
                return func
        elif self.active_backend == BackendType.JAX:
            if self.enable_jit:
                jax = importlib.import_module("jax")
                return jax.jit(func)
            else:
                return func
        elif self.active_backend == BackendType.NUMBA:
            if self.enable_jit and _spec_available("numba"):
                numba = importlib.import_module("numba")
                return numba.jit(func)
            else:
                return func
        else:
            return func


# Global backend manager instance
_backend_manager: Optional[BackendManager] = None


def get_backend_manager() -> BackendManager:
    """Get the global backend manager instance"""
    global _backend_manager
    if _backend_manager is None:
        _backend_manager = BackendManager()
    return _backend_manager


def set_backend_manager(manager: BackendManager) -> None:
    """Set the global backend manager instance"""
    global _backend_manager
    _backend_manager = manager


def get_active_backend() -> BackendType:
    """Get the currently active backend"""
    return get_backend_manager().active_backend


def switch_backend(backend: BackendType) -> bool:
    """Switch to a different backend"""
    return get_backend_manager().switch_backend(backend)
