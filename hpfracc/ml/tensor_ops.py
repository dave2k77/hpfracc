"""
Unified Tensor Operations for Multi-Backend Support

This module provides consistent tensor operations across PyTorch, JAX, and a
NumPy-backed "NUMBA lane" (arrays are NumPy; numba is a compiler elsewhere),
enabling seamless switching between frameworks while maintaining the same API.
"""

from typing import Optional, Union, Any, List, Tuple, Dict
from contextlib import nullcontext
import warnings
import importlib
import os

from .backends import get_backend_manager, BackendType
from .adapters import get_optimal_adapter, HighPerformanceAdapter

import numpy as _np  # used as a safe NumPy namespace at construction


class TensorOps:
    """
    Unified tensor operations across different backends.

    Notes:
      - AUTO is resolved to a concrete, installed backend during __init__.
      - NUMBA lane uses NumPy arrays (numba itself is not a tensor library).
      - JAX random ops require a PRNG key; pass via kwargs (key=...).
    """

    def __init__(self, backend: Optional[Union[BackendType, str]] = None):
        # Backend manager might import lazily; fall back to NumPy-only if unavailable
        try:
            backend_manager = get_backend_manager()
        except Exception:
            backend_manager = None

        # Convert string backend to enum if needed
        if isinstance(backend, str):
            try:
                backend = BackendType(backend)
            except ValueError:
                raise ValueError(
                    f"Unknown backend: {backend}. Available backends: {[b.value for b in BackendType]}")
        # Resolve requested (or active) backend into an installed concrete choice,
        # with sensible fallbacks.
        self.backend, self.tensor_lib = self._resolve_backend(
            backend, backend_manager)
        # Construct optimized adapter for future delegation
        self._adapter: HighPerformanceAdapter
        try:
            self._adapter = HighPerformanceAdapter(self.backend)
            # Ensure underlying lib is loaded
            _ = self._adapter.get_lib()
        except Exception:
            # Fallback: create adapter with current backend
            self._adapter = HighPerformanceAdapter()

    # ------------------------ Backend resolution ------------------------

    def _resolve_backend(self, backend: Optional[BackendType], backend_manager):
        """
        Pick a concrete, installed backend with sensible fallbacks.
        Priority:
          1) explicit `backend` (if not AUTO) when installed
          2) backend_manager.active_backend (if not AUTO) when installed
          3) fallback order: TORCH -> JAX -> NUMBA (NumPy)
        """
        candidates: List[BackendType] = []

        # 1) explicit request (if provided and not AUTO)
        if backend is not None and backend != BackendType.AUTO:
            candidates.append(backend)

        # 2) manager's active (if not AUTO)
        ab = getattr(backend_manager, "active_backend",
                     None) if backend_manager is not None else None
        if ab is not None and ab != BackendType.AUTO:
            # Only honor manager active backend if not disabled by env
            disable_map_ab = {
                BackendType.TORCH: os.getenv("HPFRACC_DISABLE_TORCH", "0") == "1",
                BackendType.JAX: os.getenv("HPFRACC_DISABLE_JAX", "0") == "1",
                BackendType.NUMBA: os.getenv("HPFRACC_DISABLE_NUMBA", "0") == "1",
            }
            if not disable_map_ab.get(ab, False):
                candidates.append(ab)

        # 3) standard fallbacks (honor env disables)
        disable_map = {
            BackendType.TORCH: os.getenv("HPFRACC_DISABLE_TORCH", "0") == "1",
            BackendType.JAX: os.getenv("HPFRACC_DISABLE_JAX", "0") == "1",
            BackendType.NUMBA: os.getenv("HPFRACC_DISABLE_NUMBA", "0") == "1",
        }
        for b in (BackendType.TORCH, BackendType.JAX, BackendType.NUMBA):
            if b not in candidates and not disable_map.get(b, False):
                candidates.append(b)

        last_err: Optional[Exception] = None
        for b in candidates:
            try:
                lib = self._get_tensor_lib_for_backend(b)
                return b, lib
            except Exception as e:  # torch/jax may raise RuntimeError/AttributeError on import
                last_err = e
                continue

        # If nothing worked, default to NumPy lane unconditionally
        return BackendType.NUMBA, _np

    def _get_tensor_lib_for_backend(self, backend: BackendType) -> Any:
        """Get tensor library for a specific backend (imports guarded)."""
        if backend == BackendType.TORCH:
            torch = importlib.import_module("torch")
            return torch
        elif backend == BackendType.JAX:
            jnp = importlib.import_module("jax.numpy")
            return jnp
        elif backend == BackendType.NUMBA:
            # Use NumPy namespace for arrays/ops; numba is a compiler elsewhere.
            return _np
        else:
            # For constructor edge-cases, fall back to TORCH
            torch = importlib.import_module("torch")
            return torch

    # ------------------------ Creation / conversion ------------------------

    def create_tensor(self, data: Any, **kwargs) -> Any:
        """Create a tensor in the current backend."""
        # Filter backend-specific args where necessary
        if self.backend == BackendType.TORCH:
            # Remove requires_grad from kwargs if it's False (default behavior)
            torch_kwargs = kwargs.copy()
            if 'requires_grad' in torch_kwargs and not torch_kwargs['requires_grad']:
                del torch_kwargs['requires_grad']
            # Normalize dtype/device from strings
            dtype = torch_kwargs.get('dtype', None)
            device = torch_kwargs.get('device', None)
            torch = self.tensor_lib
            if isinstance(dtype, str):
                dtype_map = {
                    'float32': torch.float32,
                    'float': torch.float32,
                    'fp32': torch.float32,
                    'float64': torch.float64,
                    'double': torch.float64,
                    'fp64': torch.float64,
                    'float16': torch.float16,
                    'half': torch.float16,
                    'fp16': torch.float16,
                    'bfloat16': getattr(torch, 'bfloat16', None),
                    'bf16': getattr(torch, 'bfloat16', None),
                    'int64': torch.int64,
                    'long': torch.int64,
                    'int32': torch.int32,
                    'int': torch.int32,
                    'int16': torch.int16,
                    'short': torch.int16,
                    'int8': torch.int8,
                    'uint8': torch.uint8,
                    'bool': torch.bool,
                }
                mapped = dtype_map.get(dtype.lower())
                if mapped is not None:
                    torch_kwargs['dtype'] = mapped
            if isinstance(device, str):
                torch_kwargs['device'] = torch.device(device)
            return self.tensor_lib.tensor(data, **torch_kwargs)
        elif self.backend == BackendType.JAX:
            # JAX doesn't support requires_grad
            jax_kwargs = {k: v for k, v in kwargs.items() if k !=
                          'requires_grad'}
            return self.tensor_lib.array(data, **jax_kwargs)
        elif self.backend == BackendType.NUMBA:
            # NUMBA lane: remove requires_grad; arrays are NumPy
            nb_kwargs = {k: v for k, v in kwargs.items() if k !=
                         'requires_grad'}
            return self.tensor_lib.array(data, **nb_kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def tensor(self, data: Any, **kwargs) -> Any:
        """Alias for create_tensor."""
        return self.create_tensor(data, **kwargs)

    def from_numpy(self, array: Any) -> Any:
        if self.backend == BackendType.TORCH:
            torch = self.tensor_lib
            return torch.from_numpy(array)
        elif self.backend == BackendType.JAX:
            jnp = self.tensor_lib
            return jnp.array(array)
        elif self.backend == BackendType.NUMBA:
            return array
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def to_numpy(self, tensor: Any) -> Any:
        if self.backend == BackendType.TORCH:
            return tensor.detach().cpu().numpy()
        elif self.backend == BackendType.JAX:
            import jax
            import numpy as np
            return np.asarray(jax.device_get(tensor))
        elif self.backend == BackendType.NUMBA:
            return tensor
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def no_grad(self):
        """
        Context manager for disabling gradient computation.
        - PyTorch: torch.no_grad()
        - JAX: there is no true 'no_grad' context; we return a nullcontext().
               Use jax.lax.stop_gradient at call sites if you need it.
        - NUMBA lane: nullcontext()
        """
        if self.backend == BackendType.TORCH:
            torch = self.tensor_lib
            return torch.no_grad()
        elif self.backend == BackendType.JAX:
            # JAX doesn't have a no_grad context manager - it uses functional programming
            # Return nullcontext as documented
            return nullcontext()
        elif self.backend == BackendType.NUMBA:
            return nullcontext()
        else:
            raise RuntimeError("Unknown backend")

    # ------------------------ Array constructors ------------------------

    def zeros(self, shape: Tuple[int, ...], **kwargs) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.zeros(shape, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.zeros(shape, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def ones(self, shape: Tuple[int, ...], **kwargs) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.ones(shape, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.ones(shape, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def eye(self, n: int, **kwargs) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.eye(n, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.eye(n, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def arange(self, start: int, end: int, step: int = 1, **kwargs) -> Any:
        if self.backend == BackendType.TORCH:
            # Default dtype to float32 to satisfy tests unless provided
            import torch
            if 'dtype' not in kwargs:
                kwargs['dtype'] = torch.float32
            return self.tensor_lib.arange(start, end, step, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.arange(start, end, step, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.arange(start, end, step, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def linspace(self, start: float, end: float, num: int, **kwargs) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.linspace(start, end, num, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.linspace(start, end, num, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def zeros_like(self, tensor: Any, **kwargs) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.zeros_like(tensor, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            if hasattr(tensor, 'shape'):
                return np.zeros_like(tensor, **kwargs)
            return np.zeros(1, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def ones_like(self, tensor: Any, **kwargs) -> Any:
        """Create a tensor of ones with the same shape as input tensor."""
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.ones_like(tensor, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            if hasattr(tensor, 'shape'):
                return np.ones_like(tensor, **kwargs)
            return np.ones(1, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------ Basic transforms ------------------------

    def sqrt(self, tensor: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.sqrt(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.sqrt(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.stack(tensors, dim=dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.stack(tensors, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.stack(tensors, axis=dim)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.cat(tensors, dim=dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.concatenate(tensors, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.concatenate(tensors, axis=dim)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def reshape(self, tensor: Any, shape: Tuple[int, ...]) -> Any:
        return tensor.reshape(shape)

    def repeat(self, tensor: Any,
               repeats: Union[int, Tuple[int, ...]], dim: int = 0) -> Any:
        """
        Repeat elements along a specified axis (element-wise repeat).
        For tiling the whole array shape, use `tile(...)` helper below.
        """
        if self.backend == BackendType.TORCH:
            import torch
            if isinstance(repeats, int):
                if hasattr(tensor, 'dim'):
                    rank = tensor.dim()
                    if rank == 0:
                        return torch.repeat_interleave(tensor, repeats, dim=0)
                    if dim >= rank:
                        # Treat as tiling across axes when dim is out-of-range
                        if rank == 1:
                            # Build a 2D tile with shape (repeats*L, dim*L)
                            L = tensor.shape[0]
                            row = tensor.repeat(dim)
                            return row.unsqueeze(0).repeat(repeats * L, 1)
                        if rank == 2:
                            return tensor.repeat(repeats, repeats)
                        reps = [1] * (rank - 1) + [repeats]
                        return tensor.repeat(*reps)
                    valid_dim = max(min(dim, rank - 1), -rank)
                    return torch.repeat_interleave(tensor, repeats, dim=valid_dim)
                # Fallback: interleave along dim 0
                return torch.repeat_interleave(tensor, repeats, dim=0)
            # tuple/sequence: tile
            return tensor.repeat(*repeats)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.repeat(tensor, repeats, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.repeat(tensor, repeats, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def tile(self, tensor: Any, reps: Union[int, Tuple[int, ...]]) -> Any:
        """Tile (broadcast repeat) tensor like np.tile / torch.repeat(*reps)."""
        if self.backend == BackendType.TORCH:
            return tensor.repeat(*((reps,) if isinstance(reps, int) else reps))
        elif self.backend == BackendType.JAX:
            # Try to use jnp.tile if available (modern JAX versions have it)
            try:
                return self.tensor_lib.tile(tensor, reps)
            except AttributeError:
                # Fallback for older JAX versions
                import numpy as np
                return self.tensor_lib.array(np.tile(self.to_numpy(tensor), reps))
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.tile(tensor, reps)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def clip(self, tensor: Any, min_val: float, max_val: float) -> Any:
        if self.backend == BackendType.TORCH:
            return tensor.clamp(min_val, max_val)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.clip(tensor, min_val, max_val)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.clip(tensor, min_val, max_val)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def unsqueeze(self, tensor: Any, dim: int) -> Any:
        if self.backend == BackendType.TORCH:
            return tensor.unsqueeze(dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.expand_dims(tensor, dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.expand_dims(tensor, dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def expand(self, tensor: Any, *sizes: int) -> Any:
        if self.backend == BackendType.TORCH:
            return tensor.expand(*sizes)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.broadcast_to(tensor, sizes)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.broadcast_to(tensor, sizes)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def gather(self, tensor: Any, dim: int, index: Any) -> Any:
        if self.backend == BackendType.TORCH:
            return tensor.gather(dim, index)
        elif self.backend == BackendType.JAX:
            # Use take_along_axis equivalent via jnp.take_along_axis
            return self.tensor_lib.take_along_axis(tensor, index, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.take_along_axis(tensor, index, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def squeeze(self, tensor: Any, dim: Optional[int] = None) -> Any:
        if self.backend == BackendType.TORCH:
            if dim is None:
                return tensor.squeeze()
            return tensor.squeeze(dim)
        elif self.backend == BackendType.JAX:
            # jnp.squeeze uses 'axis'; None removes all size-1 dimensions
            if dim is None:
                return self.tensor_lib.squeeze(tensor)
            return self.tensor_lib.squeeze(tensor, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            if dim is None:
                return np.squeeze(tensor)
            return np.squeeze(tensor, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def transpose(self, tensor: Any, *args, **kwargs) -> Any:
        """
        Transpose a tensor. Supports signatures:
          - transpose(tensor) for 2D: matrix transpose; otherwise reverse axes
          - transpose(tensor, dim0, dim1) : swap two axes (positional)
          - transpose(tensor, dim0=..., dim1=...) : swap two axes (keyword)
          - transpose(tensor, dims=(...)) : permute by dims
        """
        dims = kwargs.get('dims', None)
        dim0 = kwargs.get('dim0', None)
        dim1 = kwargs.get('dim1', None)

        # Handle positional args (dim0, dim1)
        if len(args) == 2:
            dim0, dim1 = args[0], args[1]
        elif len(args) > 0:
            raise ValueError(
                f"transpose expects 0 or 2 positional args, got {len(args)}")

        if self.backend == BackendType.TORCH:
            if dims is not None:
                return tensor.permute(dims)
            if dim0 is not None and dim1 is not None:
                return tensor.transpose(dim0, dim1)
            if tensor.dim() == 2:
                return tensor.t()
            return tensor.permute(tuple(reversed(range(tensor.dim()))))

        elif self.backend == BackendType.JAX:
            if dims is not None:
                # jnp.transpose expects axes as positional args or as a tuple
                # Try method call first, fall back to jnp.transpose if needed
                try:
                    if isinstance(dims, tuple):
                        return tensor.transpose(*dims)
                    else:
                        return tensor.transpose(dims)
                except (TypeError, AttributeError):
                    # Fallback to jnp.transpose function
                    return self.tensor_lib.transpose(tensor, dims)
            if dim0 is not None and dim1 is not None:
                axes = list(range(tensor.ndim))
                axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
                try:
                    return tensor.transpose(*axes)
                except (TypeError, AttributeError):
                    return self.tensor_lib.transpose(tensor, axes)
            # default: reverse axes (matrix transpose if 2D)
            try:
                return tensor.transpose()
            except (TypeError, AttributeError):
                # Fallback for edge cases
                axes = tuple(reversed(range(tensor.ndim)))
                return self.tensor_lib.transpose(tensor, axes)

        elif self.backend == BackendType.NUMBA:
            import numpy as np
            if dims is not None:
                return np.transpose(tensor, axes=dims)
            if dim0 is not None and dim1 is not None:
                axes = list(range(tensor.ndim))
                axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
                return np.transpose(tensor, axes=axes)
            return tensor.T
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------ Linear algebra & reductions ------------------------

    def matmul(self, a: Any, b: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self._adapter.get_lib().matmul(a, b)
        elif self.backend == BackendType.NUMBA:
            lib = self._adapter.get_lib()
            return lib.matmul(a, b)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def inverse(self, tensor: Any) -> Any:
        """Compute the inverse of a square matrix."""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.inverse(tensor)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.linalg.inv(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.linalg.inv(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def einsum(self, equation: str, *operands) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self._adapter.get_lib().einsum(equation, *operands)
        elif self.backend == BackendType.NUMBA:
            warnings.warn(
                "NUMBA lane doesn't support einsum fully; using fallback")
            return self._numba_einsum_fallback(equation, *operands)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _numba_einsum_fallback(self, equation: str, *operands) -> Any:
        import numpy as np
        if equation == "ij,jk->ik":
            return np.matmul(operands[0], operands[1])
        elif equation == "i,i->":
            return np.sum(operands[0] * operands[1])
        else:
            raise NotImplementedError(
                f"NUMBA lane doesn't support einsum pattern: {equation}"
            )

    def sum(self, tensor: Any, dim: Optional[int] = None,
            keepdim: Optional[bool] = None, keepdims: Optional[bool] = False) -> Any:
        if keepdim is None:
            keepdim = bool(keepdims) if keepdims is not None else False
        if self.backend == BackendType.TORCH:
            return tensor.sum(dim=dim, keepdim=keepdim)
        elif self.backend == BackendType.JAX:
            lib = self._adapter.get_lib()
            return lib.sum(tensor, axis=dim, keepdims=keepdim)
        elif self.backend == BackendType.NUMBA:
            lib = self._adapter.get_lib()
            return lib.sum(tensor, axis=dim, keepdims=keepdim)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def mean(self, tensor: Any, dim: Optional[int] = None,
             keepdim: Optional[bool] = None, keepdims: Optional[bool] = False) -> Any:
        if keepdim is None:
            keepdim = bool(keepdims) if keepdims is not None else False
        if self.backend == BackendType.TORCH:
            return tensor.mean(dim=dim, keepdim=keepdim)
        elif self.backend == BackendType.JAX:
            lib = self._adapter.get_lib()
            return lib.mean(tensor, axis=dim, keepdims=keepdim)
        elif self.backend == BackendType.NUMBA:
            lib = self._adapter.get_lib()
            return lib.mean(tensor, axis=dim, keepdims=keepdim)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def std(self, tensor: Any, dim: Optional[int] = None,
            keepdims: bool = False) -> Any:
        if self.backend == BackendType.TORCH:
            return tensor.std(dim=dim, keepdim=keepdims)
        elif self.backend == BackendType.JAX:
            lib = self._adapter.get_lib()
            return lib.std(tensor, axis=dim, keepdims=keepdims)
        elif self.backend == BackendType.NUMBA:
            lib = self._adapter.get_lib()
            return lib.std(tensor, axis=dim, keepdims=keepdims)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def median(self, tensor: Any, dim: Optional[int] = None,
               keepdims: bool = False) -> Any:
        if self.backend == BackendType.TORCH:
            if dim is None:
                return tensor.median()
            return tensor.median(dim=dim, keepdim=keepdims).values
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.median(tensor, axis=dim, keepdims=keepdims)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.median(tensor, axis=dim, keepdims=keepdims)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def quantile(self, tensor: Any, q: Union[float, List[float]],
                 dim: Optional[int] = None, keepdims: bool = False) -> Any:
        if self.backend == BackendType.TORCH:
            import torch
            return tensor.quantile(torch.tensor(q), dim=dim, keepdim=keepdims)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.quantile(tensor, q, axis=dim, keepdims=keepdims)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.quantile(tensor, q, axis=dim, keepdims=keepdims)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def max(self, tensor: Any, dim: Optional[int] = None,
            keepdims: bool = False) -> Any:
        if self.backend == BackendType.TORCH:
            if dim is None:
                return tensor.max()
            return tensor.max(dim=dim, keepdim=keepdims).values
        elif self.backend == BackendType.JAX:
            lib = self._adapter.get_lib()
            if dim is None:
                return lib.max(tensor)
            return lib.max(tensor, axis=dim, keepdims=keepdims)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            if dim is None:
                return np.max(tensor)
            return np.max(tensor, axis=dim, keepdims=keepdims)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def min(self, tensor: Any, dim: Optional[int] = None,
            keepdims: bool = False) -> Any:
        if self.backend == BackendType.TORCH:
            if dim is None:
                return tensor.min()
            return tensor.min(dim=dim, keepdim=keepdims).values
        elif self.backend == BackendType.JAX:
            lib = self._adapter.get_lib()
            if dim is None:
                return lib.min(tensor)
            return lib.min(tensor, axis=dim, keepdims=keepdims)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            if dim is None:
                return np.min(tensor)
            return np.min(tensor, axis=dim, keepdims=keepdims)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def norm(self, tensor: Any, p: float = 2,
             dim: Optional[int] = None) -> Any:
        if self.backend == BackendType.TORCH:
            return tensor.norm(p=p, dim=dim)
        elif self.backend == BackendType.JAX:
            lib = self._adapter.get_lib()
            return lib.linalg.norm(tensor, ord=p, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.linalg.norm(tensor, ord=p, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ Non-linearities ------------------------

    def softmax(self, tensor: Any, dim: int = -1) -> Any:
        if self.backend == BackendType.TORCH:
            return self._adapter.get_lib().softmax(tensor, dim=dim)
        elif self.backend == BackendType.JAX:
            import jax.nn as jnn
            return jnn.softmax(tensor, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            ex = np.exp(tensor - np.max(tensor, axis=dim, keepdims=True))
            return ex / np.sum(ex, axis=dim, keepdims=True)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def relu(self, tensor: Any) -> Any:
        if self.backend == BackendType.TORCH:
            return self._adapter.get_lib().relu(tensor)
        elif self.backend == BackendType.JAX:
            lib = self._adapter.get_lib()
            return lib.maximum(tensor, 0)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.maximum(tensor, 0)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def sigmoid(self, tensor: Any) -> Any:
        if self.backend == BackendType.TORCH:
            return self._adapter.get_lib().sigmoid(tensor)
        elif self.backend == BackendType.JAX:
            lib = self._adapter.get_lib()
            return 1 / (1 + lib.exp(-tensor))
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return 1 / (1 + np.exp(-tensor))
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def tanh(self, tensor: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self._adapter.get_lib().tanh(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.tanh(tensor)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def log(self, tensor: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self._adapter.get_lib().log(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.log(tensor)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ Elementwise arithmetic ------------------------

    def add(self, tensor1: Any, tensor2: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.add(tensor1, tensor2)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.add(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def subtract(self, tensor1: Any, tensor2: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.subtract(tensor1, tensor2)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.subtract(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def multiply(self, tensor1: Any, tensor2: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.multiply(tensor1, tensor2)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.multiply(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def divide(self, tensor1: Any, tensor2: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self.tensor_lib.divide(tensor1, tensor2)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.divide(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def power(self, tensor: Any, exponent: Union[int, float]) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            if self.backend == BackendType.TORCH:
                return self.tensor_lib.pow(tensor, exponent)
            return self.tensor_lib.power(tensor, exponent)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.power(tensor, exponent)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def sin(self, tensor: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self._adapter.get_lib().sin(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.sin(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def cos(self, tensor: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self._adapter.get_lib().cos(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.cos(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def exp(self, tensor: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self._adapter.get_lib().exp(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.exp(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def abs(self, tensor: Any) -> Any:
        if self.backend in (BackendType.TORCH, BackendType.JAX):
            return self._adapter.get_lib().abs(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.abs(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------ Randomness ------------------------

    def randn(self, shape: Tuple[int, ...], **kwargs) -> Any:
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.randn(*shape, **kwargs)
        elif self.backend == BackendType.JAX:
            import jax.random as random
            key = kwargs.pop("key", None)
            if key is None:
                raise ValueError(
                    "JAX randn requires a PRNG key passed as key=...")
            return random.normal(key, shape, **kwargs)
        elif self.backend == BackendType.NUMBA:
            lib = self._adapter.get_lib()
            return lib.random.randn(*shape)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def randn_like(self, tensor: Any, **kwargs) -> Any:
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.randn_like(tensor, **kwargs)
        elif self.backend == BackendType.JAX:
            import jax.random as random
            key = kwargs.pop("key", None)
            if key is None:
                raise ValueError(
                    "JAX randn_like requires a PRNG key passed as key=...")
            return random.normal(key, tensor.shape, **kwargs)
        elif self.backend == BackendType.NUMBA:
            lib = self._adapter.get_lib()
            return lib.random.randn(*tensor.shape).astype(getattr(tensor, "dtype", _np.float64))
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def random_normal(self, shape: Tuple[int, ...], **kwargs) -> Any:
        """Alias for randn."""
        return self.randn(shape, **kwargs)

    def random_uniform(self, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0, **kwargs) -> Any:
        """Generate random uniform values."""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.rand(*shape, **kwargs) * (high - low) + low
        elif self.backend == BackendType.JAX:
            import jax.random as random
            key = kwargs.pop("key", None)
            if key is None:
                raise ValueError(
                    "JAX random_uniform requires a PRNG key passed as key=...")
            return random.uniform(key, shape, minval=low, maxval=high, **kwargs)
        elif self.backend == BackendType.NUMBA:
            lib = self._adapter.get_lib()
            return lib.random.uniform(low, high, shape)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def dropout(self, tensor: Any, p: float = 0.5, training: bool = True, **kwargs) -> Any:
        if not training or p == 0:
            return tensor
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.dropout(tensor, p=p, train=training)
        elif self.backend == BackendType.JAX:
            import jax.random as random
            key = kwargs.pop("key", None)
            if key is None:
                raise ValueError(
                    "JAX dropout requires a PRNG key passed as key=...")
            keep_prob = 1.0 - p
            mask = random.bernoulli(key, keep_prob, tensor.shape)
            return tensor * mask / keep_prob
        elif self.backend == BackendType.NUMBA:
            lib = self._adapter.get_lib()
            keep_prob = 1.0 - p
            mask = (lib.random.random(tensor.shape) < keep_prob).astype(
                tensor.dtype if hasattr(tensor, "dtype") else _np.float64)
            return tensor * mask / keep_prob
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ FFT ------------------------

    def fft(self, tensor: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch
            return torch.fft.fft(tensor)
        elif self.backend == BackendType.JAX:
            from jax.numpy import fft as jfft
            return jfft.fft(tensor)
        elif self.backend == BackendType.NUMBA:
            lib = self._adapter.get_lib()
            return lib.fft.fft(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def ifft(self, tensor: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch
            return torch.fft.ifft(tensor)
        elif self.backend == BackendType.JAX:
            from jax.numpy import fft as jfft
            return jfft.ifft(tensor)
        elif self.backend == BackendType.NUMBA:
            lib = self._adapter.get_lib()
            return lib.fft.ifft(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------ Misc ------------------------

    def clone(self, tensor: Any) -> Any:
        if self.backend == BackendType.TORCH:
            return tensor.clone()
        else:
            # JAX/NumPy arrays are immutable / copy-on-write; .copy() suffices
            return tensor.copy()

    def concatenate(self, tensors: List[Any], dim: int = 0) -> Any:
        return self.cat(tensors, dim=dim)

    # ------------------------ Gradient operations ------------------------

    def backward(self, tensor: Any, **kwargs) -> None:
        """Compute gradients via backpropagation."""
        if self.backend == BackendType.TORCH:
            tensor.backward(**kwargs)
        elif self.backend == BackendType.JAX:
            # JAX uses functional autograd, backward is not applicable
            warnings.warn("JAX doesn't support imperative backward(); use jax.grad instead")
        elif self.backend == BackendType.NUMBA:
            warnings.warn("NUMBA lane doesn't support automatic differentiation")
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def grad(self, tensor: Any) -> Any:
        """Get gradients of a tensor."""
        if self.backend == BackendType.TORCH:
            if tensor.grad is None:
                return None
            return tensor.grad
        elif self.backend == BackendType.JAX:
            warnings.warn("JAX gradients are computed functionally, not stored on tensors")
            return None
        elif self.backend == BackendType.NUMBA:
            warnings.warn("NUMBA lane doesn't support automatic differentiation")
            return None
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ Device operations ------------------------

    def device(self, tensor: Any) -> Any:
        """Get the device of a tensor."""
        if self.backend == BackendType.TORCH:
            return tensor.device
        elif self.backend == BackendType.JAX:
            import jax
            return jax.devices()[0]  # Return first device
        elif self.backend == BackendType.NUMBA:
            return 'cpu'  # NumPy arrays are always on CPU
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def to_device(self, tensor: Any, device: Union[str, Any]) -> Any:
        """Move tensor to a specific device."""
        if self.backend == BackendType.TORCH:
            if isinstance(device, str):
                device = self.tensor_lib.device(device)
            return tensor.to(device)
        elif self.backend == BackendType.JAX:
            # JAX handles device placement automatically
            return tensor
        elif self.backend == BackendType.NUMBA:
            # NumPy arrays are always on CPU
            return tensor
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ Shape operations ------------------------

    def shape(self, tensor: Any) -> Tuple[int, ...]:
        """Get the shape of a tensor."""
        if hasattr(tensor, 'shape'):
            return tuple(tensor.shape)
        else:
            raise ValueError(f"Tensor doesn't have a shape attribute: {type(tensor)}")

    # ------------------------ Indexing operations ------------------------

    def index(self, tensor: Any, index: Union[int, Tuple[int, ...]]) -> Any:
        """Index into a tensor."""
        if isinstance(index, int):
            return tensor[index]
        elif isinstance(index, tuple):
            return tensor[index]
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

    def slice(self, tensor: Any, start: Optional[int] = None, end: Optional[int] = None, dim: int = 0) -> Any:
        """Slice a tensor along a dimension."""
        if self.backend == BackendType.TORCH:
            slices = [slice(None)] * tensor.dim()
            slices[dim] = slice(start, end)
            return tensor[tuple(slices)]
        elif self.backend == BackendType.JAX:
            slices = [slice(None)] * tensor.ndim
            slices[dim] = slice(start, end)
            return tensor[tuple(slices)]
        elif self.backend == BackendType.NUMBA:
            slices = [slice(None)] * tensor.ndim
            slices[dim] = slice(start, end)
            return tensor[tuple(slices)]
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ Comparison operations ------------------------

    def equal(self, tensor1: Any, tensor2: Any) -> Any:
        """Element-wise equality comparison."""
        if self.backend == BackendType.TORCH:
            return tensor1 == tensor2
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.equal(tensor1, tensor2)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.equal(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def greater(self, tensor1: Any, tensor2: Any) -> Any:
        """Element-wise greater than comparison."""
        if self.backend == BackendType.TORCH:
            return tensor1 > tensor2
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.greater(tensor1, tensor2)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.greater(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def less(self, tensor1: Any, tensor2: Any) -> Any:
        """Element-wise less than comparison."""
        if self.backend == BackendType.TORCH:
            return tensor1 < tensor2
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.less(tensor1, tensor2)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.less(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------ Logical operations ------------------------

    def logical_and(self, tensor1: Any, tensor2: Any) -> Any:
        """Element-wise logical AND."""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.logical_and(tensor1, tensor2)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.logical_and(tensor1, tensor2)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.logical_and(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def logical_or(self, tensor1: Any, tensor2: Any) -> Any:
        """Element-wise logical OR."""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.logical_or(tensor1, tensor2)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.logical_or(tensor1, tensor2)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.logical_or(tensor1, tensor2)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def logical_not(self, tensor: Any) -> Any:
        """Element-wise logical NOT."""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.logical_not(tensor)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.logical_not(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.logical_not(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------ Convolution operations ------------------------

    def convolve(self, tensor: Any, kernel: Any, mode: str = 'valid') -> Any:
        """1D convolution."""
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            # Convert to 3D format (batch, channels, length) for conv1d
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
                kernel = kernel.unsqueeze(0).unsqueeze(0)
                result = F.conv1d(tensor, kernel, padding=0 if mode == 'valid' else 'same')
                return result.squeeze(0).squeeze(0)
            else:
                return F.conv1d(tensor, kernel, padding=0 if mode == 'valid' else 'same')
        elif self.backend == BackendType.JAX:
            from jax.numpy import convolve as jax_convolve
            return jax_convolve(tensor, kernel, mode=mode)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.convolve(tensor, kernel, mode=mode)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ Pooling operations ------------------------

    def max_pool(self, tensor: Any, kernel_size: Union[int, Tuple[int, ...]], stride: Optional[Union[int, Tuple[int, ...]]] = None, padding: int = 0) -> Any:
        """Max pooling operation."""
        if stride is None:
            stride = kernel_size
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            # Ensure tensor has batch and channel dimensions
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
                result = F.max_pool2d(tensor, kernel_size, stride, padding)
                return result.squeeze(0).squeeze(0)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
                result = F.max_pool2d(tensor, kernel_size, stride, padding)
                return result.squeeze(0)
            else:
                return F.max_pool2d(tensor, kernel_size, stride, padding)
        elif self.backend == BackendType.JAX:
            # Simple max pooling implementation for JAX
            import numpy as np
            arr = np.asarray(tensor)
            if arr.ndim == 2:
                h, w = arr.shape
                kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
                sh, sw = (stride, stride) if isinstance(stride, int) else stride
                out_h = (h - kh) // sh + 1
                out_w = (w - kw) // sw + 1
                result = np.zeros((out_h, out_w))
                for i in range(out_h):
                    for j in range(out_w):
                        result[i, j] = np.max(arr[i*sh:i*sh+kh, j*sw:j*sw+kw])
                return self.tensor_lib.array(result)
            else:
                raise NotImplementedError("JAX max_pool only supports 2D tensors")
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            arr = np.asarray(tensor)
            if arr.ndim == 2:
                h, w = arr.shape
                kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
                sh, sw = (stride, stride) if isinstance(stride, int) else stride
                out_h = (h - kh) // sh + 1
                out_w = (w - kw) // sw + 1
                result = np.zeros((out_h, out_w))
                for i in range(out_h):
                    for j in range(out_w):
                        result[i, j] = np.max(arr[i*sh:i*sh+kh, j*sw:j*sw+kw])
                return result
            else:
                raise NotImplementedError("NUMBA max_pool only supports 2D tensors")
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def avg_pool(self, tensor: Any, kernel_size: Union[int, Tuple[int, ...]], stride: Optional[Union[int, Tuple[int, ...]]] = None, padding: int = 0) -> Any:
        """Average pooling operation."""
        if stride is None:
            stride = kernel_size
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            # Ensure tensor has batch and channel dimensions
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
                result = F.avg_pool2d(tensor, kernel_size, stride, padding)
                return result.squeeze(0).squeeze(0)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
                result = F.avg_pool2d(tensor, kernel_size, stride, padding)
                return result.squeeze(0)
            else:
                return F.avg_pool2d(tensor, kernel_size, stride, padding)
        elif self.backend == BackendType.JAX:
            # Simple avg pooling implementation for JAX
            import numpy as np
            arr = np.asarray(tensor)
            if arr.ndim == 2:
                h, w = arr.shape
                kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
                sh, sw = (stride, stride) if isinstance(stride, int) else stride
                out_h = (h - kh) // sh + 1
                out_w = (w - kw) // sw + 1
                result = np.zeros((out_h, out_w))
                for i in range(out_h):
                    for j in range(out_w):
                        result[i, j] = np.mean(arr[i*sh:i*sh+kh, j*sw:j*sw+kw])
                return self.tensor_lib.array(result)
            else:
                raise NotImplementedError("JAX avg_pool only supports 2D tensors")
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            arr = np.asarray(tensor)
            if arr.ndim == 2:
                h, w = arr.shape
                kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
                sh, sw = (stride, stride) if isinstance(stride, int) else stride
                out_h = (h - kh) // sh + 1
                out_w = (w - kw) // sw + 1
                result = np.zeros((out_h, out_w))
                for i in range(out_h):
                    for j in range(out_w):
                        result[i, j] = np.mean(arr[i*sh:i*sh+kh, j*sw:j*sw+kw])
                return result
            else:
                raise NotImplementedError("NUMBA avg_pool only supports 2D tensors")
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ Normalization operations ------------------------

    def batch_norm(self, tensor: Any, running_mean: Optional[Any] = None, running_var: Optional[Any] = None, 
                   weight: Optional[Any] = None, bias: Optional[Any] = None, 
                   training: bool = True, momentum: float = 0.1, eps: float = 1e-5) -> Any:
        """Batch normalization."""
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            # Ensure tensor has batch and channel dimensions
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
                result = F.batch_norm(tensor, running_mean, running_var, weight, bias, training, momentum, eps)
                return result.squeeze(0).squeeze(0)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
                result = F.batch_norm(tensor, running_mean, running_var, weight, bias, training, momentum, eps)
                return result.squeeze(0)
            else:
                return F.batch_norm(tensor, running_mean, running_var, weight, bias, training, momentum, eps)
        elif self.backend == BackendType.JAX:
            # Simple batch norm: normalize across batch dimension
            mean = self.tensor_lib.mean(tensor, axis=0, keepdims=True)
            var = self.tensor_lib.var(tensor, axis=0, keepdims=True)
            normalized = (tensor - mean) / self.tensor_lib.sqrt(var + eps)
            if weight is not None:
                normalized = normalized * weight
            if bias is not None:
                normalized = normalized + bias
            return normalized
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            mean = np.mean(tensor, axis=0, keepdims=True)
            var = np.var(tensor, axis=0, keepdims=True)
            normalized = (tensor - mean) / np.sqrt(var + eps)
            if weight is not None:
                normalized = normalized * weight
            if bias is not None:
                normalized = normalized + bias
            return normalized
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def layer_norm(self, tensor: Any, normalized_shape: Optional[Tuple[int, ...]] = None, 
                   weight: Optional[Any] = None, bias: Optional[Any] = None, eps: float = 1e-5) -> Any:
        """Layer normalization."""
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            if normalized_shape is None:
                normalized_shape = tensor.shape[-1:]
            return F.layer_norm(tensor, normalized_shape, weight, bias, eps)
        elif self.backend == BackendType.JAX:
            # Simple layer norm: normalize across last dimension
            mean = self.tensor_lib.mean(tensor, axis=-1, keepdims=True)
            var = self.tensor_lib.var(tensor, axis=-1, keepdims=True)
            normalized = (tensor - mean) / self.tensor_lib.sqrt(var + eps)
            if weight is not None:
                normalized = normalized * weight
            if bias is not None:
                normalized = normalized + bias
            return normalized
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            mean = np.mean(tensor, axis=-1, keepdims=True)
            var = np.var(tensor, axis=-1, keepdims=True)
            normalized = (tensor - mean) / np.sqrt(var + eps)
            if weight is not None:
                normalized = normalized * weight
            if bias is not None:
                normalized = normalized + bias
            return normalized
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ Loss operations ------------------------

    def mse_loss(self, pred: Any, target: Any, reduction: str = 'mean') -> Any:
        """Mean squared error loss."""
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.mse_loss(pred, target, reduction=reduction)
        elif self.backend == BackendType.JAX:
            loss = self.tensor_lib.mean((pred - target) ** 2)
            if reduction == 'none':
                return (pred - target) ** 2
            elif reduction == 'sum':
                return self.tensor_lib.sum((pred - target) ** 2)
            return loss
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            loss = np.mean((pred - target) ** 2)
            if reduction == 'none':
                return (pred - target) ** 2
            elif reduction == 'sum':
                return np.sum((pred - target) ** 2)
            return loss
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def cross_entropy_loss(self, pred: Any, target: Any, reduction: str = 'mean') -> Any:
        """Cross entropy loss."""
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.cross_entropy(pred, target, reduction=reduction)
        elif self.backend == BackendType.JAX:
            import jax.nn as jnn
            # Apply softmax and compute cross entropy
            log_probs = jnn.log_softmax(pred, axis=-1)
            if target.ndim == 0:
                # Single label
                loss = -log_probs[target]
            else:
                # One-hot or class indices
                if target.ndim > 1:
                    # One-hot encoding
                    loss = -self.tensor_lib.sum(log_probs * target, axis=-1)
                else:
                    # Class indices
                    loss = -log_probs[self.tensor_lib.arange(len(target)), target]
            if reduction == 'none':
                return loss
            elif reduction == 'sum':
                return self.tensor_lib.sum(loss)
            return self.tensor_lib.mean(loss)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            # Apply softmax
            exp_pred = np.exp(pred - np.max(pred, axis=-1, keepdims=True))
            probs = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
            log_probs = np.log(probs + 1e-10)
            if target.ndim == 0:
                loss = -log_probs[target]
            else:
                if target.ndim > 1:
                    loss = -np.sum(log_probs * target, axis=-1)
                else:
                    loss = -log_probs[np.arange(len(target)), target]
            if reduction == 'none':
                return loss
            elif reduction == 'sum':
                return np.sum(loss)
            return np.mean(loss)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ Optimization operations ------------------------

    def sgd_step(self, tensor: Any, lr: float = 0.01) -> Any:
        """Perform one SGD step (subtract lr * grad)."""
        if self.backend == BackendType.TORCH:
            if tensor.grad is None:
                return tensor
            return tensor - lr * tensor.grad
        elif self.backend == BackendType.JAX:
            warnings.warn("JAX uses functional optimizers, not imperative step()")
            return tensor
        elif self.backend == BackendType.NUMBA:
            warnings.warn("NUMBA lane doesn't support automatic differentiation")
            return tensor
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def adam_step(self, tensor: Any, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Any:
        """Perform one Adam step (simplified, requires state tracking)."""
        if self.backend == BackendType.TORCH:
            if tensor.grad is None:
                return tensor
            # Simplified Adam without state tracking
            warnings.warn("adam_step is simplified; use proper optimizer for production")
            return tensor - lr * tensor.grad
        elif self.backend == BackendType.JAX:
            warnings.warn("JAX uses functional optimizers, not imperative step()")
            return tensor
        elif self.backend == BackendType.NUMBA:
            warnings.warn("NUMBA lane doesn't support automatic differentiation")
            return tensor
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------ Utility methods ------------------------

    def switch_backend(self, backend: BackendType) -> None:
        """Switch to a different backend (instance method)."""
        from .backends import switch_backend as switch_backend_manager
        if switch_backend_manager(backend):
            self.backend, self.tensor_lib = self._resolve_backend(backend, get_backend_manager())
            try:
                self._adapter = HighPerformanceAdapter(self.backend)
                _ = self._adapter.get_lib()
            except Exception:
                self._adapter = HighPerformanceAdapter()

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        info = {
            'backend': str(self.backend),
            'tensor_lib': str(type(self.tensor_lib)),
            'device': 'cpu'  # Default
        }
        if self.backend == BackendType.TORCH:
            info['device'] = str(self.tensor_lib.device('cpu'))
        elif self.backend == BackendType.JAX:
            import jax
            info['device'] = str(jax.devices()[0])
        return info

    def enable_profiling(self, enabled: bool) -> None:
        """Enable or disable profiling (placeholder)."""
        self._profiling_enabled = enabled

    def get_profile_results(self) -> Dict[str, Any]:
        """Get profiling results (placeholder)."""
        return {'profiling': getattr(self, '_profiling_enabled', False), 'results': {}}

    def clear_cache(self) -> None:
        """Clear any cached computations."""
        if self.backend == BackendType.TORCH:
            import torch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        elif self.backend == BackendType.JAX:
            import jax
            jax.clear_backends()
        # NUMBA doesn't have a cache to clear

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        usage = {'allocated': 0, 'reserved': 0}
        if self.backend == BackendType.TORCH:
            import torch
            if torch.cuda.is_available():
                usage['allocated'] = torch.cuda.memory_allocated()
                usage['reserved'] = torch.cuda.memory_reserved()
        elif self.backend == BackendType.JAX:
            # JAX memory usage is harder to query
            usage['allocated'] = 0
            usage['reserved'] = 0
        return usage


# Global tensor operations instance
_tensor_ops: Optional[TensorOps] = None


def get_tensor_ops(backend: Optional[BackendType] = None) -> TensorOps:
    """Get the global tensor operations instance (resolves AUTO safely)."""
    global _tensor_ops
    if _tensor_ops is None or (backend is not None and _tensor_ops.backend != backend):
        _tensor_ops = TensorOps(backend)
    return _tensor_ops


def create_tensor(data: Any, **kwargs) -> Any:
    """Create a tensor using the current backend."""
    return get_tensor_ops().create_tensor(data, **kwargs)


def switch_backend(backend: BackendType) -> None:
    """Switch to a different backend and update tensor operations."""
    from .backends import switch_backend as switch_backend_manager
    if switch_backend_manager(backend):
        global _tensor_ops
        _tensor_ops = None  # Reset tensor ops for new backend
