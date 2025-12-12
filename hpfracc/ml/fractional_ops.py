"""
Backend-agnostic fractional derivative operations for ML modules.
"""
import torch
import numpy as np
from typing import Union, Iterable, Sequence, Tuple, Optional

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from ..core.definitions import FractionalOrder

_Number = Union[int, float]
_Alpha = Union[_Number, torch.Tensor]
_DimType = Union[int, Sequence[int], None]

_COMPLEX_DTYPES = {torch.complex64, torch.complex128}


def _is_complex_dtype(dtype: torch.dtype) -> bool:
    return dtype in _COMPLEX_DTYPES


def _complex_dtype_for(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    return torch.complex64


def _real_dtype_for(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.complex64, torch.float32):
        return torch.float32
    if dtype in (torch.complex128, torch.float64):
        return torch.float64
    return torch.float32


def _normalize_dims(x: torch.Tensor, dim: _DimType) -> Tuple[int, ...]:
    if dim is None:
        return tuple(range(x.ndim))
    if isinstance(dim, Iterable) and not isinstance(dim, (int, torch.Tensor)):
        dims = list(dim)
    else:
        dims = [int(dim)]
    resolved = []
    for axis in dims:
        axis = int(axis)
        if axis < 0:
            axis += x.ndim
        if axis < 0 or axis >= x.ndim:
            raise ValueError(
                f"Invalid dimension {axis} for tensor with {x.ndim} dims")
        resolved.append(axis)
    return tuple(resolved)


def _ensure_alpha_tensor(alpha: _Alpha, reference: torch.Tensor) -> torch.Tensor:
    target_dtype = _real_dtype_for(reference.dtype)
    target_device = reference.device
    if isinstance(alpha, torch.Tensor):
        return alpha.to(device=target_device, dtype=target_dtype)
    return torch.tensor(float(alpha), device=target_device, dtype=target_dtype)


def _validate_alpha(alpha: torch.Tensor) -> None:
    alpha_value = float(alpha.detach().cpu())
    if not (0.0 < alpha_value <= 2.0):
        raise ValueError("Alpha must be in (0, 2]")


def _frequency_grid(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length == 0:
        return torch.zeros(0, dtype=dtype, device=device)
    return torch.fft.fftfreq(length, d=1.0, device=device, dtype=dtype)


def _build_kernel_from_freqs(
    freqs: torch.Tensor,
    alpha: torch.Tensor,
    kernel_type: str,
    epsilon: float,
) -> torch.Tensor:
    if freqs.numel() == 0:
        return torch.zeros_like(freqs)

    freq_abs = freqs.abs().clamp_min(epsilon)
    alpha = alpha.view(1).to(freqs.dtype)

    if kernel_type == "riesz":
        return torch.pow(freq_abs, alpha)
    if kernel_type == "tempered":
        base = freq_abs + epsilon
        return torch.pow(base, alpha)
    if kernel_type == "weyl":
        magnitude = torch.pow(freq_abs, alpha)
        phase = torch.sign(freqs) * (alpha * torch.pi / 2.0)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return torch.complex(real, imag)
    raise ValueError(f"Unsupported kernel type '{kernel_type}'")


def _to_complex(kernel: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    if torch.is_complex(kernel):
        return kernel.to(target_dtype)
    complex_dtype = target_dtype
    if not _is_complex_dtype(complex_dtype):
        complex_dtype = _complex_dtype_for(target_dtype)
    zero_imag = torch.zeros_like(kernel)
    return torch.complex(kernel, zero_imag).to(complex_dtype)


def _reshape_kernel(kernel: torch.Tensor, ndim: int, axis: int) -> torch.Tensor:
    shape = [1] * ndim
    if kernel.numel() == 0:
        return kernel.reshape(shape)
    shape[axis] = kernel.shape[0]
    return kernel.view(shape)


def _get_fractional_kernel(
    alpha: _Alpha,
    n: int,
    kernel_type: str = "riesz",
    epsilon: float = 1e-6,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the 1D spectral kernel for the requested configuration."""

    if n < 0:
        raise ValueError("Kernel size must be non-negative")
    dtype = dtype or torch.float32
    device = device or torch.device("cpu")
    reference = torch.empty(1, device=device, dtype=dtype)
    alpha_tensor = _ensure_alpha_tensor(alpha, reference)
    _validate_alpha(alpha_tensor)

    freq_dtype = dtype if dtype in (
        torch.float32, torch.float64) else _real_dtype_for(dtype)
    freqs = _frequency_grid(n, device=device, dtype=freq_dtype)
    kernel = _build_kernel_from_freqs(
        freqs, alpha_tensor, kernel_type, epsilon)
    if torch.is_complex(kernel):
        target_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        return kernel.to(target_dtype)
    return kernel.to(dtype)


def spectral_derivative_torch(x, alpha, dim=-1, kernel_type="riesz"):
    """PyTorch implementation of spectral derivative."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor for torch backend")

    # Handle empty tensors
    if x.numel() == 0:
        return x.clone()

    # Normalize dim
    if dim < 0:
        dim = x.ndim + dim
    if dim < 0 or dim >= x.ndim:
        raise ValueError(f"Invalid dimension {dim} for tensor with {x.ndim} dims")
    
    # Handle zero-length dimension
    if x.shape[dim] == 0:
        return x.clone()

    fft = torch.fft.fft(x, dim=dim, norm="ortho")
    kernel = _get_fractional_kernel(
        alpha,
        x.shape[dim],
        kernel_type=kernel_type,
        device=x.device,
    )

    transformed = fft * _reshape_kernel(kernel, x.ndim, dim)
    ifft = torch.fft.ifft(transformed, dim=dim, norm="ortho")

    return ifft.real if not torch.is_complex(x) else ifft


if JAX_AVAILABLE:
    from functools import partial

    @partial(jit, static_argnames=['n', 'kernel_type', 'dtype'])
    def _get_fractional_kernel_jax(alpha, n, kernel_type="riesz", dtype=jnp.float32):
        """JAX implementation of _get_fractional_kernel."""
        freqs = jnp.fft.fftfreq(n, d=1.0)

        if kernel_type == "riesz":
            kernel = jnp.power(jnp.abs(freqs), alpha)
        else:
            # Add other kernel types as needed
            raise NotImplementedError(
                f"Kernel type '{kernel_type}' not implemented for JAX.")

        return kernel.astype(dtype)

    @jit
    def spectral_derivative_jax(x, alpha, dim=-1, kernel_type="riesz"):
        """JAX implementation of spectral derivative."""
        if not isinstance(x, jnp.ndarray):
            raise TypeError(
                "Input must be a jax.numpy.ndarray for jax backend")

        fft = jnp.fft.fft(x, axis=dim, norm="ortho")
        kernel = _get_fractional_kernel_jax(
            alpha,
            x.shape[dim],
            kernel_type=kernel_type,
            dtype=x.dtype,
        )

        if not jnp.iscomplexobj(kernel):
            kernel = kernel.astype(jnp.complex64)

        if x.ndim == 1:
            transformed = fft * kernel
        else:
            transformed = fft * \
                kernel.reshape([1] * dim + [-1] + [1] * (x.ndim - dim - 1))

        ifft = jnp.fft.ifft(transformed, axis=dim, norm="ortho")

        return ifft.real if not jnp.iscomplexobj(x) else ifft
