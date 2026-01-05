"""
CUDA backend implementations for fractional calculus operations using CuPy.
"""

import warnings
import numpy as np
from ...special.gamma_beta import gamma as gamma_func

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def _check_cupy():
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not available.")


def _riemann_liouville_cuda(f, alpha, n, h):
    """
    Compute Riemann-Liouville derivative using CuPy (Spectral Method).
    """
    _check_cupy()
    
    # Move data to GPU
    f_gpu = cp.asarray(f)
    N = len(f)
    
    # Precompute gamma
    gamma_val = gamma_func(n - alpha)
    
    # RL kernel in time domain
    # Use array generation directly on GPU
    t_gpu = cp.arange(N, dtype=cp.float64) * h
    
    kernel_gpu = cp.where(
        t_gpu > 0.0, 
        (t_gpu ** (n - alpha - 1.0)) / gamma_val, 
        0.0
    )
    
    # Zero-padding
    pad_size = 1 << (N - 1).bit_length()
    if pad_size < 2 * N:
        pad_size = 2 * N
        
    f_pad = cp.pad(f_gpu, (0, pad_size - N))
    kernel_pad = cp.pad(kernel_gpu, (0, pad_size - N))
    
    # FFT Convolution
    Ff = cp.fft.fft(f_pad)
    Fk = cp.fft.fft(kernel_pad)
    Gpad = cp.real(cp.fft.ifft(Ff * Fk))
    
    # Spectral n-th derivative
    freqs = cp.fft.fftfreq(pad_size, d=h) * (2.0 * cp.pi)
    iomegaN = (1j * freqs) ** n
    
    FG = cp.fft.fft(Gpad)
    dGpad = cp.real(cp.fft.ifft(FG * iomegaN))
    
    # Crop and enforce RL convention (first n values 0)
    out = dGpad[:N]
    if n > 0:
        mask = cp.arange(N) < n
        out = cp.where(mask, 0.0, out)
        
    return cp.asnumpy(out)


def _caputo_cuda(f, alpha, h):
    """
    Compute Caputo derivative using CuPy.
    """
    _check_cupy()
    raise NotImplementedError("CuPy implementation for Caputo not yet available.")


def _grunwald_letnikov_cuda(f, alpha, h):
    """
    Compute GL derivative using CuPy (FFT Convolution).
    """
    _check_cupy()
    
    f_gpu = cp.asarray(f)
    N = len(f)
    
    # Binomial coefficients
    coeffs = cp.zeros(N)
    coeffs[0] = 1.0
    
    from ...special.binomial_coeffs import binomial_sequence_fast
    # Compute coeffs on CPU (fast Numba) then move to GPU
    coeffs_cpu = binomial_sequence_fast(alpha, N - 1)
    coeffs_gpu = cp.asarray(coeffs_cpu)
    
    signs = (-1.0) ** cp.arange(N)
    gl_coeffs = signs * coeffs_gpu
    
    # FFT Convolution
    pad_size = 1 << (N - 1).bit_length()
    if pad_size < 2 * N:
        pad_size = 2 * N
        
    f_pad = cp.pad(f_gpu, (0, pad_size - N))
    coeffs_pad = cp.pad(gl_coeffs, (0, pad_size - N))
    
    f_fft = cp.fft.fft(f_pad)
    c_fft = cp.fft.fft(coeffs_pad)
    
    result = cp.real(cp.fft.ifft(f_fft * c_fft))[:N]
    
    return cp.asnumpy(result * (h ** (-alpha)))
