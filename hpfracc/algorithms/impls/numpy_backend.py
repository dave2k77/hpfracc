"""
NumPy backend implementations for fractional calculus operations.
"""

import numpy as np
from scipy.signal import convolve
from ...special.gamma_beta import gamma as gamma_func
from ...special.binomial_coeffs import binomial_sequence_fast


def _grunwald_letnikov_numpy(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
    """
    Compute Grünwald-Letnikov derivative using FFT convolution.
    Using centralized binomial coefficients.
    """
    N = len(f)
    # Use centralized, optimized binomial sequence generator
    coeffs = binomial_sequence_fast(alpha, N - 1)
    signs = (-1) ** np.arange(N)
    gl_coeffs = signs * coeffs
    
    # Use FFT convolution for O(N log N)
    # Pad to power of 2 for efficiency
    size = int(2 ** np.ceil(np.log2(2 * N - 1)))
    
    f_padded = np.zeros(size)
    f_padded[:N] = f
    
    coeffs_padded = np.zeros(size)
    coeffs_padded[:N] = gl_coeffs
    
    f_fft = np.fft.fft(f_padded)
    coeffs_fft = np.fft.fft(coeffs_padded)
    
    result = np.fft.ifft(f_fft * coeffs_fft).real[:N]
    return result * (h ** (-alpha))


def _riemann_liouville_numpy(f: np.ndarray, alpha: float, n: int, h: float) -> np.ndarray:
    """
    Compute Riemann-Liouville derivative using Grünwald-Letnikov approximation
    which is equivalent for small h.
    """
    return _grunwald_letnikov_numpy(f, alpha, h)


def _caputo_numpy(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
    """
    Compute Caputo derivative using the L1 scheme (for 0 < alpha < 1) 
    or L2 scheme (for 1 < alpha < 2).
    """
    N = len(f)
    result = np.zeros_like(f)
    
    if 0 < alpha < 1:
        # L1 Scheme: O(h^(2-alpha))
        c = 1.0 / (gamma_func(2 - alpha) * h**alpha)
        
        # Precompute weights
        k = np.arange(N)
        weights = (k + 1)**(1 - alpha) - k**(1 - alpha)
        
        # Compute differences
        df = np.diff(f)
        df = np.insert(df, 0, f[0]) # Handle t=0
        
        # Convolution sum
        size = int(2 ** np.ceil(np.log2(2 * N - 1)))
        
        weights_padded = np.zeros(size)
        weights_padded[:N] = weights
        
        df_padded = np.zeros(size)
        df_padded[:N] = df
        
        w_fft = np.fft.fft(weights_padded)
        df_fft = np.fft.fft(df_padded)
        
        conv = np.fft.ifft(w_fft * df_fft).real[:N]
        result = c * conv
        
    elif 1 < alpha < 2:
        # L2 Scheme: D^alpha f(t) = I^(2-alpha) f''(t)
        beta = 2 - alpha
        # Second derivative approximation
        d2f = np.zeros_like(f)
        d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / h**2
        d2f[0] = (f[2] - 2*f[1] + f[0]) / h**2 
        d2f[-1] = (f[-1] - 2*f[-2] + f[-3]) / h**2
        
        # Using GL for integral part (-beta)
        integral = _grunwald_letnikov_numpy(d2f, -beta, h)
        result = integral
        
    else:
        # Fallback for integer or other orders
        if alpha == 1.0:
            result = np.gradient(f, h, edge_order=2)
        elif alpha == 0.0:
            result = f
        else:
            result = _grunwald_letnikov_numpy(f, alpha, h)
            
    return result
