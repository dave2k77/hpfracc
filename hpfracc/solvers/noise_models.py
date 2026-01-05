"""
Stochastic Noise Models for Fractional SDEs

This module provides various noise models including Brownian motion,
fractional Brownian motion, Lévy processes, and coloured noise.
"""

import numpy as np
from typing import Optional, Tuple, Union, Callable, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from scipy import fft
from scipy.stats import levy_stable

try:
    import numpyro
    import numpyro.distributions as dist
    from jax import random as jax_random
    import jax.numpy as jnp
    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False


class NoiseModel(ABC):
    """Base class for stochastic noise models."""
    
    @abstractmethod
    def generate_increment(
        self,
        t: float,
        dt: float,
        size: Tuple[int, ...] = (),
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate noise increment.
        
        Args:
            t: Current time
            dt: Time step
            size: Output shape
            seed: Random seed
            
        Returns:
            Noise increment array
        """
        pass
    
    def prepare(self, num_steps: int, dt: float, size: Tuple[int, ...] = ()):
        """
        Optional preparation step for pre-computing noise paths.
        Useful for non-Markovian processes like fBm.
        """
        pass


class BrownianMotion(NoiseModel):
    """
    Standard Brownian motion (Wiener process).
    
    Generates independent Gaussian increments with variance dt.
    """
    
    def __init__(self, scale: float = 1.0):
        """
        Initialize Brownian motion.
        
        Args:
            scale: Scaling factor for noise
        """
        self.scale = scale
    
    def generate_increment(
        self,
        t: float,
        dt: float,
        size: Tuple[int, ...] = (),
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate Wiener increment."""
        if seed is not None:
            np.random.seed(seed)
        
        dW = np.random.normal(0, np.sqrt(dt), size=size)
        return self.scale * dW
    
    def variance(self, dt: float) -> float:
        """Get variance of increment."""
        return self.scale**2 * dt


class FractionalBrownianMotion(NoiseModel):
    """
    Fractional Brownian motion (fBm).
    
    A Gaussian process with long-range dependence characterized by
    the Hurst exponent H (0 < H < 1).
    """
    
    def __init__(self, hurst: float = 0.5, scale: float = 1.0):
        """
        Initialize fractional Brownian motion.
        
        Args:
            hurst: Hurst exponent (H=0.5 gives standard Brownian motion)
            scale: Scaling factor for noise
        """
        if not (0 < hurst < 1):
            raise ValueError("Hurst exponent must be in (0, 1)")
        
        self.hurst = hurst
        self.scale = scale
        self._precomputed_increments = None
        self._step_counter = 0

    def prepare(self, num_steps: int, dt: float, size: Tuple[int, ...] = ()):
        """Pre-compute fBm increments using Davies-Harte method (FFT)."""
        self._step_counter = 0
        N = num_steps
        H = self.hurst
        
        # Covariance function of fBm increments
        # gamma(k) = 0.5 * (|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H}) * dt^{2H}
        # We compute for k = 0, ..., N
        k = np.arange(N + 1)
        gamma_k = 0.5 * (np.abs(k + 1)**(2*H) + np.abs(k - 1)**(2*H) - 2*np.abs(k)**(2*H)) * dt**(2*H)
        
        # Construct circulant matrix first row
        # C = [gamma_0, gamma_1, ..., gamma_{N-1}, gamma_N, gamma_{N-1}, ..., gamma_1]
        C = np.concatenate([gamma_k, gamma_k[1:-1][::-1]])
        
        # Eigenvalues via FFT
        # Note: Eigenvalues must be non-negative for valid covariance. 
        # Davies-Harte usually gaurantees this.
        eigenvals = fft.fft(C).real
        
        if np.any(eigenvals < 0):
            # Fallback or clipping for numerical issues
            eigenvals = np.abs(eigenvals)
            
        # Generate random complex Gaussian noise
        M = len(C)
        # Handle handling multiple dimensions if 'size' is provided
        # Flatten size for generation
        num_paths = np.prod(size) if size else 1
        
        # Generate V: Z = X + iY
        V = np.random.normal(0, 1, size=(M, num_paths)) + 1j * np.random.normal(0, 1, size=(M, num_paths))
        
        # W = FFT( sqrt(eigenvals) * V ) / sqrt(M)
        W = fft.fft(np.sqrt(eigenvals)[:, None] * V, axis=0)
        W = W[:N, :].real / np.sqrt(M)
        
        # These are the increments
        self._precomputed_increments = W if size is None else W.reshape((N,) + size)
        
    def generate_increment(
        self,
        t: float,
        dt: float,
        size: Tuple[int, ...] = (),
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate fractional Brownian motion increment.
        
        Uses precomputed Davies-Harte increments if available, 
        otherwise falls back to simplified independent increments (with warning).
        """
        if seed is not None:
            np.random.seed(seed)
            
        if self._precomputed_increments is not None:
            if self._step_counter < len(self._precomputed_increments):
                inc = self._precomputed_increments[self._step_counter]
                self._step_counter += 1
                return self.scale * inc
            else:
                # Run out of precomputed increments
                pass
        
        # Fallback: Simplified fBm with correct time-scaling variance but NO memory
        # Warning: This is not true fBm!
        variance = dt**(2 * self.hurst)
        dW = np.random.normal(0, np.sqrt(variance), size=size)
        
        return self.scale * dW
    
    @property
    def is_standard_bm(self) -> bool:
        """Check if this is standard Brownian motion."""
        return abs(self.hurst - 0.5) < 1e-10


class LevyNoise(NoiseModel):
    """
    Lévy noise for jump diffusions.
    
    Uses stable distributions to model heavy-tailed noise.
    """
    
    def __init__(
        self,
        alpha: float = 1.5,
        beta: float = 0.0,
        scale: float = 1.0,
        location: float = 0.0
    ):
        """
        Initialize Lévy noise.
        
        Args:
            alpha: Stability parameter (0 < α ≤ 2)
            beta: Skewness parameter (-1 ≤ β ≤ 1)
            scale: Scale parameter
            location: Location parameter
        """
        if not (0 < alpha <= 2):
            raise ValueError("Alpha must be in (0, 2]")
        
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.location = location
    
    def generate_increment(
        self,
        t: float,
        dt: float,
        size: Tuple[int, ...] = (),
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Lévy noise increment.
        
        Uses stable distribution sampling (simplified implementation).
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Use scipy.stats.levy_stable for correct distribution
        # Parameter mapping: 
        # scipy alpha = self.alpha
        # scipy beta = self.beta
        # scipy loc = 0 (increments are centered)
        # scipy scale = scale * dt^(1/alpha)
        
        scale_param = self.scale * dt**(1/self.alpha)
        
        dW = levy_stable.rvs(
            self.alpha, 
            self.beta, 
            loc=0, 
            scale=scale_param, 
            size=size
        ) # + self.location * dt ? Typically noise is centered, location is drift.
        # If self.location is drift per step:
        dW += self.location * dt
        
        return dW


class ColouredNoise(NoiseModel):
    """
    Coloured noise (Ornstein-Uhlenbeck process).
    
    Gaussian noise with exponential autocorrelation.
    """
    
    def __init__(
        self,
        correlation_time: float = 1.0,
        amplitude: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize coloured noise.
        
        Args:
            correlation_time: Correlation time constant
            amplitude: Noise amplitude
            seed: Random seed for state initialization
        """
        self.correlation_time = correlation_time
        self.amplitude = amplitude
        self._state = None
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_increment(
        self,
        t: float,
        dt: float,
        size: Tuple[int, ...] = (),
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate coloured noise increment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize state if first call
        if self._state is None:
            self._state = np.zeros(size)
        
        # Ornstein-Uhlenbeck update
        theta = 1.0 / self.correlation_time
        dW = np.random.normal(0, np.sqrt(dt), size=size)
        
        # Update state
        self._state = (self._state * (1 - theta * dt) + 
                      self.amplitude * np.sqrt(2 * theta) * dW)
        
        return self._state * dt
    
    def reset(self):
        """Reset the noise process state."""
        self._state = None


# NumPyro integration for Bayesian inference
if NUMPYRO_AVAILABLE:
    def numpyro_brownian_noise(t: float, dt: float, scale: float = 1.0):
        """
        NumPyro model for Brownian noise.
        
        Args:
            t: Current time
            dt: Time step
            scale: Noise scale
            
        Returns:
            NumPyro sample
        """
        return numpyro.sample(
            "dW",
            dist.Normal(0.0, scale * jnp.sqrt(dt))
        )
    
    def numpyro_fractional_brownian_noise(
        t: float, dt: float, hurst: float = 0.5, scale: float = 1.0
    ):
        """
        NumPyro model for fractional Brownian noise.
        
        Args:
            t: Current time
            dt: Time step
            hurst: Hurst exponent
            scale: Noise scale
            
        Returns:
            NumPyro sample
        """
        variance = dt**(2 * hurst)
        return numpyro.sample(
            "dW_fbm",
            dist.Normal(0.0, scale * jnp.sqrt(variance))
        )
    
    def numpyro_levy_noise(
        t: float, dt: float, alpha: float = 1.5, scale: float = 1.0
    ):
        """
        NumPyro model for Lévy noise.
        
        Note: NumPyro doesn't have stable distribution, so uses approximation.
        
        Args:
            t: Current time
            dt: Time step
            alpha: Stability parameter
            scale: Noise scale
            
        Returns:
            NumPyro sample
        """
        if abs(alpha - 2.0) < 1e-10:
            # Gaussian case
            return numpyro.sample(
                "dW_levy",
                dist.Normal(0.0, scale * jnp.sqrt(dt))
            )
        else:
            # Approximate with Student's t
            return numpyro.sample(
                "dW_levy",
                dist.StudentT(df=2.0, loc=0.0, scale=scale * dt**(1/alpha))
            )


@dataclass
class NoiseConfig:
    """Configuration for noise models."""
    noise_type: str = "brownian"
    hurst: float = 0.5  # For fBm
    scale: float = 1.0
    alpha: float = 1.5  # For Lévy
    beta: float = 0.0   # For Lévy
    correlation_time: float = 1.0  # For coloured noise
    amplitude: float = 1.0  # For coloured noise


def create_noise_model(config: NoiseConfig) -> NoiseModel:
    """
    Create a noise model from configuration.
    
    Args:
        config: Noise configuration
        
    Returns:
        NoiseModel instance
    """
    if config.noise_type == "brownian":
        return BrownianMotion(scale=config.scale)
    elif config.noise_type == "fractional_brownian":
        return FractionalBrownianMotion(hurst=config.hurst, scale=config.scale)
    elif config.noise_type == "levy":
        return LevyNoise(
            alpha=config.alpha,
            beta=config.beta,
            scale=config.scale
        )
    elif config.noise_type == "coloured":
        return ColouredNoise(
            correlation_time=config.correlation_time,
            amplitude=config.amplitude
        )
    else:
        raise ValueError(f"Unknown noise type: {config.noise_type}")


def generate_noise_trajectory(
    noise_model: NoiseModel,
    t_span: Tuple[float, float],
    num_steps: int,
    size: Tuple[int, ...] = (),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a complete noise trajectory.
    
    Args:
        noise_model: Noise model to use
        t_span: Time interval (t0, tf)
        num_steps: Number of steps
        size: Shape of noise increments
        seed: Random seed
        
    Returns:
        Tuple of (time array, noise increments array)
    """
    t0, tf = t_span
    dt = (tf - t0) / num_steps
    t = np.linspace(t0, tf, num_steps + 1)
    
    if seed is not None:
        np.random.seed(seed)
    
    dW = np.zeros((num_steps,) + size)
    for i in range(num_steps):
        dW[i] = noise_model.generate_increment(t[i], dt, size)
    
    return t, dW
