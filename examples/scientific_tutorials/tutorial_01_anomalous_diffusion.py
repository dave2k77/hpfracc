"""
Tutorial 01: Anomalous Diffusion and Fractional Dynamics
========================================================

This tutorial introduces the concept of anomalous diffusion, a hallmark of 
fractional dynamics in complex systems. We will:

1. Simulate normal diffusion (Brownian Motion)
2. Simulate anomalous diffusion (Fractional Brownian Motion)
3. Analyze the Mean Squared Displacement (MSD) to quantify diffusion type
4. Demonstrate how fractional calculus models these phenomena

References:
- Mandelbrot, B. B., & Van Ness, J. W. (1968). Fractional Brownian motions, fractional noises and applications.
- Metzler, R., & Klafter, J. (2000). The random walk's guide to anomalous diffusion.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from hpfracc.core.definitions import FractionalOrder

def generate_fbm(n_samples, hurst, length=1.0):
    """
    Generate Fractional Brownian Motion (fBm) using spectral synthesis method.
    
    Parameters:
    -----------
    n_samples : int
        Number of points
    hurst : float
        Hurst exponent (0 < H < 1)
        H = 0.5: Normal diffusion (Brownian motion)
        H < 0.5: Sub-diffusion (Anti-correlated)
        H > 0.5: Super-diffusion (Correlated)
    """
    # Frequencies
    freqs = np.fft.rfftfreq(n_samples * 2)
    
    # Spectral density S(f) proportional to 1/f^(2H+1)
    beta = 2 * hurst + 1
    
    # Generate magnitude (ignore f=0 to avoid infinity)
    magnitude = np.zeros_like(freqs)
    magnitude[1:] = freqs[1:] ** (-beta / 2.0)
    
    # Generate random phase
    phase = np.random.uniform(0, 2*np.pi, size=len(freqs))
    
    # Construct Fourier coefficients
    coeffs = magnitude * np.exp(1j * phase)
    
    # Inverse FFT to get time domain signal
    fgn = np.fft.irfft(coeffs)
    
    # fGn (Fractional Gaussian Noise) is the increment process
    # Integrate to get fBm
    fbm = np.cumsum(fgn[:n_samples])
    
    # Normalize to desired scale (optional)
    t = np.linspace(0, length, n_samples)
    fbm = fbm / np.std(fbm) * (t[-1] ** hurst)
    
    return t, fbm

def calculate_msd(trajectory, max_lag=None):
    """
    Calculate Mean Squared Displacement (MSD) for a trajectory.
    MSD(tau) = <(x(t+tau) - x(t))^2>
    """
    if max_lag is None:
        max_lag = len(trajectory) // 4
        
    msd = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        diff = trajectory[lag:] - trajectory[:-lag]
        msd[lag-1] = np.mean(diff ** 2)
        
    return np.arange(1, max_lag + 1), msd

def power_law(x, a, b):
    return a * (x ** b)

def main():
    print("=" * 60)
    print("TUTORIAL 01: ANOMALOUS DIFFUSION")
    print("=" * 60)
    
    # Parameters
    n_samples = 2000
    hurst_values = [0.3, 0.5, 0.7]
    display_names = {0.3: "Sub-diffusion (H=0.3)", 
                     0.5: "Normal Diffusion (H=0.5)", 
                     0.7: "Super-diffusion (H=0.7)"}
    
    plt.figure(figsize=(15, 10))
    
    # 1. Trajectories
    plt.subplot(2, 2, 1)
    trajectories = {}
    
    print("\nGenerating trajectories...")
    for H in hurst_values:
        t, x = generate_fbm(n_samples, H)
        trajectories[H] = x
        plt.plot(t, x, label=f"{display_names[H]}")
        print(f"  Generated {display_names[H]}")
        
    plt.title("Fractional Brownian Motion Trajectories")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    
    # 2. Increments distribution
    plt.subplot(2, 2, 2)
    for H in hurst_values:
        increments = np.diff(trajectories[H])
        plt.hist(increments, bins=50, density=True, alpha=0.5, label=f"H={H}")
        
    plt.title("Increment Distribution (Step Sizes)")
    plt.xlabel("Step Size")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    
    # 3. MSD Analysis
    plt.subplot(2, 2, 3)
    print("\nCalculating Mean Squared Displacement (MSD)...")
    
    for H in hurst_values:
        lags, msd = calculate_msd(trajectories[H])
        
        # Fit power law: MSD ~ t^alpha
        # Ideally alpha approx 2*H
        # Log-log fit is more robust
        log_lags = np.log(lags)
        log_msd = np.log(msd)
        slope, intercept = np.polyfit(log_lags, log_msd, 1)
        estimated_alpha = slope
        
        plt.loglog(lags, msd, label=f"H={H} (α_est={estimated_alpha:.2f})")
        print(f"  H={H}: Estimated MSD exponent α = {estimated_alpha:.2f} (Expected: {2*H:.1f})")
        
        # Plot theoretical slope line (shifted)
        theoretical_msd = np.exp(intercept) * (lags ** (2*H))
        plt.loglog(lags, theoretical_msd, '--', alpha=0.5)
        
    plt.title("Mean Squared Displacement (Log-Log)")
    plt.xlabel("Time Lag (τ)")
    plt.ylabel("MSD(τ)")
    plt.legend()
    plt.grid(True)
    
    # 4. Connection to Fractional Calculus
    plt.subplot(2, 2, 4)
    plt.axis('off')
    text = (
        "Connection to Fractional Calculus:\n\n"
        "1. Anomalous diffusion is governed by the\n"
        "   Fractional Diffusion Equation:\n\n"
        "   ∂P(x,t)/∂t = D_α * ∂^α P(x,t)/∂|x|^α\n\n"
        "2. The Mean Squared Displacement scales as:\n"
        "   <x^2(t)> ~ t^α\n\n"
        "3. Relation to Hurst Exponent:\n"
        "   α = 2H\n\n"
        "4. HPFRACC allows solving these equations\n"
        "   numerically for complex boundary conditions."
    )
    plt.text(0.1, 0.5, text, fontsize=12, va='center')
    plt.title("Theoretical Background")
    
    plt.tight_layout()
    plt.savefig("anomalous_diffusion.png")
    print(f"\n✅ Plot saved to {os.path.abspath('anomalous_diffusion.png')}")
    
    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
