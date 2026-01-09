Unified Fractional Autograd Guide
=================================

Fractional calculus presents a unique challenge for automatic differentiation: **non-locality**. A gradient at point :math:`x` depends on the entire history of the signal, not just the local neighborhood. HPFRACC solves this using two primary frameworks: **Spectral Autograd** and **Stochastic Memory Sampling**.

1. Spectral Autograd Framework
------------------------------

The Spectral Autograd framework maps fractional derivatives into the frequency or transform domain, where non-local convolution becomes local multiplication.

Key Engines
~~~~~~~~~~~

*   **FFT Engine**: Uses the Fast Fourier Transform for :math:`O(N \log N)` computation. Best for long, periodic signals or stationary processes.
*   **Mellin Engine**: Leverages the Mellin transform for scale-invariant dynamics and power-law kernels.
*   **Laplacian Engine**: Specifically designed for spatial fractional diffusion (:math:`(-\Delta)^{\alpha/2}`) in multi-dimensional grids.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from hpfracc.ml import SpectralFractionalDerivative

   x = torch.randn(32, 128, requires_grad=True)
   alpha = 0.5
   
   # Forward pass (automatically selects optimal engine)
   y = SpectralFractionalDerivative.apply(x, alpha)
   
   # Backpropagate through the spectral transform
   y.sum().backward()
   print(f"Gradient norm: {x.grad.norm()}")

2. Stochastic Memory Sampling
-----------------------------

For extremely long trajectories where spectral filters become memory-prohibitive, HPFRACC uses stochastic approximations.

How it Works
~~~~~~~~~~~~

Instead of computing the full convolution sum:

.. math::

   D^\alpha x(t) = \int_0^t K(t-\tau) x(\tau) d\tau

We sample :math:`k` points from the past using importance sampling, significantly reducing the computational and memory cost.

Available Samplers
~~~~~~~~~~~~~~~~~~

*   **Importance Sampler**: Samples points with a probability proportional to the kernel weight :math:`K(t-\tau)`.
*   **Stratified Sampler**: Ensures recent history is sampled more densely than the distant past.
*   **Control Variate Sampler**: Uses a local derivative as a baseline to reduce the variance of the stochastic estimate.

3. Probabilistic Fractional Orders
----------------------------------

In many research scenarios, the exact fractional order :math:`\alpha` is unknown. HPFRACC allows you to treat :math:`\alpha` as a learnable distribution.

.. code-block:: python

   from hpfracc.ml.probabilistic_fractional_orders import NormalAlphaLayer

   # Alpha is learned as a Gaussian distribution N(μ, σ)
   layer = NormalAlphaLayer(mean=0.5, std=0.1)
   x = torch.randn(32, 16)
   output = layer(x)

4. Performance Tips
-------------------

*   **GPU Acceleration**: Both Spectral and Stochastic frameworks are fully compatible with CUDA.
*   **Intelligent Backend**: If using `hpfracc.ml.backends.BackendManager`, the system will automatically choose the fastest implementation (e.g., using JAX for spectral transforms if available).
*   **Memory Efficiency**: Use the `StochasticFractionalLayer` for sequence-based memory or the **Adjoint Method** for training deep fractional ODE/SDE solvers without storing full trajectories.

.. tip::
   For v3.1.0, we recommend the **Adjoint Optimizer** as the default choice for all deep learning tasks in HPFRACC to ensure maximum hardware utilization.
