# HPFRACC Library: Methods Section Description

## Overview

HPFRACC (High-Performance Fractional Calculus) is a production-ready Python library (v3.0.1) designed for high-performance fractional calculus operations with seamless machine learning integration, GPU acceleration, and state-of-the-art neural network architectures for stochastic differential equations. The library was developed at the University of Reading, Department of Biomedical Engineering, with a focus on computational physics and biophysics-based fractional-order machine learning applications.

## Architectural Design

### Core Architecture

The library employs a modular, layered architecture organized into several key subsystems:

1. **Core Fractional Calculus Module** (`hpfracc.core`)
   - **Definitions** (`definitions.py`): Implements fundamental fractional calculus definitions including Riemann-Liouville, Caputo, Grünwald-Letnikov, Weyl, Marchaud, Hadamard, and Reiz-Feller formulations
   - **Derivatives** (`derivatives.py`): Provides fractional derivative operations with support for multiple definition types
   - **Integrals** (`integrals.py`): Implements fractional integration operators including RL, Caputo, Weyl, and Hadamard types
   - **Fractional Implementations** (`fractional_implementations.py`): Contains optimized numerical implementations of fractional operators
   - **Utilities** (`utilities.py`): Provides helper functions, numerical utilities, and validation tools
   - **JAX Configuration** (`jax_config.py`): Manages JAX backend configuration for automatic differentiation and GPU acceleration

2. **Solver Module** (`hpfracc.solvers`)
   - **SDE Solvers** (`sde_solvers.py`): Implements fractional stochastic differential equation solvers including Fractional Euler-Maruyama and Fractional Milstein methods with FFT-based history accumulation for O(N log N) complexity
   - **ODE Solvers** (`ode_solvers.py`): Provides fractional ordinary differential equation solvers with adaptive step size control
   - **PDE Solvers** (`pde_solvers.py`): Implements fractional partial differential equation solvers for spatial-temporal problems
   - **Noise Models** (`noise_models.py`): Contains stochastic noise models including Brownian motion, fractional Brownian motion, Lévy noise, and colored noise
   - **Coupled Solvers** (`coupled_solvers.py`): Provides operator splitting and monolithic methods for coupled fractional systems

3. **Machine Learning Module** (`hpfracc.ml`)
   - **Neural Fractional SDE** (`neural_fsde.py`): Implements learnable neural network-based fractional SDEs with adjoint training methods
   - **Graph-SDE Coupling** (`graph_sde_coupling.py`): Provides spatio-temporal dynamics with graph neural networks coupled to SDEs
   - **Probabilistic SDE** (`probabilistic_sde.py`): Implements Bayesian neural fSDEs with NumPyro integration for uncertainty quantification
   - **Neural ODE** (`neural_ode.py`): Learning-based fractional differential equation solvers
   - **Spectral Autograd** (`spectral_autograd.py`): Revolutionary framework for gradient flow through fractional operations
   - **Fractional Autograd** (`fractional_autograd.py`): Automatic differentiation support for fractional operators
   - **Intelligent Backend Selector** (`intelligent_backend_selector.py`): Automatic workload-aware backend optimization
   - **GPU Optimization** (`gpu_optimization.py`): Implements AMP (Automatic Mixed Precision) support, chunked FFT, and performance profiling
   - **Graph Neural Network Layers** (`gnn_layers.py`, `hybrid_gnn_layers.py`): Fractional GCN, GAT, and GraphSAGE implementations
   - **Fractional Layers** (`layers.py`): Neural network layers with fractional derivative components
   - **Loss Functions** (`losses.py`): SDE-specific loss functions including trajectory matching, KL divergence, pathwise loss, and moment matching
   - **Optimizers** (`optimized_optimizers.py`): Fractional Adam and Fractional SGD with fractional momentum
   - **Training** (`training.py`): Training loops and utilities for neural fractional models
   - **Variance-Aware Training** (`variance_aware_training.py`): Adaptive sampling and stochastic seed management
   - **Adjoint Optimization** (`adjoint_optimization.py`): Memory-efficient adjoint methods for backpropagation through SDEs

4. **Backend System**
   - **Multi-Backend Support** (`backends.py`): Seamless integration with NumPy, PyTorch, JAX, and Numba
   - **Intelligent Selection**: Automatic workload-aware backend selection with performance learning
   - **Graceful Fallback**: Automatic CPU fallback if GPU unavailable
   - **Memory-Safe GPU Operations**: Dynamic thresholds to prevent out-of-memory errors

## Key Features and Methodologies

### 1. Fractional Calculus Operations

The library implements seven major fractional derivative definitions:

- **Riemann-Liouville**: $D^{\alpha} f(x) = \frac{1}{\Gamma(n-\alpha)} \frac{d^n}{dx^n} \int_0^x \frac{f(t)}{(x-t)^{\alpha-n+1}} dt$
- **Caputo**: $^C D^{\alpha} f(x) = \frac{1}{\Gamma(n-\alpha)} \int_0^x \frac{f^{(n)}(t)}{(x-t)^{\alpha-n+1}} dt$
- **Grünwald-Letnikov**: $^{GL}D^{\alpha} f(x) = \lim_{h \to 0} h^{-\alpha} \sum_{k=0}^{\infty} (-1)^k \binom{\alpha}{k} f(x-kh)$
- **Weyl**: For functions defined on the entire real line
- **Marchaud**: Alternative regularized formulation
- **Hadamard**: Logarithmic kernel formulation
- **Reiz-Feller**: Symmetric fractional derivative

Special functions include the Mittag-Leffler function $E_{\alpha,\beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)}$, fractional Laplacian, fractional Fourier transform, fractional Z-transform, and fractional Mellin transform.

### 2. Neural Fractional SDE Framework

The library's neural fractional SDE solver (`NeuralFractionalSDE`) implements learnable stochastic differential equations of the form:

$$D^{\alpha} X(t) = f_{\theta}(t, X(t)) dt + g_{\phi}(t, X(t)) dW(t)$$

where:
- $D^{\alpha}$ is the fractional derivative operator (Caputo or Riemann-Liouville)
- $f_{\theta}(t, X(t))$ is a neural network parameterizing the drift function
- $g_{\phi}(t, X(t))$ is a neural network parameterizing the diffusion function
- $dW(t)$ is the Wiener process increment

**Key Components**:
- **Learnable Drift and Diffusion Networks**: Fully connected neural networks with configurable architecture
- **Learnable Fractional Orders**: End-to-end learning of memory effects via learnable $\alpha$ parameter
- **Adjoint Training**: Memory-efficient backpropagation through SDEs using the adjoint sensitivity method
- **Automatic Checkpointing**: Dynamic memory management for long trajectory simulations
- **FFT-Based History Accumulation**: Achieves O(N log N) computational complexity for memory kernel convolutions

**Solver Methods**:
- **Fractional Euler-Maruyama**: First-order strong convergence method with FFT-optimized history tracking
- **Fractional Milstein**: Second-order convergence method for improved accuracy
- **Adaptive Step Size**: Automatic step size selection based on local error estimates

### 3. Stochastic Noise Models

The library provides four noise model types (`hpfracc.solvers.noise_models`):

1. **Brownian Motion**: Standard Wiener process $dW(t) \sim \mathcal{N}(0, dt)$
2. **Fractional Brownian Motion**: Correlated noise with Hurst parameter $H \in (0,1)$
3. **Lévy Noise**: Jump diffusions with stable distributions for heavy-tailed processes
4. **Colored Noise**: Ornstein-Uhlenbeck process for temporally correlated noise

### 4. Graph-SDE Coupling

The `GraphFractionalSDELayer` enables spatio-temporal dynamics by coupling graph neural networks with fractional SDEs:

- **Bidirectional Coupling**: Graph-to-SDE and SDE-to-graph information flow
- **Attention-Based Mechanisms**: Selective information propagation between spatial and temporal components
- **Multi-Scale Systems**: Handle systems at different spatial and temporal scales
- **Fractional Message Passing**: GNN layers with fractional derivative components

### 5. Bayesian Neural fSDEs

The probabilistic module (`probabilistic_sde.py`) implements Bayesian inference for neural fractional SDEs:

- **Variational Inference**: NumPyro-based probabilistic programming for parameter uncertainty
- **Posterior Predictive Sampling**: Generate predictions with uncertainty quantification
- **Prior Specification**: Flexible prior distributions over drift, diffusion, and fractional order parameters
- **MCMC and SVI**: Support for both Markov Chain Monte Carlo and Stochastic Variational Inference

### 6. SDE Loss Functions

The library provides four specialized loss functions for SDE training (`losses.py`):

1. **Trajectory Matching Loss**: Direct MSE on observed vs. predicted trajectories
2. **KL Divergence Loss**: Match probability distributions of observed and predicted paths
3. **Pathwise Loss**: Point-wise comparison along stochastic paths
4. **Moment Matching Loss**: Match statistical moments (mean, variance, higher-order)

### 7. Intelligent Backend Selection

The `IntelligentBackendSelector` automatically optimizes computational performance:

- **Workload Characterization**: Analyzes data size, operation type, and hardware availability
- **Performance Learning**: Adapts over time based on observed execution characteristics
- **Sub-microsecond Overhead**: Selection decision takes < 0.001 ms
- **Memory-Safe GPU Thresholds**: Dynamic allocation prevents OOM errors
- **Multi-GPU Support**: Intelligent distribution across multiple GPUs

**Performance Characteristics**:
- Small data (< 1K): 10-100x speedup with NumPy/Numba
- Medium data (1K-100K): 1.5-3x speedup with optimal backend
- Large data (> 100K): Reliable performance with GPU backends
- Neural networks: 1.2-5x speedup with auto-selected backend

### 8. High-Performance Computing Features

**GPU Acceleration**:
- JAX integration with XLA compilation
- PyTorch native CUDA support with Automatic Mixed Precision (AMP)
- Multi-GPU automatic distribution
- Dynamic memory management and cleanup

**CPU Optimization**:
- Numba JIT compilation for CPU-bound operations
- Multi-threaded execution for parallel operations
- SIMD vectorization for element-wise computations
- FFTW integration for spectral methods

### 9. Spectral Autograd Framework

The `spectral_autograd.py` module provides a revolutionary framework for automatic differentiation through fractional operators:

- **Spectral Domain Differentiation**: Computes gradients in frequency domain for efficiency
- **Chain Rule Support**: Proper gradient flow through composed fractional operations
- **Mixed-Precision Training**: Native support for FP16/BF16 computation
- **Custom Backward Functions**: Optimized backward passes for fractional derivatives

## Implementation Details

### Numerical Accuracy

The library maintains high numerical accuracy across all operations:
- Fractional derivatives: Relative error < 1e-10 for Caputo (α=0.5)
- Fractional integrals: Relative error < 1e-9 for Riemann-Liouville (α=0.3)
- Mittag-Leffler function: Relative error < 1e-8
- FFT operations: Relative error < 1e-12
- SDE solutions: Relative error < 1e-6

### Memory Efficiency

- Small data (< 1K): 1-10 MB, 95% efficiency
- Medium data (1K-100K): 10-100 MB, 90% efficiency
- Large data (> 100K): 100-1000 MB, 85% efficiency
- GPU operations: 500 MB - 8 GB, 80% efficiency
- SDE solving: Adaptive memory with FFT optimization, 75-85% efficiency

### Software Dependencies

**Core Requirements**:
- Python ≥ 3.9
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0

**Machine Learning**:
- PyTorch ≥ 1.12.0
- JAX ≥ 0.4.0, JAXlib ≥ 0.4.0
- Optax ≥ 0.1.0

**Performance**:
- Numba ≥ 0.56.0
- Joblib ≥ 1.1.0

**Probabilistic Programming** (Optional):
- NumPyro ≥ 0.13.0

**GPU Acceleration** (Optional):
- CUDA ≥ 12.1
- CuPy ≥ 10.0.0

### Testing and Validation

The library achieves 100% integration test coverage with comprehensive validation:
- All 20+ core methods fully tested
- Neural fSDE solvers validated against analytical solutions
- GPU acceleration tested across multiple hardware configurations
- Cross-platform compatibility (Windows, Linux, macOS)

## Use Cases

The library is designed for research applications in:

1. **Computational Physics**: Fractional PDEs, viscoelastic materials, anomalous transport, memory effects
2. **Biophysics**: Protein dynamics, membrane transport, drug delivery kinetics, neural network modeling
3. **Machine Learning**: Fractional neural networks, physics-informed ML, graph neural networks, uncertainty quantification
4. **Stochastic Modeling**: Complex stochastic processes with memory, non-Markovian dynamics, financial modeling

## Citation

When using HPFRACC in scientific publications, please cite:

```bibtex
@software{hpfracc2025,
  title={HPFRACC: High-Performance Fractional Calculus Library with Neural Fractional SDE Solvers},
  author={Chin, Davian R.},
  year={2025},
  version={3.0.1},
  doi={10.5281/zenodo.17476041},
  url={https://github.com/dave2k77/hpfracc},
  publisher={Zenodo},
  note={Department of Biomedical Engineering, University of Reading}
}
```

## License

The library is released under the MIT License, ensuring open access for academic and commercial use.
