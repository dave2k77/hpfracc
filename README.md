# HPFRACC: High-Performance Fractional Calculus

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/hpfracc.svg)](https://badge.fury.io/py/hpfracc)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17476040.svg)](https://doi.org/10.5281/zenodo.17476040)
[![Documentation](https://readthedocs.org/projects/hpfracc/badge/?version=latest)](https://fractional-calculus-library.readthedocs.io/en/latest/)

**HPFRACC** is the state-of-the-art Python library for high-performance fractional calculus. It combines mathematical rigor with deep learning integration, featuring **intelligent backend selection** (Autotuning between PyTorch, JAX, and Numba) to provide up to 100x speedup over standard implementations.

---

## ✨ Features at a Glance

*   **🚀 High-Performance Engines**: Intelligent selection between PyTorch (GPU), JAX (XLA), and Numba (JIT).
*   **🧠 Neural Fractional SDEs**: Advanced solvers for learning stochastic dynamics with long-range memory.
*   **📉 Spectral Autograd**: A revolutionary framework for differentiating through fractional operators.
*   **📦 Production Ready**: Robust implementations of Riemann-Liouville, Caputo, and Grünwald-Letnikov operators.
*   **🔗 Graph-SDE Coupling**: Modeling spatio-temporal dynamics on complex networks.

---

## ⚡ Quick Start

### Installation

```bash
# Basic installation
pip install hpfracc

# With Machine Learning and GPU support
pip install hpfracc[ml,gpu]
```

### 🧠 A Simple Learnable Fractional Model

```python
import torch
import hpfracc
from hpfracc.ml import SpectralFractionalDerivative, BoundedAlphaParameter

# 1. Create a signal
t = torch.linspace(0, 10, 1000, requires_grad=True)
x = torch.sin(t)

# 2. Define a learnable fractional order (initialized at 0.5)
alpha_param = BoundedAlphaParameter(alpha_init=0.5)
alpha = alpha_param()

# 3. Compute fractional derivative with full Autograd support
# This automatically selects the optimal backend for your hardware
result = SpectralFractionalDerivative.apply(x, alpha)

# 4. Integrate into your loss and optimize
loss = torch.sum(result**2)
loss.backward()

print(f"Learned Alpha Gradient: {alpha_param.rho.grad.item():.6f}")
```

---

## 📚 Documentation & Resources

For detailed tutorials, mathematical theory, and API references, please visit our **[Official Documentation](https://fractional-calculus-library.readthedocs.io/en/latest/)**.

*   **[User Guide](https://fractional-calculus-library.readthedocs.io/en/latest/user_manual/index.html)**: Comprehensive walkthroughs for researchers.
*   **[Theoretical Foundations](https://fractional-calculus-library.readthedocs.io/en/latest/user_manual/science_and_theory.html)**: The math behind the magic.
*   **[API Reference](https://fractional-calculus-library.readthedocs.io/en/latest/api/index.html)**: Detailed class and function documentation.

---

## 🔬 Research & Citation

Developed at the University of Reading, HPFRACC is designed to empower research in computational physics, biophysics, and fractional-order machine learning.

If you use HPFRACC in your research, please cite:

```bibtex
@software{hpfracc2025,
  title={HPFRACC: High-Performance Fractional Calculus Library with Neural Fractional SDE Solvers},
  author={Chin, Davian R.},
  year={2025},
  version={3.1.0},
  doi={10.5281/zenodo.17476041},
  url={https://github.com/dave2k77/hpfracc}
}
```

---

*© 2025 Davian R. Chin | Department of Biomedical Engineering, University of Reading*
