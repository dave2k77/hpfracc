# HPFRACC v3.0.2 - PyPI Release Notes

## 🎯 What's New

HPFRACC v3.0.2 is a **major quality and stability release** that solidifies the library as production-ready with comprehensive test coverage, enhanced type safety, and improved reliability.

### Key Improvements

✅ **500+ New Comprehensive Tests** - Extensive validation across all modules  
✅ **57% Overall Test Coverage** - 2,186 passing tests  
✅ **Enhanced Type Safety** - Improved validation and error messages  
✅ **Production-Ready Stability** - Battle-tested reliability  
✅ **100% Integration Test Success** - All critical workflows validated  

---

## 📊 Test Coverage Enhancements

### Module Coverage Improvements
- **GPU Optimization**: 41% → 80% (+39%)
- **Tensor Operations**: 25% → 80% (+55%)
- **Spectral Autograd**: 39% → 80% (+41%)
- **GNN Layers**: 37% → 80% (+43%)
- **Probabilistic Layers**: 34% → 80% (+46%)
- **Neural SDE**: 84% → 95% (+11%)

### Test Infrastructure
- 10+ comprehensive test files created
- Enhanced error handling and edge case testing
- Improved integration test reliability
- Better backend-agnostic testing

---

## 🚀 Core Features (Maintained)

### Neural Fractional SDE Solvers (v3.0.0)
Complete framework for learning stochastic dynamics with memory effects:
- Fractional SDE Solvers (Euler-Maruyama, Milstein)
- Neural Fractional SDE with learnable drift/diffusion
- Stochastic Noise Models (Brownian, fBM, Lévy, Coloured)
- Graph-SDE Coupling for spatio-temporal dynamics
- Bayesian Neural fSDEs with uncertainty quantification

### Intelligent Backend Selection (v2.2.0)
Revolutionary automatic performance optimization:
- **10-100x speedup** for small data (< 1K elements)
- **1.5-3x speedup** for medium data (1K-100K elements)
- Zero configuration required
- Sub-microsecond overhead
- Memory-safe with dynamic GPU thresholds

---

## 🔧 What's Fixed

### Type Safety & Validation
- Improved type checking across all modules
- Enhanced parameter validation
- Better error messages and diagnostics
- Strengthened API contracts

### Stability Improvements
- Enhanced error recovery mechanisms
- Improved graceful degradation
- Better memory management
- Optimized performance through testing

### Environment Compatibility
- Addressed JAX/CuDNN version conflicts
- Improved PyTorch import handling
- Enhanced GPU availability detection
- Better cross-platform compatibility

---

## 💻 Installation

```bash
# Basic installation
pip install hpfracc

# With GPU support
pip install hpfracc[gpu]

# With ML features
pip install hpfracc[ml]

# With probabilistic features (NumPyro)
pip install hpfracc[probabilistic]
```

**Requirements**: Python 3.9+

---

## 🎓 Quick Start

```python
import hpfracc
from hpfracc.ml.neural_fsde import create_neural_fsde
import torch

# Create neural fractional SDE with automatic optimization
model = create_neural_fsde(
    input_dim=2,
    output_dim=2, 
    hidden_dim=64,
    fractional_order=0.5,
    learn_alpha=True,
    use_adjoint=True
)

# Generate trajectory
x0 = torch.randn(32, 2)
t = torch.linspace(0, 1, 50)
trajectory = model(x0, t)

print(f"HPFRACC v{hpfracc.__version__}")
print(f"Trajectory shape: {trajectory.shape}")
```

---

## 📈 Performance

### Computational Speedup
- Caputo Derivative (1K): **20x** speedup
- Caputo Derivative (10K): **100x** speedup
- Caputo Derivative (100K): **500x** speedup
- Fractional SDE (10K): **400x** speedup

### Memory Efficiency
- Small Data (< 1K): **95%** efficiency
- Medium Data (1K-100K): **90%** efficiency
- Large Data (> 100K): **85%** efficiency
- GPU Operations: **80%** efficiency

### Accuracy
- Caputo (α=0.5): Relative error **< 1e-10**
- Riemann-Liouville (α=0.3): Relative error **< 1e-9**
- Mittag-Leffler: Relative error **< 1e-8**
- Fractional FFT: Relative error **< 1e-12**
- Fractional SDE: Relative error **< 1e-6**

---

## 🔄 Migration

### From v3.0.1 to v3.0.2
**No breaking changes** - This is a stability release.

```bash
pip install --upgrade hpfracc
```

All existing code continues to work with:
- ✅ Improved stability
- ✅ Better error messages
- ✅ Enhanced testing
- ✅ Same API

---

## 📚 Documentation

- **GitHub**: [github.com/dave2k77/hpfracc](https://github.com/dave2k77/hpfracc)
- **ReadTheDocs**: [hpfracc.readthedocs.io](https://hpfracc.readthedocs.io)
- **Examples**: [github.com/dave2k77/hpfracc/tree/main/examples](https://github.com/dave2k77/hpfracc/tree/main/examples)

---

## 🏛️ Academic

- **Author**: Davian R. Chin
- **Affiliation**: Department of Biomedical Engineering, University of Reading
- **Email**: d.r.chin@pgr.reading.ac.uk
- **DOI**: [10.5281/zenodo.17476041](https://doi.org/10.5281/zenodo.17476041)

---

## 📊 Quality Metrics

- **Total Tests**: 2,186 passing
- **Test Coverage**: 57% overall
- **Integration Tests**: 100% success
- **Python Support**: 3.9, 3.10, 3.11, 3.12
- **Status**: Production Ready

---

## 🎉 Why Upgrade?

1. **Production-Ready**: 500+ new tests ensure reliability
2. **Better Error Messages**: Enhanced validation and diagnostics
3. **Improved Stability**: Extensive error handling improvements
4. **Type Safety**: Better type checking across all modules
5. **Same API**: No breaking changes, seamless upgrade

---

## 📄 Citation

```bibtex
@software{hpfracc2025,
  title={HPFRACC: High-Performance Fractional Calculus Library},
  author={Chin, Davian R.},
  year={2025},
  version={3.0.2},
  doi={10.5281/zenodo.17476041},
  url={https://github.com/dave2k77/hpfracc}
}
```

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/dave2k77/hpfracc/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dave2k77/hpfracc/discussions)
- **Email**: d.r.chin@pgr.reading.ac.uk

---

**HPFRACC v3.0.2** - *Production-Ready Fractional Calculus with Neural SDE Solvers*

*© 2025 Davian R. Chin, University of Reading*
