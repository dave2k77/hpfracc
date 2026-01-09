HPFRACC Documentation
=====================

Welcome to the **HPFRACC** (High-Performance Fractional Calculus) documentation!

What is HPFRACC?
----------------

**HPFRACC** is a cutting-edge Python library that provides high-performance implementations of fractional calculus operations with **revolutionary intelligent backend selection**, seamless machine learning integration, and state-of-the-art neural network architectures.

Current Status - PRODUCTION READY (v3.1.0)
-----------------------------------------

*   **Intelligent Backend Selection**: Revolutionary automatic optimization (100% complete)
*   **Spectral Autograd**: Production-ready implementation with FFT/Mellin/Laplacian engines (100% complete)
*   **Neural Fractional SDEs**: Complete framework with adjoint training and stochastic sampling (100% complete)
*   **Performance Benchmarking**: Comprehensive benchmarks showing 10-100x speedups (100% complete)
*   **Status**: ✅ PRODUCTION READY FOR RESEARCH AND INDUSTRY

Getting Started
---------------

If you are new to HPFRACC, we recommend starting with the **User Manual**:

*   :doc:`user_manual/index` - Comprehensive guide covering everything from installation to advanced research.

API Reference
-------------

Deep dive into the technical details of every function and class:

*   :doc:`api/index` - Sectional API documentation organized by functional area.

Practical Introduction Guide
----------------------------

For a comprehensive, printable book-style introduction to the library, we provide a rigorous LaTeX guide:

*   **Source**: `docs/practical_guide/`
*   **Content**: 31 pages covering foundations, ML, and scientific applications.
*   **Usage**: Compile using `pdflatex` to generate the full PDF manual.

Quick Links
-----------

*   **GitHub Repository**: `hpfracc <https://github.com/dave2k77/hpfracc>`_
*   **PyPi Package**: `hpfracc <https://pypi.org/project/hpfracc/>`_
*   **Academic Contact**: `d.r.chin@pgr.reading.ac.uk <mailto:d.r.chin@pgr.reading.ac.uk>`_

Citation
--------

If you use HPFRACC in your research, please cite:

.. code-block:: bibtex

   @software{hpfracc2025,
     title={HPFRACC: High-Performance Fractional Calculus Library with Neural Fractional SDE Solvers},
     author={Chin, Davian R.},
     year={2025},
     version={3.1.0},
     doi={10.5281/zenodo.17476041},
     url={https://github.com/dave2k77/hpfracc},
     publisher={Zenodo},
     note={Department of Biomedical Engineering, University of Reading}
   }

----

**HPFRACC v3.1.0** | © 2025 Davian R. Chin

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   user_manual/index
   api/index

.. toctree::
   :maxdepth: 2
   :caption: Deep Dives:

   deep_dives/unified_autograd_guide
   neural_fsde_guide
   neural_fode_guide
   JAX_GPU_SETUP
   PERFORMANCE_OPTIMIZATION_GUIDE
