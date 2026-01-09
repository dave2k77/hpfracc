Advanced Usage & Optimization
=============================

1. Configuration & Precision
----------------------------

HPFRACC allows for fine-grained control over numerical precision and logging.

.. code-block:: python

   from hpfracc.core.utilities import set_default_precision, setup_logging

   # Set to 64-bit precision for high-stakes research
   set_default_precision(64)

   # Enable detailed logging to a file
   setup_logging(level="DEBUG", log_file="research_run.log")

2. Intelligent Backend Selection
---------------------------------

The library includes an autonomous system that selects the best computational engine based on your hardware and data size.

*   **Learning Mode**: If enabled, the library benchmarks operations on your specific hardware and "remembers" which backend was fastest for a given data size.
*   **Manual Override**: You can force a specific backend if required (e.g., for debugging).

.. code-block:: python

   from hpfracc.ml.backends import BackendManager, BackendType

   # Force JAX for XLA-optimized spectral operations
   BackendManager.set_backend(BackendType.JAX)

3. Known Limitations & Workarounds
-----------------------------------

We maintain high standards of mathematical rigor. Where a feature is not yet fully mature, it is documented here:

*   **Matrix Diffusion in fSDEs**: Full matrix noise is not yet implemented. Use **Scalar** or **Vector** (diagonal) noise as a substitute.
*   **FFT Convolution Axis**: Currently only supported on the first dimension (time axis). Transpose your tensors accordingly.
*   **Adaptive ODE Solving**: This feature is currently in "experimental" status. We recommend using **Fixed-Step Solvers** with a small $h$ for production research.

4. Troubleshooting
------------------

*   **PyTorch/JAX CUDA Conflict**: If you have version mismatches between backends, use the included environment setup scripts:
    .. code-block:: bash
       source scripts/setup_jax_gpu_env.sh
*   **Memory Issues**: For large 3D datasets, use the **Stochastic Layers** to reduce the cache size.
