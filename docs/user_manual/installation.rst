Installation & Setup
====================

Basic Installation
------------------

Install the core library via pip:

.. code-block:: bash

   pip install hpfracc

GPU & Machine Learning Support
------------------------------

For high-performance research involving GPU acceleration and ML, we recommend the following setup:

.. code-block:: bash

   # 1. Install PyTorch with CUDA 12.8
   pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
   
   # 2. Install JAX with CUDA support
   pip install --upgrade "jax[cuda12]"
   
   # 3. Install HPFRACC with ML extras
   pip install hpfracc[ml,gpu]

Requirements
------------

*   **Python**: 3.9+ 
*   **Backends**: PyTorch (>=1.12), JAX (>=0.4.0), Numba (>=0.56.0)
*   **GPU**: CUDA-compatible hardware (optional)

Intelligent Backend Selection
-----------------------------

HPFRACC automatically detects your hardware and selects the best backend for each operation. 

*   **PyTorch** is generally selected for large-scale training.
*   **JAX** is used for complex spectral operations.
*   **Numba** provides sub-microsecond latency for small-scale local calculations.

Verification
------------

Verify your installation with this simple check:

.. code-block:: python

   import hpfracc
   from hpfracc.ml.backends import BackendManager
   
   print(f"HPFRACC version: {hpfracc.__version__}")
   print(f"Active Backends: {BackendManager.get_available_backends()}")
