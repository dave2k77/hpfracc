API Reference
============

This section provides comprehensive documentation for all functions, classes, and methods in the HPFRACC library.

Core Module
----------

Fractional Order Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc
   :members:
   :undoc-members:
   :show-inheritance:

Core Algorithms
~~~~~~~~~~~~~~

.. automodule:: hpfracc.algorithms.optimized_methods
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hpfracc.algorithms.advanced_methods
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hpfracc.algorithms.special_methods
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.core.fractional_implementations
   :members:
   :undoc-members:
   :show-inheritance:

Core Derivatives
~~~~~~~~~~~~~~~

.. automodule:: hpfracc.core.derivatives
   :members:
   :undoc-members:
   :show-inheritance:

Core Integrals
~~~~~~~~~~~~~

.. automodule:: hpfracc.core.integrals
   :members:
   :undoc-members:
   :show-inheritance:

Machine Learning Module
----------------------

Fractional Autograd Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.spectral_autograd
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hpfracc.ml.stochastic_memory_sampling
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hpfracc.ml.probabilistic_fractional_orders
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hpfracc.ml.variance_aware_training
   :members:
   :undoc-members:
   :show-inheritance:

GPU Optimization
~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.gpu_optimization
   :members:
   :undoc-members:
   :show-inheritance:

Backend Management
~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.backends
   :members:
   :undoc-members:
   :show-inheritance:

Tensor Operations
~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.tensor_ops
   :members:
   :undoc-members:
   :show-inheritance:

Core ML Components
~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.core
   :members:
   :undoc-members:
   :show-inheritance:

Neural Network Layers
~~~~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.layers
   :members:
   :undoc-members:
   :show-inheritance:

Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.gnn_layers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hpfracc.ml.gnn_models
   :members:
   :undoc-members:
   :show-inheritance:

Loss Functions
~~~~~~~~~~~~~

.. automodule:: hpfracc.ml.losses
   :members:
   :undoc-members:
   :show-inheritance:

Optimizers
~~~~~~~~~

.. automodule:: hpfracc.ml.optimizers
   :members:
   :undoc-members:
   :show-inheritance:

Detailed API Documentation
-------------------------

Core Definitions
~~~~~~~~~~~~~~~

FractionalOrder
^^^^^^^^^^^^^^

.. autoclass:: hpfracc.FractionalOrder
   :members:
   :undoc-members:
   :special-members: __init__, __str__, __repr__

   .. automethod:: __init__

Core Fractional Calculus Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OptimizedRiemannLiouville
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.OptimizedRiemannLiouville
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

OptimizedCaputo
^^^^^^^^^^^^^^

.. autoclass:: hpfracc.OptimizedCaputo
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

OptimizedGrunwaldLetnikov
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.OptimizedGrunwaldLetnikov
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: compute

RiemannLiouvilleDerivative
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.core.derivatives.RiemannLiouvilleDerivative
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

CaputoDerivative
^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.core.derivatives.CaputoDerivative
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

GrunwaldLetnikovDerivative
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.core.derivatives.GrunwaldLetnikovDerivative
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Backend Management
~~~~~~~~~~~~~~~~~

BackendType
^^^^^^^^^^

.. autoclass:: hpfracc.ml.backends.BackendType
   :members:
   :undoc-members:
   :no-index:

BackendManager
^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.backends.BackendManager
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Tensor Operations
~~~~~~~~~~~~~~~~

TensorOps
^^^^^^^^^

.. autoclass:: hpfracc.ml.tensor_ops.TensorOps
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Neural Networks
~~~~~~~~~~~~~~

FractionalNeuralNetwork
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.core.FractionalNeuralNetwork
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward

Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~

FractionalGCN
^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_models.FractionalGCN
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: get_parameters

FractionalGAT
^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_models.FractionalGAT
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

FractionalGraphSAGE
^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_models.FractionalGraphSAGE
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

FractionalGraphUNet
^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_models.FractionalGraphUNet
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

GNN Factory
^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_models.FractionalGNNFactory
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

GNN Layers
^^^^^^^^^^

FractionalGCNLayer
^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_layers.FractionalGCNLayer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward

FractionalGATLayer
^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_layers.FractionalGATLayer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward

FractionalGraphSAGELayer
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gnn_layers.FractionalGraphSAGELayer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward

Attention Mechanisms
~~~~~~~~~~~~~~~~~~~

FractionalAttention
^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.core.FractionalAttention
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward

Fractional Autograd Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SpectralAutogradEngine
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.spectral_autograd.SpectralAutogradEngine
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: backward

StochasticMemorySampler
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.stochastic_memory_sampling.StochasticMemorySampler
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: sample
   .. automethod:: compute_variance

ProbabilisticFractionalLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.probabilistic_fractional_orders.ProbabilisticFractionalLayer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: sample_alpha

VarianceAwareTrainer
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.variance_aware_training.VarianceAwareTrainer
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: train_step
   .. automethod:: monitor_variance

GPU Optimization
~~~~~~~~~~~~~~~~

GPUProfiler
^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gpu_optimization.GPUProfiler
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: start_profiling
   .. automethod:: stop_profiling

ChunkedFFT
^^^^^^^^^^

.. autoclass:: hpfracc.ml.gpu_optimization.ChunkedFFT
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: backward

AMPFractionalEngine
^^^^^^^^^^^^^^^^^^^

.. autoclass:: hpfracc.ml.gpu_optimization.AMPFractionalEngine
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: backward

Utility Functions
----------------

Fractional Derivative Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hpfracc.core.derivatives.create_fractional_derivative

.. autofunction:: hpfracc.core.derivatives.riemann_liouville

.. autofunction:: hpfracc.core.derivatives.caputo

.. autofunction:: hpfracc.core.derivatives.grunwald_letnikov

Backend Utilities
~~~~~~~~~~~~~~~~

.. autofunction:: hpfracc.ml.backends.get_backend_ops

.. autofunction:: hpfracc.ml.backends.set_default_backend

.. autofunction:: hpfracc.ml.backends.check_backend_compatibility

Tensor Utilities
~~~~~~~~~~~~~~~

.. autofunction:: hpfracc.ml.tensor_ops.create_tensor_ops

.. autofunction:: hpfracc.ml.tensor_ops.convert_tensor

.. autofunction:: hpfracc.ml.tensor_ops.get_tensor_info

Model Utilities
~~~~~~~~~~~~~~

.. note::

   Model creation utilities are available through the main ML module.
   See :py:mod:`hpfracc.ml` for factory functions and model constructors.

Configuration
-------------

Default Parameters
~~~~~~~~~~~~~~~~~

.. data:: hpfracc.core.definitions.DEFAULT_FRACTIONAL_ORDER
   :annotation: = 0.5

.. data:: hpfracc.ml.backends.DEFAULT_BACKEND
   :annotation: = BackendType.JAX

.. data:: hpfracc.ml.tensor_ops.DEFAULT_DTYPE
   :annotation: = 'float32'

Supported Backends
~~~~~~~~~~~~~~~~~

.. data:: hpfracc.ml.backends.SUPPORTED_BACKENDS
   :annotation: = [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]

Supported GNN Types
~~~~~~~~~~~~~~~~~~

.. data:: hpfracc.ml.gnn_models.SUPPORTED_GNN_TYPES
   :annotation: = ['gcn', 'gat', 'sage', 'unet']

Supported Derivative Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: hpfracc.core.derivatives.SUPPORTED_METHODS
   :annotation: = ['RL', 'Caputo', 'GL']

Type Information
---------------

The library uses Python type hints throughout. Key types include:

- **FractionalOrder**: Core class for representing fractional orders
- **BackendType**: Enum for backend selection (TORCH, JAX, NUMBA, AUTO)
- **TensorType**: Union of numpy arrays, PyTorch tensors, and JAX arrays

See the source code for detailed type annotations.

Usage Examples
-------------

Basic Fractional Calculus
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.core.derivatives import create_fractional_derivative
   import numpy as np

   # Create fractional derivative
   alpha = FractionalOrder(0.5)
   deriv = create_fractional_derivative(alpha, method="RL")

   # Test function
   def f(x):
       return np.exp(-x)

   # Compute derivative
   x = np.linspace(0, 1, 100)
   result = deriv(f, x)

Neural Network Usage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml import FractionalNeuralNetwork
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendType
   import numpy as np

   # Create model
   model = FractionalNeuralNetwork(
       input_dim=10,
       hidden_dims=[32, 16],
       output_dim=1,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.JAX
   )

   # Forward pass
   X = np.random.randn(100, 10)
   output = model.forward(X)

GNN Usage
~~~~~~~~~

.. code-block:: python

   from hpfracc.ml import FractionalGNNFactory
   from hpfracc.core.definitions import FractionalOrder
   from hpfracc.ml.backends import BackendType
   import numpy as np

   # Create GNN
   gnn = FractionalGNNFactory.create_model(
       model_type='gcn',
       input_dim=16,
       hidden_dim=32,
       output_dim=4,
       fractional_order=FractionalOrder(0.5),
       backend=BackendType.TORCH
   )

   # Graph data
   node_features = np.random.randn(50, 16)
   edge_index = np.random.randint(0, 50, (2, 100))

   # Forward pass
   output = gnn.forward(node_features, edge_index)

Backend Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.ml.backends import BackendManager, BackendType

   # Check available backends
   available = BackendManager.get_available_backends()
   print(f"Available: {available}")

   # Set backend
   BackendManager.set_backend(BackendType.JAX)

   # Get current backend
   current = BackendManager.get_current_backend()
   print(f"Current: {current}")

Fractional Autograd Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from hpfracc.ml.spectral_autograd import SpectralAutogradEngine
   from hpfracc.ml.stochastic_memory_sampling import StochasticMemorySampler
   from hpfracc.ml.probabilistic_fractional_orders import ProbabilisticFractionalLayer

   # Create spectral autograd engine
   spectral_engine = SpectralAutogradEngine(alpha=0.5, method="mellin")
   
   # Create stochastic memory sampler
   sampler = StochasticMemorySampler(k=32, method="importance")
   
   # Create probabilistic fractional layer
   prob_layer = ProbabilisticFractionalLayer(mean=0.5, std=0.1, learnable=True)
   
   # Forward pass with autograd
   x = torch.randn(100, 10, requires_grad=True)
   result = spectral_engine(x)
   
   # Backward pass
   loss = result.sum()
   loss.backward()
   print(f"Gradients computed: {x.grad is not None}")

GPU Optimization
~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from hpfracc.ml.gpu_optimization import GPUProfiler, ChunkedFFT, gpu_optimization_context

   # GPU profiling
   profiler = GPUProfiler()
   with profiler:
       # Your GPU operations here
       x = torch.randn(1000, 1000, device='cuda')
       result = torch.fft.fft(x)

   # Chunked FFT for large sequences
   chunked_fft = ChunkedFFT(chunk_size=1024)
   x = torch.randn(10000, device='cuda')
   result = chunked_fft.forward(x)

   # GPU optimization context
   with gpu_optimization_context(use_amp=True, chunk_size=512):
       # Your fractional calculus operations here
       pass

Performance Considerations
-------------------------

Backend Selection
~~~~~~~~~~~~~~~~

- **PyTorch**: Best for GPU acceleration and complex neural networks
- **JAX**: Best for functional programming and TPU acceleration
- **NUMBA**: Best for CPU optimization and lightweight deployment

Memory Management
~~~~~~~~~~~~~~~~

- Use batch processing for large datasets
- Clear intermediate tensors when possible
- Monitor memory usage with large models

Computation Optimization
~~~~~~~~~~~~~~~~~~~~~~~

- Choose appropriate fractional derivative method for your use case
- Use JIT compilation when available
- Profile performance with different backends

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Backend not available**
.. code-block:: python

   # Check available backends
   from hpfracc.ml.backends import BackendManager
   available = BackendManager.get_available_backends()
   print(f"Available: {available}")

**Invalid fractional order**
.. code-block:: python

   # Valid orders: -1 < order < 2
   from hpfracc.core.definitions import FractionalOrder
   try:
       order = FractionalOrder(0.5)  # Valid
   except ValueError as e:
       print(f"Error: {e}")

**Tensor shape mismatch**
.. code-block:: python

   # Ensure input dimensions match model expectations
   model = FractionalNeuralNetwork(input_dim=10, ...)
   X = np.random.randn(100, 10)  # Correct shape
   # X = np.random.randn(100, 5)  # Wrong shape - will fail

Debugging Tips
~~~~~~~~~~~~~

1. **Enable debug logging**
2. **Check tensor shapes and types**
3. **Verify backend compatibility**
4. **Test with small datasets first**

For more detailed examples, see :doc:`user_manual/tutorials`.
