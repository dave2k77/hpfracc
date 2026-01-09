Tutorials & Examples
====================

This section provides practical examples for common workflows in HPFRACC.

1. Basic Calculus Operations
----------------------------

Computing a Fractional Derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most common operation is computing a fractional derivative of a 1D signal.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc import OptimizedCaputo, OptimizedRiemannLiouville

   # Create time grid
   t = np.linspace(0.01, 5, 100)
   f = t**2  # Test function

   # Compute Caputo derivative (order 0.5)
   caputo = OptimizedCaputo(order=0.5)
   result = caputo.compute(f, t)

   plt.plot(t, f, "k-", label="Original t²")
   plt.plot(t, result, "r--", label="Caputo (α=0.5)")
   plt.legend()
   plt.show()

3. High-Performance Machine Learning
------------------------------------

HPFRACC provides a complete ecosystem for fractional research, including optimized optimizers and memory-efficient layers.

Optimized Optimizers
~~~~~~~~~~~~~~~~~~~~

The new **Optimized Optimizers** (Adam, SGD, RMSprop) are designed specifically for fractional-order updates, featuring efficient parameter caching and backend-aware computation.

.. code-block:: python

   from hpfracc.ml import OptimizedFractionalAdam, FractionalNeuralNetwork
   from hpfracc.core.definitions import FractionalOrder

   # Create model
   model = FractionalNeuralNetwork(input_dim=10, hidden_dims=[32], output_dim=1)
   
   # Setup optimizer with fractional support
   optimizer = OptimizedFractionalAdam(
       model.parameters(), 
       lr=1e-3, 
       fractional_order=FractionalOrder(0.8)
   )

Adjoint Memory Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~

For deep fractional networks or fSDE solvers, the **Adjoint Method** reduces memory from $O(T)$ to $O(1)$.

.. code-block:: python

   from hpfracc.ml import AdjointFractionalLayer
   import torch

   # A memory-efficient fractional layer using the Adjoint method
   layer = AdjointFractionalLayer(
       in_features=64, 
       out_features=64, 
       order=0.5,
       checkpoint_steps=10  # Balance speed vs memory
   )
   
   x = torch.randn(32, 64)
   y = layer(x)

Data Caching
~~~~~~~~~~~~

The `FractionalDataset` now supports intelligent **Data Caching**, which stores computed derivatives on disk or in RAM to accelerate subsequent epochs by 10-50x.

.. code-block:: python

   from hpfracc.ml import FractionalDataset

   dataset = FractionalDataset(
       raw_data, 
       fractional_orders=[0.1, 0.5, 0.9],
       use_cache=True, 
       cache_dir="./hpc_cache"
   )
   # The first epoch computes derivatives; subsequent epochs load from cache.

4. Graph Neural Networks
------------------------

Fractional Graph Convolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modeling anomalous diffusion on complex networks using fractional Laplacians.

.. code-block:: python

   import networkx as nx
   from hpfracc.ml.gnn_layers import FractionalGraphConvolution

   G = nx.erdos_renyi_graph(20, 0.3)
   adj = nx.adjacency_matrix(G).toarray()
   features = np.random.randn(20, 16)
   
   layer = FractionalGraphConvolution(input_dim=16, output_dim=8, order=0.5)
   output = layer(adj, features)
   print(f"Graph Features Shape: {output.shape}")
