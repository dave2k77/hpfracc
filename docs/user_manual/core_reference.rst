Core Component Reference
========================

This section provides a deep dive into the core components of HPFRACC, including mathematical operators and neural network layers.

1. Fractional Operators
-----------------------

HPFRACC provides a comprehensive collection of derivatives and integrals. For internal details on methods, see :doc:`../api/derivatives_integrals_api`.

Classical Derivatives
~~~~~~~~~~~~~~~~~~~~~

*   **Caputo**: Ideal for physics problems with well-defined initial conditions.
*   **Riemann-Liouville**: The fundamental operator for most fractional calculus theory.
*   **Grünwald-Letnikov**: A robust discrete approximation for numerical stability.

Advanced Operators
~~~~~~~~~~~~~~~~~~

*   **Fractional Laplacian**: Spectral implementation of the :math:`(-\Delta)^{\alpha/2}` operator.
*   **Caputo-Fabrizio**: Uses a non-singular exponential kernel, perfect for viscoelasticity.
*   **Atangana-Baleanu**: Uses a Mittag-Leffler kernel for modeling complex memory crossover.

Fractional Integrals
~~~~~~~~~~~~~~~~~~~~

HPFRACC supports standard Riemann-Liouville, Weyl, and Hadamard integrals. The **Caputo Integral** in v3.1.0 now supports all orders $\alpha \geq 0$ via a unified decomposition method.

2. Fractional Neural Networks
----------------------------

HPFRACC layers are designed to be seamless drop-in replacements for standard layers, with full Autograd support.

Spectral Autograd Layers
~~~~~~~~~~~~~~~~~~~~~~~~

The `SpectralFractionalDerivative` class implements gradient flow via the frequency domain. It supports three engines:

1.  **FFT Engine**: Optimized for sequences and periodic signals ($O(N \log N)$).
2.  **Mellin Engine**: Designed for scale-invariant and power-law dynamics.
3.  **Laplacian Engine**: Specifically for spatial diffusion in higher dimensions.

Stochastic & Probabilistic Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **Stochastic Fractional Layer**: Uses sampling techniques (Importance, Stratified, Control Variate) to approximate memory history, significantly reducing GPU memory footprint.
*   **Probabilistic Layer**: Treats the fractional order $\alpha$ as a learnable distribution (Normal, Beta, or Uniform), enabling uncertainty quantification.

3. Graph Neural Networks
------------------------

Fractional Graph Neural Networks extend standard GNNs to handle anomalous diffusion on networks.

*   **FractionalGCNLayer**: Standard graph convolution with fractional Laplacian normalization.
*   **FractionalGATLayer**: Attention-based message passing with fractional order scaling.
*   **FractionalGraphSAGELayer**: Inductive learning on graphs with fractional neighbors.

4. Neural Solver Frameworks
---------------------------

Neural fODEs & fSDEs
~~~~~~~~~~~~~~~~~~~~

These frameworks learn the underlying dynamics of differential equations from data.

*   **NeuralFODE**: Learns the drift of a fractional ODE.
*   **NeuralFSDE**: Learns both the drift (:math:`f`) and diffusion (:math:`g`) of a fractional SDE.

Adjoint Method
~~~~~~~~~~~~~~

All solvers support **Adjoint Training**, which allows backpropagation through time without storing the entire trajectory, enabling the training of very long-range memory models on consumer hardware.

.. code-block:: python

   from hpfracc.ml import NeuralFSDE, AdjointOptimizer
   
   # Define a model using the Adjoint solver
   model = NeuralFSDE(adjoint=True)
   
   # Use the specialized AdjointOptimizer for memory efficiency
   optimizer = AdjointOptimizer(model.parameters(), lr=1e-3)
   
   # Loss and backward pass (memory-efficient)
   loss = model.compute_loss(input_data)
   loss.backward()
   optimizer.step()
