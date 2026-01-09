SDE Examples and Tutorials
===========================

This section provides practical examples and tutorials for the Neural Fractional SDE Solvers.

Basic SDE Examples
==================

Simple Stochastic Processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples demonstrate basic fractional SDE solving using the HPFRACC solvers.

Ornstein-Uhlenbeck Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Ornstein-Uhlenbeck process is a mean-reverting stochastic process:

.. code-block:: python

   from hpfracc.solvers import solve_fractional_sde
   import numpy as np
   
   def drift(t, x):
       return 1.5 * (0.5 - x)  # Mean reversion
   
   def diffusion(t, x):
       return 0.3  # Constant volatility
   
   x0 = np.array([0.0])
   solution = solve_fractional_sde(
       drift, diffusion, x0,
       t_span=(0, 5),
       fractional_order=0.5,
       num_steps=200
   )

Geometric Brownian Motion
^^^^^^^^^^^^^^^^^^^^^^^^^

Geometric Brownian Motion with fractional derivatives:

.. code-block:: python

   def drift(t, x):
       return 0.1 * x  # Exponential drift
   
   def diffusion(t, x):
       return 0.2 * x  # Proportional diffusion
   
   x0 = np.array([1.0])
   solution = solve_fractional_sde(
       drift, diffusion, x0,
       t_span=(0, 2),
       fractional_order=0.5,
       method="milstein"
   )

Noise Models
~~~~~~~~~~~~

Different Stochastic Noise Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from hpfracc.solvers import BrownianMotion, FractionalBrownianMotion
   
   # Standard Brownian motion
   bm = BrownianMotion(scale=1.0)
   dw = bm.generate_increment(t=0.0, dt=0.01, size=(100,))
   
   # Fractional Brownian motion (correlated noise)
   fbm = FractionalBrownianMotion(hurst=0.7, scale=1.0)
   dw_fbm = fbm.generate_increment(t=0.0, dt=0.01, size=(100,))

Neural Fractional SDE Examples
===============================

Basic Neural fSDE Training
~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn stochastic dynamics from data:

.. code-block:: python

   from hpfracc.ml.neural_fsde import create_neural_fsde
   import torch
   
   # Create neural fSDE
   model = create_neural_fsde(
       input_dim=2,
       output_dim=2,
       hidden_dim=64,
       fractional_order=0.5,
       noise_type="additive"
   )
   
   # Forward pass
   x0 = torch.randn(32, 2)
   t = torch.linspace(0, 1, 50)
   trajectory = model(x0, t, method="euler_maruyama")

Learnable Fractional Orders
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train the fractional order as a learnable parameter:

.. code-block:: python

   model = create_neural_fsde(
       input_dim=2,
       output_dim=2,
       fractional_order=0.5,
       learn_alpha=True  # Make alpha learnable
   )
   
   # During training, alpha updates automatically
   alpha = model.get_fractional_order()
   print(f"Current fractional order: {alpha}")

Advanced Training
~~~~~~~~~~~~~~~~

Adjoint Methods for Memory-Efficient Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from hpfracc.ml.sde_adjoint_utils import (
       SDEAdjointOptimizer, CheckpointConfig, MixedPrecisionConfig
   )
   
   optimizer = torch.optim.Adam(model.parameters())
   
   sde_optimizer = SDEAdjointOptimizer(
       model, optimizer,
       checkpoint_config=CheckpointConfig(checkpoint_frequency=10),
       mixed_precision_config=MixedPrecisionConfig(enable_amp=True)
   )
   
   # Training loop
   for epoch in range(100):
       loss = loss_fn(model(x0, t), target)
       sde_optimizer.step(loss)

Uncertainty Quantification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Bayesian Neural fSDE with NumPyro:

.. code-block:: python

   from hpfracc.ml.probabilistic_sde import create_bayesian_fsde
   import numpyro.infer as infer
   
   bayesian_model = create_bayesian_fsde(
       input_dim=2,
       output_dim=2,
       fractional_order=0.5
   )
   
   # Variational inference
   svi = infer.SVI(
       bayesian_model.model,
       bayesian_model.create_guide(),
       infer.optim.Adam(step_size=1e-3),
       infer.Trace_ELBO()
   )
   
   # Training
   for epoch in range(1000):
       elbo = svi.step(x0, t, observations)

Graph-SDE Coupling
==================

Spatio-Temporal Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~

Combine graph neural networks with temporal SDE evolution:

.. code-block:: python

   from hpfracc.ml.graph_sde_coupling import GraphFractionalSDELayer
   
   layer = GraphFractionalSDELayer(
       input_dim=10,
       output_dim=10,
       fractional_order=0.5,
       coupling_type="bidirectional"
   )
   
   # Apply to graph
   features = torch.randn(32, 10)  # Node features
   adjacency = torch.ones(32, 32)  # Graph edges
   output = layer(features, adjacency)

Physics and Scientific Applications
====================================

Stochastic Oscillator
~~~~~~~~~~~~~~~~~~~~~

Fractional damped oscillator with noise:

.. code-block:: python

   def oscillator_drift(t, x):
       omega = 2.0  # Natural frequency
       gamma = 0.1  # Damping
       return np.array([x[1], -omega**2 * x[0] - gamma * x[1]])
   
   def oscillator_diffusion(t, x):
       sigma = 0.1  # Noise amplitude
       return np.array([0, sigma])
   
   x0 = np.array([1.0, 0.0])
   solution = solve_fractional_sde(
       oscillator_drift, oscillator_diffusion, x0,
       t_span=(0, 10),
       fractional_order=0.5
   )

Anomalous Diffusion
~~~~~~~~~~~~~~~~~~~

Model subdiffusive and superdiffusive processes:

.. code-block:: python

   # Subdiffusion (alpha < 1)
   solution_sub = solve_fractional_sde(
       drift, diffusion, x0,
       t_span=(0, 5),
       fractional_order=0.3  # Subdiffusion
   )
   
   # Superdiffusion (alpha > 1)
   solution_super = solve_fractional_sde(
       drift, diffusion, x0,
       t_span=(0, 5),
       fractional_order=0.7  # Superdiffusion
   )

See Also
========

* :doc:`neural_fsde_guide` - Comprehensive guide to neural fractional SDEs
* :doc:`sde_api_reference` - Complete API reference
* :doc:`user_manual/tutorials` - More general examples
