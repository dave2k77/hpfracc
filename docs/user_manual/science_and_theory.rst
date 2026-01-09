Science & Theory
================

HPFRACC is built on a rigorous mathematical foundation, enabling its use in high-stakes scientific research.

1. Scientific Applications
--------------------------

Computational Physics
~~~~~~~~~~~~~~~~~~~~~

*   **Fractional Diffusion**: Modeling anomalous transport in porous media or plasma physics.
*   **Viscoelasticity**: Simulating materials with both fluid and solid properties using fractional-order stress-strain relations.
*   **Fractional Oscillators**: Analyzing systems with non-local damping effects.

Biophysics
~~~~~~~~~~

*   **Protein Folding**: Modeling the kinetics of conformational changes with memory.
*   **Membrane Transport**: Simulating sub-diffusive movement of ions through cellular membranes.
*   **Pharmacokinetics**: Modeling drug concentration decay with anomalous clearance rates.

2. Mathematical Foundations
---------------------------

Fractional calculus generalizes derivatives and integrals to any real order $\alpha$.

Key Definitions
~~~~~~~~~~~~~~~

*   **Gamma Function**: :math:`\Gamma(z) = \int_0^{\infty} t^{z-1} e^{-t} dt`
*   **Riemann-Liouville Derivative**: 

    .. math::

       D^\alpha_{RL} f(t) = \frac{1}{\Gamma(n-\alpha)} \frac{d^n}{dt^n} \int_0^t (t-\tau)^{n-\alpha-1} f(\tau) d\tau

*   **Caputo Derivative**: 

    .. math::

       D^\alpha_C f(t) = \frac{1}{\Gamma(n-\alpha)} \int_0^t (t-\tau)^{n-\alpha-1} f^{(n)}(\tau) d\tau

For a comprehensive derivation of all implemented operators, please refer to the :doc:`/mathematical_theory` deep dive.

3. Numerical Stability & Accuracy
---------------------------------

HPFRACC implements several strategies to ensure scientific validity:

*   **Error Analysis**: Built-in tools to compare numerical results against analytical solutions for common functions ($t^k$, $\sin(t)$, etc.).
*   **Convergence Monitoring**: Automatic tracking of numerical error relative to step size $h$.
*   **Intelligent Backend Choice**: Backends are selected not just for speed, but for numerical precision (e.g., using JAX for high-precision spectral operations).
