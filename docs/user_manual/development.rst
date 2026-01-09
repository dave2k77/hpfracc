Development & Contribution
==========================

This section is intended for developers contributing to HPFRACC or looking to understand the internal architecture.

1. Reports & Analysis
---------------------

HPFRACC maintains comprehensive internal documentation generated from its CI/CD pipeline:

*   **Assessment Reports**: Deep-dive analysis of individual modules.
*   **Coverage Reports**: Statements on test coverage and code reliability.
*   **Test Reports**: Results from the latest rigorous integration tests.

2. Design Documents
-------------------

For those interested in the underlying math, the `design_documents` folder contains the technical specifications for major features like the **Spectral Autograd Framework** and the **Adjoint Optimizer**.

3. Contributing
---------------

We welcome contributions! Please follow these steps:

1.  **Fork the repo** on GitHub.
2.  **Install dev dependencies**: `pip install -e .[dev]`
3.  **Run the test suite**: Use `pytest` to ensure no regressions were introduced.
4.  **Submit a PR** with a clear description of the changes.

For more detailed guides, see the internal development index.
