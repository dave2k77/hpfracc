"""
Base classes for fractional derivatives.

This module provides abstract base classes and common interfaces
for implementing different fractional derivative definitions.
"""

import numpy as np
from abc import ABC, abstractmethod

# Optional JAX import
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
from typing import Union, Optional, Callable, Dict, Any, List
from .definitions import FractionalOrder, DefinitionType, FractionalDefinition


class BaseFractionalDerivative(ABC):
    """
    Abstract base class for fractional derivatives.

    This class defines the common interface that all fractional
    derivative implementations must follow.
    """

    def __init__(
        self,
        order: Union[float, FractionalOrder],
        definition: Optional[FractionalDefinition] = None,
        use_jax: bool = False,
        use_numba: bool = True,
    ):
        """
        Initialize fractional derivative.

        Args:
            order: Fractional order
            definition: Mathematical definition
            use_jax: Whether to use JAX optimizations
            use_numba: Whether to use NUMBA optimizations
        """
        if isinstance(order, FractionalOrder):
            # Preserve FractionalOrder instance when provided
            self._alpha_order = order
            self._alpha_value = float(order.alpha)
            self.alpha = order
            self.fractional_order = order  # Add for test compatibility
        else:
            alpha_order = FractionalOrder(order)
            self._alpha_order = alpha_order
            self._alpha_value = float(alpha_order.alpha)
            # Expose FractionalOrder to satisfy tests
            self.alpha = alpha_order
            self.fractional_order = alpha_order  # Add for test compatibility
        self.definition = definition
        self.use_jax = use_jax
        self.use_numba = use_numba

        # Validation
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate input parameters."""
        if self._alpha_order.alpha < 0:
            raise ValueError("Fractional order must be non-negative")

        if self.definition is not None and not isinstance(
            self.definition, FractionalDefinition
        ):
            raise TypeError(
                "Definition must be a FractionalDefinition instance")

    @abstractmethod
    def compute(
        self, f: Callable, x: Union[float, np.ndarray, "jnp.ndarray"], **kwargs
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the fractional derivative of function f at point(s) x.

        Args:
            f: Function to differentiate
            x: Point(s) at which to compute the derivative
            **kwargs: Additional parameters

        Returns:
            Fractional derivative value(s)
        """

    @abstractmethod
    def compute_numerical(
        self,
        f_values: Union[np.ndarray, "jnp.ndarray"],
        x_values: Union[np.ndarray, "jnp.ndarray"],
        **kwargs,
    ) -> Union[np.ndarray, "jnp.ndarray"]:
        """
        Compute the fractional derivative numerically from function values.

        Args:
            f_values: Function values at x_values
            x_values: Points where function is evaluated
            **kwargs: Additional parameters

        Returns:
            Fractional derivative values
        """

    def get_definition_info(self) -> Dict[str, Any]:
        """Get information about the mathematical definition."""
        if self.definition is None:
            return {"type": "unknown", "formula": "not specified"}

        return {
            "type": self.definition.definition_type.value,
            "formula": self.definition.get_definition_formula(),
            "properties": self.definition.get_properties(),
        }

    def __repr__(self) -> str:
        definition_type = (
            self.definition.definition_type.value if self.definition else "unknown")
        return f"{definition_type.title()}Derivative(α={self.alpha})"

    def __str__(self) -> str:
        return f"D^{self._alpha_value} (using {self.get_definition_info()['type']})"


class FractionalDerivativeOperator:
    """
    High-level operator for fractional derivatives.

    This class provides a unified interface for different
    fractional derivative definitions and implementations.
    """

    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        definition_type: Union[str, DefinitionType] = DefinitionType.CAPUTO,
        use_jax: bool = False,
        use_numba: bool = True,
    ):
        """
        Initialize fractional derivative operator.

        Args:
            alpha: Fractional order
            definition_type: Type of fractional definition
            use_jax: Whether to use JAX optimizations
            use_numba: Whether to use NUMBA optimizations
        """
        self.alpha = (
            FractionalOrder(alpha) if isinstance(
                alpha, (int, float)) else alpha
        )
        self.definition_type = definition_type
        self.use_jax = use_jax
        self.use_numba = use_numba

        # Create definition
        from .definitions import create_definition

        self.definition = create_definition(definition_type, self.alpha)

        # Initialize implementation (will be set by subclasses)
        self._implementation = None

    def __call__(
        self, f: Callable, x: Union[float, np.ndarray, "jnp.ndarray"], **kwargs
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the fractional derivative.

        Args:
            f: Function to differentiate (can be callable or array)
            x: Point(s) at which to compute the derivative
            **kwargs: Additional parameters

        Returns:
            Fractional derivative value(s)
        """
        # If no implementation is set, try to auto-create a default one
        if self._implementation is None:
            # Auto-create a default implementation using Riemann-Liouville
            from .fractional_implementations import RiemannLiouvilleDerivative
            self._implementation = RiemannLiouvilleDerivative(self.alpha)
        
        # If f is an array (not callable), use compute_numerical instead
        if not callable(f) and isinstance(f, (np.ndarray, list)):
            if isinstance(x, (np.ndarray, list)):
                return self.compute_numerical(np.asarray(f), np.asarray(x), **kwargs)
            else:
                # For scalar x with array f, we need to interpolate or use array x
                raise ValueError("Cannot compute derivative: array f requires array x")
        
        return self._implementation.compute(f, x, **kwargs)

    def compute_numerical(
        self,
        f_values: Union[np.ndarray, "jnp.ndarray"],
        x_values: Union[np.ndarray, "jnp.ndarray"],
        **kwargs,
    ) -> Union[np.ndarray, "jnp.ndarray"]:
        """
        Compute the fractional derivative numerically.

        Args:
            f_values: Function values at x_values
            x_values: Points where function is evaluated
            **kwargs: Additional parameters

        Returns:
            Fractional derivative values
        """
        if self._implementation is None:
            raise NotImplementedError("No implementation available")

        return self._implementation.compute_numerical(
            f_values, x_values, **kwargs)

    def set_implementation(self, implementation: BaseFractionalDerivative):
        """Set the implementation for this operator."""
        self._implementation = implementation

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive information about this operator."""
        return {
            "alpha": self.alpha.alpha,
            "definition_type": (
                self.definition_type.value
                if isinstance(self.definition_type, DefinitionType)
                else self.definition_type
            ),
            "definition_info": self.definition.get_properties(),
            "use_jax": self.use_jax,
            "use_numba": self.use_numba,
            "implementation_available": self._implementation is not None,
        }


class FractionalDerivativeFactory:
    """
    Factory class for creating fractional derivative implementations.

    This class provides a convenient way to create different types
    of fractional derivative implementations.
    """

    def __init__(self):
        """Initialize the factory."""
        self._implementations = {}

    def register_implementation(
        self, definition_type: DefinitionType, implementation_class: type
    ):
        """
        Register an implementation for a specific definition type.

        Args:
            definition_type: Type of fractional definition
            implementation_class: Implementation class
        """
        self._implementations[definition_type] = implementation_class

    def create(
        self,
        definition_type: Union[str, DefinitionType],
        alpha: Union[float, FractionalOrder],
        use_jax: bool = False,
        use_numba: bool = True,
        **kwargs,
    ) -> BaseFractionalDerivative:
        """
        Create a fractional derivative implementation.

        Args:
            definition_type: Type of fractional definition
            alpha: Fractional order
            use_jax: Whether to use JAX optimizations
            use_numba: Whether to use NUMBA optimizations
            **kwargs: Additional parameters for the implementation

        Returns:
            Fractional derivative implementation
        """
        # Handle string keys directly if they exist in the implementations
        if isinstance(definition_type, str):
            # First try to find the string key directly
            if definition_type in self._implementations:
                # Keep the string key as is - don't convert to enum
                pass
            else:
                # Try to convert to DefinitionType enum
                try:
                    definition_type = DefinitionType(definition_type.lower())
                except ValueError:
                    # If conversion fails, check if the string key exists
                    if definition_type not in self._implementations:
                        raise ValueError(
                            f"No implementation registered for {definition_type}")

        if definition_type not in self._implementations:
            raise ValueError(
                f"No implementation registered for {definition_type}")

        implementation_class = self._implementations[definition_type]
        return implementation_class(
            alpha, use_jax=use_jax, use_numba=use_numba, **kwargs
        )

    def get_available_implementations(self) -> List[str]:
        """Get list of available implementation types."""
        result = []
        for impl in self._implementations.keys():
            if hasattr(impl, 'value'):
                result.append(impl.value)
            else:
                result.append(str(impl))
        return result


class FractionalDerivativeChain:
    """
    Chain of fractional derivatives for composition.

    This class allows composing multiple fractional derivatives
    to create higher-order or mixed-order derivatives.
    """

    def __init__(self, derivatives: List[BaseFractionalDerivative]):
        """
        Initialize derivative chain.

        Args:
            derivatives: List of fractional derivatives to chain
        """
        self.derivatives = derivatives
        self._validate_chain()

    def _validate_chain(self):
        """Validate the derivative chain."""
        if not self.derivatives:
            raise ValueError("Derivative chain cannot be empty")

        for derivative in self.derivatives:
            if not isinstance(derivative, BaseFractionalDerivative):
                raise TypeError(
                    "All elements must be BaseFractionalDerivative instances"
                )

    def compute(
        self, f: Callable, x: Union[float, np.ndarray, "jnp.ndarray"], **kwargs
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the chained fractional derivative.

        Args:
            f: Function to differentiate
            x: Point(s) at which to compute the derivative
            **kwargs: Additional parameters

        Returns:
            Chained fractional derivative value(s)
        """
        result = f(x)

        for derivative in self.derivatives:
            # Create a function that represents the current result
            def current_function(x_val):
                return derivative.compute(lambda t: result, x_val, **kwargs)

            result = current_function(x)

        return result

    def get_total_order(self) -> float:
        """Get the total fractional order of the chain."""
        return sum(derivative.alpha.alpha for derivative in self.derivatives)

    def get_chain_info(self) -> List[Dict[str, Any]]:
        """Get information about each derivative in the chain."""
        return [derivative.get_definition_info()
                for derivative in self.derivatives]


class FractionalDerivativeProperties:
    """
    Properties and utilities for fractional derivatives.
    """

    @staticmethod
    def check_linearity(
        derivative: BaseFractionalDerivative,
        f: Callable,
        g: Callable,
        x: Union[float, np.ndarray],
        a: float = 1.0,
        b: float = 1.0,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Check if a fractional derivative satisfies linearity.

        Args:
            derivative: Fractional derivative to test
            f, g: Functions to test
            x: Point(s) to test at
            a, b: Coefficients
            tolerance: Numerical tolerance

        Returns:
            True if linearity is satisfied
        """

        # Compute D^α (af + bg)
        def combined_function(t):
            return a * f(t) + b * g(t)

        left_side = derivative.compute(combined_function, x)

        # Compute a * D^α f + b * D^α g
        right_side = a * \
            derivative.compute(f, x) + b * derivative.compute(g, x)

        return np.allclose(left_side, right_side, atol=tolerance)

    @staticmethod
    def check_semigroup_property(
        derivative_class: type,
        alpha: float,
        beta: float,
        f: Callable,
        x: Union[float, np.ndarray],
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Check if a fractional derivative satisfies the semigroup property.

        Args:
            derivative_class: Class of fractional derivative
            alpha, beta: Fractional orders
            f: Function to test
            x: Point(s) to test at
            tolerance: Numerical tolerance

        Returns:
            True if semigroup property is satisfied
        """
        # Create derivatives
        d_alpha = derivative_class(alpha)
        d_beta = derivative_class(beta)
        d_alpha_beta = derivative_class(alpha + beta)

        # Compute D^α D^β f
        def d_beta_f(t):
            return d_beta.compute(f, t)

        left_side = d_alpha.compute(d_beta_f, x)

        # Compute D^(α+β) f
        right_side = d_alpha_beta.compute(f, x)

        return np.allclose(left_side, right_side, atol=tolerance)

    @staticmethod
    def get_analytical_solutions() -> Dict[str, Callable]:
        """
        Get analytical solutions for common functions.

        Returns:
            Dictionary of analytical solutions
        """
        return {
            "power": lambda x, alpha, n: "Γ(n+1) / Γ(n-α+1) * x^(n-α)",
            "exponential": lambda x, alpha: "x^(-α) / Γ(1-α)",
            "constant": lambda x, alpha: "x^(-α) / Γ(1-α)",
            "linear": lambda x, alpha: "x^(1-α) / Γ(2-α)",
        }


# Global factory instance
derivative_factory = FractionalDerivativeFactory()

# Register implementations
try:
    from hpfracc.core.fractional_implementations import (
        RiemannLiouvilleDerivative,
        CaputoDerivative,
        GrunwaldLetnikovDerivative
    )

    # Register derivative implementations
    derivative_factory.register_implementation(
        DefinitionType.RIEMANN_LIOUVILLE,
        RiemannLiouvilleDerivative
    )
    derivative_factory.register_implementation(
        DefinitionType.CAPUTO,
        CaputoDerivative
    )
    derivative_factory.register_implementation(
        DefinitionType.GRUNWALD_LETNIKOV,
        GrunwaldLetnikovDerivative
    )

    print("Fractional derivative implementations registered successfully!")
except ImportError as e:
    # If the implementations module isn't available yet, skip registration
    print(f"Could not register implementations: {e}")


# Convenience functions
def create_fractional_derivative(
    definition_type: Union[str, DefinitionType],
    alpha: Union[float, FractionalOrder],
    use_jax: bool = False,
    use_numba: bool = True,
    **kwargs,
) -> BaseFractionalDerivative:
    """
    Create a fractional derivative implementation.

    Args:
        definition_type: Type of fractional definition
        alpha: Fractional order
        use_jax: Whether to use JAX optimizations
        use_numba: Whether to use NUMBA optimizations
        **kwargs: Additional parameters

    Returns:
        Fractional derivative implementation
    """
    return derivative_factory.create(
        definition_type, alpha, use_jax, use_numba, **kwargs
    )


def create_derivative_operator(
    definition_type: Union[str, DefinitionType],
    alpha: Union[float, FractionalOrder],
    use_jax: bool = False,
    use_numba: bool = True,
) -> FractionalDerivativeOperator:
    """
    Create a fractional derivative operator.

    Args:
        definition_type: Type of fractional definition
        alpha: Fractional order
        use_jax: Whether to use JAX optimizations
        use_numba: Whether to use NUMBA optimizations

    Returns:
        Fractional derivative operator
    """
    return FractionalDerivativeOperator(
        alpha, definition_type, use_jax, use_numba)


def caputo(f, alpha, **kwargs):
    """Convenience function for Caputo derivative."""
    from hpfracc.algorithms.optimized_methods import OptimizedCaputo
    return OptimizedCaputo(alpha).compute(f, **kwargs)


def riemann_liouville(f, alpha, **kwargs):
    """Convenience function for Riemann-Liouville derivative."""
    from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
    return OptimizedRiemannLiouville(alpha).compute(f, **kwargs)


def grunwald_letnikov(f, alpha, **kwargs):
    """Convenience function for Grünwald-Letnikov derivative."""
    from hpfracc.algorithms.optimized_methods import OptimizedGrunwaldLetnikov
    return OptimizedGrunwaldLetnikov(alpha).compute(f, **kwargs)
