from abc import ABC, abstractmethod
from typing import Optional, Union, Any
from hpfracc.ml.backends import get_backend_manager, BackendType
from hpfracc.core.definitions import FractionalOrder
from hpfracc.core.fractional_implementations import _AlphaCompatibilityWrapper
from hpfracc.ml.tensor_ops import get_tensor_ops

class BaseFractionalGNNLayer(ABC):
    """
    Base class for fractional GNN layers
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fractional_order: Union[float, FractionalOrder] = 0.5,
        method: str = "RL",
        use_fractional: bool = True,
        activation: str = "relu",
        dropout: float = 0.1,
        bias: bool = True,
        backend: Optional[BackendType] = None
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Alpha handling
        if isinstance(fractional_order, float):
             self.fractional_order = _AlphaCompatibilityWrapper(
                 FractionalOrder(fractional_order))
        elif isinstance(fractional_order, FractionalOrder):
             self.fractional_order = fractional_order
        else:
             self.fractional_order = _AlphaCompatibilityWrapper(fractional_order)
             
        self.method = method
        self.use_fractional = use_fractional
        self.activation = activation
        self.dropout = dropout
        self.bias = bias if bias else None
        
        # Backend handling
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
        
        # Initialize layer-specific parameters (weights, biases, etc.)
        self._initialize_layer()


    @abstractmethod
    def forward(
            self,
            x: Any,
            edge_index: Any,
            edge_weight: Optional[Any] = None,
            **kwargs) -> Any:
        """Forward pass through the layer"""
        pass

    def apply_fractional_derivative(self, x: Any) -> Any:
        """Apply fractional derivative to input features."""
        if not self.use_fractional:
            return x

        alpha = self.fractional_order.alpha
        
        # Delegate to backend-specific implementations that must be provided
        # by subclasses or mixins, or handled via tensor_ops if possible.
        # For now, we'll keep the logic that switches based on backend 
        # but implement it cleanly.
        
        if self.backend == BackendType.TORCH or self.backend == BackendType.AUTO:
            return self._torch_fractional_derivative(x, alpha)
        elif self.backend == BackendType.JAX:
            return self._jax_fractional_derivative(x, alpha)
        elif self.backend == BackendType.NUMBA:
            return self._numba_fractional_derivative(x, alpha)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
            
    # These internal implementations will be provided by backend-specific logic
    # injected or inherited via mixins.
    # They are NOT abstract so that mixin composition works correctly.
    
    def _torch_fractional_derivative(self, x: Any, alpha: float) -> Any:
        raise NotImplementedError("Subclass must provide _torch_fractional_derivative via mixin")
        
    def _jax_fractional_derivative(self, x: Any, alpha: float) -> Any:
        raise NotImplementedError("Subclass must provide _jax_fractional_derivative via mixin")
        
    def _numba_fractional_derivative(self, x: Any, alpha: float) -> Any:
        raise NotImplementedError("Subclass must provide _numba_fractional_derivative via mixin")

    def __call__(self, *args, **kwargs):
        """Callable layer wrapper"""
        return self.forward(*args, **kwargs)
