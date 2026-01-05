from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Optional, Union, Sequence, ContextManager, Dict

class TensorOps(ABC):
    """
    Abstract base class for tensor operations across different backends.
    """
    
    @abstractmethod
    def create_tensor(self, data: Any, **kwargs) -> Any:
        pass

    @abstractmethod
    def shape(self, tensor: Any) -> Any:
        pass
        
    @abstractmethod
    def from_numpy(self, array: Any) -> Any:
        pass
        
    @abstractmethod
    def to_numpy(self, tensor: Any) -> Any:
        pass
        
    @abstractmethod
    def no_grad(self) -> ContextManager:
        pass
        
    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], **kwargs) -> Any:
        pass
        
    @abstractmethod
    def ones(self, shape: Tuple[int, ...], **kwargs) -> Any:
        pass
        
    @abstractmethod
    def eye(self, n: int, **kwargs) -> Any:
        pass
        
    @abstractmethod
    def arange(self, start: int, end: int, step: int = 1, **kwargs) -> Any:
        pass
        
    @abstractmethod
    def linspace(self, start: float, end: float, num: int, **kwargs) -> Any:
        pass
        
    @abstractmethod
    def zeros_like(self, tensor: Any, **kwargs) -> Any:
        pass
        
    @abstractmethod
    def ones_like(self, tensor: Any, **kwargs) -> Any:
        pass
        
    @abstractmethod
    def sqrt(self, tensor: Any) -> Any:
        pass
        
    @abstractmethod
    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        pass
        
    @abstractmethod
    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        pass
        
    @abstractmethod
    def reshape(self, tensor: Any, shape: Tuple[int, ...]) -> Any:
        pass
        
    @abstractmethod
    def repeat(self, tensor: Any, repeats: Union[int, Tuple[int, ...]], dim: int = 0) -> Any:
        pass
        
    @abstractmethod
    def tile(self, tensor: Any, reps: Union[int, Tuple[int, ...]]) -> Any:
        pass
        
    @abstractmethod
    def clip(self, tensor: Any, min_val: float, max_val: float) -> Any:
        pass
        
    @abstractmethod
    def unsqueeze(self, tensor: Any, dim: int) -> Any:
        pass
        
    @abstractmethod
    def expand(self, tensor: Any, *sizes: int) -> Any:
        pass
        
    @abstractmethod
    def gather(self, tensor: Any, dim: int, index: Any) -> Any:
        pass
        
    @abstractmethod
    def squeeze(self, tensor: Any, dim: Optional[int] = None) -> Any:
        pass
        
    @abstractmethod
    def transpose(self, tensor: Any, *args, **kwargs) -> Any:
        pass
        
    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        pass
        
    @abstractmethod
    def inverse(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def mean(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        pass

    @abstractmethod
    def sum(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        pass

    @abstractmethod
    def max(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        pass

    @abstractmethod
    def min(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        pass
        
    def tensor(self, data: Any, **kwargs) -> Any:
        """Alias for create_tensor"""
        return self.create_tensor(data, **kwargs)

    # --- Arithmetic ---
    @abstractmethod
    def add(self, a: Any, b: Any) -> Any: pass
    
    @abstractmethod
    def subtract(self, a: Any, b: Any) -> Any: pass
    
    @abstractmethod
    def multiply(self, a: Any, b: Any) -> Any: pass
    
    @abstractmethod
    def divide(self, a: Any, b: Any) -> Any: pass
    
    @abstractmethod
    def power(self, a: Any, b: Any) -> Any: pass

    # --- Math ---
    @abstractmethod
    def sin(self, tensor: Any) -> Any: pass
    
    @abstractmethod
    def cos(self, tensor: Any) -> Any: pass
    
    @abstractmethod
    def exp(self, tensor: Any) -> Any: pass
    
    @abstractmethod
    def log(self, tensor: Any) -> Any: pass
    
    @abstractmethod
    def abs(self, tensor: Any) -> Any: pass
    
    @abstractmethod
    def sign(self, tensor: Any) -> Any: pass

    # --- Activation ---
    @abstractmethod
    def relu(self, tensor: Any) -> Any: pass
    
    @abstractmethod
    def sigmoid(self, tensor: Any) -> Any: pass
    
    @abstractmethod
    def tanh(self, tensor: Any) -> Any: pass
    
    @abstractmethod
    def softmax(self, tensor: Any, dim: int = -1) -> Any: pass
    
    # --- NN Operations ---
    @abstractmethod
    def dropout(self, tensor: Any, p: float = 0.5, training: bool = True, **kwargs) -> Any: pass
    
    @abstractmethod
    def batch_norm(self, tensor: Any, mean: Any = None, var: Any = None, weight: Any = None, bias: Any = None, training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> Any: pass
    
    @abstractmethod
    def layer_norm(self, tensor: Any, normalized_shape: Any = None, weight: Any = None, bias: Any = None, eps: float = 1e-5) -> Any: pass
    
    @abstractmethod
    def convolve(self, input: Any, weight: Any, bias: Optional[Any] = None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> Any: pass
    
    @abstractmethod
    def max_pool(self, input: Any, kernel_size: int, stride: Optional[int] = None, padding: int = 0) -> Any: pass
    
    @abstractmethod
    def avg_pool(self, input: Any, kernel_size: int, stride: Optional[int] = None, padding: int = 0) -> Any: pass

    # --- Gradient/Autograd ---
    @abstractmethod
    def backward(self, tensor: Any, grad_tensor: Optional[Any] = None) -> Any: pass
    
    @abstractmethod
    def grad(self, tensor: Any) -> Any: pass

    # --- Random ---
    @abstractmethod
    def random_normal(self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, **kwargs) -> Any: pass
    
    @abstractmethod
    def random_uniform(self, shape: Tuple[int, ...], min_val: float = 0.0, max_val: float = 1.0, **kwargs) -> Any: pass

    # --- Loss ---
    @abstractmethod
    def mse_loss(self, input: Any, target: Any, reduction: str = 'mean') -> Any: pass
    
    @abstractmethod
    def cross_entropy_loss(self, input: Any, target: Any, reduction: str = 'mean', **kwargs) -> Any: pass

    # --- Device/Misc ---
    @abstractmethod
    def to_device(self, tensor: Any, device: str) -> Any: pass
    
    @abstractmethod
    def device(self, tensor: Any) -> str: pass

    # --- Logic ---
    @abstractmethod
    def equal(self, a: Any, b: Any) -> Any: pass
    
    @abstractmethod
    def greater(self, a: Any, b: Any) -> Any: pass
    
    @abstractmethod
    def less(self, a: Any, b: Any) -> Any: pass
    
    @abstractmethod
    def logical_and(self, a: Any, b: Any) -> Any: pass
    
    @abstractmethod
    def logical_or(self, a: Any, b: Any) -> Any: pass
    
    @abstractmethod
    def logical_not(self, a: Any) -> Any: pass
    
    # --- FFT ---
    @abstractmethod
    def fft(self, tensor: Any, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Any: pass
    
    @abstractmethod
    def ifft(self, tensor: Any, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Any:
        pass

    # --- Extra Utils for Tests ---
    def concatenate(self, tensors: List[Any], dim: int = 0) -> Any:
        return self.cat(tensors, dim)

    @abstractmethod
    def index(self, tensor: Any, index: Any) -> Any:
        pass

    @abstractmethod
    def slice(self, tensor: Any, start: int, end: int, dim: int = 0) -> Any:
        pass

    @abstractmethod
    def sgd_step(self, tensor: Any, lr: float) -> Any:
        pass

    @abstractmethod
    def adam_step(self, tensor: Any, lr: float) -> Any:
        pass

    def switch_backend(self, backend: Any) -> bool:
         # Default implementation calling global manager
         from hpfracc.ml.backends import get_backend_manager
         return get_backend_manager().switch_backend(backend)

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        pass

    def enable_profiling(self, enable: bool):
        pass

    def get_profile_results(self) -> Dict[str, Any]:
        return {}
    
    def clear_cache(self):
        pass
    
    def get_memory_usage(self) -> Dict[str, Any]:
        return {}
