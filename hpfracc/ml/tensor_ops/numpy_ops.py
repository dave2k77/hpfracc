from .base import TensorOps
from typing import Any, Tuple, List, Optional, Union, ContextManager, Dict
import numpy as np
from contextlib import nullcontext
from hpfracc.ml.backends import BackendType

class NumpyTensorOps(TensorOps):
    def __init__(self):
        self._lib = np
        self.tensor_lib = np
        self.backend = BackendType.NUMBA

    def create_tensor(self, data: Any, **kwargs) -> Any:
        return np.array(data, **kwargs)

    def shape(self, tensor: Any) -> Any:
        return tensor.shape

    def from_numpy(self, array: Any) -> Any:
        return np.array(array)

    def to_numpy(self, tensor: Any) -> Any:
        return tensor

    def no_grad(self) -> ContextManager:
        return nullcontext()

    def zeros(self, shape: Tuple[int, ...], **kwargs) -> Any:
        return np.zeros(shape, **kwargs)

    def ones(self, shape: Tuple[int, ...], **kwargs) -> Any:
        return np.ones(shape, **kwargs)

    def eye(self, n: int, **kwargs) -> Any:
        return np.eye(n, **kwargs)

    def arange(self, start: int, end: int, step: int = 1, **kwargs) -> Any:
        return np.arange(start, end, step, **kwargs)

    def linspace(self, start: float, end: float, num: int, **kwargs) -> Any:
        return np.linspace(start, end, num, **kwargs)

    def zeros_like(self, tensor: Any, **kwargs) -> Any:
        return np.zeros_like(tensor, **kwargs)

    def ones_like(self, tensor: Any, **kwargs) -> Any:
        return np.ones_like(tensor, **kwargs)

    def sqrt(self, tensor: Any) -> Any:
        return np.sqrt(tensor)

    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        return np.stack(tensors, axis=dim)

    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        return np.concatenate(tensors, axis=dim)

    def reshape(self, tensor: Any, shape: Tuple[int, ...]) -> Any:
        return np.reshape(tensor, shape)

    def repeat(self, tensor: Any, repeats: Union[int, Tuple[int, ...]], dim: int = 0) -> Any:
        return np.repeat(tensor, repeats, axis=dim)

    def tile(self, tensor: Any, reps: Union[int, Tuple[int, ...]]) -> Any:
        return np.tile(tensor, reps)

    def clip(self, tensor: Any, min_val: float, max_val: float) -> Any:
        return np.clip(tensor, min_val, max_val)

    def unsqueeze(self, tensor: Any, dim: int) -> Any:
        return np.expand_dims(tensor, axis=dim)

    def expand(self, tensor: Any, *sizes: int) -> Any:
        return np.broadcast_to(tensor, sizes)

    def gather(self, tensor: Any, dim: int, index: Any) -> Any:
        return np.take_along_axis(tensor, index, axis=dim)

    def squeeze(self, tensor: Any, dim: Optional[int] = None) -> Any:
        return np.squeeze(tensor, axis=dim)

    def transpose(self, tensor: Any, *args, **kwargs) -> Any:
        if 'dims' in kwargs:
             return np.transpose(tensor, axes=kwargs['dims'])
        return np.transpose(tensor, axes=args if args else None)

    def matmul(self, a: Any, b: Any) -> Any:
        return np.matmul(a, b)

    def inverse(self, tensor: Any) -> Any:
        return np.linalg.inv(tensor)

    def mean(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        return np.mean(tensor, axis=dim, keepdims=keepdims)

    def sum(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        return np.sum(tensor, axis=dim, keepdims=keepdims)

    def max(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        return np.max(tensor, axis=dim, keepdims=keepdims)

    def min(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        return np.min(tensor, axis=dim, keepdims=keepdims)
        
    def relu(self, tensor: Any) -> Any:
        return np.maximum(tensor, 0)
    
    def sigmoid(self, tensor: Any) -> Any:
        return 1 / (1 + np.exp(-tensor))
    
    def tanh(self, tensor: Any) -> Any:
        return np.tanh(tensor)
    
    def log(self, tensor: Any) -> Any:
        return np.log(tensor)
        
    def softmax(self, tensor: Any, dim: int = -1) -> Any:
        e_x = np.exp(tensor - np.max(tensor, axis=dim, keepdims=True))
        return e_x / e_x.sum(axis=dim, keepdims=True)

    def einsum(self, equation: str, *operands: Any) -> Any:
        return np.einsum(equation, *operands)

    def std(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        return np.std(tensor, axis=dim, keepdims=keepdims)

    def var(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        return np.var(tensor, axis=dim, keepdims=keepdims)

    def dropout(self, tensor: Any, p: float = 0.5, training: bool = True, **kwargs) -> Any:
        # Numpy backend usually for inference or simple ops, so dropout is identity
        return tensor

    # --- Arithmetic ---
    def add(self, a: Any, b: Any) -> Any:
        return np.add(a, b)
    
    def subtract(self, a: Any, b: Any) -> Any:
        return np.subtract(a, b)
    
    def multiply(self, a: Any, b: Any) -> Any:
        return np.multiply(a, b)
    
    def divide(self, a: Any, b: Any) -> Any:
        return np.divide(a, b)
    
    def power(self, a: Any, b: Any) -> Any:
        return np.power(a, b)

    # --- Math ---
    def sin(self, tensor: Any) -> Any:
        return np.sin(tensor)
    
    def cos(self, tensor: Any) -> Any:
        return np.cos(tensor)
    
    def exp(self, tensor: Any) -> Any:
        return np.exp(tensor)
    
    def abs(self, tensor: Any) -> Any:
        return np.abs(tensor)
    
    def sign(self, tensor: Any) -> Any:
        return np.sign(tensor)

    # --- NN Operations ---
    def batch_norm(self, tensor: Any, mean: Any = None, var: Any = None, weight: Any = None, bias: Any = None, training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> Any:
        if mean is None: mean = np.mean(tensor, axis=0)
        if var is None: var = np.var(tensor, axis=0)
        
        inv = 1.0 / np.sqrt(var + eps)
        if weight is not None:
            inv = inv * weight
            
        res = (tensor - mean) * inv
        
        if bias is not None:
            res = res + bias
        return res

    def layer_norm(self, tensor: Any, normalized_shape: Any = None, weight: Any = None, bias: Any = None, eps: float = 1e-5) -> Any:
        axis = -1 if normalized_shape is None else -1
        mean = np.mean(tensor, axis=axis, keepdims=True)
        var = np.var(tensor, axis=axis, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        out = (tensor - mean) * inv
        if weight is not None: out = out * weight
        if bias is not None: out = out + bias
        return out

    def convolve(self, input: Any, weight: Any, bias: Optional[Any] = None, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> Any:
        if input.ndim == 1:
            return np.convolve(input, weight, mode='same')
        else:
            return input # Placeholder

    def max_pool(self, input: Any, kernel_size: int, stride: Optional[int] = None, padding: int = 0) -> Any:
        return input # Placeholder

    def avg_pool(self, input: Any, kernel_size: int, stride: Optional[int] = None, padding: int = 0) -> Any:
        return input # Placeholder

    # --- Gradient/Autograd ---
    def backward(self, tensor: Any, grad_tensor: Optional[Any] = None) -> Any:
        pass

    def grad(self, tensor: Any) -> Any:
        return None

    # --- Random ---
    def random_normal(self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, **kwargs) -> Any:
        return np.random.normal(mean, std, size=shape)

    def random_uniform(self, shape: Tuple[int, ...], min_val: float = 0.0, max_val: float = 1.0, **kwargs) -> Any:
        return np.random.uniform(min_val, max_val, size=shape)

    # --- Loss ---
    def mse_loss(self, input: Any, target: Any, reduction: str = 'mean') -> Any:
        loss = (input - target) ** 2
        if reduction == 'mean': return np.mean(loss)
        elif reduction == 'sum': return np.sum(loss)
        return loss

    def cross_entropy_loss(self, input: Any, target: Any, reduction: str = 'mean', **kwargs) -> Any:
        # Naive implementation
        e_x = np.exp(input - np.max(input, axis=-1, keepdims=True))
        probs = e_x / e_x.sum(axis=-1, keepdims=True)
        log_probs = np.log(probs + 1e-10)
        loss = -np.sum(target * log_probs, axis=-1)
        if reduction == 'mean': return np.mean(loss)
        return loss

    # --- Device/Misc ---
    def to_device(self, tensor: Any, device: str) -> Any:
        return tensor

    def device(self, tensor: Any) -> str:
        return "cpu"

    # --- Logic ---
    def equal(self, a: Any, b: Any) -> Any:
        return np.equal(a, b)

    def greater(self, a: Any, b: Any) -> Any:
        return np.greater(a, b)

    def less(self, a: Any, b: Any) -> Any:
        return np.less(a, b)

    def logical_and(self, a: Any, b: Any) -> Any:
        return np.logical_and(a, b)

    def logical_or(self, a: Any, b: Any) -> Any:
        return np.logical_or(a, b)

    def logical_not(self, a: Any) -> Any:
        return np.logical_not(a)

    # --- FFT ---
    def fft(self, tensor: Any, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Any:
        return np.fft.fft(tensor, n=n, axis=dim, norm=norm)

    def ifft(self, tensor: Any, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> Any:
        return np.fft.ifft(tensor, n=n, axis=dim, norm=norm)

    # --- Extra Utils for Tests ---
    def index(self, tensor: Any, index: Any) -> Any:
        return tensor[index]

    def slice(self, tensor: Any, start: int, end: int, dim: int = 0) -> Any:
        # Standard numpy slicing
        slHandling = [slice(None)] * tensor.ndim
        slHandling[dim] = slice(start, end)
        return tensor[tuple(slHandling)]

    def sgd_step(self, tensor: Any, lr: float) -> Any:
        return tensor

    def adam_step(self, tensor: Any, lr: float) -> Any:
        return tensor

    def get_backend_info(self) -> Dict[str, Any]:
         return {
            'backend': 'numpy',
            'device': 'cpu',
            'tensor_lib': 'numpy'
        }
