import numpy as np
from typing import Any, Optional

class NumbaFractionalGNNMixin:
    """
    Mixin for Numba-specific fractional GNN logic.
    """
    
    def _numba_fractional_derivative(self, x: Any, alpha: float) -> Any:
        if alpha == 0:
            return x
        elif alpha == 1:
            if x.ndim > 1:
                diff = np.diff(x, axis=-1)
                padding_shape = list(x.shape)
                padding_shape[-1] = 1
                padding = np.zeros(padding_shape, dtype=x.dtype)
                return np.concatenate([diff, padding], axis=-1)
            else:
                diff = np.diff(x, axis=0)
                padding = np.zeros(1, dtype=x.dtype)
                return np.concatenate([diff, padding], axis=0)
        else:
            if 0 < alpha < 1:
                if x.ndim > 1:
                    derivative = np.diff(x, axis=-1)
                    derivative = np.concatenate([derivative, np.zeros_like(x[..., :1])], axis=-1)
                else:
                    derivative = np.diff(x, axis=0)
                    derivative = np.concatenate([derivative, np.zeros(1, dtype=x.dtype)], axis=0)
                return (1 - alpha) * x + alpha * derivative
            else:
                result = x
                n = int(alpha)
                beta = alpha - n
                for _ in range(n):
                    if result.ndim > 1:
                        result = np.diff(result, axis=-1)
                        result = np.concatenate([result, np.zeros_like(result[..., :1])], axis=-1)
                    else:
                        result = np.diff(result, axis=0)
                        result = np.concatenate([result, np.zeros(1, dtype=result.dtype)], axis=0)
                if beta > 0:
                    if result.ndim > 1:
                        derivative = np.diff(result, axis=-1)
                        derivative = np.concatenate([derivative, np.zeros_like(result[..., :1])], axis=-1)
                    else:
                        derivative = np.diff(result, axis=0)
                        derivative = np.concatenate([derivative, np.zeros(1, dtype=result.dtype)], axis=0)
                    result = (1 - beta) * result + beta * derivative
                return result

    def _numba_forward_impl(self, x, edge_index, edge_weight, weight, bias, activation):
        out = np.matmul(x, weight)
        
        # Numba scatter add placeholder (matching original no-op)
        if edge_index is not None:
             pass

        if bias is not None:
            out = out + bias
            
        if activation == "relu":
            out = np.maximum(out, 0)
        elif activation == "sigmoid":
            out = 1 / (1 + np.exp(-out))
        elif activation == "tanh":
            out = np.tanh(out)
            
        return out
