import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

class TorchFractionalGNNMixin:
    """
    Mixin for PyTorch-specific fractional GNN logic.
    """
    
    def _torch_fractional_derivative(self, x: Any, alpha: float) -> Any:
        # Implementation from original gnn_layers.py
        if alpha == 0:
            return x
        elif alpha == 1:
            if x.dim() > 1:
                gradients = torch.gradient(x, dim=-1)[0]
                if gradients.shape != x.shape:
                    if gradients.shape[-1] < x.shape[-1]:
                        padding = x.shape[-1] - gradients.shape[-1]
                        gradients = torch.cat(
                            [gradients, torch.zeros_like(gradients[..., :padding])], dim=-1)
                    else:
                        gradients = gradients[..., :x.shape[-1]]
                return gradients
            else:
                diff = torch.diff(x, dim=-1)
                padding = torch.zeros(1, dtype=x.dtype, device=x.device)
                return torch.cat([diff, padding], dim=-1)
        else:
            if alpha == 0:
                return x
            elif 0 < alpha < 1:
                derivative = torch.diff(x, dim=-1)
                derivative = torch.cat([derivative, torch.zeros_like(x[..., :1])], dim=-1)
                return (1 - alpha) * x + alpha * derivative
            else:
                result = x
                n = int(alpha)
                beta = alpha - n
                for _ in range(n):
                    result = torch.diff(result, dim=-1)
                    result = torch.cat([result, torch.zeros_like(result[..., :1])], dim=-1)
                if beta > 0:
                    derivative = torch.diff(result, dim=-1)
                    derivative = torch.cat([derivative, torch.zeros_like(result[..., :1])], dim=-1)
                    result = (1 - beta) * result + beta * derivative
                return result

    def _torch_forward_impl(self, x, edge_index, edge_weight, weight, bias, activation, dropout, training):
        # Helper to avoid massive duplication in forward
        # Assumes 'self.tensor_ops' is available
        
        weight = weight.to(x.dtype).to(x.device)
        out = torch.matmul(x, weight)
        
        if edge_index is not None and edge_index.shape[1] > 0:
            if edge_index.dim() == 1:
                edge_index = self.tensor_ops.reshape(edge_index, (1, -1))
            if edge_index.shape[0] == 1:
                edge_index = self.tensor_ops.repeat(edge_index, 2, dim=0)
            elif edge_index.shape[0] > 2:
                edge_index = edge_index[:2, :]
                
            num_nodes = x.shape[0]
            edge_index = self.tensor_ops.clip(edge_index, 0, num_nodes - 1)
            row, col = edge_index
            
            if edge_weight is not None:
                edge_weight = edge_weight.to(x.device)
                if edge_weight.dim() == 1:
                    edge_weight = self.tensor_ops.unsqueeze(edge_weight, -1)
                weighted_features = out[col] * edge_weight
                index = self.tensor_ops.unsqueeze(row, -1).expand(-1, out.shape[-1])
                index = index.to(out.device)
                out = out.scatter_add(0, index, weighted_features)
            else:
                index = self.tensor_ops.unsqueeze(row, -1).expand(-1, out.shape[-1])
                index = index.to(out.device)
                out = out.scatter_add(0, index, out[col])

        if bias is not None:
            bias = bias.to(x.dtype).to(x.device)
            out = out + bias
            
        if activation == "relu":
            out = F.relu(out)
        elif activation == "sigmoid":
            out = torch.sigmoid(out)
        elif activation == "tanh":
            out = torch.tanh(out)
        elif activation != "identity":
             try:
                out = getattr(F, activation)(out)
             except AttributeError:
                out = F.relu(out)

        if training:
            out = F.dropout(out, p=dropout, training=True)
            
        return out
