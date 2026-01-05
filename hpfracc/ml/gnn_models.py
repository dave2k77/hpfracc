"""
Complete Fractional Graph Neural Network Architectures

This module provides complete GNN architectures with fractional calculus integration,
including various model types and configurations for different graph learning tasks.
"""

from typing import Optional, Union, Any, List, Dict
from abc import ABC, abstractmethod

from .backends import get_backend_manager, BackendType
from .gnn_layers import FractionalGraphConv, FractionalGraphAttention, FractionalGraphPooling
from .tensor_ops import get_tensor_ops
from hpfracc.core.definitions import FractionalOrder


class BaseFractionalGNN(ABC):
    """
    Base class for fractional Graph Neural Networks

    This abstract class defines the interface for all fractional GNN models,
    ensuring consistency across different architectures and backends.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        fractional_order: Union[float, FractionalOrder] = 0.5,
        method: str = "RL",
        use_fractional: bool = True,
        activation: str = "relu",
        dropout: float = 0.1,
        backend: Optional[BackendType] = None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.fractional_order = FractionalOrder(fractional_order) if isinstance(
            fractional_order, float) else fractional_order
        self.method = method
        self.use_fractional = use_fractional
        self.activation = activation
        self.dropout = dropout
        self.backend = backend or get_backend_manager().active_backend

        # Initialize tensor operations
        self.tensor_ops = get_tensor_ops(self.backend)

        # Build the network
        self._build_network()

    @abstractmethod
    def _build_network(self):
        """Build the specific network architecture"""

    @abstractmethod
    def forward(
            self,
            x: Any,
            edge_index: Any,
            batch: Optional[Any] = None,
            **kwargs) -> Any:
        """Forward pass through the network"""

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend"""
        return {
            'backend': self.backend.value,
            'tensor_lib': str(self.tensor_ops.tensor_lib),
            'fractional_order': self.fractional_order.alpha,
            'method': self.method,
            'use_fractional': self.use_fractional
        }

    def __call__(
            self,
            x: Any,
            edge_index: Any,
            batch: Optional[Any] = None,
            **kwargs) -> Any:
        """Make models callable like torch modules"""
        return self.forward(x, edge_index, batch, **kwargs)

    def parameters(self) -> List[Any]:
        """Collect learnable parameters from sub-layers for testing/optimizers"""
        params: List[Any] = []
        layer_attrs = []
        # Gather potential layer lists/attributes
        for attr in [
            'layers', 'encoder_layers', 'decoder_layers', 'output_layer'
        ]:
            if hasattr(self, attr):
                layer_attrs.append(getattr(self, attr))
        for entry in layer_attrs:
            if isinstance(entry, list):
                iterable = entry
            else:
                iterable = [entry]
            for layer in iterable:
                for name in [
                    'weight',
                    'bias',
                    'query_weight',
                    'key_weight',
                    'value_weight',
                    'output_weight',
                        'score_network']:
                    if hasattr(layer, name):
                        params.append(getattr(layer, name))
        return params


class FractionalGCN(BaseFractionalGNN):
    """
    Fractional Graph Convolutional Network

    A GNN architecture that uses fractional graph convolution layers
    for node classification, graph classification, and other tasks.
    """

    def _build_network(self):
        """Build the GCN architecture"""
        self.layers = []

        # Input layer
        self.layers.append(
            FractionalGraphConv(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim,
                fractional_order=self.fractional_order,
                method=self.method,
                use_fractional=self.use_fractional,
                activation=self.activation,
                dropout=self.dropout,
                backend=self.backend
            )
        )

        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.layers.append(
                FractionalGraphConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    fractional_order=self.fractional_order,
                    method=self.method,
                    use_fractional=self.use_fractional,
                    activation=self.activation,
                    dropout=self.dropout,
                    backend=self.backend
                )
            )

        # Output layer
        self.layers.append(
            FractionalGraphConv(
                in_channels=self.hidden_dim,
                out_channels=self.output_dim,
                fractional_order=self.fractional_order,
                method=self.method,
                use_fractional=self.use_fractional,
                activation="identity",  # No activation for output
                dropout=0.0,  # No dropout for output
                backend=self.backend
            )
        )

    def forward(
            self,
            x: Any,
            edge_index: Any,
            batch: Optional[Any] = None,
            **kwargs) -> Any:
        """
        Forward pass through the fractional GCN

        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Pass through all layers
        for layer in self.layers:
            x = layer.forward(x, edge_index, **kwargs)

        return x


class FractionalGAT(BaseFractionalGNN):
    """
    Fractional Graph Attention Network

    A GNN architecture that uses fractional graph attention layers
    for enhanced graph representation learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        fractional_order: Union[float, FractionalOrder] = 0.5,
        method: str = "RL",
        use_fractional: bool = True,
        activation: str = "relu",
        dropout: float = 0.1,
        backend: Optional[BackendType] = None
    ):
        self.num_heads = num_heads
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            fractional_order,
            method,
            use_fractional,
            activation,
            dropout,
            backend)

    def _build_network(self):
        """Build the GAT architecture"""
        self.layers = []

        # Input layer
        self.layers.append(
            FractionalGraphAttention(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim,
                heads=self.num_heads,
                fractional_order=self.fractional_order,
                method=self.method,
                use_fractional=self.use_fractional,
                activation=self.activation,
                dropout=self.dropout,
                backend=self.backend
            )
        )

        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.layers.append(
                FractionalGraphAttention(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    heads=self.num_heads,
                    fractional_order=self.fractional_order,
                    method=self.method,
                    use_fractional=self.use_fractional,
                    activation=self.activation,
                    dropout=self.dropout,
                    backend=self.backend
                )
            )

        # Output layer
        self.layers.append(
            FractionalGraphAttention(
                in_channels=self.hidden_dim,
                out_channels=self.output_dim,
                heads=1,  # Single head for output
                fractional_order=self.fractional_order,
                method=self.method,
                use_fractional=self.use_fractional,
                activation="identity",
                dropout=0.0,
                backend=self.backend
            )
        )

    def forward(
            self,
            x: Any,
            edge_index: Any,
            batch: Optional[Any] = None,
            **kwargs) -> Any:
        """
        Forward pass through the fractional GAT

        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Pass through all layers
        for layer in self.layers:
            x = layer.forward(x, edge_index, **kwargs)

        return x


class FractionalGraphSAGE(BaseFractionalGNN):
    """
    Fractional GraphSAGE Network

    A GNN architecture that uses fractional graph convolution layers
    with neighbor sampling for scalable graph learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        num_samples: int = 25,
        fractional_order: Union[float, FractionalOrder] = 0.5,
        method: str = "RL",
        use_fractional: bool = True,
        activation: str = "relu",
        dropout: float = 0.1,
        backend: Optional[BackendType] = None
    ):
        self.num_samples = num_samples
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            fractional_order,
            method,
            use_fractional,
            activation,
            dropout,
            backend)

    def _build_network(self):
        """Build the GraphSAGE architecture"""
        self.layers = []

        # Input layer
        self.layers.append(
            FractionalGraphConv(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim,
                fractional_order=self.fractional_order,
                method=self.method,
                use_fractional=self.use_fractional,
                activation=self.activation,
                dropout=self.dropout,
                backend=self.backend
            )
        )

        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.layers.append(
                FractionalGraphConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    fractional_order=self.fractional_order,
                    method=self.method,
                    use_fractional=self.use_fractional,
                    activation=self.activation,
                    dropout=self.dropout,
                    backend=self.backend
                )
            )

        # Output layer
        self.layers.append(
            FractionalGraphConv(
                in_channels=self.hidden_dim,
                out_channels=self.output_dim,
                fractional_order=self.fractional_order,
                method=self.method,
                use_fractional=self.use_fractional,
                activation="identity",
                dropout=0.0,
                backend=self.backend
            )
        )

    def forward(
            self,
            x: Any,
            edge_index: Any,
            batch: Optional[Any] = None,
            **kwargs) -> Any:
        """
        Forward pass through the fractional GraphSAGE

        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Pass through all layers
        for layer in self.layers:
            x = layer.forward(x, edge_index, **kwargs)

        return x


class FractionalGraphUNet(BaseFractionalGNN):
    """
    Fractional Graph U-Net

    A hierarchical GNN architecture that uses fractional calculus
    for multi-scale graph representation learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        pooling_ratio: float = 0.5,
        fractional_order: Union[float, FractionalOrder] = 0.5,
        method: str = "RL",
        use_fractional: bool = True,
        activation: str = "relu",
        dropout: float = 0.1,
        backend: Optional[BackendType] = None
    ):
        self.pooling_ratio = pooling_ratio
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            fractional_order,
            method,
            use_fractional,
            activation,
            dropout,
            backend)

    def _build_network(self):
        """Build the Graph U-Net architecture"""
        # Encoder layers
        self.encoder_layers = []
        current_dim = self.input_dim

        for i in range(self.num_layers):
            self.encoder_layers.append(
                FractionalGraphConv(
                    in_channels=current_dim,
                    out_channels=self.hidden_dim,
                    fractional_order=self.fractional_order,
                    method=self.method,
                    use_fractional=self.use_fractional,
                    activation=self.activation,
                    dropout=self.dropout,
                    backend=self.backend
                )
            )
            current_dim = self.hidden_dim

        # Pooling layers (skip for small networks to preserve node count)
        self.pooling_layers = []
        if self.num_layers > 2:  # Only use pooling for deeper networks
            for _ in range(self.num_layers - 1):
                self.pooling_layers.append(
                    FractionalGraphPooling(
                        in_channels=self.hidden_dim,
                        pooling_ratio=self.pooling_ratio,
                        fractional_order=self.fractional_order,
                        method=self.method,
                        use_fractional=self.use_fractional,
                        backend=self.backend
                    )
                )

        # Decoder layers (skip for small networks)
        self.decoder_layers = []
        if self.num_layers > 2:  # Only use decoder layers for deeper networks
            for i in range(self.num_layers - 1):
                self.decoder_layers.append(
                    FractionalGraphConv(
                        in_channels=self.hidden_dim * 2,  # Skip connection
                        out_channels=self.hidden_dim,
                        fractional_order=self.fractional_order,
                        method=self.method,
                        use_fractional=self.use_fractional,
                        activation=self.activation,
                        dropout=self.dropout,
                        backend=self.backend
                    )
                )

        # Output layer
        self.output_layer = FractionalGraphConv(
            in_channels=self.hidden_dim,
            out_channels=self.output_dim,
            fractional_order=self.fractional_order,
            method=self.method,
            use_fractional=self.use_fractional,
            activation="identity",
            dropout=0.0,
            backend=self.backend
        )

    def forward(
            self,
            x: Any,
            edge_index: Any,
            batch: Optional[Any] = None,
            **kwargs) -> Any:
        """
        Forward pass through the fractional Graph U-Net

        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Encoder path
        encoder_outputs = [x]
        current_x = x
        current_edge_index = edge_index
        current_batch = batch

        for i, layer in enumerate(self.encoder_layers):
            current_x = layer.forward(current_x, current_edge_index, **kwargs)
            encoder_outputs.append(current_x)

            # Apply pooling (except for the last layer and if pooling layers
            # exist)
            if i < len(self.pooling_layers) and len(self.pooling_layers) > 0:
                current_x, current_edge_index, current_batch = self.pooling_layers[i].forward(
                    current_x, current_edge_index, current_batch, **kwargs)

        # Decoder path with skip connections (only if decoder layers exist)
        if len(self.decoder_layers) > 0:
            for i, layer in enumerate(self.decoder_layers):
                # Skip connection
                skip_x = encoder_outputs[-(i + 2)]

                # Ensure skip_x has compatible dimensions with current_x
                if skip_x.shape[0] != current_x.shape[0]:
                    # Reshape skip_x to match current_x dimensions
                    if skip_x.shape[0] > current_x.shape[0]:
                        # Truncate skip_x to match current_x
                        skip_x = skip_x[:current_x.shape[0], :]
                    else:
                        # Pad skip_x to match current_x
                        padding = current_x.shape[0] - skip_x.shape[0]
                        if padding > 0:
                            # Create padding tensor using tensor_ops
                            padding_tensor = self.tensor_ops.zeros(
                                (padding, skip_x.shape[1]))
                            skip_x = self.tensor_ops.cat(
                                [skip_x, padding_tensor], dim=0)

                # Ensure feature dimensions are compatible for concatenation
                if skip_x.shape[-1] != current_x.shape[-1]:
                    # Reshape skip_x to match current_x feature dimensions
                    if skip_x.shape[-1] > current_x.shape[-1]:
                        # Truncate features
                        skip_x = skip_x[..., :current_x.shape[-1]]
                    else:
                        # Pad features with zeros
                        feature_padding = current_x.shape[-1] - \
                            skip_x.shape[-1]
                        if feature_padding > 0:
                            padding_tensor = self.tensor_ops.zeros(
                                (skip_x.shape[0], feature_padding))
                            skip_x = self.tensor_ops.cat(
                                [skip_x, padding_tensor], dim=-1)

                # Concatenate with current features
                current_x = self.tensor_ops.cat([current_x, skip_x], dim=-1)

                # Pass through decoder layer
                current_x = layer.forward(current_x, current_edge_index, **kwargs)

        # Output layer
        output = self.output_layer.forward(current_x, current_edge_index, **kwargs)

        return output


class FractionalGNNFactory:
    """
    Factory class for creating fractional GNN models

    This class provides a convenient interface for creating different
    types of fractional GNN architectures with consistent configurations.
    """

    @staticmethod
    def create_model(
        model_type: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        **kwargs
    ) -> BaseFractionalGNN:
        """
        Create a fractional GNN model of the specified type

        Args:
            model_type: Type of GNN model ('gcn', 'gat', 'sage', 'unet')
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            **kwargs: Additional arguments for the model

        Returns:
            Configured fractional GNN model
        """
        model_type = model_type.lower()

        if model_type == 'gcn':
            return FractionalGCN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                **kwargs
            )
        elif model_type == 'gat':
            return FractionalGAT(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                **kwargs
            )
        elif model_type == 'sage':
            return FractionalGraphSAGE(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                **kwargs
            )
        elif model_type == 'unet':
            return FractionalGraphUNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. "
                             f"Available types: gcn, gat, sage, unet")

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types"""
        return ['gcn', 'gat', 'sage', 'unet']

    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """Get information about a specific model type"""
        model_type = model_type.lower()

        info = {
            'gcn': {
                'name': 'Fractional Graph Convolutional Network',
                'description': 'Standard GCN with fractional calculus integration',
                'best_for': ['Node classification', 'Graph classification', 'Link prediction'],
                'complexity': 'Low',
                'memory_efficient': True
            },
            'gat': {
                'name': 'Fractional Graph Attention Network',
                'description': 'GNN with attention mechanisms and fractional calculus',
                'best_for': ['Node classification', 'Graph classification', 'Attention analysis'],
                'complexity': 'Medium',
                'memory_efficient': False
            },
            'sage': {
                'name': 'Fractional GraphSAGE',
                'description': 'Scalable GNN with neighbor sampling and fractional calculus',
                'best_for': ['Large graphs', 'Inductive learning', 'Node classification'],
                'complexity': 'Low',
                'memory_efficient': True
            },
            'unet': {
                'name': 'Fractional Graph U-Net',
                'description': 'Hierarchical GNN with skip connections and fractional calculus',
                'best_for': ['Multi-scale learning', 'Graph segmentation', 'Hierarchical tasks'],
                'complexity': 'High',
                'memory_efficient': False
            }
        }

        return info.get(
            model_type, {
                'error': f'Unknown model type: {model_type}'})
