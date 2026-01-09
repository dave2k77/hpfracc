"""
Multi-Backend Fractional Neural Network Demo

This script demonstrates the comprehensive multi-backend support across all
fractional neural network components including:
- Core neural networks
- Neural network layers
- Loss functions
- Optimizers
- Graph Neural Networks

Supports PyTorch, JAX, and NUMBA backends.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Any
import warnings
import sys
import os

# Add library to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import HPFRACC ML components
from hpfracc.ml import (
    # Backend management
    BackendType, get_backend_manager, switch_backend,

    # Core components
    MLConfig, FractionalNeuralNetwork, FractionalAttention,

    # Layers
    LayerConfig, FractionalConv1D, FractionalConv2D, FractionalLSTM,
    FractionalTransformer, FractionalPooling, FractionalBatchNorm1d,

    # Loss functions
    FractionalMSELoss, FractionalCrossEntropyLoss, FractionalHuberLoss,
    FractionalSmoothL1Loss, FractionalKLDivLoss, FractionalBCELoss,

    # Optimizers
    FractionalAdam, FractionalSGD, FractionalRMSprop,

    # GNN components
    FractionalGNNFactory, get_tensor_ops
)

# Import FractionalOrder for proper typing
from hpfracc.core.definitions import FractionalOrder

warnings.filterwarnings('ignore')


def create_synthetic_data(backend: BackendType, num_samples: int = 100, input_dim: int = 16, num_classes: int = 4) -> Tuple[Any, Any]:
    """Create synthetic data for the specified backend"""
    tensor_ops = get_tensor_ops(backend)

    # Create random input data based on backend
    if backend == BackendType.JAX:
        import jax.random as random
        key = random.PRNGKey(0)
        X_data = random.normal(key, (num_samples, input_dim))
        X = tensor_ops.create_tensor(X_data, requires_grad=True)
    elif backend == BackendType.TORCH:
        import torch
        X_data = torch.randn(num_samples, input_dim)
        X = tensor_ops.create_tensor(X_data, requires_grad=True)
    elif backend == BackendType.NUMBA:
        import numpy as np
        X_data = np.random.randn(num_samples, input_dim)
        X = tensor_ops.create_tensor(X_data, requires_grad=True)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Create random labels
    if num_classes > 1:
        if backend == BackendType.JAX:
            import jax.random as random
            key = random.PRNGKey(1)
            y_data = random.randint(key, (num_samples,), 0, num_classes)
            y = tensor_ops.create_tensor(y_data, requires_grad=False)
        elif backend == BackendType.TORCH:
            import torch
            y_data = torch.randint(0, num_classes, (num_samples,))
            y = tensor_ops.create_tensor(y_data, requires_grad=False)
        elif backend == BackendType.NUMBA:
            import numpy as np
            y_data = np.random.randint(0, num_classes, num_samples)
            y = tensor_ops.create_tensor(y_data, requires_grad=False)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    else:
        if backend == BackendType.JAX:
            import jax.random as random
            key = random.PRNGKey(2)
            y_data = random.normal(key, (num_samples, 1))
            y = tensor_ops.create_tensor(y_data, requires_grad=False)
        elif backend == BackendType.TORCH:
            import torch
            y_data = torch.randn(num_samples, 1)
            y = tensor_ops.create_tensor(y_data, requires_grad=False)
        elif backend == BackendType.NUMBA:
            import numpy as np
            y_data = np.random.randn(num_samples, 1)
            y = tensor_ops.create_tensor(y_data, requires_grad=False)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    return X, y


def benchmark_core_networks(backend: BackendType, num_runs: int = 5) -> Dict[str, float]:
    """Benchmark core neural network components"""
    print(f"\n🔬 Benchmarking Core Networks ({backend.value.upper()})")

    # Create synthetic data
    X, y = create_synthetic_data(
        backend, num_samples=50, input_dim=16, num_classes=4)

    results = {}

    # Test FractionalNeuralNetwork
    print("  Testing FractionalNeuralNetwork...")
    start_time = time.time()

    model = FractionalNeuralNetwork(
        input_size=16,
        hidden_sizes=[32, 16],
        output_size=4,
        fractional_order=FractionalOrder(0.5),
        backend=backend
    )

    for _ in range(num_runs):
        output = model(X, use_fractional=True, method="RL")

    end_time = time.time()
    results['FractionalNeuralNetwork'] = (end_time - start_time) / num_runs

    # Test FractionalAttention
    print("  Testing FractionalAttention...")
    start_time = time.time()

    attention = FractionalAttention(
        d_model=16,  # 16 is divisible by 4 heads
        n_heads=4,
        fractional_order=FractionalOrder(0.5),
        backend=backend
    )

    # Reshape input for attention (batch_size, seq_len, d_model)
    # Ensure d_model matches what the attention expects
    X_reshaped = X.reshape(-1, 1, attention.d_model)

    for _ in range(num_runs):
        output = attention(X_reshaped, method="RL")

    end_time = time.time()
    results['FractionalAttention'] = (end_time - start_time) / num_runs

    return results


def benchmark_layers(backend: BackendType, num_runs: int = 5) -> Dict[str, float]:
    """Benchmark neural network layers"""
    print(f"\n🔬 Benchmarking Neural Network Layers ({backend.value.upper()})")

    results = {}

    # Test FractionalConv1D
    print("  Testing FractionalConv1D...")
    start_time = time.time()

    conv1d = FractionalConv1D(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        config=LayerConfig(
            fractional_order=FractionalOrder(0.5), backend=backend),
        backend=backend
    )

    # Create 1D input (batch_size, channels, seq_len)
    if backend == BackendType.TORCH:
        import torch
        X_1d_data = torch.randn(32, 16, 64)
    elif backend == BackendType.JAX:
        import jax.random as random
        key = random.PRNGKey(42)
        X_1d_data = random.normal(key, (32, 16, 64))
    elif backend == BackendType.NUMBA:
        import numpy as np
        X_1d_data = np.random.randn(32, 16, 64)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    X_1d = get_tensor_ops(backend).create_tensor(X_1d_data, requires_grad=True)

    for _ in range(num_runs):
        output = conv1d.forward(X_1d)

    end_time = time.time()
    results['FractionalConv1D'] = (end_time - start_time) / num_runs

    # Test FractionalConv2D
    print("  Testing FractionalConv2D...")
    start_time = time.time()

    conv2d = FractionalConv2D(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        config=LayerConfig(
            fractional_order=FractionalOrder(0.5), backend=backend),
        backend=backend
    )

    # Create 2D input (batch_size, channels, height, width)
    if backend == BackendType.TORCH:
        import torch
        X_2d_data = torch.randn(16, 16, 32, 32)
    elif backend == BackendType.JAX:
        import jax.random as random
        key = random.PRNGKey(43)
        X_2d_data = random.normal(key, (16, 16, 32, 32))
    elif backend == BackendType.NUMBA:
        import numpy as np
        X_2d_data = np.random.randn(16, 16, 32, 32)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    X_2d = get_tensor_ops(backend).create_tensor(X_2d_data, requires_grad=True)

    for _ in range(num_runs):
        output = conv2d.forward(X_2d)

    end_time = time.time()
    results['FractionalConv2D'] = (end_time - start_time) / num_runs

    # Test FractionalLSTM
    print("  Testing FractionalLSTM...")
    start_time = time.time()

    lstm = FractionalLSTM(
        input_size=16,
        hidden_size=32,
        config=LayerConfig(
            fractional_order=FractionalOrder(0.5), backend=backend),
        backend=backend
    )

    # Create sequence input (batch_size, seq_len, input_size)
    if backend == BackendType.TORCH:
        import torch
        X_seq_data = torch.randn(16, 20, 16)
    elif backend == BackendType.JAX:
        import jax.random as random
        key = random.PRNGKey(44)
        X_seq_data = random.normal(key, (16, 20, 16))
    elif backend == BackendType.NUMBA:
        import numpy as np
        X_seq_data = np.random.randn(16, 20, 16)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    X_seq = get_tensor_ops(backend).create_tensor(
        X_seq_data, requires_grad=True)

    for _ in range(num_runs):
        output, (h, c) = lstm.forward(X_seq)

    end_time = time.time()
    results['FractionalLSTM'] = (end_time - start_time) / num_runs

    # Test FractionalTransformer
    print("  Testing FractionalTransformer...")
    start_time = time.time()

    transformer = FractionalTransformer(
        d_model=16,
        n_heads=4,
        config=LayerConfig(
            fractional_order=FractionalOrder(0.5), backend=backend),
        backend=backend
    )

    # Create transformer input (batch_size, seq_len, d_model)
    if backend == BackendType.TORCH:
        import torch
        X_trans_data = torch.randn(16, 20, 16)
    elif backend == BackendType.JAX:
        import jax.random as random
        key = random.PRNGKey(45)
        X_trans_data = random.normal(key, (16, 20, 16))
    elif backend == BackendType.NUMBA:
        import numpy as np
        X_trans_data = np.random.randn(16, 20, 16)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    X_trans = get_tensor_ops(backend).create_tensor(
        X_trans_data, requires_grad=True)

    for _ in range(num_runs):
        transformer.forward(X_trans)

    end_time = time.time()
    results['FractionalTransformer'] = (end_time - start_time) / num_runs

    return results


def benchmark_loss_functions(backend: BackendType, num_runs: int = 5) -> Dict[str, float]:
    """Benchmark loss functions"""
    print(f"\n🔬 Benchmarking Loss Functions ({backend.value.upper()})")

    # Create synthetic data
    predictions, targets = create_synthetic_data(
        backend, num_samples=50, input_dim=16, num_classes=4)

    results = {}

    # Test various loss functions
    # For classification, we need predictions and targets with compatible shapes
    # predictions: (50, 4) - 50 samples, 4 classes
    # targets: (50,) - 50 samples, integer class labels

    # Create predictions that match the neural network output
    model = FractionalNeuralNetwork(
        input_size=16,
        hidden_sizes=[32, 16],
        output_size=4,
        fractional_order=FractionalOrder(0.5),
        backend=backend
    )
    # Use predictions as input to get output
    predictions = model(predictions, use_fractional=False, method="RL")

    # Ensure targets are the right dtype for the backend
    if backend == BackendType.TORCH:
        import torch
        targets = torch.tensor(targets, dtype=torch.long)

    loss_functions = {
        'FractionalCrossEntropyLoss': FractionalCrossEntropyLoss(fractional_order=FractionalOrder(0.5), backend=backend),
    }

    for name, loss_fn in loss_functions.items():
        print(f"  Testing {name}...")
        start_time = time.time()

        for _ in range(num_runs):
            loss = loss_fn(predictions, targets, use_fractional=True)

        end_time = time.time()
        results[name] = (end_time - start_time) / num_runs

    return results


def benchmark_optimizers(backend: BackendType, num_runs: int = 5) -> Dict[str, float]:
    """Benchmark optimizers"""
    print(f"\n🔬 Benchmarking Optimizers ({backend.value.upper()})")

    # Create synthetic data
    X, y = create_synthetic_data(
        backend, num_samples=50, input_dim=16, num_classes=4)

    results = {}

    # Create a simple model for testing optimizers
    model = FractionalNeuralNetwork(
        input_size=16,
        hidden_sizes=[32, 16],
        output_size=4,
        fractional_order=FractionalOrder(0.5),
        backend=backend
    )

    # Test various optimizers using the new simplified versions
    optimizers = {
        'FractionalSGD': FractionalSGD(
            lr=0.01,
            fractional_order=FractionalOrder(0.5),
            backend=backend
        ),
        'FractionalAdam': FractionalAdam(
            lr=0.001,
            fractional_order=FractionalOrder(0.5),
            backend=backend
        ),
        'FractionalRMSprop': FractionalRMSprop(
            lr=0.001,
            fractional_order=FractionalOrder(0.5),
            backend=backend
        )
    }

    # Test each optimizer
    for name, optimizer in optimizers.items():
        print(f"  Testing {name}...")
        start_time = time.time()

        for _ in range(num_runs):
            # Create simple gradients for testing
            if backend == BackendType.TORCH:
                import torch
                gradients = [torch.randn(1) for _ in range(3)]  # 3 parameters
            elif backend == BackendType.JAX:
                import jax.random as random
                key = random.PRNGKey(48)
                gradients = [random.normal(key, (1,))
                             for _ in range(3)]  # 3 parameters
            elif backend == BackendType.NUMBA:
                import numpy as np
                gradients = [np.random.randn(1)
                             for _ in range(3)]  # 3 parameters
            else:
                raise ValueError(f"Unsupported backend: {backend}")

            # Create simple parameters for testing
            if backend == BackendType.TORCH:
                import torch
                params = [torch.randn(1, requires_grad=True) for _ in range(3)]
            elif backend == BackendType.JAX:
                import jax.random as random
                key = random.PRNGKey(49)
                params = [random.normal(key, (1,)) for _ in range(3)]
            elif backend == BackendType.NUMBA:
                import numpy as np
                params = [np.random.randn(1) for _ in range(3)]
            else:
                raise ValueError(f"Unsupported backend: {backend}")

            # Test optimizer step
            try:
                optimizer.step(params, gradients)
            except Exception as e:
                print(f"    ⚠️  {name} step failed: {e}")

        end_time = time.time()
        results[name] = (end_time - start_time) / num_runs

    return results


def benchmark_gnn_models(backend: BackendType, num_runs: int = 5) -> Dict[str, float]:
    """Benchmark Graph Neural Network models"""
    print(f"\n🔬 Benchmarking GNN Models ({backend.value.upper()})")

    # Create synthetic graph data
    num_nodes = 100
    num_features = 16
    num_classes = 4

    # Create random node features
    if backend == BackendType.TORCH:
        import torch
        node_features_data = torch.randn(num_nodes, num_features)
    elif backend == BackendType.JAX:
        import jax.random as random
        key = random.PRNGKey(46)
        node_features_data = random.normal(key, (num_nodes, num_features))
    elif backend == BackendType.NUMBA:
        import numpy as np
        node_features_data = np.random.randn(num_nodes, num_features)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    node_features = get_tensor_ops(backend).create_tensor(
        node_features_data, requires_grad=True)

    # Create random edge index
    if backend == BackendType.TORCH:
        import torch
        edge_index_data = torch.randint(0, num_nodes, (2, 200))
    elif backend == BackendType.JAX:
        import jax.random as random
        key = random.PRNGKey(47)
        edge_index_data = random.randint(key, (2, 200), 0, num_nodes)
    elif backend == BackendType.NUMBA:
        import numpy as np
        edge_index_data = np.random.randint(0, num_nodes, (2, 200))
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    edge_index = get_tensor_ops(backend).create_tensor(
        edge_index_data, requires_grad=False)

    results = {}

    # Test various GNN models
    gnn_models = {
        'FractionalGCN': FractionalGNNFactory.create_model(
            'gcn', num_features, 32, num_classes,
            fractional_order=FractionalOrder(0.5), backend=backend
        ),
        'FractionalGAT': FractionalGNNFactory.create_model(
            'gat', num_features, 32, num_classes,
            fractional_order=FractionalOrder(0.5), backend=backend
        ),
        'FractionalGraphSAGE': FractionalGNNFactory.create_model(
            'graphsage', num_features, 32, num_classes,
            fractional_order=FractionalOrder(0.5), backend=backend
        ),
        'FractionalGraphUNet': FractionalGNNFactory.create_model(
            'unet', num_features, 32, num_classes,
            fractional_order=FractionalOrder(0.5), backend=backend
        )
    }

    for name, model in gnn_models.items():
        print(f"  Testing {name}...")
        start_time = time.time()

        for _ in range(num_runs):
            model.forward(node_features, edge_index)

        end_time = time.time()
        results[name] = (end_time - start_time) / num_runs

    return results


def compare_fractional_orders(backend: BackendType, model_type: str = "gcn") -> Dict[str, List[float]]:
    """Compare performance across different fractional orders"""
    print(f"\n🔬 Comparing Fractional Orders ({backend.value.upper()})")

    # Create synthetic data using working components
    num_samples = 50
    input_dim = 16
    num_classes = 4

    # Create input data using backend-appropriate random generation
    if backend == BackendType.TORCH:
        import torch
        X_data = torch.randn(num_samples, input_dim)
    elif backend == BackendType.JAX:
        import jax.random as random
        key = random.PRNGKey(42)
        X_data = random.normal(key, (num_samples, input_dim))
    elif backend == BackendType.NUMBA:
        import numpy as np
        X_data = np.random.randn(num_samples, input_dim)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    X = get_tensor_ops(backend).create_tensor(X_data, requires_grad=True)

    # Stay within 0 < α < 1 to avoid L1 scheme validation issues
    fractional_orders = [0.25, 0.5, 0.75]
    results = {'fractional_orders': fractional_orders,
               'inference_time': [], 'memory_usage': []}

    for alpha in fractional_orders:
        print(f"  Testing fractional order α = {alpha}")

        # Create a simple working model with specific fractional order
        model = FractionalNeuralNetwork(
            input_size=input_dim,
            hidden_sizes=[32, 16],
            output_size=num_classes,
            fractional_order=FractionalOrder(alpha),
            backend=backend
        )

        # Measure inference time
        start_time = time.time()
        for _ in range(10):
            output = model(X, use_fractional=True, method="RL")
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        results['inference_time'].append(avg_time)

        # Estimate memory usage (simplified)
        if backend == BackendType.TORCH:
            # Check if model has parameters attribute, otherwise use rough estimate
            try:
                memory_usage = sum(p.numel() * p.element_size()
                                   for p in model.parameters())
            except AttributeError:
                memory_usage = num_samples * input_dim * 4  # 4 bytes per float
        else:
            # Rough estimate for JAX/NUMBA
            memory_usage = num_samples * input_dim * 4  # 4 bytes per float

        results['memory_usage'].append(memory_usage)

    return results


def visualize_performance_comparison(all_results: Dict[str, Dict[str, float]]) -> None:
    """Visualize performance comparison across backends"""
    print("\n📊 Creating Performance Visualization...")

    # Prepare data for plotting
    backends = list(all_results.keys())
    components = list(all_results[backends[0]].keys())

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Backend Performance Comparison',
                 fontsize=16, fontweight='bold')

    # Core Networks
    ax1 = axes[0, 0]
    core_components = [
        comp for comp in components if 'FractionalNeuralNetwork' in comp or 'FractionalAttention' in comp]
    if core_components:
        x = np.arange(len(core_components))
        width = 0.25

        for i, backend in enumerate(backends):
            values = [all_results[backend].get(
                comp, 0) for comp in core_components]
            ax1.bar(x + i * width, values, width,
                    label=backend.value.upper(), alpha=0.8)

        ax1.set_xlabel('Components')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Core Networks Performance')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([comp.replace('Fractional', '')
                            for comp in core_components], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Layers
    ax2 = axes[0, 1]
    layer_components = [comp for comp in components if any(
        layer in comp for layer in ['Conv', 'LSTM', 'Transformer', 'Pooling', 'BatchNorm'])]
    if layer_components:
        x = np.arange(len(layer_components))
        width = 0.25

        for i, backend in enumerate(backends):
            values = [all_results[backend].get(
                comp, 0) for comp in layer_components]
            ax2.bar(x + i * width, values, width,
                    label=backend.value.upper(), alpha=0.8)

        ax2.set_xlabel('Components')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Neural Network Layers Performance')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([comp.replace('Fractional', '')
                            for comp in layer_components], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Loss Functions
    ax3 = axes[1, 0]
    loss_components = [comp for comp in components if 'Loss' in comp]
    if loss_components:
        x = np.arange(len(loss_components))
        width = 0.25

        for i, backend in enumerate(backends):
            values = [all_results[backend].get(
                comp, 0) for comp in loss_components]
            ax3.bar(x + i * width, values, width,
                    label=backend.value.upper(), alpha=0.8)

        ax3.set_xlabel('Components')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Loss Functions Performance')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([comp.replace('Fractional', '').replace(
            'Loss', '') for comp in loss_components], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Optimizers
    ax4 = axes[1, 1]
    optimizer_components = [comp for comp in components if any(
        opt in comp for opt in ['Adam', 'SGD', 'RMSprop', 'Adagrad'])]
    if optimizer_components:
        x = np.arange(len(optimizer_components))
        width = 0.25

        for i, backend in enumerate(backends):
            values = [all_results[backend].get(
                comp, 0) for comp in optimizer_components]
            ax4.bar(x + i * width, values, width,
                    label=backend.value.upper(), alpha=0.8)

        ax4.set_xlabel('Components')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Optimizers Performance')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels([comp.replace('Fractional', '')
                            for comp in optimizer_components], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multi_backend_performance_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def visualize_fractional_effects(fractional_results: Dict[str, Dict[str, List[float]]]) -> None:
    """Visualize the effects of different fractional orders"""
    print("\n📊 Creating Fractional Order Effects Visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Fractional Order Effects on Performance',
                 fontsize=16, fontweight='bold')

    # Inference time comparison
    ax1.set_xlabel('Fractional Order (α)')
    ax1.set_ylabel('Inference Time (seconds)')
    ax1.set_title('Inference Time vs Fractional Order')
    ax1.grid(True, alpha=0.3)

    for backend, results in fractional_results.items():
        ax1.plot(results['fractional_orders'], results['inference_time'],
                 marker='o', linewidth=2, label=f'{backend.value.upper()}', alpha=0.8)

    ax1.legend()

    # Memory usage comparison
    ax2.set_xlabel('Fractional Order (α)')
    ax2.set_ylabel('Memory Usage (bytes)')
    ax2.set_title('Memory Usage vs Fractional Order')
    ax2.grid(True, alpha=0.3)

    for backend, results in fractional_results.items():
        ax2.plot(results['fractional_orders'], results['memory_usage'],
                 marker='s', linewidth=2, label=f'{backend.value.upper()}', alpha=0.8)

    ax2.legend()

    plt.tight_layout()
    plt.savefig('fractional_order_effects.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main demonstration function"""
    print("🚀 HPFRACC Multi-Backend Fractional Neural Network Demo")
    print("=" * 60)

    # Get available backends
    backend_manager = get_backend_manager()
    available_backends = backend_manager.available_backends

    print(f"Available backends: {[b.value for b in available_backends]}")
    print(f"Active backend: {backend_manager.active_backend.value}")

    # Store all results
    all_results = {}
    fractional_results = {}

    # Test each available backend
    for backend in available_backends:
        if backend == BackendType.AUTO:
            continue

        print(f"\n{'='*20} Testing {backend.value.upper()} Backend {'='*20}")

        # Switch to the backend
        switch_backend(backend)

        try:
            # Benchmark core networks
            core_results = benchmark_core_networks(backend)

            # Benchmark layers
            layer_results = benchmark_layers(backend)

            # Benchmark loss functions
            loss_results = benchmark_loss_functions(backend)

            # Benchmark optimizers
            optimizer_results = benchmark_optimizers(backend)

            # Benchmark GNN models
            gnn_results = benchmark_gnn_models(backend)

            # Combine all results
            all_results[backend] = {
                **core_results,
                **layer_results,
                **loss_results,
                **optimizer_results,
                **gnn_results
            }

            # Test fractional order effects
            fractional_results[backend] = compare_fractional_orders(backend)

            print(
                f"✅ {backend.value.upper()} backend testing completed successfully!")

        except Exception as e:
            print(f"❌ Error testing {backend.value.upper()} backend: {e}")
            continue

    # Display summary
    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY")
    print("="*60)

    for backend, results in all_results.items():
        print(f"\n{backend.value.upper()} Backend:")
        for component, time_taken in results.items():
            print(f"  {component}: {time_taken:.4f}s")

    # Create visualizations
    if len(all_results) > 1:
        visualize_performance_comparison(all_results)

    if fractional_results:
        visualize_fractional_effects(fractional_results)

    print("\n🎉 Multi-backend demo completed successfully!")
    print("📁 Generated visualizations:")
    print("  - multi_backend_performance_comparison.png")
    print("  - fractional_order_effects.png")


if __name__ == "__main__":
    main()
