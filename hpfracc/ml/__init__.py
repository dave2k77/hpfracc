"""
Machine Learning Integration for Fractional Calculus

This module provides comprehensive ML components that integrate fractional calculus
with neural networks, including:

- **Core Neural Networks**: FractionalNeuralNetwork, FractionalAttention
- **Neural Network Layers**: FractionalConv1D, FractionalConv2D, FractionalLSTM, FractionalTransformer, FractionalPooling, FractionalBatchNorm1d
- **Loss Functions**: FractionalMSELoss, FractionalCrossEntropyLoss, FractionalHuberLoss, and more
- **Optimizers**: FractionalAdam, FractionalSGD, FractionalRMSprop
- **Fractional Graph Neural Networks (GNNs) with multi-backend support**
- **Multi-backend support (PyTorch, JAX, NUMBA)**
- **Backend Management System**: BackendManager, BackendType, unified tensor operations
- **Unified Tensor Operations**: Cross-backend tensor manipulations
- **MLOps**: ModelRegistry, DevelopmentWorkflow, ProductionWorkflow, QualityGate
"""

# Backend Management
from .backends import (
    BackendType,
    BackendManager,
    get_backend_manager,
    set_backend_manager,
    get_active_backend,
    switch_backend,
)

# Tensor Operations
from .tensor_ops import (
    TensorOps,
    get_tensor_ops,
    create_tensor,
)

# Core ML Components
from .core import (
    MLConfig,
    FractionalNeuralNetwork,
    FractionalAttention,
    FractionalLossFunction,
    FractionalAutoML,
)

# Neural Network Layers
from .layers import (
    LayerConfig,
    FractionalConv1D,
    FractionalConv2D,
    FractionalLSTM,
    FractionalTransformer,
    FractionalPooling,
    FractionalBatchNorm1d,
)

# Loss Functions
from .losses import (
    FractionalMSELoss,
    FractionalCrossEntropyLoss,
    FractionalHuberLoss,
    FractionalSmoothL1Loss,
    FractionalKLDivLoss,
    FractionalBCELoss,
    FractionalNLLLoss,
    FractionalPoissonNLLLoss,
    FractionalCosineEmbeddingLoss,
    FractionalMarginRankingLoss,
    FractionalMultiMarginLoss,
    FractionalTripletMarginLoss,
    FractionalCTCLoss,
    FractionalCustomLoss,
    FractionalCombinedLoss,
)

# Optimizers (using actual class names from optimized_optimizers.py)
from .optimized_optimizers import (
    OptimizedBaseOptimizer as FractionalOptimizer,
    OptimizedFractionalAdam as FractionalAdam,
    OptimizedFractionalSGD as FractionalSGD,
    OptimizedFractionalRMSprop as FractionalRMSprop,
)

# Fractional GNN Components
from .gnn_layers import (
    BaseFractionalGNNLayer,
    FractionalGraphConv,
    FractionalGraphAttention,
    FractionalGraphAttention,
    FractionalGraphPooling,
)

from .gnn_models import (
    BaseFractionalGNN,
    FractionalGCN,
    FractionalGAT,
    FractionalGraphSAGE,
    FractionalGraphUNet,
    FractionalGNNFactory,
)

# Spectral Fractional Autograd
from .spectral_autograd import (
    SpectralFractionalDerivative,
    SpectralFractionalLayer,
    SpectralFractionalNetwork,
    BoundedAlphaParameter,
    spectral_fractional_derivative,
    create_fractional_layer,
)

# Stochastic Memory Sampling
from .stochastic_memory_sampling import (
    StochasticFractionalLayer,
    stochastic_fractional_derivative,
    create_stochastic_fractional_layer,
)

# Import probabilistic fractional orders
from .probabilistic_fractional_orders import (
    ProbabilisticFractionalOrder,
    ProbabilisticFractionalLayer,
    create_probabilistic_fractional_layer,
    create_normal_alpha_layer,
    create_uniform_alpha_layer,
    create_beta_alpha_layer,
)

# MLOps Components
from .registry import (
    ModelRegistry,
    ModelMetadata,
    ModelVersion,
    DeploymentStatus,
)

from .workflow import (
    DevelopmentWorkflow,
    ProductionWorkflow,
    ModelValidator,
    QualityGate,
    QualityMetric,
    QualityThreshold,
)

# Export all components
__all__ = [
    # Backend Management
    'BackendType',
    'BackendManager',
    'get_backend_manager',
    'set_backend_manager',
    'get_active_backend',
    'switch_backend',
    # Tensor Operations
    'TensorOps',
    'get_tensor_ops',
    'create_tensor',
    # Core ML Components
    'MLConfig',
    'FractionalNeuralNetwork',
    'FractionalAttention',
    'FractionalLossFunction',
    'FractionalAutoML',
    # Neural Network Layers
    'LayerConfig',
    'FractionalConv1D',
    'FractionalConv2D',
    'FractionalLSTM',
    'FractionalTransformer',
    'FractionalPooling',
    'FractionalBatchNorm1d',
    # Loss Functions
    'FractionalMSELoss',
    'FractionalCrossEntropyLoss',
    'FractionalHuberLoss',
    'FractionalSmoothL1Loss',
    'FractionalKLDivLoss',
    'FractionalBCELoss',
    'FractionalNLLLoss',
    'FractionalPoissonNLLLoss',
    'FractionalCosineEmbeddingLoss',
    'FractionalMarginRankingLoss',
    'FractionalMultiMarginLoss',
    'FractionalTripletMarginLoss',
    'FractionalCTCLoss',
    'FractionalCustomLoss',
    'FractionalCombinedLoss',
    # Optimizers
    'FractionalOptimizer',
    'FractionalAdam',
    'FractionalSGD',
    'FractionalRMSprop',
    # Fractional GNN Components
    'BaseFractionalGNNLayer',
    'FractionalGraphConv',
    'FractionalGraphAttention',
    'FractionalGraphAttention',
    'FractionalGraphPooling',
    'BaseFractionalGNN',
    'FractionalGCN',
    'FractionalGAT',
    'FractionalGraphSAGE',
    'FractionalGraphUNet',
    'FractionalGNNFactory',
    # Spectral
    'SpectralFractionalDerivative',
    'SpectralFractionalLayer',
    'SpectralFractionalNetwork',
    'BoundedAlphaParameter',
    'spectral_fractional_derivative',
    'create_fractional_layer',
    # Stochastic
    'StochasticFractionalLayer',
    'stochastic_fractional_derivative',
    'create_stochastic_fractional_layer',
    # Probabilistic
    'ProbabilisticFractionalOrder',
    'ProbabilisticFractionalLayer',
    'create_probabilistic_fractional_layer',
    'create_normal_alpha_layer',
    'create_uniform_alpha_layer',
    'create_beta_alpha_layer',
    # MLOps
    'ModelRegistry',
    'ModelMetadata',
    'ModelVersion',
    'DeploymentStatus',
    'DevelopmentWorkflow',
    'ProductionWorkflow',
    'ModelValidator',
    'QualityGate',
    'QualityMetric',
    'QualityThreshold',
]
__version__ = "3.2.0"
__author__ = "Davian R. Chin"
__email__ = "d.r.chin@pgr.reading.ac.uk"
__institution__ = "Department of Biomedical Engineering, University of Reading"
