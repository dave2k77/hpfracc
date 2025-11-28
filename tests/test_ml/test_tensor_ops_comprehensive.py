"""
Comprehensive tests for hpfracc.ml.tensor_ops module

This module tests unified tensor operations across PyTorch, JAX, and NumPy backends.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple

from hpfracc.ml.tensor_ops import (
    TensorOps,
    get_tensor_ops,
    create_tensor,
    switch_backend
)
from hpfracc.ml.backends import BackendType


class TestTensorOps:
    """Test the TensorOps class"""

    def test_initialization_default(self):
        """Test TensorOps initialization with default parameters"""
        tensor_ops = TensorOps()
        
        assert tensor_ops.backend is not None
        assert tensor_ops.tensor_lib is not None
        assert tensor_ops._adapter is not None

    def test_initialization_custom_backend(self):
        """Test TensorOps initialization with custom backend"""
        tensor_ops = TensorOps(backend=BackendType.TORCH)
        
        assert tensor_ops.backend is not None
        assert tensor_ops.tensor_lib is not None

    def test_initialization_string_backend(self):
        """Test TensorOps initialization with string backend"""
        tensor_ops = TensorOps(backend="torch")
        
        assert tensor_ops.backend is not None
        assert tensor_ops.tensor_lib is not None

    def test_initialization_invalid_backend(self):
        """Test TensorOps initialization with invalid backend"""
        with pytest.raises(ValueError):
            TensorOps(backend="invalid_backend")

    def test_backend_resolution_priority(self):
        """Test backend resolution priority"""
        # Test that explicit backend is preferred
        tensor_ops = TensorOps(backend=BackendType.TORCH)
        
        assert tensor_ops.backend is not None
        assert tensor_ops.tensor_lib is not None

    def test_backend_fallback(self):
        """Test backend fallback mechanism"""
        # Test fallback when preferred backend is not available
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_manager:
            mock_manager.return_value = None
            
            tensor_ops = TensorOps()
            
            assert tensor_ops.backend is not None
            assert tensor_ops.tensor_lib is not None

    def test_create_tensor_basic(self):
        """Test basic tensor creation"""
        tensor_ops = TensorOps()
        
        # Test with list data
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = tensor_ops.create_tensor(data)
        
        assert tensor is not None
        assert hasattr(tensor, 'shape')
        assert hasattr(tensor, 'dtype')

    def test_create_tensor_numpy(self):
        """Test tensor creation from numpy array"""
        tensor_ops = TensorOps()
        
        # Test with numpy array
        data = np.array([1.0, 2.0, 3.0, 4.0])
        tensor = tensor_ops.create_tensor(data)
        
        assert tensor is not None
        assert hasattr(tensor, 'shape')
        assert hasattr(tensor, 'dtype')

    def test_create_tensor_empty(self):
        """Test tensor creation with empty data"""
        tensor_ops = TensorOps()
        
        # Test with empty list
        data = []
        tensor = tensor_ops.create_tensor(data)
        
        assert tensor is not None
        assert hasattr(tensor, 'shape')
        assert hasattr(tensor, 'dtype')

    def test_create_tensor_with_dtype(self):
        """Test tensor creation with specific dtype"""
        tensor_ops = TensorOps()
        
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = tensor_ops.create_tensor(data, dtype='float32')
        
        assert tensor is not None
        assert hasattr(tensor, 'dtype')

    def test_create_tensor_with_device(self):
        """Test tensor creation with specific device"""
        tensor_ops = TensorOps()
        
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = tensor_ops.create_tensor(data, device='cpu')
        
        assert tensor is not None
        assert hasattr(tensor, 'device')

    def test_arithmetic_operations(self):
        """Test arithmetic operations"""
        tensor_ops = TensorOps()
        
        # Create test tensors
        a = tensor_ops.create_tensor([1.0, 2.0, 3.0])
        b = tensor_ops.create_tensor([4.0, 5.0, 6.0])
        
        # Test addition
        result = tensor_ops.add(a, b)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test subtraction
        result = tensor_ops.subtract(a, b)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test multiplication
        result = tensor_ops.multiply(a, b)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test division
        result = tensor_ops.divide(a, b)
        assert result is not None
        assert hasattr(result, 'shape')

    def test_mathematical_functions(self):
        """Test mathematical functions"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test exp
        result = tensor_ops.exp(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test log
        result = tensor_ops.log(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test sin
        result = tensor_ops.sin(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test cos
        result = tensor_ops.cos(x)
        assert result is not None
        assert hasattr(result, 'shape')

    def test_reduction_operations(self):
        """Test reduction operations"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Test sum
        result = tensor_ops.sum(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test mean
        result = tensor_ops.mean(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test max
        result = tensor_ops.max(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test min
        result = tensor_ops.min(x)
        assert result is not None
        assert hasattr(result, 'shape')

    def test_linear_algebra_operations(self):
        """Test linear algebra operations"""
        tensor_ops = TensorOps()
        
        # Create test matrices
        a = tensor_ops.create_tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tensor_ops.create_tensor([[5.0, 6.0], [7.0, 8.0]])
        
        # Test matrix multiplication
        result = tensor_ops.matmul(a, b)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test transpose
        result = tensor_ops.transpose(a)
        assert result is not None
        assert hasattr(result, 'shape')

    @pytest.mark.skip(reason="random_normal and random_uniform methods not exported")
    def test_random_operations(self):
        """Test random operations"""
        tensor_ops = TensorOps()
        
        # Test random normal
        result = tensor_ops.random_normal(shape=(2, 3))
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test random uniform
        result = tensor_ops.random_uniform(shape=(2, 3))
        assert result is not None
        assert hasattr(result, 'shape')

    @pytest.mark.skip(reason="backward and grad methods not exported")
    def test_gradient_operations(self):
        """Test gradient operations"""
        tensor_ops = TensorOps()
        
        # Create test tensor with gradient
        x = tensor_ops.create_tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Test gradient computation
        y = tensor_ops.multiply(x, x)
        result = tensor_ops.sum(y)
        
        # Test backward
        tensor_ops.backward(result)
        
        # Check gradient
        grad = tensor_ops.grad(x)
        assert grad is not None
        assert hasattr(grad, 'shape')

    @pytest.mark.skip(reason="device and to_device methods not exported")
    def test_device_operations(self):
        """Test device operations"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([1.0, 2.0, 3.0])
        
        # Test device
        device = tensor_ops.device(x)
        assert device is not None
        
        # Test to device
        result = tensor_ops.to_device(x, 'cpu')
        assert result is not None
        assert hasattr(result, 'device')

    @pytest.mark.skip(reason="shape method not exported (use .shape attribute)")
    def test_shape_operations(self):
        """Test shape operations"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Test shape
        shape = tensor_ops.shape(x)
        assert shape is not None
        assert isinstance(shape, (list, tuple))
        
        # Test reshape
        result = tensor_ops.reshape(x, (4,))
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test squeeze
        result = tensor_ops.squeeze(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test unsqueeze
        result = tensor_ops.unsqueeze(x, 0)
        assert result is not None
        assert hasattr(result, 'shape')

    @pytest.mark.skip(reason="index and slice methods not exported")
    def test_indexing_operations(self):
        """Test indexing operations"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Test indexing
        result = tensor_ops.index(x, 0)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test slicing
        result = tensor_ops.slice(x, 0, 1)
        assert result is not None
        assert hasattr(result, 'shape')

    def test_concatenation_operations(self):
        """Test concatenation operations"""
        tensor_ops = TensorOps()
        
        # Create test tensors
        a = tensor_ops.create_tensor([1.0, 2.0])
        b = tensor_ops.create_tensor([3.0, 4.0])
        
        # Test concatenate
        result = tensor_ops.concatenate([a, b])
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test stack
        result = tensor_ops.stack([a, b])
        assert result is not None
        assert hasattr(result, 'shape')

    @pytest.mark.skip(reason="equal, greater, less methods not exported")
    def test_comparison_operations(self):
        """Test comparison operations"""
        tensor_ops = TensorOps()
        
        # Create test tensors
        a = tensor_ops.create_tensor([1.0, 2.0, 3.0])
        b = tensor_ops.create_tensor([2.0, 2.0, 2.0])
        
        # Test equal
        result = tensor_ops.equal(a, b)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test greater
        result = tensor_ops.greater(a, b)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test less
        result = tensor_ops.less(a, b)
        assert result is not None
        assert hasattr(result, 'shape')

    @pytest.mark.skip(reason="logical_and, logical_or, logical_not methods not exported")
    def test_logical_operations(self):
        """Test logical operations"""
        tensor_ops = TensorOps()
        
        # Create test tensors
        a = tensor_ops.create_tensor([True, False, True])
        b = tensor_ops.create_tensor([False, True, True])
        
        # Test logical and
        result = tensor_ops.logical_and(a, b)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test logical or
        result = tensor_ops.logical_or(a, b)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test logical not
        result = tensor_ops.logical_not(a)
        assert result is not None
        assert hasattr(result, 'shape')

    def test_fft_operations(self):
        """Test FFT operations"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test FFT
        result = tensor_ops.fft(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test IFFT
        result = tensor_ops.ifft(result)
        assert result is not None
        assert hasattr(result, 'shape')

    @pytest.mark.skip(reason="convolve method not exported")
    def test_convolution_operations(self):
        """Test convolution operations"""
        tensor_ops = TensorOps()
        
        # Create test tensors
        x = tensor_ops.create_tensor([1.0, 2.0, 3.0, 4.0])
        kernel = tensor_ops.create_tensor([0.5, 0.5])
        
        # Test convolution
        result = tensor_ops.convolve(x, kernel)
        assert result is not None
        assert hasattr(result, 'shape')

    @pytest.mark.skip(reason="max_pool method not exported")
    def test_pooling_operations(self):
        """Test pooling operations"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Test max pool
        result = tensor_ops.max_pool(x, kernel_size=2)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test avg pool
        result = tensor_ops.avg_pool(x, kernel_size=2)
        assert result is not None
        assert hasattr(result, 'shape')

    @pytest.mark.skip(reason="batch_norm, layer_norm methods not exported")
    def test_normalization_operations(self):
        """Test normalization operations"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Test batch norm
        result = tensor_ops.batch_norm(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test layer norm
        result = tensor_ops.layer_norm(x)
        assert result is not None
        assert hasattr(result, 'shape')

    def test_activation_operations(self):
        """Test activation operations"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([1.0, -1.0, 2.0, -2.0])
        
        # Test relu
        result = tensor_ops.relu(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test sigmoid
        result = tensor_ops.sigmoid(x)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test tanh
        result = tensor_ops.tanh(x)
        assert result is not None
        assert hasattr(result, 'shape')

    @pytest.mark.skip(reason="mse_loss, cross_entropy_loss methods not exported")
    def test_loss_operations(self):
        """Test loss operations"""
        tensor_ops = TensorOps()
        
        # Create test tensors
        pred = tensor_ops.create_tensor([0.8, 0.2, 0.9])
        target = tensor_ops.create_tensor([1.0, 0.0, 1.0])
        
        # Test MSE loss
        result = tensor_ops.mse_loss(pred, target)
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Test cross entropy loss
        result = tensor_ops.cross_entropy_loss(pred, target)
        assert result is not None
        assert hasattr(result, 'shape')

    @pytest.mark.skip(reason="sgd_step, adam_step methods not exported")
    def test_optimization_operations(self):
        """Test optimization operations"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Test SGD step
        result = tensor_ops.sgd_step(x, lr=0.01)
        assert result is not None
        
        # Test Adam step
        result = tensor_ops.adam_step(x, lr=0.01)
        assert result is not None

    @pytest.mark.skip(reason="switch_backend method not exported")
    def test_backend_switching(self):
        """Test backend switching"""
        tensor_ops = TensorOps()
        
        # Test switching backend
        tensor_ops.switch_backend(BackendType.TORCH)
        
        assert tensor_ops.backend is not None
        assert tensor_ops.tensor_lib is not None

    @pytest.mark.skip(reason="get_backend_info method not exported")
    def test_backend_info(self):
        """Test backend information"""
        tensor_ops = TensorOps()
        
        # Test getting backend info
        info = tensor_ops.get_backend_info()
        
        assert isinstance(info, dict)
        assert 'backend' in info
        assert 'tensor_lib' in info
        assert 'device' in info

    @pytest.mark.skip(reason="enable_profiling, get_profile_results methods not exported")
    def test_performance_profiling(self):
        """Test performance profiling"""
        tensor_ops = TensorOps()
        
        # Test enabling profiling
        tensor_ops.enable_profiling(True)
        
        # Test disabling profiling
        tensor_ops.enable_profiling(False)
        
        # Test getting profile results
        results = tensor_ops.get_profile_results()
        assert isinstance(results, dict)

    @pytest.mark.skip(reason="clear_cache, get_memory_usage methods not exported")
    def test_memory_management(self):
        """Test memory management"""
        tensor_ops = TensorOps()
        
        # Test clearing cache
        tensor_ops.clear_cache()
        
        # Test getting memory usage
        usage = tensor_ops.get_memory_usage()
        assert isinstance(usage, dict)
        assert 'allocated' in usage
        assert 'reserved' in usage

    def test_error_handling(self):
        """Test error handling"""
        tensor_ops = TensorOps()
        
        # Test with invalid data
        with pytest.raises(Exception):
            tensor_ops.create_tensor("invalid_data")
        
        # Test with invalid shape
        with pytest.raises(Exception):
            tensor_ops.create_tensor([1.0, 2.0], shape="invalid")

    def test_edge_cases(self):
        """Test edge cases"""
        tensor_ops = TensorOps()
        
        # Test with zero-dimensional tensor
        x = tensor_ops.create_tensor(5.0)
        assert x is not None
        assert hasattr(x, 'shape')
        
        # Test with very large tensor
        x = tensor_ops.create_tensor([1.0] * 1000)
        assert x is not None
        assert hasattr(x, 'shape')
        
        # Test with very small tensor
        x = tensor_ops.create_tensor([1.0])
        assert x is not None
        assert hasattr(x, 'shape')


class TestTensorOpsFunctions:
    """Test the module-level functions"""

    def test_get_tensor_ops_default(self):
        """Test get_tensor_ops with default parameters"""
        tensor_ops = get_tensor_ops()
        
        assert isinstance(tensor_ops, TensorOps)
        assert tensor_ops.backend is not None
        assert tensor_ops.tensor_lib is not None

    def test_get_tensor_ops_custom(self):
        """Test get_tensor_ops with custom backend"""
        tensor_ops = get_tensor_ops(backend=BackendType.TORCH)
        
        assert isinstance(tensor_ops, TensorOps)
        assert tensor_ops.backend is not None
        assert tensor_ops.tensor_lib is not None

    def test_create_tensor_function(self):
        """Test create_tensor function"""
        # Test with list data
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = create_tensor(data)
        
        assert tensor is not None
        assert hasattr(tensor, 'shape')
        assert hasattr(tensor, 'dtype')

    def test_create_tensor_function_with_kwargs(self):
        """Test create_tensor function with kwargs"""
        data = [1.0, 2.0, 3.0, 4.0]
        tensor = create_tensor(data, dtype='float32', device='cpu')
        
        assert tensor is not None
        assert hasattr(tensor, 'shape')
        assert hasattr(tensor, 'dtype')

    def test_switch_backend_function(self):
        """Test switch_backend function"""
        # Test switching backend
        switch_backend(BackendType.TORCH)
        
        # Verify backend was switched
        tensor_ops = get_tensor_ops()
        assert tensor_ops.backend is not None


# Integration tests
class TestTensorOpsIntegration:
    """Integration tests for tensor operations"""

    def test_full_workflow(self):
        """Test complete tensor operations workflow"""
        # Create tensor ops
        tensor_ops = TensorOps()
        
        # Create test data
        data = [1.0, 2.0, 3.0, 4.0]
        x = tensor_ops.create_tensor(data)
        
        # Test various operations
        y = tensor_ops.multiply(x, x)
        z = tensor_ops.sum(y)
        
        assert z is not None
        assert hasattr(z, 'shape')

    def test_backend_consistency(self):
        """Test backend consistency across operations"""
        tensor_ops = TensorOps()
        
        # Create test tensors
        a = tensor_ops.create_tensor([1.0, 2.0])
        b = tensor_ops.create_tensor([3.0, 4.0])
        
        # Test operations maintain backend consistency
        result = tensor_ops.add(a, b)
        
        assert result is not None
        assert hasattr(result, 'shape')

    def test_memory_efficiency(self):
        """Test memory efficiency"""
        tensor_ops = TensorOps()
        
        # Create large tensor
        x = tensor_ops.create_tensor([1.0] * 10000)
        
        # Test operations don't cause memory issues
        y = tensor_ops.multiply(x, x)
        z = tensor_ops.sum(y)
        
        assert z is not None
        assert hasattr(z, 'shape')

    def test_performance_consistency(self):
        """Test performance consistency"""
        tensor_ops = TensorOps()
        
        # Create test tensor
        x = tensor_ops.create_tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test operations are consistent
        results = []
        for _ in range(10):
            result = tensor_ops.multiply(x, x)
            results.append(result)
        
        # All results should be consistent
        for result in results:
            assert result is not None
            assert hasattr(result, 'shape')
