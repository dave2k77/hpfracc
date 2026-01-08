"""
Comprehensive tests for hpfracc.ml.gpu_optimization module

This module tests GPU acceleration features including AMP, chunked FFT operations,
and performance profiling for fractional calculus operations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
import time

from hpfracc.ml.gpu_optimization import (
    PerformanceMetrics,
    GPUProfiler,
    ChunkedFFT,
    AMPFractionalEngine,
    GPUOptimizedSpectralEngine,
    GPUOptimizedStochasticSampler,
    gpu_optimization_context,
    benchmark_gpu_optimization,
    create_gpu_optimized_components,
    test_gpu_optimization
)


class TestPerformanceMetrics:
    """Test the PerformanceMetrics dataclass"""

    def test_initialization_default(self):
        """Test PerformanceMetrics initialization with default parameters"""
        metrics = PerformanceMetrics(
            operation="test_op",
            device="cuda:0",
            dtype="torch.float32",
            input_shape=(10, 20),
            execution_time=0.1,
            memory_used=1.5,
            memory_peak=2.0,
            throughput=1000.0,
            timestamp=time.time()
        )
        
        assert metrics.operation == "test_op"
        assert metrics.device == "cuda:0"
        assert metrics.dtype == "torch.float32"
        assert metrics.input_shape == (10, 20)
        assert metrics.execution_time == 0.1
        assert metrics.memory_used == 1.5
        assert metrics.memory_peak == 2.0
        assert metrics.throughput == 1000.0
        assert isinstance(metrics.timestamp, float)

    def test_initialization_custom(self):
        """Test PerformanceMetrics initialization with custom parameters"""
        timestamp = time.time()
        metrics = PerformanceMetrics(
            operation="custom_op",
            device="cpu",
            dtype="torch.float64",
            input_shape=(5, 10, 15),
            execution_time=0.05,
            memory_used=0.8,
            memory_peak=1.2,
            throughput=2000.0,
            timestamp=timestamp
        )
        
        assert metrics.operation == "custom_op"
        assert metrics.device == "cpu"
        assert metrics.dtype == "torch.float64"
        assert metrics.input_shape == (5, 10, 15)
        assert metrics.execution_time == 0.05
        assert metrics.memory_used == 0.8
        assert metrics.memory_peak == 1.2
        assert metrics.throughput == 2000.0
        assert metrics.timestamp == timestamp


class TestGPUProfiler:
    """Test the GPUProfiler class"""

    def test_initialization_default(self):
        """Test GPUProfiler initialization with default parameters"""
        profiler = GPUProfiler()
        
        assert profiler.device == "cuda"
        assert profiler.metrics_history == []
        assert profiler.current_metrics == {}

    def test_initialization_custom(self):
        """Test GPUProfiler initialization with custom parameters"""
        profiler = GPUProfiler(device="cpu")
        
        assert profiler.device == "cpu"
        assert profiler.metrics_history == []
        assert profiler.current_metrics == {}

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.synchronize')
    def test_start_timer_cuda(self, mock_sync, mock_available):
        """Test start_timer with CUDA device"""
        profiler = GPUProfiler(device="cuda")
        
        profiler.start_timer("test_operation")
        
        assert hasattr(profiler, 'start_time')
        assert hasattr(profiler, 'operation')
        assert profiler.operation == "test_operation"
        assert isinstance(profiler.start_time, float)
        mock_sync.assert_called_once()

    @patch('torch.cuda.is_available', return_value=False)
    def test_start_timer_cpu(self, mock_available):
        """Test start_timer with CPU device"""
        profiler = GPUProfiler(device="cpu")
        
        profiler.start_timer("test_operation")
        
        assert hasattr(profiler, 'start_time')
        assert hasattr(profiler, 'operation')
        assert profiler.operation == "test_operation"
        assert isinstance(profiler.start_time, float)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.memory_allocated', return_value=1024**3)  # 1GB
    @patch('torch.cuda.max_memory_allocated', return_value=2*1024**3)  # 2GB
    def test_end_timer_cuda(self, mock_max_mem, mock_mem, mock_sync, mock_available):
        """Test end_timer with CUDA device"""
        profiler = GPUProfiler(device="cuda")
        
        # Start timer
        profiler.start_timer("test_operation")
        time.sleep(0.01)  # Small delay
        
        # Create mock tensors
        input_tensor = MagicMock()
        input_tensor.device = torch.device('cuda:0')
        input_tensor.numel.return_value = 3
        input_tensor.dtype = torch.float32
        input_tensor.shape = (3,)
        
        output_tensor = MagicMock()
        output_tensor.numel.return_value = 3
        
        metrics = profiler.end_timer(input_tensor, output_tensor)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation == "test_operation"
        assert metrics.device == "cuda:0"
        assert metrics.input_shape == (3,)
        assert metrics.execution_time > 0
        assert metrics.memory_used == 1.0  # 1GB in GB
        assert metrics.memory_peak == 2.0  # 2GB in GB
        assert metrics.throughput > 0
        
        # Check that metrics were stored
        assert "test_operation" in profiler.current_metrics
        assert len(profiler.metrics_history) == 1

    @patch('torch.cuda.is_available', return_value=False)
    def test_end_timer_cpu(self, mock_available):
        """Test end_timer with CPU device"""
        profiler = GPUProfiler(device="cpu")
        
        # Start timer
        profiler.start_timer("test_operation")
        time.sleep(0.01)  # Small delay
        
        # Create mock tensors
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output_tensor = torch.tensor([4.0, 5.0, 6.0])
        
        metrics = profiler.end_timer(input_tensor, output_tensor)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation == "test_operation"
        assert metrics.device == "cpu"
        assert metrics.input_shape == (3,)
        assert metrics.execution_time > 0
        assert metrics.memory_used == 0.0  # No CUDA memory
        assert metrics.memory_peak == 0.0  # No CUDA memory
        assert metrics.throughput > 0

    def test_end_timer_no_output(self):
        """Test end_timer without output tensor"""
        profiler = GPUProfiler(device="cpu")
        
        # Start timer
        profiler.start_timer("test_operation")
        time.sleep(0.01)  # Small delay
        
        # Create mock tensor
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        
        metrics = profiler.end_timer(input_tensor)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation == "test_operation"
        assert metrics.input_shape == (3,)
        assert metrics.execution_time > 0
        assert metrics.throughput > 0

    def test_get_summary_empty(self):
        """Test get_summary with no metrics"""
        profiler = GPUProfiler()
        
        summary = profiler.get_summary()
        
        assert summary == {}
        
        assert summary == {}


class TestChunkedFFT:
    """Test the ChunkedFFT class"""

    def test_initialization_default(self):
        """Test ChunkedFFT initialization with default parameters"""
        fft = ChunkedFFT()
        
        assert fft.chunk_size == 1024
        assert fft.overlap == 0.1
        assert fft.window_type == "hann"

    def test_initialization_custom(self):
        """Test ChunkedFFT initialization with custom parameters"""
        fft = ChunkedFFT(chunk_size=2048, overlap=0.2, window_type="blackman")
        
        assert fft.chunk_size == 2048
        assert fft.overlap == 0.2
        assert fft.window_type == "blackman"

    def test_forward_basic(self):
        """Test basic forward FFT operation"""
        fft = ChunkedFFT(chunk_size=4)
        
        # Create test signal
        signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        result = fft.forward(signal)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == signal.shape[0]  # Same length
        assert result.dtype == torch.complex64

    def test_forward_single_chunk(self):
        """Test forward FFT with signal smaller than chunk size"""
        fft = ChunkedFFT(chunk_size=10)
        
        # Create test signal smaller than chunk size
        signal = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        result = fft.forward(signal)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == signal.shape[0]
        assert result.dtype == torch.complex64

    def test_forward_empty_signal(self):
        """Test forward FFT with empty signal"""
        fft = ChunkedFFT()
        
        signal = torch.tensor([])
        
        result = fft.forward(signal)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 0

    def test_inverse_basic(self):
        """Test basic inverse FFT operation"""
        fft = ChunkedFFT(chunk_size=4)
        
        # Create test signal
        signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # Forward transform
        fft_result = fft.forward(signal)
        
        # Inverse transform
        result = fft.inverse(fft_result)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == signal.shape
        assert result.dtype == torch.complex64
        
        # Should be close to original (within numerical precision)
        assert torch.allclose(result.real, signal, atol=1e-5)

    def test_inverse_empty_signal(self):
        """Test inverse FFT with empty signal"""
        fft = ChunkedFFT()
        
        signal = torch.tensor([], dtype=torch.complex64)
        
        result = fft.inverse(signal)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 0

    def test_roundtrip_consistency(self):
        """Test roundtrip consistency of forward and inverse FFT"""
        fft = ChunkedFFT(chunk_size=8)
        
        # Create test signal
        signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # Forward then inverse
        fft_result = fft.forward(signal)
        reconstructed = fft.inverse(fft_result)
        
        # Should be close to original
        assert torch.allclose(reconstructed.real, signal, atol=1e-5)

    def test_different_window_types(self):
        """Test FFT with different window types"""
        signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        window_types = ["hann", "hamming", "blackman"]
        
        for window_type in window_types:
            fft = ChunkedFFT(window_type=window_type)
            result = fft.forward(signal)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape[0] == signal.shape[0]

    def test_different_overlap_values(self):
        """Test FFT with different overlap values"""
        signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        overlap_values = [0.0, 0.1, 0.2, 0.5]
        
        for overlap in overlap_values:
            fft = ChunkedFFT(overlap=overlap)
            result = fft.forward(signal)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape[0] == signal.shape[0]


class TestAMPFractionalEngine:
    """Test the AMPFractionalEngine class"""

    @patch('torch.cuda.is_available', return_value=True)
    def test_initialization_default(self, mock_available):
        """Test AMPFractionalEngine initialization with default parameters"""
        base_engine = MagicMock()
        engine = AMPFractionalEngine(base_engine)
        
        assert engine.base_engine == base_engine
        assert engine.use_amp == True
        assert engine.scaler is not None
        # GradScaler can be in torch.amp or torch.cuda.amp depending on PyTorch version
        assert 'GradScaler' in type(engine.scaler).__name__

    def test_initialization_custom(self):
        """Test AMPFractionalEngine initialization with custom parameters"""
        base_engine = MagicMock()
        engine = AMPFractionalEngine(base_engine, use_amp=False)
        
        assert engine.base_engine == base_engine
        assert engine.use_amp == False
        assert engine.scaler is None

    def test_forward_with_amp(self):
        """Test forward pass with AMP enabled"""
        base_engine = MagicMock()
        engine = AMPFractionalEngine(base_engine, use_amp=True)
        
        # Create test input
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        
        # Configure mock
        base_engine.forward.return_value = x * 2
        
        result = engine.forward(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        base_engine.forward.assert_called_once()

    def test_forward_without_amp(self):
        """Test forward pass with AMP disabled"""
        base_engine = MagicMock()
        engine = AMPFractionalEngine(base_engine, use_amp=False)
        
        # Create test input
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        
        # Configure mock
        base_engine.forward.return_value = x * 2
        
        result = engine.forward(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        base_engine.forward.assert_called_once()

    def test_forward_empty_input(self):
        """Test forward pass with empty input"""
        base_engine = MagicMock()
        engine = AMPFractionalEngine(base_engine)
        
        x = torch.tensor([])
        
        # Configure mock
        base_engine.forward.return_value = x
        
        result = engine.forward(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_backward_with_amp(self):
        """Test backward pass with AMP enabled"""
        base_engine = MagicMock()
        engine = AMPFractionalEngine(base_engine, use_amp=True)
        
        # Create test input
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, requires_grad=True)
        
        # Configure mock
        base_engine.forward.return_value = x * 2
        
        result = engine.forward(x, alpha=0.5)
        
        # Compute loss and backward
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        base_engine.forward.assert_called_once()

    def test_backward_without_amp(self):
        """Test backward pass with AMP disabled"""
        base_engine = MagicMock()
        engine = AMPFractionalEngine(base_engine, use_amp=False)
        
        # Create test input
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, requires_grad=True)
        
        # Configure mock
        base_engine.forward.return_value = x * 2
        
        result = engine.forward(x, alpha=0.5)
        
        # Compute loss and backward
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        base_engine.forward.assert_called_once()

    def test_update_scaler(self):
        """Test scaler update"""
        base_engine = MagicMock()
        engine = AMPFractionalEngine(base_engine, use_amp=True)
        
        # Mock scaler
        mock_scaler = Mock()
        engine.scaler = mock_scaler
        
        engine.update_scaler()
        
        mock_scaler.update.assert_called_once()

    def test_get_scaler_state(self):
        """Test getting scaler state"""
        base_engine = MagicMock()
        engine = AMPFractionalEngine(base_engine, use_amp=True)
        
        # Mock scaler
        mock_scaler = Mock()
        mock_scaler.get_scale.return_value = 2.0
        engine.scaler = mock_scaler
        
        state = engine.get_scaler_state()
        
        assert state['scale'] == 2.0


class TestGPUOptimizedSpectralEngine:
    """Test the GPUOptimizedSpectralEngine class"""

    def test_initialization_default(self):
        """Test GPUOptimizedSpectralEngine initialization with default parameters"""
        engine = GPUOptimizedSpectralEngine()
        
        assert engine.use_amp == True
        assert engine.chunk_size == 1024
        assert engine.chunked_fft is not None
        assert isinstance(engine.chunked_fft, ChunkedFFT)

    def test_initialization_custom(self):
        """Test GPUOptimizedSpectralEngine initialization with custom parameters"""
        engine = GPUOptimizedSpectralEngine(use_amp=False, chunk_size=2048)
        
        assert engine.use_amp == False
        assert engine.chunk_size == 2048
        assert engine.chunked_fft is not None

    def test_spectral_derivative_basic(self):
        """Test basic spectral derivative computation"""
        engine = GPUOptimizedSpectralEngine()
        
        # Create test signal
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        result = engine.spectral_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_spectral_derivative_empty_input(self):
        """Test spectral derivative with empty input"""
        engine = GPUOptimizedSpectralEngine()
        
        x = torch.tensor([])
        
        result = engine.spectral_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_spectral_derivative_different_alpha(self):
        """Test spectral derivative with different alpha values"""
        engine = GPUOptimizedSpectralEngine()
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        alpha_values = [0.1, 0.5, 1.0, 1.5]
        
        for alpha in alpha_values:
            result = engine.spectral_derivative(x, alpha=alpha)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_spectral_integral_basic(self):
        """Test basic spectral integral computation"""
        engine = GPUOptimizedSpectralEngine()
        
        # Create test signal
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        result = engine.spectral_integral(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_spectral_integral_empty_input(self):
        """Test spectral integral with empty input"""
        engine = GPUOptimizedSpectralEngine()
        
        x = torch.tensor([])
        
        result = engine.spectral_integral(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_spectral_transform_basic(self):
        """Test basic spectral transform computation"""
        engine = GPUOptimizedSpectralEngine()
        
        # Create test signal
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        result = engine.spectral_transform(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.complex64

    def test_spectral_transform_empty_input(self):
        """Test spectral transform with empty input"""
        engine = GPUOptimizedSpectralEngine()
        
        x = torch.tensor([])
        
        result = engine.spectral_transform(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_gpu_fallback(self):
        """Test GPU fallback when GPU is not available"""
        engine = GPUOptimizedSpectralEngine(use_amp=True)
        
        # Mock GPU unavailability
        with patch('torch.cuda.is_available', return_value=False):
            x = torch.tensor([1.0, 2.0, 3.0, 4.0])
            
            result = engine.spectral_derivative(x, alpha=0.5)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_performance_profiling(self):
        """Test performance profiling integration"""
        engine = GPUOptimizedSpectralEngine()
        
        # Create test signal
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # Test with profiling
        result = engine.spectral_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Check that profiler was used
        assert engine.profiler is not None


class TestGPUOptimizedStochasticSampler:
    """Test the GPUOptimizedStochasticSampler class"""

    def test_initialization_default(self):
        """Test GPUOptimizedStochasticSampler initialization with default parameters"""
        base_sampler = MagicMock()
        sampler = GPUOptimizedStochasticSampler(base_sampler)
        
        assert sampler.base_sampler == base_sampler
        assert sampler.use_amp == True
        assert sampler.batch_size == 1024

    def test_initialization_custom(self):
        """Test GPUOptimizedStochasticSampler initialization with custom parameters"""
        base_sampler = MagicMock()
        sampler = GPUOptimizedStochasticSampler(
            base_sampler,
            use_amp=False,
            batch_size=512
        )
        
        assert sampler.base_sampler == base_sampler
        assert sampler.use_amp == False
        assert sampler.batch_size == 512

    def test_sample_indices_basic(self):
        """Test basic sample_indices operation"""
        base_sampler = MagicMock()
        base_sampler.sample_indices.return_value = torch.tensor([0, 1, 2, 3, 4])
        sampler = GPUOptimizedStochasticSampler(base_sampler)
        
        result = sampler.sample_indices(n=100, k=5)
        
        assert isinstance(result, torch.Tensor)
        base_sampler.sample_indices.assert_called()

    def test_sample_indices_with_amp_disabled(self):
        """Test sample_indices with AMP disabled"""
        base_sampler = MagicMock()
        base_sampler.sample_indices.return_value = torch.tensor([0, 2, 4, 6, 8])
        sampler = GPUOptimizedStochasticSampler(base_sampler, use_amp=False)
        
        result = sampler.sample_indices(n=50, k=5)
        
        assert isinstance(result, torch.Tensor)
        base_sampler.sample_indices.assert_called_with(50, 5)

    def test_get_performance_summary(self):
        """Test performance summary retrieval"""
        base_sampler = MagicMock()
        sampler = GPUOptimizedStochasticSampler(base_sampler)
        
        summary = sampler.get_performance_summary()
        
        assert isinstance(summary, dict)



class TestGPUOptimizationContext:
    """Test the gpu_optimization_context function"""

    def test_context_default(self):
        """Test GPU optimization context with default parameters"""
        with gpu_optimization_context() as context:
            assert context['use_amp'] == True
            assert context['dtype'] == torch.float16

    def test_context_custom(self):
        """Test GPU optimization context with custom parameters"""
        with gpu_optimization_context(use_amp=False, dtype=torch.float32) as context:
            assert context['use_amp'] == False
            assert context['dtype'] == torch.float32

    def test_context_within_context(self):
        """Test nested GPU optimization contexts"""
        with gpu_optimization_context(use_amp=True, dtype=torch.float16) as outer:
            with gpu_optimization_context(use_amp=False, dtype=torch.float32) as inner:
                assert outer['use_amp'] == True
                assert outer['dtype'] == torch.float16
                assert inner['use_amp'] == False
                assert inner['dtype'] == torch.float32


class TestBenchmarkGPUOptimization:
    """Test the benchmark_gpu_optimization function"""

    @patch('torch.cuda.is_available', return_value=True)
    def test_benchmark_with_gpu(self, mock_available):
        """Test benchmark with GPU available"""
        results = benchmark_gpu_optimization()
        
        assert isinstance(results, dict)
        assert 'gpu_available' in results
        assert results['gpu_available'] == True
        assert 'benchmarks' in results

    @patch('torch.cuda.is_available', return_value=False)
    def test_benchmark_without_gpu(self, mock_available):
        """Test benchmark without GPU"""
        results = benchmark_gpu_optimization()
        
        assert isinstance(results, dict)
        assert 'gpu_available' in results
        assert results['gpu_available'] == False
        assert 'benchmarks' in results

    def test_benchmark_results_structure(self):
        """Test benchmark results structure"""
        results = benchmark_gpu_optimization()
        
        assert isinstance(results, dict)
        assert 'gpu_available' in results
        assert 'benchmarks' in results
        assert 'summary' in results
        
        benchmarks = results['benchmarks']
        assert isinstance(benchmarks, dict)
        
        # Check that common operations are benchmarked
        expected_ops = ['fft', 'convolution', 'matrix_multiply']
        for op in expected_ops:
            if op in benchmarks:
                assert isinstance(benchmarks[op], dict)
                assert 'execution_time' in benchmarks[op]
                assert 'memory_used' in benchmarks[op]


class TestCreateGPUOptimizedComponents:
    """Test the create_gpu_optimized_components function"""

    def test_create_components_default(self):
        """Test creating components with default parameters"""
        components = create_gpu_optimized_components()
        
        assert isinstance(components, dict)
        assert 'amp_engine' in components
        assert 'spectral_engine' in components
        assert 'stochastic_sampler' in components
        assert 'profiler' in components
        
        assert isinstance(components['amp_engine'], AMPFractionalEngine)
        assert isinstance(components['spectral_engine'], GPUOptimizedSpectralEngine)
        assert isinstance(components['stochastic_sampler'], GPUOptimizedStochasticSampler)
        assert isinstance(components['profiler'], GPUProfiler)

    def test_create_components_custom(self):
        """Test creating components with custom parameters"""
        components = create_gpu_optimized_components(
            use_amp=False,
            chunk_size=2048
        )
        
        assert isinstance(components, dict)
        assert 'amp_engine' in components
        assert 'spectral_engine' in components
        assert 'stochastic_sampler' in components
        assert 'profiler' in components
        
        assert components['amp_engine'].use_amp == False
        assert components['spectral_engine'].use_amp == False
        assert components['spectral_engine'].chunk_size == 2048

    def test_create_components_with_profiler(self):
        """Test creating components with profiler"""
        components = create_gpu_optimized_components(enable_profiling=True)
        
        assert isinstance(components, dict)
        assert 'profiler' in components
        assert isinstance(components['profiler'], GPUProfiler)


class TestTestGPUOptimization:
    """Test the test_gpu_optimization function"""

    def test_gpu_optimization_test(self):
        """Test the GPU optimization test function"""
        results = test_gpu_optimization()
        
        assert isinstance(results, dict)
        assert 'test_passed' in results
        assert 'test_results' in results
        assert 'performance_metrics' in results
        
        assert isinstance(results['test_passed'], bool)
        assert isinstance(results['test_results'], dict)
        assert isinstance(results['performance_metrics'], dict)

    def test_gpu_optimization_test_with_custom_params(self):
        """Test GPU optimization test with custom parameters"""
        results = test_gpu_optimization(
            test_size=100,
            num_iterations=5,
            use_amp=True
        )
        
        assert isinstance(results, dict)
        assert 'test_passed' in results
        assert 'test_results' in results
        assert 'performance_metrics' in results

    def test_gpu_optimization_test_performance_metrics(self):
        """Test GPU optimization test performance metrics"""
        results = test_gpu_optimization()
        
        performance_metrics = results['performance_metrics']
        
        assert isinstance(performance_metrics, dict)
        
        # Check that performance metrics contain expected keys
        expected_keys = ['execution_time', 'memory_used', 'throughput']
        for key in expected_keys:
            if key in performance_metrics:
                assert isinstance(performance_metrics[key], (int, float))


# Integration tests
class TestGPUOptimizationIntegration:
    """Integration tests for GPU optimization module"""

    def test_full_gpu_optimization_workflow(self):
        """Test complete GPU optimization workflow"""
        # Create optimized components
        components = create_gpu_optimized_components(use_amp=True)
        
        amp_engine = components['amp_engine']
        spectral_engine = components['spectral_engine']
        stochastic_sampler = components['stochastic_sampler']
        profiler = components['profiler']
        
        # Test data
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # Test AMP engine
        # Test AMP engine
        # MockBaseEngine returns x * alpha
        amp_result = amp_engine.forward(x, alpha=0.5)
        assert isinstance(amp_result, torch.Tensor)
        assert amp_result.shape == x.shape
        
        # Test spectral engine
        spectral_result = spectral_engine.spectral_derivative(x, alpha=0.5)
        assert isinstance(spectral_result, torch.Tensor)
        assert spectral_result.shape == x.shape
        
        # Test stochastic sampler
        mu = torch.tensor([0.0, 1.0, 2.0])
        sigma = torch.tensor([1.0, 0.5, 1.5])
        
        samples = stochastic_sampler.sample(mu, sigma, num_samples=100)
        assert isinstance(samples, torch.Tensor)
        assert samples.shape[0] == 100
        assert samples.shape[1] == 3
        
        # Test profiler
        profiler.start_timer("test_operation")
        time.sleep(0.01)
        metrics = profiler.end_timer(x)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation == "test_operation"

    def test_gpu_fallback_workflow(self):
        """Test GPU fallback workflow"""
        # Mock GPU unavailability
        with patch('torch.cuda.is_available', return_value=False):
            components = create_gpu_optimized_components(use_amp=True)
            
            spectral_engine = components['spectral_engine']
            stochastic_sampler = components['stochastic_sampler']
            
            # Test that components still work without GPU
            x = torch.tensor([1.0, 2.0, 3.0, 4.0])
            
            spectral_result = spectral_engine.spectral_derivative(x, alpha=0.5)
            assert isinstance(spectral_result, torch.Tensor)
            
            mu = torch.tensor([0.0, 1.0])
            sigma = torch.tensor([1.0, 0.5])
            
            samples = stochastic_sampler.sample(mu, sigma, num_samples=50)
            assert isinstance(samples, torch.Tensor)

    def test_performance_profiling_integration(self):
        """Test performance profiling integration"""
        components = create_gpu_optimized_components(enable_profiling=True)
        
        profiler = components['profiler']
        spectral_engine = components['spectral_engine']
        
        # Enable profiling on spectral engine
        spectral_engine.profiler = profiler
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # Run operation with profiling
        result = spectral_engine.spectral_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        
        # Check that profiler collected metrics
        summary = profiler.get_summary()
        assert isinstance(summary, dict)

    def test_amp_integration_with_spectral_engine(self):
        """Test AMP integration with spectral engine"""
        # Mock base engine
        base_engine = MagicMock()
        
        # Configure mock behavior
        base_engine.forward.side_effect = lambda x, alpha, **kwargs: x * 2
        
        amp_engine = AMPFractionalEngine(base_engine, use_amp=True)
        spectral_engine = GPUOptimizedSpectralEngine()
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, requires_grad=True)
        
        # Test AMP with spectral operations
        amp_result = amp_engine.forward(x, alpha=0.5)
        spectral_result = spectral_engine.spectral_derivative(x, alpha=0.5)
            
        assert isinstance(amp_result, torch.Tensor)
        assert isinstance(spectral_result, torch.Tensor)
        
        # Test gradient computation
        loss = amp_result.sum() + spectral_result.sum()
        loss.backward()
        
        assert x.grad is not None
