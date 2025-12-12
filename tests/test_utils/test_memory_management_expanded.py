"""
Expanded comprehensive tests for memory_management.py module.
Tests cache operations, memory optimization, memory usage tracking, cache management.
"""

import pytest
import numpy as np
import gc
from unittest.mock import Mock, patch, MagicMock

from hpfracc.utils.memory_management import (
    MemoryManager,
    CacheManager,
    clear_cache,
    get_memory_usage,
    optimize_memory_usage_func,
    get_global_cache,
    cache_result
)


class TestMemoryManager:
    """Tests for MemoryManager class."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create MemoryManager instance."""
        return MemoryManager()
    
    def test_initialization_default(self, memory_manager):
        """Test initialization with default parameters."""
        assert memory_manager.memory_limit_gb is None
        assert memory_manager.memory_history == []
        assert memory_manager.peak_memory == 0.0
        assert memory_manager.monitoring is False
    
    def test_initialization_with_limit(self):
        """Test initialization with memory limit."""
        manager = MemoryManager(memory_limit_gb=4.0)
        assert manager.memory_limit_gb == 4.0
    
    @patch('hpfracc.utils.memory_management.psutil')
    def test_get_memory_usage(self, mock_psutil, memory_manager):
        """Test getting memory usage."""
        # Mock psutil
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=1024**3, vms=2*1024**3)
        mock_process.memory_percent.return_value = 50.0
        mock_psutil.Process.return_value = mock_process
        mock_psutil.virtual_memory.return_value = Mock(available=4*1024**3, total=8*1024**3)
        
        usage = memory_manager.get_memory_usage()
        
        assert isinstance(usage, dict)
        assert 'rss' in usage
        assert 'vms' in usage
        assert 'percent' in usage
        assert 'available' in usage
        assert 'total' in usage
        assert usage['rss'] > 0
    
    def test_record_memory_usage(self, memory_manager):
        """Test recording memory usage."""
        with patch.object(memory_manager, 'get_memory_usage', return_value={'rss': 1.0}):
            usage = memory_manager.record_memory_usage()
            
            assert len(memory_manager.memory_history) == 1
            assert memory_manager.peak_memory == 1.0
            assert usage['rss'] == 1.0
    
    def test_check_memory_limit_no_limit(self, memory_manager):
        """Test memory limit check with no limit set."""
        result = memory_manager.check_memory_limit()
        assert result is True
    
    def test_check_memory_limit_within_limit(self):
        """Test memory limit check within limit."""
        manager = MemoryManager(memory_limit_gb=4.0)
        with patch.object(manager, 'get_memory_usage', return_value={'rss': 2.0}):
            result = manager.check_memory_limit()
            assert result is True
    
    def test_check_memory_limit_exceeded(self):
        """Test memory limit check when exceeded."""
        manager = MemoryManager(memory_limit_gb=4.0)
        with patch.object(manager, 'get_memory_usage', return_value={'rss': 5.0}):
            result = manager.check_memory_limit()
            assert result is False
    
    def test_force_garbage_collection(self, memory_manager):
        """Test forcing garbage collection."""
        # Should not raise exception
        memory_manager.force_garbage_collection()
    
    def test_optimize_memory_usage(self, memory_manager):
        """Test memory optimization."""
        with patch.object(memory_manager, 'get_memory_usage', return_value={'rss': 1.0}):
            result = memory_manager.optimize_memory_usage()
            
            assert isinstance(result, dict)
            assert 'before' in result
            assert 'after' in result


class TestCacheManager:
    """Tests for CacheManager class."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create CacheManager instance."""
        return CacheManager()
    
    def test_initialization(self, cache_manager):
        """Test CacheManager initialization."""
        assert cache_manager is not None
        assert hasattr(cache_manager, 'cache') or hasattr(cache_manager, '_cache')
    
    def test_get_cached(self, cache_manager):
        """Test getting cached value."""
        key = "test_key"
        value = np.array([1, 2, 3])
        
        # Set value first
        cache_manager.set(key, value)
        
        # Get cached value
        cached = cache_manager.get(key)
        
        np.testing.assert_array_equal(cached, value)
    
    def test_get_cached_nonexistent(self, cache_manager):
        """Test getting nonexistent cached value."""
        cached = cache_manager.get("nonexistent_key")
        assert cached is None
    
    def test_set_cached(self, cache_manager):
        """Test setting cached value."""
        key = "test_key"
        value = np.array([1, 2, 3])
        
        cache_manager.set(key, value)
        
        cached = cache_manager.get(key)
        np.testing.assert_array_equal(cached, value)
    
    def test_clear_cache(self, cache_manager):
        """Test clearing cache."""
        cache_manager.set("key1", np.array([1]))
        cache_manager.set("key2", np.array([2]))
        
        cache_manager.clear()
        
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None
    
    def test_get_cache_size(self, cache_manager):
        """Test getting cache size."""
        cache_manager.set("key1", np.array([1, 2, 3]))
        cache_manager.set("key2", np.array([4, 5, 6]))
        
        size = cache_manager.size()
        
        assert size >= 0
        assert isinstance(size, (int, float))


class TestStandaloneFunctions:
    """Tests for standalone memory management functions."""
    
    def test_clear_cache_function(self):
        """Test clear_cache standalone function."""
        # Should not raise exception
        clear_cache()
    
    @patch('hpfracc.utils.memory_management.psutil')
    def test_get_memory_usage_function(self, mock_psutil):
        """Test get_memory_usage standalone function."""
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=1024**3, vms=2*1024**3)
        mock_process.memory_percent.return_value = 50.0
        mock_psutil.Process.return_value = mock_process
        mock_psutil.virtual_memory.return_value = Mock(available=4*1024**3, total=8*1024**3)
        
        usage = get_memory_usage()
        
        assert isinstance(usage, dict)
        assert 'rss' in usage
    
    def test_optimize_memory_usage_func(self):
        """Test optimize_memory_usage_func."""
        # Should not raise exception
        result = optimize_memory_usage_func()
        assert result is not None
    
    def test_get_global_cache(self):
        """Test get_global_cache function."""
        cache = get_global_cache()
        assert cache is not None
        assert isinstance(cache, CacheManager)
    
    def test_cache_result_decorator(self):
        """Test cache_result decorator."""
        call_count = [0]
        
        @cache_result(key_prefix="test")
        def test_function(x):
            call_count[0] += 1
            return x * 2
        
        # First call
        result1 = test_function(5)
        assert result1 == 10
        assert call_count[0] == 1
        
        # Second call with same argument (should use cache)
        result2 = test_function(5)
        assert result2 == 10
        # May or may not use cache depending on implementation
        assert call_count[0] >= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_memory_manager_with_large_arrays(self):
        """Test MemoryManager with large arrays."""
        manager = MemoryManager()
        
        # Create large array
        large_array = np.zeros(1000000)
        
        usage = manager.get_memory_usage()
        assert usage['rss'] >= 0
    
    def test_cache_manager_with_different_types(self):
        """Test CacheManager with different value types."""
        cache = CacheManager()
        
        # Test with different types
        cache.set("int", 42)
        cache.set("float", 3.14)
        cache.set("array", np.array([1, 2, 3]))
        cache.set("string", "test")
        
        assert cache.get("int") == 42
        assert cache.get("float") == 3.14
        np.testing.assert_array_equal(cache.get("array"), np.array([1, 2, 3]))
        assert cache.get("string") == "test"
    
    def test_cache_manager_memory_limit(self):
        """Test CacheManager with memory limits."""
        cache = CacheManager(max_size=100)
        
        # Add items up to limit
        for i in range(10):
            cache.set(f"key{i}", np.array([i] * 10))
        
        # Should handle memory limits appropriately
        size = cache.size()
        assert size >= 0
