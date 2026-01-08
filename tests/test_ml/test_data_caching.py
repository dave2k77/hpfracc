import torch
import time
import unittest
from hpfracc.ml.data import FractionalTensorDataset

class TestDataCaching(unittest.TestCase):
    def test_caching_speedup(self):
        """Test that caching fractional derivatives speeds up access."""
        # Create large tensor
        data_size = 1000
        tensor_len = 500
        x = torch.randn(data_size, tensor_len)
        y = torch.randn(data_size, 1)
        
        # 1. Non-cached dataset
        ds_no_cache = FractionalTensorDataset([x, y], apply_fractional=True, cache_fractional=False)
        
        start = time.time()
        # Access 100 items
        for i in range(100):
            _ = ds_no_cache[i]
        no_cache_time = time.time() - start
        
        # 2. Cached dataset
        # This will incur init time but fast access
        start_init = time.time()
        ds_cached = FractionalTensorDataset([x, y], apply_fractional=True, cache_fractional=True)
        init_time = time.time() - start_init
        
        start = time.time()
        # Access 100 items
        for i in range(100):
            _ = ds_cached[i]
        cache_time = time.time() - start
        
        print(f"\nNon-cached time (100 items): {no_cache_time:.4f}s")
        print(f"Cached init time: {init_time:.4f}s")
        print(f"Cached access time (100 items): {cache_time:.4f}s")
        
        # Cached access should be significantly faster than non-cached access
        self.assertLess(cache_time, no_cache_time * 0.5, "Caching should provide significant speedup")

    def test_caching_correctness(self):
        """Test that cached values match computed values."""
        x = torch.randn(10, 20)
        ds_no_cache = FractionalTensorDataset([x], apply_fractional=True, cache_fractional=False)
        ds_cached = FractionalTensorDataset([x], apply_fractional=True, cache_fractional=True)
        
        val_no_cache, _ = ds_no_cache[0]
        val_cached, _ = ds_cached[0]
        
        # Should be identical
        self.assertTrue(torch.allclose(val_no_cache, val_cached, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
