"""
[DEPRECATED] This module is kept for backward compatibility.
Please use `hpfracc.algorithms.derivatives` for unified interfaces.
"""

from .derivatives import RiemannLiouville as OptimizedRiemannLiouville
from .derivatives import Caputo as OptimizedCaputo
from .derivatives import GrunwaldLetnikov as OptimizedGrunwaldLetnikov
from .base import FractionalOperator

# For backward compatibility with things asking for 'optimized_*' functions
def optimized_riemann_liouville(f, t, alpha, h=None):
    return OptimizedRiemannLiouville(alpha).compute(f, t, h)

def optimized_caputo(f, t, alpha, h=None):
    return OptimizedCaputo(alpha).compute(f, t, h)

def optimized_grunwald_letnikov(f, t, alpha, h=None):
    return OptimizedGrunwaldLetnikov(alpha).compute(f, t, h)

# Classes that were present but maybe not strictly "OptimizedX"
class OptimizedFractionalMethods:
     pass

class ParallelConfig:
    def __init__(self, n_jobs=1, enabled=False, **kwargs):
        self.n_jobs = n_jobs
        self.enabled = enabled

class AdvancedFFTMethods:
    def __init__(self, method="spectral", *args, **kwargs):
        pass # Placeholder

class L1L2Schemes:
    pass

class ParallelLoadBalancer:
    pass

# Aliases for "Parallel" versions (unified handles dispatch/threading eventually)
class ParallelOptimizedRiemannLiouville(OptimizedRiemannLiouville): pass
class ParallelOptimizedCaputo(OptimizedCaputo): pass
class ParallelOptimizedGrunwaldLetnikov(OptimizedGrunwaldLetnikov): pass

class NumbaOptimizer: pass
class NumbaFractionalKernels: pass
class NumbaParallelManager: pass

def benchmark_parallel_vs_serial(*args): pass
def optimize_parallel_parameters(*args): pass
def memory_efficient_caputo(*args): pass
def block_processing_kernel(*args): pass
