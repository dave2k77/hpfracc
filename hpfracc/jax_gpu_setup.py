"""
JAX GPU Setup for HPFRACC Library
Automatically configures JAX to use GPU when available with proper conflict resolution.

This module now uses the centralized JAX configuration from core.jax_config
to prevent PJRT plugin conflicts and system-level errors.
"""

import os
import warnings
from typing import Optional, Dict, Any

# Use centralized JAX configuration
try:
    from ..core.jax_config import (
        initialize_jax_once, 
        get_jax_safely, 
        is_jax_available, 
        is_jax_gpu_available
    )
except ImportError:
    # Fallback if core module not available
    def initialize_jax_once(*args, **kwargs):
        return {'available': False, 'gpu_available': False, 'initialized': False}
    def get_jax_safely():
        return None, None
    def is_jax_available():
        return False
    def is_jax_gpu_available():
        return False


def clear_jax_plugins():
    """Clear any existing JAX plugins to prevent conflicts."""
    try:
        # Clear environment variables that might cause conflicts
        env_vars_to_clear = [
            'JAX_PLATFORM_NAME',
            'JAX_ENABLE_XLA', 
            'XLA_FLAGS',
            'CUDA_VISIBLE_DEVICES'
        ]
        
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
                
    except Exception as e:
        warnings.warn(f"Failed to clear JAX environment variables: {e}")


def check_cudnn_compatibility() -> Dict[str, Any]:
    """Check CuDNN compatibility and return status information."""
    try:
        import jaxlib
        
        # Try to get CuDNN version info
        try:
            import ctypes
            cudnn_lib = ctypes.CDLL("libcudnn.so")
            # This is a simplified check - actual version detection is more complex
            cudnn_available = True
        except:
            cudnn_available = False
            
        return {
            'cudnn_available': cudnn_available,
            'jaxlib_version': jaxlib.__version__,
            'warning': 'CuDNN version mismatch detected. Consider upgrading CuDNN to 9.20.0+ (via jax[cuda13] for Blackwell) for optimal performance.'
        }
        
    except Exception as e:
        return {
            'cudnn_available': False,
            'error': str(e)
        }


def setup_jax_gpu_safe() -> bool:
    """
    Set up JAX to use GPU when available with proper conflict resolution.
    
    This function uses the centralized JAX configuration to prevent
    PJRT plugin conflicts and handle system-level errors gracefully.
    
    Returns:
        bool: True if GPU is available and configured, False if using CPU fallback
    """
    # Use centralized initialization (handles all conflicts automatically)
    config = initialize_jax_once()
    
    if not config['available']:
        return False
    
    # Check GPU availability
    gpu_available = is_jax_gpu_available()
    
    if gpu_available:
        try:
            jax, jnp = get_jax_safely()
            if jax is not None:
                devices = jax.devices()
                gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
                if gpu_devices:
                    return True
        except Exception:
            pass
    
    return False


def setup_jax_gpu() -> bool:
    """
    Legacy function for backward compatibility.
    Use setup_jax_gpu_safe() for new code.
    """
    return setup_jax_gpu_safe()


def get_jax_info() -> dict:
    """
    Get comprehensive JAX device and compatibility information.
    
    Returns:
        dict: JAX device, backend, and compatibility information
    """
    try:
        import jax
        devices = jax.devices()
        
        # Get CuDNN compatibility info
        cudnn_info = check_cudnn_compatibility()
        
        return {
            'version': jax.__version__,
            'devices': [str(d) for d in devices],
            'device_count': len(devices),
            'backend': jax.default_backend(),
            'gpu_available': any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices),
            'cudnn_info': cudnn_info,
            'platform': jax.default_backend()
        }
    except Exception as e:
        return {'error': str(e)}


def force_cpu_fallback() -> bool:
    """
    Force JAX to use CPU even if GPU is available.
    Useful for debugging or when GPU causes issues.
    
    Returns:
        bool: True if successfully forced to CPU
    """
    try:
        # Set environment variable to force CPU
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        
        # Clear any cached JAX state
        import jax
        jax.clear_caches()
        
        devices = jax.devices()
        cpu_devices = [d for d in devices if 'cpu' in str(d).lower()]
        
        if cpu_devices:
            print(f"✅ Forced JAX to use CPU: {cpu_devices}")
            return True
        else:
            print("⚠️  Failed to force CPU fallback")
            return False
            
    except Exception as e:
        warnings.warn(f"Failed to force CPU fallback: {e}")
        return False


# Auto-configure JAX on import with safe setup
# Note: This uses centralized initialization which prevents conflicts
_jax_gpu_available = setup_jax_gpu_safe()

# Export the configuration status
JAX_GPU_AVAILABLE = _jax_gpu_available
