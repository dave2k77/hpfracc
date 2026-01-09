import torch
import jax
import numpy as np
import scipy
import numba

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch Device: {torch.cuda.get_device_name(0)}")

print(f"\nJAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

print(f"\nNumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")
print(f"Numba version: {numba.__version__}")
