import numpy as np
import pytest

from hpfracc.ml.tensor_ops import get_tensor_ops, switch_backend
from hpfracc.ml.backends import BackendType


@pytest.mark.parametrize("backend", ["numpy", "torch"])  # 'numpy' maps to NUMBA lane
def test_fft_forward_inverse_numpy(backend, set_seed):
    set_seed(42)

    if backend == "torch":
        pytest.importorskip("torch")
        switch_backend(BackendType.TORCH)
    else:
        switch_backend(BackendType.NUMBA)

    ops = get_tensor_ops()

    x = np.random.randn(16).astype(np.float32)
    xt = ops.create_tensor(x)

    X = ops.fft(xt)
    x_rec = ops.ifft(X)

    x_back = ops.to_numpy(x_rec)
    # Allow small numerical error
    np.testing.assert_allclose(x_back.real, x, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backend", ["numpy", "torch"])  # 'numpy' maps to NUMBA lane
def test_einsum_matrix_multiply(backend, set_seed):
    set_seed(0)

    if backend == "torch":
        pytest.importorskip("torch")
        switch_backend(BackendType.TORCH)
    else:
        switch_backend(BackendType.NUMBA)

    ops = get_tensor_ops()

    A = np.random.randn(3, 4).astype(np.float32)
    B = np.random.randn(4, 2).astype(np.float32)

    At = ops.create_tensor(A)
    Bt = ops.create_tensor(B)

    Ct = ops.einsum("ij,jk->ik", At, Bt)
    C = ops.to_numpy(Ct)

    np.testing.assert_allclose(C, A @ B, rtol=1e-5, atol=1e-5)
