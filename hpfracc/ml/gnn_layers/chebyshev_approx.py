
import torch
from typing import Optional

def chebyshev_spectral_fractional(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
    k: int = 3,
    normalization: str = 'sym',
    lambda_max: float = 2.0
) -> torch.Tensor:
    """
    Compute fractional Laplacian power L^alpha * x using Chebyshev polynomial approximation.
    
    This avoids expensive eigendecomposition O(N^3) by approximating the filter
    f(lambda) = lambda^alpha using Chebyshev polynomials T_k(x).
    Complexity: O(K * E), where E is number of edges.
    
    Args:
        x: Node features [N, D]
        edge_index: Graph connectivity [2, E]
        alpha: Fractional order
        k: Order of Chebyshev approximation
        normalization: Laplacian normalization ('sym' or 'rw')
        lambda_max: Maximum eigenvalue (upper bound needed for scaling)
        
    Returns:
        Filtered features [N, D]
    """
    num_nodes = x.size(0)
    
    # 1. Compute Sparse Laplacian L
    # L = I - D^{-1/2} A D^{-1/2} (for sym)
    if edge_index.size(0) != 2:
         # Handle case where edge_index might be adjacency matrix? 
         # Assuming standard PyG format [2, E]
         raise ValueError("edge_index must be shape [2, E]")
         
    row, col = edge_index
    
    # Compute degree
    deg = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=x.dtype, device=x.device))
    
    # Compute normalized Laplacian coefficients
    # Standard GCN normalization: deg_inv_sqrt[i] * deg_inv_sqrt[j]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # We implement L operation implicitly via sparse matrix-vector multiplication (SpMM)
    # L x = x - D^{-1/2} A D^{-1/2} x
    
    def apply_laplacian(vec: torch.Tensor) -> torch.Tensor:
        # D^{-1/2} x
        norm_vec = vec * deg_inv_sqrt.view(-1, 1)
        
        # A (D^{-1/2} x) -> scatter add
        # source nodes are col, target are row
        msg = norm_vec[col]
        aggr = torch.zeros_like(vec)
        aggr.scatter_add_(0, row.view(-1, 1).expand(-1, vec.size(1)), msg)
        
        # D^{-1/2} (A ...)
        norm_aggr = aggr * deg_inv_sqrt.view(-1, 1)
        
        # L x = x - ...
        return vec - norm_aggr

    # 2. Scale Laplacian to [-1, 1] for Chebyshev
    # L_scaled = 2L / lambda_max - I
    def apply_scaled_laplacian(vec: torch.Tensor) -> torch.Tensor:
        Lx = apply_laplacian(vec)
        return (2.0 / lambda_max) * Lx - vec

    # 3. Chebyshev Recurrence
    # T_0(x) = 1
    # T_1(x) = x
    # T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
    
    # Coefficients c_k for f(lambda) = lambda^alpha approximation
    # We interpret this as filtering: y = sum(c_k * T_k(L_scaled) * x)
    # However, calculating c_k for fractional power x^alpha is non-trivial analytically for the recurrence.
    # Standard approach:
    # Instead of full expansion, we can use a simpler approach for "fractional diffusion":
    # (I + L)^-alpha or e^{-tL}.
    # User requested "Fractional Laplacian Power" L^alpha.
    # We will use the recurrence directly if we had coefficients.
    
    # ALTERNATIVE:
    # Use simpler truncated Taylor expansion of binomial series if L is close to I?
    # No, Chebyshev is requested.
    
    # Let's compute Chebyshev coefficients c_j for function g(y) = ((y+1)*lambda_max/2)^alpha
    # mapping y in [-1, 1] back to lambda in [0, lambda_max]
    
    # We compute coefficients numerically via DCT
    import math
    
    coeffs = []
    N_cheb = k + 1
    for j in range(N_cheb):
        # Nodes for numerical integration / interpolation
        y_node = math.cos(math.pi * (j + 0.5) / N_cheb)
        # Function value
        # y -> lambda: lambda = (y + 1) * lambda_max / 2
        lam_val = (y_node + 1) * lambda_max / 2.0
        func_val = lam_val ** alpha
        coeffs.append(func_val)
        
    # Discrete Cosine Transform to get Chebyshev coefficients
    # c_m = (2/N) * sum_{j=0}^{N-1} f(y_j) * cos(pi * m * (j+0.5) / N)
    cheb_coeffs = []
    for m in range(N_cheb):
        sum_val = 0.0
        for j in range(N_cheb):
            y_j = math.cos(math.pi * (j + 0.5) / N_cheb)
            # T_m(y_j) = cos(m * acos(y_j)) = cos(m * pi * (j+0.5) / N)
            term = coeffs[j] * math.cos(math.pi * m * (j + 0.5) / N_cheb)
            sum_val += term
        c_m = (2.0 / N_cheb) * sum_val
        cheb_coeffs.append(c_m)
        
    # 4. Filter Evaluation via Recurrence
    # y = c_0/2 * T_0(L)x + sum_{m=1}^k c_m T_m(L)x
    # T_0(L)x = x
    # T_1(L)x = L_scaled * x
    
    # T_0 term
    # Note: definition of Series usually c0 T0 + ... but DCT gives standard coeff.
    # Adjust c0? No, standard Type-II DCT matches definition.
    # Actually standard Chebyshev expansion often defined with c0/2. 
    # Let's trust the DCT projection.
    
    # Init
    Tx_prev = x
    Tx_curr = apply_scaled_laplacian(x)
    
    # Result accumulator
    # First term T0:
    # Often standard form is sum_{k=0} c_k T_k - with c0/2??
    # Let's assume standard series: sum c_k T_k - check DCT def carefully.
    # Standard orthogonal projection: c_k = <f, T_k> / <T_k, T_k>
    # <T_0, T_0> = pi, <T_k, T_k> = pi/2.
    # So yes, c_0 usually computed with 1/N but applied as c0/2 if using 2/N formula.
    # Our formula used 2/N for all. So c0 needs /2?
    # No, DCT-II (scipy style) orthogonalizes differently. 
    # Let's stick to simplest approx:
    
    # result = c[0] * T0 + c[1] * T1 ...
    # Wait, DCT-II matches Chebyshev T_k coefficients except c0?
    
    # Let's just use the computed projection directly.
    # Correct handling of c0 implies dividing it by 2 if we used the 2/N prefactor for all terms including 0.
    # (Since norm squared is different for k=0).
    # Yes, T0 norm is Pi, Tk norm is Pi/2. So 2/N formula is correct for k>0. For k=0 it gives 2*<f,1>/Pi = 2*mean.
    # True coef is <f,1>/Pi = mean.
    # So we need to divide c0 by 2.
    
    cheb_coeffs[0] /= 2.0
    
    result = cheb_coeffs[0] * Tx_prev + cheb_coeffs[1] * Tx_curr
    
    for m in range(2, k + 1):
        # T_m = 2 * L_scaled * T_{m-1} - T_{m-2}
        Tx_next = 2.0 * apply_scaled_laplacian(Tx_curr) - Tx_prev
        result = result + cheb_coeffs[m] * Tx_next
        
        # Shift
        Tx_prev = Tx_curr
        Tx_curr = Tx_next
        
    return result
