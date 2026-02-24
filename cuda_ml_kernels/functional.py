"""User-facing functional API for CUDA ML kernels.

Provides drop-in replacements for standard PyTorch operations:
    - softmax(x) → custom CUDA softmax with online algorithm
    - matmul(a, b) → tiled GEMM with shared memory
    - layernorm(x, weight, bias) → fused LayerNorm with Welford's
    - attention(q, k, v) → FlashAttention-inspired fused attention

Each function validates inputs and falls back to PyTorch native
implementations on CPU tensors.
"""

import torch
import math
from cuda_ml_kernels.ops import CUDASoftmax, CUDALayerNorm, CUDAAttention


def softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with custom CUDA kernel.

    Uses online algorithm with warp-level reductions for numerical
    stability and efficiency. Falls back to PyTorch for non-CUDA tensors.

    Args:
        input: Input tensor (any shape).
        dim: Dimension to compute softmax over. Must be -1 (last dim)
             for the custom kernel; other dims fall back to PyTorch.

    Returns:
        Softmax output tensor (same shape as input).
    """
    if not input.is_cuda or dim != -1:
        return torch.nn.functional.softmax(input, dim=dim)

    return CUDASoftmax.apply(input)


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication with custom CUDA tiled kernel.

    Uses shared memory tiling (TILE_SIZE=32) to reduce global memory
    bandwidth. Falls back to torch.mm for non-CUDA or non-2D tensors.

    Args:
        A: Left matrix (M, K).
        B: Right matrix (K, N).

    Returns:
        Product matrix (M, N).
    """
    if not A.is_cuda or not B.is_cuda or A.dim() != 2 or B.dim() != 2:
        return torch.mm(A, B) if A.dim() == 2 else torch.matmul(A, B)

    from cuda_ml_kernels._C import matmul as _matmul
    return _matmul(A.contiguous(), B.contiguous())


def layernorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Layer normalization with custom CUDA fused kernel.

    Fuses mean/variance computation (Welford's algorithm) with the
    affine transform (gamma * x_hat + beta) in a single kernel pass.

    Args:
        input: Input tensor (..., normalized_size).
        weight: Gamma parameter (normalized_size,).
        bias: Beta parameter (normalized_size,).
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor (same shape as input).
    """
    if not input.is_cuda:
        return torch.nn.functional.layer_norm(
            input, [input.size(-1)], weight, bias, eps
        )

    return CUDALayerNorm.apply(input, weight, bias, eps)


def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention with custom CUDA fused kernel.

    Uses FlashAttention-inspired tiling to avoid materializing the
    full N×N attention matrix. Supports optional causal masking.

    Args:
        Q: Query tensor (batch, seq_len, head_dim).
        K: Key tensor (batch, seq_len, head_dim).
        V: Value tensor (batch, seq_len, head_dim).
        causal: If True, apply causal (lower-triangular) masking.

    Returns:
        Attention output tensor (batch, seq_len, head_dim).
    """
    if not Q.is_cuda or Q.dim() != 3:
        # Fallback to standard attention
        d_k = Q.size(-1)
        scale = 1.0 / math.sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if causal:
            seq_len = Q.size(-2)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
            scores.masked_fill_(mask, float("-inf"))
        weights = torch.nn.functional.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    return CUDAAttention.apply(Q, K, V, causal)
