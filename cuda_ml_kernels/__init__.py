"""CUDA ML Kernels — GPU-accelerated ML operations with PyTorch bindings.

Provides custom CUDA implementations of:
- Softmax (online algorithm with warp-level reductions)
- Matrix multiplication (shared memory tiling with double buffering)
- Layer normalization (fused Welford's algorithm)
- Scaled dot-product attention (FlashAttention-inspired tiling)

Usage:
    import cuda_ml_kernels as cmk

    out = cmk.softmax(x)
    out = cmk.matmul(a, b)
    out = cmk.layernorm(x, weight, bias)
    out = cmk.attention(q, k, v)
"""

from cuda_ml_kernels.functional import softmax, matmul, layernorm, attention

__version__ = "0.1.0"
__all__ = ["softmax", "matmul", "layernorm", "attention"]
