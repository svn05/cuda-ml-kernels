"""torch.autograd.Function wrappers for CUDA ML kernels.

Each Function class implements forward() and backward() to integrate
the custom CUDA kernels with PyTorch's autograd engine, enabling
automatic differentiation through the custom ops.
"""

import torch
from torch.autograd import Function


class CUDASoftmax(Function):
    """Autograd wrapper for custom CUDA softmax."""

    @staticmethod
    def forward(ctx, input):
        from cuda_ml_kernels._C import softmax_forward
        output = softmax_forward(input.contiguous())
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from cuda_ml_kernels._C import softmax_backward
        (output,) = ctx.saved_tensors
        grad_input = softmax_backward(grad_output.contiguous(), output)
        return grad_input


class CUDALayerNorm(Function):
    """Autograd wrapper for custom CUDA fused Layer Normalization."""

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-5):
        from cuda_ml_kernels._C import layernorm_forward
        output, mean, rstd = layernorm_forward(
            input.contiguous(), gamma.contiguous(), beta.contiguous(), eps
        )
        ctx.save_for_backward(input, gamma, mean, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from cuda_ml_kernels._C import layernorm_backward
        input, gamma, mean, rstd = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = layernorm_backward(
            grad_output.contiguous(), input, gamma, mean, rstd
        )
        return grad_input, grad_gamma, grad_beta, None  # None for eps


class CUDAAttention(Function):
    """Autograd wrapper for custom CUDA fused attention (forward only).

    Note: backward pass is not implemented for the fused attention kernel.
    Use torch.autograd.gradcheck with the standard attention as a fallback
    if gradients are needed.
    """

    @staticmethod
    def forward(ctx, Q, K, V, causal=False):
        from cuda_ml_kernels._C import attention_forward
        output = attention_forward(
            Q.contiguous(), K.contiguous(), V.contiguous(), causal
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "Backward pass for fused attention is not implemented. "
            "Use cuda_ml_kernels.functional.attention with autograd_fallback=True "
            "for gradient computation."
        )
