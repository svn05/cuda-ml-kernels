"""Tests for custom CUDA fused Layer Normalization kernel.

Validates forward and backward passes against PyTorch's native
nn.functional.layer_norm implementation.
"""

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.fixture(autouse=True)
def cuda_setup():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def import_kernel():
    from cuda_ml_kernels._C import layernorm_forward, layernorm_backward
    return layernorm_forward, layernorm_backward


class TestLayerNormForward:
    """Test LayerNorm forward pass."""

    @pytest.mark.parametrize("shape,norm_shape", [
        ((8, 64), [64]),
        ((32, 128), [128]),
        ((64, 256), [256]),
        ((16, 512), [512]),
        ((8, 1024), [1024]),
        ((4, 16, 64), [64]),     # 3D input
        ((2, 8, 16, 32), [32]),  # 4D input
    ])
    def test_shapes(self, shape, norm_shape):
        """Forward pass matches torch.nn.functional.layer_norm."""
        layernorm_forward, _ = import_kernel()

        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        normalized_size = norm_shape[0]
        gamma = torch.randn(normalized_size, device="cuda", dtype=torch.float32)
        beta = torch.randn(normalized_size, device="cuda", dtype=torch.float32)

        expected = F.layer_norm(x, norm_shape, gamma, beta, eps=1e-5)
        output, mean, rstd = layernorm_forward(x.contiguous(), gamma, beta, 1e-5)
        output = output.view_as(expected)

        assert torch.allclose(output, expected, atol=1e-4), (
            f"Shape {shape}: max diff = {(output - expected).abs().max().item()}"
        )

    def test_identity_transform(self):
        """With gamma=1, beta=0, output should be zero-mean unit-variance."""
        layernorm_forward, _ = import_kernel()

        x = torch.randn(32, 256, device="cuda", dtype=torch.float32)
        gamma = torch.ones(256, device="cuda", dtype=torch.float32)
        beta = torch.zeros(256, device="cuda", dtype=torch.float32)

        output, mean, rstd = layernorm_forward(x, gamma, beta, 1e-5)

        # Mean should be ~0 along last dim
        out_mean = output.mean(dim=-1)
        assert torch.allclose(out_mean, torch.zeros_like(out_mean), atol=1e-4)

        # Variance should be ~1 along last dim
        out_var = output.var(dim=-1, correction=0)
        assert torch.allclose(out_var, torch.ones_like(out_var), atol=1e-3)

    def test_saved_statistics(self):
        """Saved mean and rstd are correct."""
        layernorm_forward, _ = import_kernel()

        x = torch.randn(16, 128, device="cuda", dtype=torch.float32)
        gamma = torch.ones(128, device="cuda", dtype=torch.float32)
        beta = torch.zeros(128, device="cuda", dtype=torch.float32)

        _, mean, rstd = layernorm_forward(x, gamma, beta, 1e-5)

        expected_mean = x.mean(dim=-1)
        expected_var = x.var(dim=-1, correction=0)
        expected_rstd = torch.rsqrt(expected_var + 1e-5)

        assert torch.allclose(mean, expected_mean, atol=1e-4)
        assert torch.allclose(rstd, expected_rstd, atol=1e-4)


class TestLayerNormBackward:
    """Test LayerNorm backward pass."""

    @pytest.mark.parametrize("shape", [
        (8, 64),
        (32, 128),
        (16, 256),
    ])
    def test_backward_grad_input(self, shape):
        """Backward pass computes correct grad_input."""
        layernorm_forward, layernorm_backward = import_kernel()

        normalized_size = shape[-1]
        x = torch.randn(*shape, device="cuda", dtype=torch.float32, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        gamma = torch.randn(normalized_size, device="cuda", dtype=torch.float32)
        beta = torch.randn(normalized_size, device="cuda", dtype=torch.float32)

        # Custom forward + backward
        output, mean, rstd = layernorm_forward(x.detach(), gamma, beta, 1e-5)
        grad_out = torch.randn_like(output)
        grad_input, grad_gamma, grad_beta = layernorm_backward(
            grad_out, x.detach(), gamma, mean, rstd
        )

        # PyTorch reference
        ref_out = F.layer_norm(x_ref, [normalized_size], gamma, beta, eps=1e-5)
        ref_out.backward(grad_out)

        assert torch.allclose(grad_input, x_ref.grad, atol=1e-3), (
            f"Shape {shape}: max grad_input diff = "
            f"{(grad_input - x_ref.grad).abs().max().item()}"
        )

    def test_backward_grad_gamma_beta(self):
        """Backward pass computes correct grad_gamma and grad_beta."""
        layernorm_forward, layernorm_backward = import_kernel()

        shape = (32, 128)
        normalized_size = 128
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        gamma = torch.randn(normalized_size, device="cuda", dtype=torch.float32, requires_grad=True)
        beta = torch.randn(normalized_size, device="cuda", dtype=torch.float32, requires_grad=True)
        gamma_ref = gamma.detach().clone().requires_grad_(True)
        beta_ref = beta.detach().clone().requires_grad_(True)

        # Custom
        output, mean, rstd = layernorm_forward(x, gamma.detach(), beta.detach(), 1e-5)
        grad_out = torch.randn_like(output)
        _, grad_gamma, grad_beta = layernorm_backward(
            grad_out, x, gamma.detach(), mean, rstd
        )

        # Reference
        ref_out = F.layer_norm(x, [normalized_size], gamma_ref, beta_ref, eps=1e-5)
        ref_out.backward(grad_out)

        assert torch.allclose(grad_gamma, gamma_ref.grad, atol=1e-3), (
            f"grad_gamma max diff = {(grad_gamma - gamma_ref.grad).abs().max().item()}"
        )
        assert torch.allclose(grad_beta, beta_ref.grad, atol=1e-3), (
            f"grad_beta max diff = {(grad_beta - beta_ref.grad).abs().max().item()}"
        )
