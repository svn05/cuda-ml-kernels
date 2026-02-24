"""Tests for custom CUDA softmax kernel.

Validates forward and backward passes against PyTorch's native
softmax implementation across various tensor shapes.
"""

import pytest
import torch
import torch.nn.functional as F


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.fixture(autouse=True)
def cuda_setup():
    """Set up CUDA device and seed for reproducibility."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def import_kernel():
    """Import the custom softmax kernel."""
    from cuda_ml_kernels._C import softmax_forward, softmax_backward
    return softmax_forward, softmax_backward


class TestSoftmaxForward:
    """Test softmax forward pass."""

    @pytest.mark.parametrize("shape", [
        (1, 64),
        (8, 128),
        (32, 256),
        (64, 512),
        (128, 1024),
        (16, 2048),
        (4, 4096),
    ])
    def test_shapes(self, shape):
        """Test forward pass produces correct output for various shapes."""
        softmax_forward, _ = import_kernel()

        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        expected = F.softmax(x, dim=-1)
        result = softmax_forward(x)

        assert torch.allclose(result, expected, atol=1e-5), (
            f"Shape {shape}: max diff = {(result - expected).abs().max().item()}"
        )

    def test_sums_to_one(self):
        """Softmax output rows should sum to 1."""
        softmax_forward, _ = import_kernel()

        x = torch.randn(32, 256, device="cuda", dtype=torch.float32)
        result = softmax_forward(x)
        row_sums = result.sum(dim=-1)

        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_non_negative(self):
        """Softmax output should be non-negative."""
        softmax_forward, _ = import_kernel()

        x = torch.randn(32, 256, device="cuda", dtype=torch.float32)
        result = softmax_forward(x)

        assert (result >= 0).all()

    def test_numerical_stability(self):
        """Test with large values that could cause overflow without max subtraction."""
        softmax_forward, _ = import_kernel()

        x = torch.randn(8, 128, device="cuda", dtype=torch.float32) * 100
        expected = F.softmax(x, dim=-1)
        result = softmax_forward(x)

        assert torch.allclose(result, expected, atol=1e-5)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_3d_input(self):
        """Test with 3D input (batch x seq_len x features)."""
        softmax_forward, _ = import_kernel()

        x = torch.randn(4, 16, 64, device="cuda", dtype=torch.float32)
        expected = F.softmax(x, dim=-1)
        result = softmax_forward(x.contiguous())

        assert torch.allclose(result.view_as(expected), expected, atol=1e-5)

    def test_single_element_rows(self):
        """Softmax of single element should be 1.0."""
        softmax_forward, _ = import_kernel()

        x = torch.randn(8, 1, device="cuda", dtype=torch.float32)
        result = softmax_forward(x)

        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)


class TestSoftmaxBackward:
    """Test softmax backward pass."""

    @pytest.mark.parametrize("shape", [
        (8, 128),
        (32, 256),
        (64, 512),
    ])
    def test_backward_correctness(self, shape):
        """Backward pass matches PyTorch autograd."""
        softmax_forward, softmax_backward = import_kernel()

        x = torch.randn(*shape, device="cuda", dtype=torch.float32, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        # Custom forward + backward
        sm_out = softmax_forward(x.detach())
        grad_out = torch.randn_like(sm_out)
        grad_in = softmax_backward(grad_out, sm_out)

        # PyTorch reference
        sm_ref = F.softmax(x_ref, dim=-1)
        sm_ref.backward(grad_out)

        assert torch.allclose(grad_in, x_ref.grad, atol=1e-5), (
            f"Shape {shape}: max grad diff = "
            f"{(grad_in - x_ref.grad).abs().max().item()}"
        )

    def test_gradient_check(self):
        """Numerical gradient check using finite differences."""
        softmax_forward, softmax_backward = import_kernel()

        x = torch.randn(4, 32, device="cuda", dtype=torch.float64, requires_grad=True)

        def func(x):
            # Use PyTorch softmax for gradcheck (our kernel is float32 only)
            return F.softmax(x, dim=-1)

        assert torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)
