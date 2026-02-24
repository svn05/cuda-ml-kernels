"""Tests for custom CUDA tiled matrix multiplication kernel.

Validates correctness against torch.mm for various matrix shapes,
including non-tile-aligned dimensions.
"""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.fixture(autouse=True)
def cuda_setup():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def import_kernel():
    from cuda_ml_kernels._C import matmul, matmul_naive
    return matmul, matmul_naive


class TestMatmulTiled:
    """Test tiled matrix multiplication."""

    @pytest.mark.parametrize("M,K,N", [
        (32, 32, 32),       # Exact tile alignment
        (64, 128, 64),      # Multiple tiles
        (33, 65, 17),       # Non-tile-aligned (boundary checks)
        (1, 256, 1),        # Vector-matrix-vector
        (128, 1, 128),      # Rank-1 outer product
        (256, 256, 256),    # Medium square
        (512, 1024, 512),   # Large
        (1024, 2048, 1024), # Very large
    ])
    def test_shapes(self, M, K, N):
        """Output matches torch.mm for various shapes."""
        matmul, _ = import_kernel()

        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        expected = torch.mm(A, B)
        result = matmul(A, B)

        # Larger matrices accumulate more floating-point error (different
        # accumulation order between tiled kernel and cuBLAS)
        max_dim = max(M, K, N)
        atol = 1e-3 if max_dim > 512 else (1e-4 if max_dim > 128 else 1e-5)
        assert torch.allclose(result, expected, atol=atol), (
            f"({M}, {K}, {N}): max diff = {(result - expected).abs().max().item()}"
        )

    def test_identity(self):
        """Multiplying by identity matrix returns the original."""
        matmul, _ = import_kernel()

        n = 64
        A = torch.randn(n, n, device="cuda", dtype=torch.float32)
        I = torch.eye(n, device="cuda", dtype=torch.float32)

        result = matmul(A, I)
        assert torch.allclose(result, A, atol=1e-5)

    def test_zero_matrix(self):
        """Multiplying by zero matrix returns zeros."""
        matmul, _ = import_kernel()

        A = torch.randn(32, 64, device="cuda", dtype=torch.float32)
        Z = torch.zeros(64, 32, device="cuda", dtype=torch.float32)

        result = matmul(A, Z)
        assert torch.allclose(result, torch.zeros(32, 32, device="cuda"))


class TestMatmulNaive:
    """Test naive matmul (baseline reference)."""

    def test_basic(self):
        """Naive kernel produces correct result."""
        _, matmul_naive = import_kernel()

        A = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        B = torch.randn(128, 64, device="cuda", dtype=torch.float32)

        expected = torch.mm(A, B)
        result = matmul_naive(A, B)

        assert torch.allclose(result, expected, atol=1e-4)

    def test_non_aligned(self):
        """Naive kernel handles non-tile-aligned sizes."""
        _, matmul_naive = import_kernel()

        A = torch.randn(33, 65, device="cuda", dtype=torch.float32)
        B = torch.randn(65, 17, device="cuda", dtype=torch.float32)

        expected = torch.mm(A, B)
        result = matmul_naive(A, B)

        assert torch.allclose(result, expected, atol=1e-5)
