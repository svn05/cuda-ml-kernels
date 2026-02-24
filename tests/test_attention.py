"""Tests for custom CUDA fused attention kernel.

Validates the FlashAttention-inspired implementation against standard
attention (Q @ K^T / sqrt(d_k) -> softmax -> @ V) for both causal and
non-causal variants.
"""

import pytest
import torch
import torch.nn.functional as F
import math


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.fixture(autouse=True)
def cuda_setup():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def import_kernel():
    from cuda_ml_kernels._C import attention_forward
    return attention_forward


def reference_attention(Q, K, V, causal=False):
    """Standard scaled dot-product attention (PyTorch reference)."""
    d_k = Q.size(-1)
    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if causal:
        seq_len = Q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


class TestAttentionForward:
    """Test fused attention forward pass."""

    @pytest.mark.parametrize("B,N,d", [
        (1, 16, 32),
        (2, 32, 64),
        (4, 64, 64),
        (2, 128, 32),
        (1, 64, 128),
    ])
    def test_non_causal(self, B, N, d):
        """Non-causal attention matches reference."""
        attention_forward = import_kernel()

        Q = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(B, N, d, device="cuda", dtype=torch.float32)

        expected = reference_attention(Q, K, V, causal=False)
        result = attention_forward(Q, K, V, False)

        # Fused attention accumulates differently, so tolerance is slightly looser
        assert torch.allclose(result, expected, atol=1e-3), (
            f"(B={B}, N={N}, d={d}): max diff = "
            f"{(result - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("B,N,d", [
        (1, 16, 32),
        (2, 32, 64),
        (4, 64, 64),
        (1, 128, 32),
    ])
    def test_causal(self, B, N, d):
        """Causal attention matches reference."""
        attention_forward = import_kernel()

        Q = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(B, N, d, device="cuda", dtype=torch.float32)

        expected = reference_attention(Q, K, V, causal=True)
        result = attention_forward(Q, K, V, True)

        assert torch.allclose(result, expected, atol=1e-3), (
            f"Causal (B={B}, N={N}, d={d}): max diff = "
            f"{(result - expected).abs().max().item()}"
        )

    def test_causal_first_token(self):
        """First token in causal attention should only attend to itself."""
        attention_forward = import_kernel()

        B, N, d = 1, 32, 64
        Q = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(B, N, d, device="cuda", dtype=torch.float32)

        result = attention_forward(Q, K, V, True)

        # First token output should equal V[0] (only attends to itself)
        expected_first = V[0, 0]
        assert torch.allclose(result[0, 0], expected_first, atol=1e-4), (
            f"First token diff = {(result[0, 0] - expected_first).abs().max().item()}"
        )

    def test_uniform_attention(self):
        """When Q=K, attention should be nearly uniform for non-causal."""
        attention_forward = import_kernel()

        B, N, d = 1, 16, 64
        # Make Q = K = zeros → all scores equal → uniform attention
        Q = torch.zeros(B, N, d, device="cuda", dtype=torch.float32)
        K = torch.zeros(B, N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(B, N, d, device="cuda", dtype=torch.float32)

        result = attention_forward(Q, K, V, False)

        # With uniform attention, output should be mean of V across seq dim
        expected = V.mean(dim=1, keepdim=True).expand_as(V)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_single_sequence(self):
        """Single-element sequence should return V."""
        attention_forward = import_kernel()

        B, N, d = 4, 1, 64
        Q = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(B, N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(B, N, d, device="cuda", dtype=torch.float32)

        result = attention_forward(Q, K, V, False)

        # softmax of single element is 1.0, so output = V
        assert torch.allclose(result, V, atol=1e-5)
