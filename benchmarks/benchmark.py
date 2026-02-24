"""Benchmark CUDA ML kernels against PyTorch native implementations.

Measures wall-clock time using torch.cuda.Event for precise GPU timing
(avoids CPU-GPU synchronization overhead). Compares custom kernels against
PyTorch's native ops across multiple matrix sizes.

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --sizes 256 512 1024 2048
    python benchmarks/benchmark.py --warmup 20 --runs 100
"""

import argparse
import torch
import torch.nn.functional as F
import math
from tabulate import tabulate


def benchmark_fn(fn, *args, warmup=10, runs=50):
    """Benchmark a function using CUDA events for precise GPU timing.

    Args:
        fn: Function to benchmark.
        *args: Arguments to pass to fn.
        warmup: Number of warmup iterations.
        runs: Number of timed iterations.

    Returns:
        Mean time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)

    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(runs):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def benchmark_softmax(sizes, warmup, runs):
    """Benchmark softmax: custom vs torch.nn.functional.softmax."""
    from cuda_ml_kernels._C import softmax_forward

    results = []
    for n in sizes:
        x = torch.randn(256, n, device="cuda", dtype=torch.float32)

        t_custom = benchmark_fn(softmax_forward, x, warmup=warmup, runs=runs)
        t_torch = benchmark_fn(F.softmax, x, -1, warmup=warmup, runs=runs)

        speedup = t_torch / t_custom if t_custom > 0 else float("inf")
        results.append({
            "size": f"256 x {n}",
            "custom_ms": f"{t_custom:.3f}",
            "torch_ms": f"{t_torch:.3f}",
            "speedup": f"{speedup:.2f}x",
        })

    return results


def benchmark_matmul(sizes, warmup, runs):
    """Benchmark matmul: custom tiled vs torch.mm."""
    from cuda_ml_kernels._C import matmul

    results = []
    for n in sizes:
        A = torch.randn(n, n, device="cuda", dtype=torch.float32)
        B = torch.randn(n, n, device="cuda", dtype=torch.float32)

        t_custom = benchmark_fn(matmul, A, B, warmup=warmup, runs=runs)
        t_torch = benchmark_fn(torch.mm, A, B, warmup=warmup, runs=runs)

        speedup = t_torch / t_custom if t_custom > 0 else float("inf")
        results.append({
            "size": f"{n} x {n}",
            "custom_ms": f"{t_custom:.3f}",
            "torch_ms": f"{t_torch:.3f}",
            "speedup": f"{speedup:.2f}x",
        })

    return results


def benchmark_layernorm(sizes, warmup, runs):
    """Benchmark layernorm: custom fused vs torch.nn.functional.layer_norm."""
    from cuda_ml_kernels._C import layernorm_forward

    results = []
    for n in sizes:
        x = torch.randn(256, n, device="cuda", dtype=torch.float32)
        gamma = torch.ones(n, device="cuda", dtype=torch.float32)
        beta = torch.zeros(n, device="cuda", dtype=torch.float32)

        t_custom = benchmark_fn(layernorm_forward, x, gamma, beta, 1e-5,
                                warmup=warmup, runs=runs)
        t_torch = benchmark_fn(F.layer_norm, x, [n], gamma, beta, 1e-5,
                               warmup=warmup, runs=runs)

        speedup = t_torch / t_custom if t_custom > 0 else float("inf")
        results.append({
            "size": f"256 x {n}",
            "custom_ms": f"{t_custom:.3f}",
            "torch_ms": f"{t_torch:.3f}",
            "speedup": f"{speedup:.2f}x",
        })

    return results


def benchmark_attention(seq_lens, warmup, runs):
    """Benchmark attention: custom fused vs standard attention."""
    from cuda_ml_kernels._C import attention_forward

    def standard_attention(Q, K, V):
        scale = 1.0 / math.sqrt(Q.size(-1))
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        return torch.matmul(F.softmax(scores, dim=-1), V)

    results = []
    head_dim = 64
    batch = 4

    for n in seq_lens:
        if n > 512:
            # Skip very large sizes for standard attention (OOM)
            continue

        Q = torch.randn(batch, n, head_dim, device="cuda", dtype=torch.float32)
        K = torch.randn(batch, n, head_dim, device="cuda", dtype=torch.float32)
        V = torch.randn(batch, n, head_dim, device="cuda", dtype=torch.float32)

        t_custom = benchmark_fn(attention_forward, Q, K, V, False,
                                warmup=warmup, runs=runs)
        t_torch = benchmark_fn(standard_attention, Q, K, V,
                               warmup=warmup, runs=runs)

        speedup = t_torch / t_custom if t_custom > 0 else float("inf")

        # Memory comparison
        mem_standard = batch * n * n * 4  # Full attention matrix (float32)
        mem_fused = batch * 32 * head_dim * 4 * 2  # Tile buffers only
        mem_ratio = mem_standard / mem_fused if mem_fused > 0 else float("inf")

        results.append({
            "size": f"{batch} x {n} x {head_dim}",
            "custom_ms": f"{t_custom:.3f}",
            "torch_ms": f"{t_torch:.3f}",
            "speedup": f"{speedup:.2f}x",
            "mem_saved": f"{mem_ratio:.1f}x",
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA ML kernels")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[256, 512, 1024, 2048, 4096],
                        help="Matrix sizes to benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    device = torch.cuda.get_device_name(0)
    print(f"GPU: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    # Softmax
    print("=" * 70)
    print("SOFTMAX — Online Algorithm with Warp Reductions")
    print("=" * 70)
    results = benchmark_softmax(args.sizes, args.warmup, args.runs)
    print(tabulate(results, headers="keys", tablefmt="github"))
    print()

    # Matmul
    print("=" * 70)
    print("MATMUL — Shared Memory Tiling (TILE=32)")
    print("=" * 70)
    results = benchmark_matmul(args.sizes, args.warmup, args.runs)
    print(tabulate(results, headers="keys", tablefmt="github"))
    print()

    # LayerNorm
    print("=" * 70)
    print("LAYERNORM — Fused Welford's Algorithm")
    print("=" * 70)
    results = benchmark_layernorm(args.sizes, args.warmup, args.runs)
    print(tabulate(results, headers="keys", tablefmt="github"))
    print()

    # Attention
    print("=" * 70)
    print("ATTENTION — FlashAttention-Inspired Tiling (B=4, d=64)")
    print("=" * 70)
    seq_lens = [s for s in args.sizes if s <= 512]
    if seq_lens:
        results = benchmark_attention(seq_lens, args.warmup, args.runs)
        print(tabulate(results, headers="keys", tablefmt="github"))
    else:
        print("  (skipped — all sizes > 512, standard attention would OOM)")
    print()


if __name__ == "__main__":
    main()
