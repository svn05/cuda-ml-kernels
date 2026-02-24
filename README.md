# CUDA ML Kernels

GPU-accelerated ML operations implemented from scratch in **CUDA C++** with **PyTorch C++ extension** bindings. Custom implementations of core operations used in transformer architectures, benchmarked against PyTorch native ops.

## Kernels

### 1. Online Softmax
Numerically stable softmax using warp-level `__shfl_down_sync` reductions and shared memory for inter-warp communication. Single-pass max and sum computation avoids multiple data passes.

### 2. Tiled Matrix Multiplication (GEMM)
Shared memory tiling (`TILE_SIZE=32`) that reduces global memory traffic from O(MNK) to O(MNK/TILE). Includes boundary checks for non-tile-aligned dimensions and a double-buffered variant that overlaps memory loads with computation.

### 3. Fused Layer Normalization
Single-pass mean/variance via **Welford's online algorithm** with warp-level parallel reductions. Fuses the normalization and affine transform (`gamma * x_hat + beta`) into one kernel, avoiding an extra global memory round-trip. Forward and backward passes implemented.

### 4. Fused Scaled Dot-Product Attention
**FlashAttention-inspired** tiled approach that processes Q/K/V in blocks to avoid materializing the full O(N^2) attention matrix. Uses online softmax with running max/sum corrections within tiles and shared memory for K/V tile caching. Supports causal masking.

## Architecture

```
Input Tensors (PyTorch, CUDA)
    │
    ▼
Python API (cuda_ml_kernels/functional.py)
    │  ─ softmax(), matmul(), layernorm(), attention()
    │  ─ Autograd integration (ops.py)
    │  ─ CPU fallback to PyTorch native
    ▼
C++ Bindings (csrc/torch_extension.cpp)
    │  ─ pybind11 module registration
    │  ─ Input validation (contiguous, CUDA, dtype)
    ▼
CUDA Kernels (csrc/kernels/*.cu)
    │  ─ Warp-level reductions (__shfl_down_sync)
    │  ─ Shared memory tiling
    │  ─ Welford's online statistics
    │  ─ Online softmax (running max/sum)
    ▼
Output Tensors (PyTorch, CUDA)
```

## Setup

### Requirements
- NVIDIA GPU (compute capability >= 7.0)
- CUDA Toolkit >= 11.8
- PyTorch >= 2.0.0

### Install
```bash
git clone https://github.com/svn05/cuda-ml-kernels.git
cd cuda-ml-kernels
pip install -e .
```

### Docker
```bash
docker build -t cuda-ml-kernels .
docker run --gpus all cuda-ml-kernels
```

## Usage

```python
import torch
import cuda_ml_kernels as cmk

# Softmax
x = torch.randn(32, 256, device="cuda")
out = cmk.softmax(x)

# Tiled matrix multiplication
A = torch.randn(1024, 512, device="cuda")
B = torch.randn(512, 768, device="cuda")
C = cmk.matmul(A, B)

# Fused Layer Normalization
x = torch.randn(32, 128, device="cuda")
gamma = torch.ones(128, device="cuda")
beta = torch.zeros(128, device="cuda")
out = cmk.layernorm(x, gamma, beta)

# Fused Attention
Q = torch.randn(4, 64, 64, device="cuda")
K = torch.randn(4, 64, 64, device="cuda")
V = torch.randn(4, 64, 64, device="cuda")
out = cmk.attention(Q, K, V, causal=True)
```

## Tests

```bash
pytest tests/ -v
```

Tests compare all kernel outputs against PyTorch native equivalents using `torch.allclose(atol=1e-5)` and include gradient verification for kernels with backward passes.

## Benchmarks

```bash
python benchmarks/benchmark.py
python benchmarks/benchmark.py --sizes 256 512 1024 2048 4096
```

Benchmarks use `torch.cuda.Event` for precise GPU timing and report speedup ratios vs PyTorch native ops across multiple matrix sizes.

## Project Structure

```
cuda-ml-kernels/
├── csrc/
│   ├── kernels/
│   │   ├── softmax.cu          # Online softmax with warp reductions
│   │   ├── matmul.cu           # Tiled GEMM with shared memory
│   │   ├── layernorm.cu        # Fused LayerNorm (Welford's algorithm)
│   │   └── attention.cu        # FlashAttention-inspired fused attention
│   ├── include/
│   │   └── common.cuh          # Shared macros, warp/block reductions
│   └── torch_extension.cpp     # PyTorch C++ extension bindings
├── cuda_ml_kernels/
│   ├── __init__.py
│   ├── ops.py                  # torch.autograd.Function wrappers
│   └── functional.py           # User-facing API with CPU fallback
├── tests/
│   ├── test_softmax.py
│   ├── test_matmul.py
│   ├── test_layernorm.py
│   └── test_attention.py
├── benchmarks/
│   └── benchmark.py            # GPU timing benchmarks vs PyTorch
├── setup.py                    # CUDAExtension build system
├── Dockerfile                  # CUDA build + test environment
├── requirements.txt
└── README.md
```

## Tech Stack

- **CUDA C++** — Custom GPU kernels with shared memory, warp intrinsics
- **PyTorch C++ Extensions** — pybind11 bindings, autograd integration
- **Python** — Functional API, test suite, benchmarking
