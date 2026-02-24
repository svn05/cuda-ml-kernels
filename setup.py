"""Build system for CUDA ML Kernels.

Uses PyTorch's CUDAExtension to compile CUDA kernels into a
loadable Python extension module.

Usage:
    pip install -e .                 # Development install
    python setup.py build_ext        # Build only
    pip install -e . --no-build-isolation  # If torch is pre-installed
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Kernel source files
cuda_sources = [
    "csrc/torch_extension.cpp",
    "csrc/kernels/softmax.cu",
    "csrc/kernels/matmul.cu",
    "csrc/kernels/layernorm.cu",
    "csrc/kernels/attention.cu",
]

# Auto-detect GPU architecture, fallback to common targets
import subprocess
import torch

def get_cuda_arch_flags():
    """Detect GPU arch at build time; fall back to common targets."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"{cap[0]}{cap[1]}"
        return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    # Fallback: build for common architectures
    return [
        "-gencode=arch=compute_75,code=sm_75",   # T4
        "-gencode=arch=compute_80,code=sm_80",   # A100
    ]

# Compilation flags
extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "--expt-relaxed-constexpr",
    ] + get_cuda_arch_flags(),
}

setup(
    name="cuda_ml_kernels",
    version="0.1.0",
    description="GPU-accelerated ML kernels in CUDA with PyTorch C++ extension bindings",
    author="San Vo",
    author_email="san.vo@mail.utoronto.ca",
    url="https://github.com/svn05/cuda-ml-kernels",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cuda_ml_kernels._C",
            sources=cuda_sources,
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc", "include")],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0.0"],
)
