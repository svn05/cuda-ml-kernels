# CUDA ML Kernels — Build and test environment
#
# Usage:
#   docker build -t cuda-ml-kernels .
#   docker run --gpus all cuda-ml-kernels          # Run tests + benchmarks
#   docker run --gpus all cuda-ml-kernels pytest    # Tests only
#   docker run --gpus all cuda-ml-kernels python benchmarks/benchmark.py  # Benchmarks only

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build essentials
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    pytest>=7.0.0 \
    numpy>=1.20.0 \
    tabulate>=0.9.0

# Copy project source
WORKDIR /workspace
COPY . .

# Build the CUDA extension
RUN pip3 install -e . --no-build-isolation

# Default: run tests then benchmarks
CMD ["bash", "-c", "echo '=== Running Tests ===' && pytest tests/ -v && echo '=== Running Benchmarks ===' && python benchmarks/benchmark.py"]
