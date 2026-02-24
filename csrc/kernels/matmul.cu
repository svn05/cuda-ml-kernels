/**
 * matmul.cu — Tiled matrix multiplication with shared memory.
 *
 * Implements C = A @ B using shared memory tiling to reduce global memory
 * bandwidth. Each thread block computes a TILE_SIZE x TILE_SIZE output tile
 * by iterating over K-dimension tiles.
 *
 * Optimizations:
 *   - Shared memory tiling: reduces global loads from O(M*N*K) to O(M*N*K/TILE)
 *   - Coalesced global memory access patterns
 *   - Boundary checks for non-tile-aligned dimensions
 *   - Double buffering: overlaps next tile load with current tile computation
 *
 * Shapes: A is (M, K), B is (K, N), output C is (M, N).
 */

#include <torch/extension.h>
#include "common.cuh"

constexpr int TILE_SIZE = 32;

// ---------- Naive kernel (baseline for comparison) ----------

__global__ void matmul_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---------- Tiled kernel (shared memory) ----------

/**
 * Tiled GEMM using shared memory.
 *
 * Each block loads TILE_SIZE x TILE_SIZE sub-matrices of A and B into
 * shared memory, then computes the partial dot products. This is repeated
 * for each tile along the K dimension.
 *
 * Memory access pattern:
 *   - Threads load A[row][tile_k + tx] and B[tile_k + ty][col] coalesced
 *   - Shared memory eliminates redundant global loads (each element loaded
 *     once per tile, used TILE_SIZE times)
 */
__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Iterate over tiles along K dimension
    int num_tiles = cdiv(K, TILE_SIZE);
    for (int t = 0; t < num_tiles; ++t) {
        // Load A tile: A[row][t*TILE + tx]
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile: B[t*TILE + ty][col]
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ---------- Double-buffered tiled kernel ----------

/**
 * Double-buffered tiled GEMM.
 *
 * Uses two shared memory buffers to overlap loading the next tile with
 * computing the current tile. This hides memory latency at the cost
 * of 2x shared memory usage.
 */
__global__ void matmul_double_buffered_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = cdiv(K, TILE_SIZE);
    int cur = 0;  // Current buffer index

    // Pre-load first tile into buffer 0
    int a_col = threadIdx.x;
    int b_row = threadIdx.y;
    As[0][threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[0][threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        int nxt = 1 - cur;

        // Load next tile into alternate buffer (if there is a next tile)
        if (t + 1 < num_tiles) {
            int next_a_col = (t + 1) * TILE_SIZE + threadIdx.x;
            int next_b_row = (t + 1) * TILE_SIZE + threadIdx.y;
            As[nxt][threadIdx.y][threadIdx.x] =
                (row < M && next_a_col < K) ? A[row * K + next_a_col] : 0.0f;
            Bs[nxt][threadIdx.y][threadIdx.x] =
                (next_b_row < K && col < N) ? B[next_b_row * N + col] : 0.0f;
        }

        // Compute with current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[cur][threadIdx.y][k] * Bs[cur][k][threadIdx.x];
        }

        __syncthreads();
        cur = nxt;
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ---------- Host dispatch ----------

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(cdiv(N, TILE_SIZE), cdiv(M, TILE_SIZE));

    matmul_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

torch::Tensor matmul_naive_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(cdiv(N, TILE_SIZE), cdiv(M, TILE_SIZE));

    matmul_naive_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
