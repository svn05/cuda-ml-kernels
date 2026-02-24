/**
 * common.cuh — Shared macros, error checking, and warp-level utilities.
 *
 * Provides:
 *   - CUDA_CHECK: runtime error checking macro
 *   - warpReduceMax / warpReduceSum: warp-level parallel reductions
 *   - blockReduceMax / blockReduceSum: block-level reductions via shared memory
 *   - WARP_SIZE, FULL_MASK constants
 */

#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

// ---------- Error checking ----------

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------- Constants ----------

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFF;

// ---------- Warp-level reductions ----------

/**
 * Reduce to maximum across a warp using shuffle-down.
 * All threads in the warp participate; result is valid in lane 0.
 */
__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

/**
 * Reduce to sum across a warp using shuffle-down.
 * All threads in the warp participate; result is valid in lane 0.
 */
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// ---------- Block-level reductions ----------

/**
 * Reduce to maximum across a block.
 * Uses warp-level reductions + shared memory for inter-warp communication.
 * Result is broadcast to all threads via __shfl_sync.
 */
__device__ __forceinline__ float blockReduceMax(float val) {
    __shared__ float shared[32];  // Max 32 warps per block (1024 threads)
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // First warp reduces the warp-level results
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);

    // Broadcast result to all threads
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

/**
 * Reduce to sum across a block.
 * Uses warp-level reductions + shared memory for inter-warp communication.
 * Result is broadcast to all threads via shared memory.
 */
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);

    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

// ---------- Utility ----------

/**
 * Ceiling division.
 */
__host__ __device__ __forceinline__ int cdiv(int a, int b) {
    return (a + b - 1) / b;
}
