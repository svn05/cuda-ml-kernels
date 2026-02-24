/**
 * layernorm.cu — Fused Layer Normalization with Welford's algorithm.
 *
 * Implements LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * Optimizations:
 *   - Welford's online algorithm: single-pass numerically stable mean/variance
 *   - Fused normalization + affine transform in one kernel (avoids extra
 *     global memory round-trip)
 *   - Warp-level reductions for partial statistics
 *   - Shared memory for cross-warp aggregation
 *
 * Forward: computes y, saves mean and rstd (reciprocal std) for backward
 * Backward: computes dx, dgamma, dbeta
 */

#include <torch/extension.h>
#include "common.cuh"

// ---------- Welford's online statistics ----------

/**
 * Welford's online algorithm state.
 * Tracks mean, M2 (sum of squared deviations), and count.
 */
struct WelfordState {
    float mean;
    float m2;     // Sum of squared differences from mean
    float count;
};

/**
 * Update Welford state with a new value.
 */
__device__ __forceinline__ WelfordState welford_update(
    WelfordState state, float val
) {
    state.count += 1.0f;
    float delta = val - state.mean;
    state.mean += delta / state.count;
    float delta2 = val - state.mean;
    state.m2 += delta * delta2;
    return state;
}

/**
 * Combine two Welford states (for parallel reduction).
 * Uses the parallel Welford merge formula.
 */
__device__ __forceinline__ WelfordState welford_combine(
    WelfordState a, WelfordState b
) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;

    float total = a.count + b.count;
    float delta = b.mean - a.mean;

    WelfordState result;
    result.count = total;
    result.mean = a.mean + delta * b.count / total;
    result.m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / total;
    return result;
}

/**
 * Warp-level Welford reduction using shuffle.
 */
__device__ __forceinline__ WelfordState warpReduceWelford(WelfordState state) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        WelfordState other;
        other.mean  = __shfl_down_sync(FULL_MASK, state.mean, offset);
        other.m2    = __shfl_down_sync(FULL_MASK, state.m2, offset);
        other.count = __shfl_down_sync(FULL_MASK, state.count, offset);
        state = welford_combine(state, other);
    }
    return state;
}

// ---------- Forward kernel ----------

/**
 * Fused LayerNorm forward.
 *
 * Each block processes one row (one normalization group).
 * 1. Compute mean and variance via Welford's algorithm (single pass)
 * 2. Normalize and apply affine transform: y = gamma * (x - mean) * rstd + beta
 * 3. Save mean and rstd for backward pass
 */
__global__ void layernorm_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int num_rows,
    int normalized_size,
    float eps
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const float* row_in  = input  + row * normalized_size;
    float*       row_out = output + row * normalized_size;

    // --- Welford's online mean/variance (single pass) ---
    WelfordState state = {0.0f, 0.0f, 0.0f};
    for (int col = threadIdx.x; col < normalized_size; col += blockDim.x) {
        state = welford_update(state, row_in[col]);
    }

    // Warp-level reduction
    state = warpReduceWelford(state);

    // Block-level reduction via shared memory
    __shared__ float s_mean[32];
    __shared__ float s_m2[32];
    __shared__ float s_count[32];

    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        s_mean[wid]  = state.mean;
        s_m2[wid]    = state.m2;
        s_count[wid] = state.count;
    }
    __syncthreads();

    // First warp reduces inter-warp results
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (wid == 0) {
        if (lane < num_warps) {
            state.mean  = s_mean[lane];
            state.m2    = s_m2[lane];
            state.count = s_count[lane];
        } else {
            state = {0.0f, 0.0f, 0.0f};
        }
        state = warpReduceWelford(state);
    }

    // Broadcast mean and rstd
    __shared__ float shared_mean;
    __shared__ float shared_rstd;
    if (threadIdx.x == 0) {
        float variance = state.m2 / state.count;
        shared_mean = state.mean;
        shared_rstd = rsqrtf(variance + eps);
        mean_out[row] = shared_mean;
        rstd_out[row] = shared_rstd;
    }
    __syncthreads();

    float mean = shared_mean;
    float rstd = shared_rstd;

    // --- Normalize + affine transform ---
    for (int col = threadIdx.x; col < normalized_size; col += blockDim.x) {
        float x_hat = (row_in[col] - mean) * rstd;
        row_out[col] = gamma[col] * x_hat + beta[col];
    }
}

// ---------- Backward kernel ----------

/**
 * LayerNorm backward: computes grad_input, grad_gamma, and grad_beta.
 *
 * Given upstream gradient dL/dy:
 *   dx_hat = dL/dy * gamma
 *   dgamma = sum_rows(dL/dy * x_hat)
 *   dbeta  = sum_rows(dL/dy)
 *
 * For grad_input (per row):
 *   ds = sum(dx_hat * x_hat)
 *   db = sum(dx_hat)
 *   dx = rstd * (dx_hat - (x_hat * ds + db) / N)
 */
__global__ void layernorm_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ mean_saved,
    const float* __restrict__ rstd_saved,
    float* __restrict__ grad_input,
    float* __restrict__ grad_gamma,  // Accumulated atomically
    float* __restrict__ grad_beta,   // Accumulated atomically
    int num_rows,
    int normalized_size
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const float* dout   = grad_output + row * normalized_size;
    const float* x      = input       + row * normalized_size;
    float*       dx     = grad_input  + row * normalized_size;

    float mean = mean_saved[row];
    float rstd = rstd_saved[row];
    float inv_n = 1.0f / (float)normalized_size;

    // Compute partial sums for this row: ds = sum(dx_hat * x_hat), db = sum(dx_hat)
    float ds_local = 0.0f;
    float db_local = 0.0f;
    for (int col = threadIdx.x; col < normalized_size; col += blockDim.x) {
        float x_hat = (x[col] - mean) * rstd;
        float dx_hat = dout[col] * gamma[col];
        ds_local += dx_hat * x_hat;
        db_local += dx_hat;
    }
    float ds = blockReduceSum(ds_local);
    float db = blockReduceSum(db_local);

    // Compute grad_input and accumulate grad_gamma/grad_beta
    for (int col = threadIdx.x; col < normalized_size; col += blockDim.x) {
        float x_hat = (x[col] - mean) * rstd;
        float dx_hat = dout[col] * gamma[col];

        // grad_input
        dx[col] = rstd * (dx_hat - inv_n * (x_hat * ds + db));

        // grad_gamma and grad_beta (atomic across rows)
        atomicAdd(&grad_gamma[col], dout[col] * x_hat);
        atomicAdd(&grad_beta[col], dout[col]);
    }
}

// ---------- Host dispatch ----------

std::vector<torch::Tensor> layernorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Beta must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    int normalized_size = input.size(-1);
    int num_rows = input.numel() / normalized_size;

    TORCH_CHECK(gamma.numel() == normalized_size, "Gamma size must match last dim");
    TORCH_CHECK(beta.numel() == normalized_size, "Beta size must match last dim");

    auto output   = torch::empty_like(input);
    auto mean_out = torch::empty({num_rows}, input.options());
    auto rstd_out = torch::empty({num_rows}, input.options());

    int block_size = 1;
    while (block_size < normalized_size && block_size < 1024) {
        block_size <<= 1;
    }
    block_size = min(block_size, 1024);

    layernorm_forward_kernel<<<num_rows, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        mean_out.data_ptr<float>(),
        rstd_out.data_ptr<float>(),
        num_rows,
        normalized_size,
        eps
    );

    return {output, mean_out, rstd_out};
}

std::vector<torch::Tensor> layernorm_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor mean_saved,
    torch::Tensor rstd_saved
) {
    TORCH_CHECK(grad_output.is_cuda() && input.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(grad_output.is_contiguous() && input.is_contiguous(), "Inputs must be contiguous");

    int normalized_size = input.size(-1);
    int num_rows = input.numel() / normalized_size;

    auto grad_input = torch::empty_like(input);
    auto grad_gamma = torch::zeros_like(gamma);
    auto grad_beta  = torch::zeros_like(gamma);

    int block_size = 1;
    while (block_size < normalized_size && block_size < 1024) {
        block_size <<= 1;
    }
    block_size = min(block_size, 1024);

    layernorm_backward_kernel<<<num_rows, block_size>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        mean_saved.data_ptr<float>(),
        rstd_saved.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_gamma.data_ptr<float>(),
        grad_beta.data_ptr<float>(),
        num_rows,
        normalized_size
    );

    return {grad_input, grad_gamma, grad_beta};
}
