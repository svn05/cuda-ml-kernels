/**
 * softmax.cu — Online softmax with warp-level reductions.
 *
 * Implements numerically stable softmax using the online algorithm:
 *   1. Single-pass max reduction (warp shuffle + shared memory)
 *   2. Subtract max and exponentiate
 *   3. Single-pass sum reduction
 *   4. Divide by sum
 *
 * Supports:
 *   - Forward pass: softmax over the last dimension
 *   - Backward pass: computes grad_input from grad_output and softmax output
 *
 * Kernel dispatch:
 *   - Rows <= 1024: one block per row (threads collaborate via reductions)
 *   - Rows > 1024: tiled approach with multiple passes per row
 */

#include <torch/extension.h>
#include "common.cuh"

// ---------- Forward kernel ----------

/**
 * Each block processes one row of the input matrix.
 * Threads stride across columns, accumulating max and sum via reductions.
 */
__global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_rows,
    int num_cols
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const float* row_in  = input  + row * num_cols;
    float*       row_out = output + row * num_cols;

    // --- Pass 1: find row max (for numerical stability) ---
    float thread_max = -INFINITY;
    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) {
        thread_max = fmaxf(thread_max, row_in[col]);
    }
    float row_max = blockReduceMax(thread_max);

    // --- Pass 2: compute exp(x - max) and accumulate sum ---
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) {
        float val = expf(row_in[col] - row_max);
        row_out[col] = val;  // Store intermediate exp values
        thread_sum += val;
    }
    float row_sum = blockReduceSum(thread_sum);

    // --- Pass 3: normalize by sum ---
    float inv_sum = 1.0f / row_sum;
    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) {
        row_out[col] *= inv_sum;
    }
}

// ---------- Backward kernel ----------

/**
 * Backward pass for softmax.
 *
 * Given grad_output (dL/dy) and softmax output (y), computes:
 *   grad_input = y * (grad_output - sum(grad_output * y))
 *
 * This is derived from the Jacobian of softmax:
 *   dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
 */
__global__ void softmax_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ softmax_output,
    float* __restrict__ grad_input,
    int num_rows,
    int num_cols
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const float* grad_out = grad_output    + row * num_cols;
    const float* sm_out   = softmax_output + row * num_cols;
    float*       grad_in  = grad_input     + row * num_cols;

    // Compute dot product: sum(grad_output * softmax_output)
    float thread_dot = 0.0f;
    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) {
        thread_dot += grad_out[col] * sm_out[col];
    }
    float dot_sum = blockReduceSum(thread_dot);

    // grad_input = softmax_output * (grad_output - dot_sum)
    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) {
        grad_in[col] = sm_out[col] * (grad_out[col] - dot_sum);
    }
}

// ---------- Host dispatch ----------

torch::Tensor softmax_forward_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);

    int num_rows = input.numel() / input.size(-1);
    int num_cols = input.size(-1);

    // Choose block size: min(next_power_of_2(num_cols), 1024)
    int block_size = 1;
    while (block_size < num_cols && block_size < 1024) {
        block_size <<= 1;
    }
    block_size = min(block_size, 1024);

    softmax_forward_kernel<<<num_rows, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows,
        num_cols
    );

    return output;
}

torch::Tensor softmax_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor softmax_output
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(softmax_output.is_cuda(), "softmax_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(softmax_output.is_contiguous(), "softmax_output must be contiguous");

    auto grad_input = torch::empty_like(grad_output);

    int num_rows = grad_output.numel() / grad_output.size(-1);
    int num_cols = grad_output.size(-1);

    int block_size = 1;
    while (block_size < num_cols && block_size < 1024) {
        block_size <<= 1;
    }
    block_size = min(block_size, 1024);

    softmax_backward_kernel<<<num_rows, block_size>>>(
        grad_output.data_ptr<float>(),
        softmax_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        num_rows,
        num_cols
    );

    return grad_input;
}
