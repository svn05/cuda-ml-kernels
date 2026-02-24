/**
 * attention.cu — Fused Scaled Dot-Product Attention (FlashAttention-inspired).
 *
 * Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * Standard attention materializes the full N×N attention matrix, which is
 * O(N²) in memory. This kernel uses a tiled approach inspired by
 * FlashAttention (Dao et al., 2022) that processes Q/K/V in blocks:
 *
 *   1. Load a tile of Q into shared memory
 *   2. Iterate over tiles of K and V:
 *      a. Compute S_tile = Q_tile @ K_tile^T / sqrt(d_k)
 *      b. Apply causal mask (optional)
 *      c. Update running softmax using online algorithm:
 *         - Track running max (m) and running sum (l) per query
 *         - Rescale previous accumulator when max changes
 *      d. Accumulate O_tile += softmax(S_tile) @ V_tile
 *   3. Final rescaling: O /= l
 *
 * This avoids materializing the full N×N matrix, reducing memory from
 * O(N²) to O(N) (only tile-sized buffers in shared memory).
 *
 * Shapes: Q, K, V are (batch, seq_len, head_dim). Output is same shape.
 * Supports optional causal masking.
 */

#include <torch/extension.h>
#include "common.cuh"

// Tile size for sequence dimension (number of queries/keys per tile)
constexpr int ATTN_TILE = 32;

// ---------- Forward kernel ----------

/**
 * Fused attention forward.
 *
 * Grid: (num_heads, batch_size) — one block per (batch, head) pair.
 * Each block processes all seq_len queries by tiling over Q and K/V.
 *
 * For simplicity, this implementation uses a single block per head with
 * threads iterating over tiles. head_dim must be <= 128.
 */
__global__ void attention_forward_kernel(
    const float* __restrict__ Q,     // (B, N, d)
    const float* __restrict__ K,     // (B, N, d)
    const float* __restrict__ V,     // (B, N, d)
    float* __restrict__ O,           // (B, N, d)
    int batch_size,
    int seq_len,
    int head_dim,
    float scale,
    bool causal
) {
    int b = blockIdx.x;  // batch index
    if (b >= batch_size) return;

    // Pointers for this batch element
    const float* q = Q + b * seq_len * head_dim;
    const float* k = K + b * seq_len * head_dim;
    const float* v = V + b * seq_len * head_dim;
    float*       o = O + b * seq_len * head_dim;

    // Shared memory for K and V tiles
    extern __shared__ float smem[];
    float* k_tile = smem;                            // ATTN_TILE x head_dim
    float* v_tile = smem + ATTN_TILE * head_dim;     // ATTN_TILE x head_dim

    // Process each query position
    for (int qi = threadIdx.x; qi < seq_len; qi += blockDim.x) {

        // Per-query accumulators for online softmax
        float m_i = -INFINITY;   // Running max
        float l_i = 0.0f;        // Running sum of exp

        // Temporary output accumulator (in registers)
        // Since head_dim can be up to 128, we accumulate in a loop
        // We'll write directly to global memory and rescale

        // Initialize output to zero
        for (int d = 0; d < head_dim; d++) {
            o[qi * head_dim + d] = 0.0f;
        }

        // Iterate over K/V tiles
        int num_kv_tiles = (seq_len + ATTN_TILE - 1) / ATTN_TILE;
        for (int t = 0; t < num_kv_tiles; t++) {
            __syncthreads();

            // Cooperative loading of K and V tiles into shared memory
            int tile_start = t * ATTN_TILE;
            for (int idx = threadIdx.x; idx < ATTN_TILE * head_dim; idx += blockDim.x) {
                int tile_row = idx / head_dim;
                int tile_col = idx % head_dim;
                int kv_idx = tile_start + tile_row;

                if (kv_idx < seq_len) {
                    k_tile[tile_row * head_dim + tile_col] = k[kv_idx * head_dim + tile_col];
                    v_tile[tile_row * head_dim + tile_col] = v[kv_idx * head_dim + tile_col];
                } else {
                    k_tile[tile_row * head_dim + tile_col] = 0.0f;
                    v_tile[tile_row * head_dim + tile_col] = 0.0f;
                }
            }
            __syncthreads();

            // Compute attention scores for this query against K tile
            int tile_end = min(tile_start + ATTN_TILE, seq_len);
            for (int kj = tile_start; kj < tile_end; kj++) {
                int local_kj = kj - tile_start;

                // Causal mask: skip future positions
                if (causal && kj > qi) continue;

                // Compute dot product: q[qi] . k[kj]
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q[qi * head_dim + d] * k_tile[local_kj * head_dim + d];
                }
                score *= scale;

                // Online softmax update
                float m_prev = m_i;
                m_i = fmaxf(m_i, score);

                // Rescale previous accumulator
                float correction = expf(m_prev - m_i);
                l_i = l_i * correction + expf(score - m_i);

                // Rescale existing output and add new contribution
                float weight = expf(score - m_i);
                for (int d = 0; d < head_dim; d++) {
                    o[qi * head_dim + d] = o[qi * head_dim + d] * correction
                                         + weight * v_tile[local_kj * head_dim + d];
                }
            }
        }

        // Final normalization: O /= l
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < head_dim; d++) {
            o[qi * head_dim + d] *= inv_l;
        }
    }
}

// ---------- Standard (non-fused) attention for comparison ----------

/**
 * Standard attention: materializes the full N×N attention matrix.
 * Used as a baseline for correctness and performance comparison.
 */
torch::Tensor attention_standard_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool causal
) {
    float scale = 1.0f / sqrtf((float)Q.size(-1));
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) * scale;

    if (causal) {
        int seq_len = Q.size(-2);
        auto mask = torch::triu(
            torch::ones({seq_len, seq_len}, Q.options()) * (-INFINITY),
            /*diagonal=*/1
        );
        scores = scores + mask;
    }

    auto attn_weights = torch::softmax(scores, /*dim=*/-1);
    return torch::matmul(attn_weights, V);
}

// ---------- Host dispatch ----------

torch::Tensor attention_forward_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Inputs must be contiguous");
    TORCH_CHECK(Q.dim() == 3, "Q must be 3D (batch, seq_len, head_dim)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "Q, K, V must have the same shape");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Inputs must be float32");

    int batch_size = Q.size(0);
    int seq_len    = Q.size(1);
    int head_dim   = Q.size(2);

    TORCH_CHECK(head_dim <= 128, "head_dim must be <= 128");

    auto O = torch::zeros_like(Q);
    float scale = 1.0f / sqrtf((float)head_dim);

    // One block per batch element, threads process different query positions
    int block_size = min(seq_len, 256);

    // Shared memory: K tile + V tile, each ATTN_TILE x head_dim floats
    size_t smem_size = 2 * ATTN_TILE * head_dim * sizeof(float);

    attention_forward_kernel<<<batch_size, block_size, smem_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        batch_size,
        seq_len,
        head_dim,
        scale,
        causal
    );

    return O;
}
