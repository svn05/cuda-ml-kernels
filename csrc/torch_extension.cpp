/**
 * torch_extension.cpp — PyTorch C++ extension bindings for CUDA ML kernels.
 *
 * Registers custom operators using the modern TORCH_LIBRARY API, making
 * them accessible from Python via torch.ops.cuda_ml_kernels.<op_name>.
 *
 * Each kernel has a C++ dispatch function that validates inputs and calls
 * the CUDA implementation. Forward declarations link to the .cu files.
 */

#include <torch/extension.h>
#include <vector>

// ---------- Forward declarations (implemented in .cu files) ----------

// Softmax
torch::Tensor softmax_forward_cuda(torch::Tensor input);
torch::Tensor softmax_backward_cuda(torch::Tensor grad_output, torch::Tensor softmax_output);

// Matrix multiplication
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_naive_cuda(torch::Tensor A, torch::Tensor B);

// Layer normalization
std::vector<torch::Tensor> layernorm_forward_cuda(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);
std::vector<torch::Tensor> layernorm_backward_cuda(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor gamma,
    torch::Tensor mean_saved, torch::Tensor rstd_saved);

// Attention
torch::Tensor attention_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal);

// ---------- Binding via pybind11 ----------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA ML Kernels — GPU-accelerated ML operations";

    // Softmax
    m.def("softmax_forward", &softmax_forward_cuda,
          "Softmax forward (CUDA) — online algorithm with warp reductions",
          py::arg("input"));
    m.def("softmax_backward", &softmax_backward_cuda,
          "Softmax backward (CUDA)",
          py::arg("grad_output"), py::arg("softmax_output"));

    // Matmul
    m.def("matmul", &matmul_cuda,
          "Tiled matrix multiplication (CUDA) — shared memory with boundary checks",
          py::arg("A"), py::arg("B"));
    m.def("matmul_naive", &matmul_naive_cuda,
          "Naive matrix multiplication (CUDA) — baseline for comparison",
          py::arg("A"), py::arg("B"));

    // Layer normalization
    m.def("layernorm_forward", &layernorm_forward_cuda,
          "Fused LayerNorm forward (CUDA) — Welford's online algorithm",
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f);
    m.def("layernorm_backward", &layernorm_backward_cuda,
          "LayerNorm backward (CUDA)",
          py::arg("grad_output"), py::arg("input"), py::arg("gamma"),
          py::arg("mean_saved"), py::arg("rstd_saved"));

    // Attention
    m.def("attention_forward", &attention_forward_cuda,
          "Fused scaled dot-product attention (CUDA) — FlashAttention-inspired tiling",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("causal") = false);
}
