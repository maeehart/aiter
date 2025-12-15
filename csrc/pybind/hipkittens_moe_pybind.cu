// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * PyBind11 Bindings for HipKittens MoE Kernels
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Forward declarations from hk_moe_torch.cu
torch::Tensor hk_fused_moe_fwd(
    torch::Tensor hidden_states,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor topk_weight,
    torch::Tensor topk_ids,
    torch::Tensor sorted_ids,
    torch::Tensor sorted_weights,
    torch::Tensor sorted_expert_ids,
    torch::Tensor num_valid_ids,
    int topk,
    int block_m
);

torch::Tensor hk_moe_stage1_fwd(
    torch::Tensor hidden_states,
    torch::Tensor w1,
    torch::Tensor sorted_ids,
    torch::Tensor sorted_expert_ids,
    torch::Tensor num_valid_ids,
    int topk,
    int block_m
);

void hk_moe_stage2_fwd(
    torch::Tensor intermediate,
    torch::Tensor w2,
    torch::Tensor output,
    torch::Tensor sorted_ids,
    torch::Tensor sorted_expert_ids,
    torch::Tensor num_valid_ids,
    torch::Tensor topk_weights,
    int topk,
    int block_m
);

// FP8 Blockscale variants
torch::Tensor hk_fused_moe_fp8_fwd(
    torch::Tensor hidden_states,
    torch::Tensor w1_fp8,
    torch::Tensor w2_fp8,
    torch::Tensor w1_scale,
    torch::Tensor w2_scale,
    torch::Tensor topk_weight,
    torch::Tensor topk_ids,
    torch::Tensor sorted_ids,
    torch::Tensor sorted_weights,
    torch::Tensor sorted_expert_ids,
    torch::Tensor num_valid_ids,
    int topk,
    int block_m
);

torch::Tensor hk_moe_stage1_fp8_fwd(
    torch::Tensor hidden_states,
    torch::Tensor w1_fp8,
    torch::Tensor w1_scale,
    torch::Tensor sorted_ids,
    torch::Tensor sorted_expert_ids,
    torch::Tensor num_valid_ids,
    int topk,
    int block_m
);

namespace py = pybind11;

PYBIND11_MODULE(module_hipkittens_moe, m) {
    m.doc() = "HipKittens MoE kernels with XCD-aware scheduling for AMD MI300X/MI325X GPUs";
    
    m.def("hk_fused_moe_fwd", &hk_fused_moe_fwd,
          "HipKittens Fused MoE Forward Pass\n"
          "\n"
          "Computes the full MoE forward pass with XCD-aware scheduling:\n"
          "1. Stage 1: hidden_states @ W1^T (gate-up projection)\n"
          "2. G1U1 Activation: gate * silu(up)\n"
          "3. Stage 2: activated @ W2^T with weighted accumulation\n"
          "\n"
          "Args:\n"
          "    hidden_states: Input tensor [num_tokens, model_dim]\n"
          "    w1: Gate-Up weights [num_experts, inter_dim*2, model_dim]\n"
          "    w2: Down weights [num_experts, model_dim, inter_dim]\n"
          "    topk_weight: Routing weights [num_tokens, topk]\n"
          "    topk_ids: Expert assignments [num_tokens, topk]\n"
          "    sorted_ids: Sorted token IDs [padded_M]\n"
          "    sorted_weights: Sorted routing weights [padded_M]\n"
          "    sorted_expert_ids: Expert ID per tile [num_tiles]\n"
          "    num_valid_ids: Valid tokens per expert [num_experts]\n"
          "    topk: Number of experts per token\n"
          "    block_m: Block size for M dimension\n"
          "\n"
          "Returns:\n"
          "    Output tensor [num_tokens, model_dim]",
          py::arg("hidden_states"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("topk_weight"),
          py::arg("topk_ids"),
          py::arg("sorted_ids"),
          py::arg("sorted_weights"),
          py::arg("sorted_expert_ids"),
          py::arg("num_valid_ids"),
          py::arg("topk"),
          py::arg("block_m"));
    
    m.def("hk_moe_stage1_fwd", &hk_moe_stage1_fwd,
          "HipKittens MoE Stage 1 (Gate-Up Projection)\n"
          "\n"
          "Computes: intermediate = hidden_states @ W1^T",
          py::arg("hidden_states"),
          py::arg("w1"),
          py::arg("sorted_ids"),
          py::arg("sorted_expert_ids"),
          py::arg("num_valid_ids"),
          py::arg("topk"),
          py::arg("block_m"));
    
    m.def("hk_moe_stage2_fwd", &hk_moe_stage2_fwd,
          "HipKittens MoE Stage 2 (Down Projection)\n"
          "\n"
          "Computes: output += weight * (intermediate @ W2^T)",
          py::arg("intermediate"),
          py::arg("w2"),
          py::arg("output"),
          py::arg("sorted_ids"),
          py::arg("sorted_expert_ids"),
          py::arg("num_valid_ids"),
          py::arg("topk_weights"),
          py::arg("topk"),
          py::arg("block_m"));
    
    // FP8 Blockscale variants
    m.def("hk_fused_moe_fp8_fwd", &hk_fused_moe_fp8_fwd,
          "HipKittens Fused MoE Forward Pass with FP8 Weights\n"
          "\n"
          "Same as hk_fused_moe_fwd but with FP8 quantized weights and blockscale.\n"
          "Supports DeepSeek R1 production workload format (e4m3fnuz, 128x128 blocks).\n"
          "\n"
          "Args:\n"
          "    hidden_states: Input tensor [num_tokens, model_dim] bf16\n"
          "    w1_fp8: Gate-Up weights [num_experts, inter_dim*2, model_dim] fp8\n"
          "    w2_fp8: Down weights [num_experts, model_dim, inter_dim] fp8\n"
          "    w1_scale: W1 scales [num_experts, ceil(N/128), ceil(K/128)] float\n"
          "    w2_scale: W2 scales [num_experts, ceil(N/128), ceil(K/128)] float\n"
          "    topk_weight: Routing weights [num_tokens, topk]\n"
          "    topk_ids: Expert assignments [num_tokens, topk]\n"
          "    sorted_ids: Sorted token IDs [padded_M]\n"
          "    sorted_weights: Sorted routing weights [padded_M]\n"
          "    sorted_expert_ids: Expert ID per tile [num_tiles]\n"
          "    num_valid_ids: Valid tokens per expert [num_experts]\n"
          "    topk: Number of experts per token\n"
          "    block_m: Block size for M dimension\n"
          "\n"
          "Returns:\n"
          "    Output tensor [num_tokens, model_dim]",
          py::arg("hidden_states"),
          py::arg("w1_fp8"),
          py::arg("w2_fp8"),
          py::arg("w1_scale"),
          py::arg("w2_scale"),
          py::arg("topk_weight"),
          py::arg("topk_ids"),
          py::arg("sorted_ids"),
          py::arg("sorted_weights"),
          py::arg("sorted_expert_ids"),
          py::arg("num_valid_ids"),
          py::arg("topk"),
          py::arg("block_m"));
    
    m.def("hk_moe_stage1_fp8_fwd", &hk_moe_stage1_fp8_fwd,
          "HipKittens MoE Stage 1 with FP8 Weights (Gate-Up Projection)\n"
          "\n"
          "Computes: intermediate = hidden_states @ W1^T with FP8 dequantization",
          py::arg("hidden_states"),
          py::arg("w1_fp8"),
          py::arg("w1_scale"),
          py::arg("sorted_ids"),
          py::arg("sorted_expert_ids"),
          py::arg("num_valid_ids"),
          py::arg("topk"),
          py::arg("block_m"));
}
