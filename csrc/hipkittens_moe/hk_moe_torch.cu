// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * PyTorch Interface for HipKittens MoE Kernels
 */

#include "hk_moe_kernel.cuh"

// G1U1 activation kernel
template <int BLOCK_SIZE_ACT>
__global__ void apply_g1u1_activation_kernel(
    bf16* __restrict__ data,  // [M, inter_dim*2]
    int M,
    int inter_dim
) {
    int idx = blockIdx.x * BLOCK_SIZE_ACT + threadIdx.x;
    int total = M * inter_dim;
    
    if (idx < total) {
        int row = idx / inter_dim;
        int col = idx % inter_dim;
        
        // Gate is first half, Up is second half
        float gate = static_cast<float>(data[row * inter_dim * 2 + col]);
        float up = static_cast<float>(data[row * inter_dim * 2 + inter_dim + col]);
        
        // Apply SiLU to gate, multiply by up (standard G1U1 convention)
        float result = silu_activation(gate) * up;
        
        // Store in first half
        data[row * inter_dim * 2 + col] = __float2bfloat16(result);
    }
}

/**
 * HipKittens Fused MoE Forward Pass
 */
torch::Tensor hk_fused_moe_fwd(
    torch::Tensor hidden_states,     // [num_tokens, model_dim]
    torch::Tensor w1,                // [num_experts, inter_dim*2, model_dim]
    torch::Tensor w2,                // [num_experts, model_dim, inter_dim]
    torch::Tensor topk_weight,       // [num_tokens, topk]
    torch::Tensor topk_ids,          // [num_tokens, topk]
    torch::Tensor sorted_ids,        // [sorted_M]
    torch::Tensor sorted_weights,    // [sorted_M]
    torch::Tensor sorted_expert_ids, // [num_tiles]
    torch::Tensor num_valid_ids,     // [num_experts]
    int topk,
    int block_m
) {
    const int num_tokens = hidden_states.size(0);
    const int model_dim = hidden_states.size(1);
    const int num_experts = w1.size(0);
    const int inter_dim = w1.size(1) / 2;  // G1U1
    const int sorted_M = sorted_ids.size(0);
    
    auto options = torch::TensorOptions()
        .dtype(hidden_states.dtype())
        .device(hidden_states.device());
    
    // Allocate intermediate buffer
    torch::Tensor intermediate = torch::zeros({sorted_M, inter_dim * 2}, options);
    
    // Allocate output
    torch::Tensor output = torch::zeros({num_tokens, model_dim}, options);
    
    hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Stage 1: Gate-Up projection
    {
        moe_stage1_globals g1 = {
            .hidden_states = make_gl_4d(
                reinterpret_cast<bf16*>(hidden_states.data_ptr()),
                (size_t)1, (size_t)1, (size_t)num_tokens, (size_t)model_dim
            ),
            .w1 = make_gl_4d(
                reinterpret_cast<bf16*>(w1.data_ptr()),
                (size_t)1, (size_t)num_experts, (size_t)(inter_dim * 2), (size_t)model_dim
            ),
            .intermediate = make_gl_4d(
                reinterpret_cast<bf16*>(intermediate.data_ptr()),
                (size_t)1, (size_t)1, (size_t)sorted_M, (size_t)(inter_dim * 2)
            ),
            .sorted_ids = sorted_ids.data_ptr<int32_t>(),
            .sorted_expert_ids = sorted_expert_ids.data_ptr<int32_t>(),
            .num_valid_ids = num_valid_ids.data_ptr<int32_t>(),
            .sorted_M = sorted_M,
            .num_tokens = num_tokens,
            .model_dim = model_dim,
            .inter_dim = inter_dim,
            .num_experts = num_experts,
            .topk = topk,
            .block_m = block_m,
            .stream = stream
        };
        
        dispatch_hk_moe_stage1(g1);
    }
    
    // Apply G1U1 activation (SiLU)
    {
        constexpr int ACT_BLOCK = 256;
        int total_elements = sorted_M * inter_dim;
        int num_blocks = (total_elements + ACT_BLOCK - 1) / ACT_BLOCK;
        
        apply_g1u1_activation_kernel<ACT_BLOCK><<<num_blocks, ACT_BLOCK, 0, stream>>>(
            reinterpret_cast<bf16*>(intermediate.data_ptr()),
            sorted_M, inter_dim
        );
    }
    
    // Stage 2: Down projection with weighted accumulation
    {
        // Slice intermediate to just the activated part (first half)
        torch::Tensor activated = intermediate.slice(1, 0, inter_dim).contiguous();
        
        // Allocate fp32 buffer for atomic adds
        torch::Tensor output_fp32 = torch::zeros({num_tokens, model_dim}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device()));
        
        moe_stage2_globals g2 = {
            .intermediate = make_gl_4d(
                reinterpret_cast<bf16*>(activated.data_ptr()),
                (size_t)1, (size_t)1, (size_t)sorted_M, (size_t)inter_dim
            ),
            .w2 = make_gl_4d(
                reinterpret_cast<bf16*>(w2.data_ptr()),
                (size_t)1, (size_t)num_experts, (size_t)model_dim, (size_t)inter_dim
            ),
            .output = make_gl_4d(
                reinterpret_cast<bf16*>(output.data_ptr()),
                (size_t)1, (size_t)1, (size_t)num_tokens, (size_t)model_dim
            ),
            .sorted_ids = sorted_ids.data_ptr<int32_t>(),
            .sorted_expert_ids = sorted_expert_ids.data_ptr<int32_t>(),
            .num_valid_ids = num_valid_ids.data_ptr<int32_t>(),
            .sorted_weights = sorted_weights.data_ptr<float>(),
            .sorted_M = sorted_M,
            .num_tokens = num_tokens,
            .model_dim = model_dim,
            .inter_dim = inter_dim,
            .num_experts = num_experts,
            .topk = topk,
            .block_m = block_m,
            .stream = stream
        };
        
        dispatch_hk_moe_stage2(g2, output_fp32.data_ptr<float>());
        
        // Convert fp32 back to bf16
        output = output_fp32.to(torch::kBFloat16);
    }
    
    return output;
}

/**
 * Stage 1 Forward (for testing)
 */
torch::Tensor hk_moe_stage1_fwd(
    torch::Tensor hidden_states,
    torch::Tensor w1,
    torch::Tensor sorted_ids,
    torch::Tensor sorted_expert_ids,
    torch::Tensor num_valid_ids,
    int topk,
    int block_m
) {
    const int num_tokens = hidden_states.size(0);
    const int model_dim = hidden_states.size(1);
    const int num_experts = w1.size(0);
    const int inter_dim = w1.size(1) / 2;
    const int sorted_M = sorted_ids.size(0);
    
    auto options = torch::TensorOptions()
        .dtype(hidden_states.dtype())
        .device(hidden_states.device());
    
    torch::Tensor intermediate = torch::zeros({sorted_M, inter_dim * 2}, options);
    
    hipStream_t stream = at::hip::getCurrentHIPStream();
    
    moe_stage1_globals g = {
        .hidden_states = make_gl_4d(
            reinterpret_cast<bf16*>(hidden_states.data_ptr()),
            (size_t)1, (size_t)1, (size_t)num_tokens, (size_t)model_dim
        ),
        .w1 = make_gl_4d(
            reinterpret_cast<bf16*>(w1.data_ptr()),
            (size_t)1, (size_t)num_experts, (size_t)(inter_dim * 2), (size_t)model_dim
        ),
        .intermediate = make_gl_4d(
            reinterpret_cast<bf16*>(intermediate.data_ptr()),
            (size_t)1, (size_t)1, (size_t)sorted_M, (size_t)(inter_dim * 2)
        ),
        .sorted_ids = sorted_ids.data_ptr<int32_t>(),
        .sorted_expert_ids = sorted_expert_ids.data_ptr<int32_t>(),
        .num_valid_ids = num_valid_ids.data_ptr<int32_t>(),
        .sorted_M = sorted_M,
        .num_tokens = num_tokens,
        .model_dim = model_dim,
        .inter_dim = inter_dim,
        .num_experts = num_experts,
        .topk = topk,
        .block_m = block_m,
        .stream = stream
    };
    
    dispatch_hk_moe_stage1(g);
    
    return intermediate;
}

/**
 * Stage 2 Forward (for testing)
 */
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
) {
    const int sorted_M = intermediate.size(0);
    const int inter_dim = intermediate.size(1);
    const int num_tokens = output.size(0);
    const int model_dim = output.size(1);
    const int num_experts = w2.size(0);
    
    hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Allocate fp32 buffer for atomic adds
    torch::Tensor output_fp32 = torch::zeros({num_tokens, model_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(intermediate.device()));
    
    moe_stage2_globals g = {
        .intermediate = make_gl_4d(
            reinterpret_cast<bf16*>(intermediate.data_ptr()),
            (size_t)1, (size_t)1, (size_t)sorted_M, (size_t)inter_dim
        ),
        .w2 = make_gl_4d(
            reinterpret_cast<bf16*>(w2.data_ptr()),
            (size_t)1, (size_t)num_experts, (size_t)model_dim, (size_t)inter_dim
        ),
        .output = make_gl_4d(
            reinterpret_cast<bf16*>(output.data_ptr()),
            (size_t)1, (size_t)1, (size_t)num_tokens, (size_t)model_dim
        ),
        .sorted_ids = sorted_ids.data_ptr<int32_t>(),
        .sorted_expert_ids = sorted_expert_ids.data_ptr<int32_t>(),
        .num_valid_ids = num_valid_ids.data_ptr<int32_t>(),
        .sorted_weights = topk_weights.to(torch::kFloat32).data_ptr<float>(),
        .sorted_M = sorted_M,
        .num_tokens = num_tokens,
        .model_dim = model_dim,
        .inter_dim = inter_dim,
        .num_experts = num_experts,
        .topk = topk,
        .block_m = block_m,
        .stream = stream
    };
    
    dispatch_hk_moe_stage2(g, output_fp32.data_ptr<float>());
    
    // Copy fp32 result to bf16 output
    output.copy_(output_fp32.to(torch::kBFloat16));
}
