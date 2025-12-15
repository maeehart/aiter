// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * PyTorch Interface for HipKittens MoE Kernels
 * 
 * Supports both BF16 and FP8 (with blockscale) weight formats.
 */

#include "hk_moe_kernel.cuh"

// Helper to get FP8 pointer from torch tensor
inline const fp8_t* get_fp8_ptr(const torch::Tensor& t) {
    // For AMD MI300 series, FP8 is stored as Float8_e4m3fnuz
    return reinterpret_cast<const fp8_t*>(t.data_ptr());
}

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
 * 
 * Uses AITER-compatible layout: intermediate is [num_tokens, topk, inter_dim]
 * indexed by (token_id, topk_slot) rather than sorted_row.
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
    
    // Allocate intermediate buffer - AITER uses [num_tokens, topk, inter_dim] layout
    // But our kernel uses sorted_row indexing, so we use [sorted_M, inter_dim*2]
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
    
    // Apply G1U1 activation (SiLU) in-place
    {
        constexpr int ACT_BLOCK = 256;
        const int sorted_M_valid = sorted_M;
        int total_elements = sorted_M_valid * inter_dim;
        int num_blocks = (total_elements + ACT_BLOCK - 1) / ACT_BLOCK;
        
        apply_g1u1_activation_kernel<ACT_BLOCK><<<num_blocks, ACT_BLOCK, 0, stream>>>(
            reinterpret_cast<bf16*>(intermediate.data_ptr()),
            sorted_M_valid, inter_dim
        );
    }
    
    // Synchronize to ensure activation completes before slicing
    // hipStreamSynchronize(stream); // Removed: Stage2 is enqueued on the same stream
    
    // Stage 2: Down projection with weighted accumulation
    {
        // Slice intermediate to just the activated part (first half)
        // torch::Tensor activated = intermediate.slice(1, 0, inter_dim).contiguous(); // Removed: Stage2 reads strided
        
        // Allocate fp32 buffer for atomic adds
        torch::Tensor output_fp32 = torch::zeros({num_tokens, model_dim}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device()));
        
        moe_stage2_globals g2 = {
            .intermediate = make_gl_4d(
                reinterpret_cast<bf16*>(intermediate.data_ptr()),
                (size_t)1, (size_t)1, (size_t)sorted_M, (size_t)(inter_dim * 2)
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
            .inter_row_stride = inter_dim * 2,
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

// ============================================================================
// FP8 Blockscale Interface Functions
// ============================================================================

/**
 * HipKittens Fused MoE Forward Pass with FP8 Weights
 * 
 * Supports FP8 (e4m3fnuz) weights with per-block dequantization scales.
 * This matches the DeepSeek R1 production workload format.
 * 
 * Weight layout: [num_experts, N, K] as FP8
 * Scale layout:  [num_experts, ceil(N/128), ceil(K/128)] as float32
 */
torch::Tensor hk_fused_moe_fp8_fwd(
    torch::Tensor hidden_states,     // [num_tokens, model_dim] bf16
    torch::Tensor w1_fp8,            // [num_experts, inter_dim*2, model_dim] fp8
    torch::Tensor w2_fp8,            // [num_experts, model_dim, inter_dim] fp8
    torch::Tensor w1_scale,          // [num_experts, num_scale_n1, num_scale_k1] float
    torch::Tensor w2_scale,          // [num_experts, num_scale_n2, num_scale_k2] float
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
    const int num_experts = w1_fp8.size(0);
    const int inter_dim = w1_fp8.size(1) / 2;  // G1U1
    const int sorted_M = sorted_ids.size(0);
    
    // Scale dimensions
    const int num_scale_n1 = w1_scale.size(1);  // ceil(inter_dim*2 / 128)
    const int num_scale_k1 = w1_scale.size(2);  // ceil(model_dim / 128)
    const int num_scale_n2 = w2_scale.size(1);  // ceil(model_dim / 128)
    const int num_scale_k2 = w2_scale.size(2);  // ceil(inter_dim / 128)
    
    auto options = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(hidden_states.device());
    
    // Allocate intermediate buffer
    torch::Tensor intermediate = torch::zeros({sorted_M, inter_dim * 2}, options);
    
    // Allocate output
    torch::Tensor output = torch::zeros({num_tokens, model_dim}, options);
    
    hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Stage 1: Gate-Up projection with FP8 weights
    {
        moe_stage1_fp8_globals g1 = {
            .hidden_states = make_gl_4d(
                reinterpret_cast<bf16*>(hidden_states.data_ptr()),
                (size_t)1, (size_t)1, (size_t)num_tokens, (size_t)model_dim
            ),
            .w1_fp8 = get_fp8_ptr(w1_fp8),
            .w1_scale = w1_scale.data_ptr<float>(),
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
            .num_scale_n = num_scale_n1,
            .num_scale_k = num_scale_k1,
            .stream = stream
        };
        
        dispatch_hk_moe_stage1_fp8(g1);
    }
    
    // Apply G1U1 activation (SiLU) in-place
    {
        constexpr int ACT_BLOCK = 256;
        const int sorted_M_valid = sorted_M;
        int total_elements = sorted_M_valid * inter_dim;
        int num_blocks = (total_elements + ACT_BLOCK - 1) / ACT_BLOCK;
        
        apply_g1u1_activation_kernel<ACT_BLOCK><<<num_blocks, ACT_BLOCK, 0, stream>>>(
            reinterpret_cast<bf16*>(intermediate.data_ptr()),
            sorted_M_valid, inter_dim
        );
    }
    
    // Stage 2: Down projection with FP8 weights
    {
        torch::Tensor output_fp32 = torch::zeros({num_tokens, model_dim}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device()));
        
        moe_stage2_fp8_globals g2 = {
            .intermediate = make_gl_4d(
                reinterpret_cast<bf16*>(intermediate.data_ptr()),
                (size_t)1, (size_t)1, (size_t)sorted_M, (size_t)(inter_dim * 2)
            ),
            .w2_fp8 = get_fp8_ptr(w2_fp8),
            .w2_scale = w2_scale.data_ptr<float>(),
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
            .inter_row_stride = inter_dim * 2,
            .num_experts = num_experts,
            .topk = topk,
            .block_m = block_m,
            .num_scale_n = num_scale_n2,
            .num_scale_k = num_scale_k2,
            .stream = stream
        };
        
        dispatch_hk_moe_stage2_fp8(g2, output_fp32.data_ptr<float>());
        
        // Convert fp32 back to bf16
        output = output_fp32.to(torch::kBFloat16);
    }
    
    return output;
}

/**
 * HipKittens Fused MoE Forward Pass with FP8 Weights + Fused Activation
 * 
 * This version fuses the G1U1 activation (silu(gate) * up) into Stage1.
 * Benefits:
 *   - One less kernel launch
 *   - 50% reduction in intermediate memory traffic
 * 
 * The intermediate buffer is now [sorted_M, inter_dim] instead of [sorted_M, 2*inter_dim].
 */
torch::Tensor hk_fused_moe_fp8_fused_act_fwd(
    torch::Tensor hidden_states,     // [num_tokens, model_dim] bf16
    torch::Tensor w1_fp8,            // [num_experts, inter_dim*2, model_dim] fp8
    torch::Tensor w2_fp8,            // [num_experts, model_dim, inter_dim] fp8
    torch::Tensor w1_scale,          // [num_experts, num_scale_n1, num_scale_k1] float
    torch::Tensor w2_scale,          // [num_experts, num_scale_n2, num_scale_k2] float
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
    const int num_experts = w1_fp8.size(0);
    const int inter_dim = w1_fp8.size(1) / 2;  // G1U1
    const int sorted_M = sorted_ids.size(0);
    
    // Scale dimensions
    const int num_scale_n1 = w1_scale.size(1);
    const int num_scale_k1 = w1_scale.size(2);
    const int num_scale_n2 = w2_scale.size(1);
    const int num_scale_k2 = w2_scale.size(2);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(hidden_states.device());
    
    // Allocate intermediate buffer - HALF the size since activation is fused!
    torch::Tensor intermediate = torch::zeros({sorted_M, inter_dim}, options);
    
    // Allocate output
    torch::Tensor output = torch::zeros({num_tokens, model_dim}, options);
    
    hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Stage 1: Gate-Up projection with FP8 weights AND fused activation
    {
        moe_stage1_fp8_globals g1 = {
            .hidden_states = make_gl_4d(
                reinterpret_cast<bf16*>(hidden_states.data_ptr()),
                (size_t)1, (size_t)1, (size_t)num_tokens, (size_t)model_dim
            ),
            .w1_fp8 = get_fp8_ptr(w1_fp8),
            .w1_scale = w1_scale.data_ptr<float>(),
            .intermediate = make_gl_4d(
                reinterpret_cast<bf16*>(intermediate.data_ptr()),
                (size_t)1, (size_t)1, (size_t)sorted_M, (size_t)inter_dim  // Only inter_dim cols!
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
            .num_scale_n = num_scale_n1,
            .num_scale_k = num_scale_k1,
            .stream = stream
        };
        
        // Use the fused activation kernel - no separate activation pass!
        dispatch_hk_moe_stage1_fp8_fused_act(g1);
    }
    
    // Stage 2: Down projection with FP8 weights
    {
        torch::Tensor output_fp32 = torch::zeros({num_tokens, model_dim}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device()));
        
        moe_stage2_fp8_globals g2 = {
            .intermediate = make_gl_4d(
                reinterpret_cast<bf16*>(intermediate.data_ptr()),
                (size_t)1, (size_t)1, (size_t)sorted_M, (size_t)inter_dim  // Only inter_dim cols!
            ),
            .w2_fp8 = get_fp8_ptr(w2_fp8),
            .w2_scale = w2_scale.data_ptr<float>(),
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
            .inter_row_stride = inter_dim,  // No longer 2*inter_dim!
            .num_experts = num_experts,
            .topk = topk,
            .block_m = block_m,
            .num_scale_n = num_scale_n2,
            .num_scale_k = num_scale_k2,
            .stream = stream
        };
        
        dispatch_hk_moe_stage2_fp8(g2, output_fp32.data_ptr<float>());
        
        // Convert fp32 back to bf16
        output = output_fp32.to(torch::kBFloat16);
    }
    
    return output;
}

/**
 * HipKittens Fused MoE Forward Pass with FP8 Weights + Fused Activation + No Atomics
 * 
 * EXPERIMENTAL: Uses atomic-free Stage 2 by writing to per-(token, slot) buffer.
 * 
 * Benefits over fused_act version:
 *   - No atomic contention in Stage 2
 *   - Better parallelism for high-collision scenarios
 * 
 * Trade-offs:
 *   - Extra memory: num_tokens * topk * model_dim * 4 bytes
 *   - Extra reduction kernel launch
 * 
 * For DeepSeek R1 (8k tokens, topk=8, model_dim=7168): ~1.87 GB temp buffer
 */
torch::Tensor hk_fused_moe_fp8_noatomic_fwd(
    torch::Tensor hidden_states,     // [num_tokens, model_dim] bf16
    torch::Tensor w1_fp8,            // [num_experts, inter_dim*2, model_dim] fp8
    torch::Tensor w2_fp8,            // [num_experts, model_dim, inter_dim] fp8
    torch::Tensor w1_scale,          // [num_experts, num_scale_n1, num_scale_k1] float
    torch::Tensor w2_scale,          // [num_experts, num_scale_n2, num_scale_k2] float
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
    const int num_experts = w1_fp8.size(0);
    const int inter_dim = w1_fp8.size(1) / 2;  // G1U1: w1 is [E, 2*inter_dim, model_dim]
    const int sorted_M = sorted_ids.size(0);
    const int num_scale_n1 = w1_scale.size(1);
    const int num_scale_k1 = w1_scale.size(2);
    const int num_scale_n2 = w2_scale.size(1);
    const int num_scale_k2 = w2_scale.size(2);
    
    auto bf16_options = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(hidden_states.device());
    auto fp32_options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(hidden_states.device());
    
    // Intermediate buffer: [sorted_M, inter_dim] (fused activation writes only inter_dim cols)
    torch::Tensor intermediate = torch::zeros({sorted_M, inter_dim}, bf16_options);
    torch::Tensor output = torch::empty({num_tokens, model_dim}, bf16_options);
    
    // Temp buffer for atomic-free Stage 2: [num_tokens, topk, model_dim]
    torch::Tensor tmp_buffer = torch::zeros({num_tokens, topk, model_dim}, fp32_options);
    
    hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Stage 1: Gate-Up with Fused Activation
    {
        moe_stage1_fp8_globals g1 = {
            .hidden_states = make_gl_4d(
                reinterpret_cast<bf16*>(hidden_states.data_ptr()),
                (size_t)1, (size_t)1, (size_t)num_tokens, (size_t)model_dim
            ),
            .w1_fp8 = get_fp8_ptr(w1_fp8),
            .w1_scale = w1_scale.data_ptr<float>(),
            .intermediate = make_gl_4d(
                reinterpret_cast<bf16*>(intermediate.data_ptr()),
                (size_t)1, (size_t)1, (size_t)sorted_M, (size_t)inter_dim
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
            .num_scale_n = num_scale_n1,
            .num_scale_k = num_scale_k1,
            .stream = stream
        };
        
        dispatch_hk_moe_stage1_fp8_fused_act(g1);
    }
    
    // Stage 2: Down projection with NO ATOMICS
    {
        torch::Tensor output_fp32 = torch::zeros({num_tokens, model_dim}, fp32_options);
        
        moe_stage2_fp8_globals g2 = {
            .intermediate = make_gl_4d(
                reinterpret_cast<bf16*>(intermediate.data_ptr()),
                (size_t)1, (size_t)1, (size_t)sorted_M, (size_t)inter_dim
            ),
            .w2_fp8 = get_fp8_ptr(w2_fp8),
            .w2_scale = w2_scale.data_ptr<float>(),
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
            .inter_row_stride = inter_dim,
            .num_experts = num_experts,
            .topk = topk,
            .block_m = block_m,
            .num_scale_n = num_scale_n2,
            .num_scale_k = num_scale_k2,
            .stream = stream
        };
        
        // Call atomic-free variant
        dispatch_hk_moe_stage2_fp8_noatomic(
            g2,
            tmp_buffer.data_ptr<float>(),
            output_fp32.data_ptr<float>()
        );
        
        // Convert fp32 back to bf16
        output = output_fp32.to(torch::kBFloat16);
    }
    
    return output;
}

/**
 * Stage 1 FP8 Forward (for testing)
 */
torch::Tensor hk_moe_stage1_fp8_fwd(
    torch::Tensor hidden_states,      // [num_tokens, model_dim] bf16
    torch::Tensor w1_fp8,             // [num_experts, inter_dim*2, model_dim] fp8
    torch::Tensor w1_scale,           // [num_experts, num_scale_n, num_scale_k] float
    torch::Tensor sorted_ids,
    torch::Tensor sorted_expert_ids,
    torch::Tensor num_valid_ids,
    int topk,
    int block_m
) {
    const int num_tokens = hidden_states.size(0);
    const int model_dim = hidden_states.size(1);
    const int num_experts = w1_fp8.size(0);
    const int inter_dim = w1_fp8.size(1) / 2;
    const int sorted_M = sorted_ids.size(0);
    const int num_scale_n = w1_scale.size(1);
    const int num_scale_k = w1_scale.size(2);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(hidden_states.device());
    
    torch::Tensor intermediate = torch::zeros({sorted_M, inter_dim * 2}, options);
    
    hipStream_t stream = at::hip::getCurrentHIPStream();
    
    moe_stage1_fp8_globals g = {
        .hidden_states = make_gl_4d(
            reinterpret_cast<bf16*>(hidden_states.data_ptr()),
            (size_t)1, (size_t)1, (size_t)num_tokens, (size_t)model_dim
        ),
        .w1_fp8 = get_fp8_ptr(w1_fp8),
        .w1_scale = w1_scale.data_ptr<float>(),
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
        .num_scale_n = num_scale_n,
        .num_scale_k = num_scale_k,
        .stream = stream
    };
    
    dispatch_hk_moe_stage1_fp8(g);
    
    return intermediate;
}
