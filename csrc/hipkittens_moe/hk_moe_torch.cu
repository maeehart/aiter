// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * PyTorch Interface for HipKittens MoE Kernels
 */

#include "hk_moe_kernel.cuh"

#include <cstdlib>
#include <cstdint>

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

// === Option B (partial): within-expert tile-local reorder by token_id ===
// We keep expert segmentation intact by sorting *within each 128-row tile* (block_m==128).
// This improves hidden_states gather locality without requiring a full segmented sort/merge.
// The sort permutes `sorted_ids` and `sorted_weights` together, in-place.
//
// Enable with: HK_MOE_TILE_SORT_TOKEN=1
constexpr int HK_TILE_SORT_M = 128;

__global__ void hk_moe_tile_sort_by_token_kernel(
    int32_t* __restrict__ sorted_ids,
    float* __restrict__ sorted_weights,
    const int sorted_M,
    const int sorted_M_valid
) {
    __shared__ int32_t s_packed[HK_TILE_SORT_M];
    __shared__ int32_t s_token[HK_TILE_SORT_M];
    __shared__ float   s_w[HK_TILE_SORT_M];

    const int lane = (int)threadIdx.x;
    if (lane >= HK_TILE_SORT_M) return;

    const int tile = (int)blockIdx.x;
    const int base = tile * HK_TILE_SORT_M;
    const int row  = base + lane;

    int32_t packed = 0;
    float w = 0.0f;
    int32_t token_key = 0x7fffffff;

    if (row < sorted_M) {
        packed = sorted_ids[row];
        w = sorted_weights[row];
    }

    // Only valid rows participate; invalid rows get a large key so they sink to the end.
    if (row < sorted_M_valid) {
        const int32_t token_id = packed & 0x00FFFFFF;
        // token_id is expected in-range; treat negative/invalid as large key too.
        token_key = (token_id >= 0) ? token_id : 0x7fffffff;
    } else {
        packed = 0;
        w = 0.0f;
        token_key = 0x7fffffff;
    }

    s_packed[lane] = packed;
    s_w[lane]      = w;
    s_token[lane]  = token_key;
    __syncthreads();

    // Bitonic sort on 128 keys (ascending).
    // Only one thread per pair performs the swap to avoid races.
    for (int k = 2; k <= HK_TILE_SORT_M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = lane ^ j;
            if (ixj > lane) {
                const bool up = ((lane & k) == 0);
                const int32_t a = s_token[lane];
                const int32_t b = s_token[ixj];
                const bool swap = up ? (a > b) : (a < b);
                if (swap) {
                    const int32_t tp = s_packed[lane];
                    const float   tw = s_w[lane];
                    const int32_t tk = s_token[lane];
                    s_packed[lane] = s_packed[ixj];
                    s_w[lane]      = s_w[ixj];
                    s_token[lane]  = s_token[ixj];
                    s_packed[ixj] = tp;
                    s_w[ixj]      = tw;
                    s_token[ixj]  = tk;
                }
            }
            __syncthreads();
        }
    }

    // Write back in-place.
    if (row < sorted_M) {
        sorted_ids[row] = s_packed[lane];
        sorted_weights[row] = s_w[lane];
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

    // Optional: tile-local within-expert reorder by token_id to improve gather coalescing.
    // This is a partial implementation of Option B (segmented sort) that avoids cross-tile merges.
    if (const char* env = std::getenv("HK_MOE_TILE_SORT_TOKEN")) {
        if (std::atoi(env) != 0) {
            // sorted_M_valid is num_valid_ids[0] (stored by moe_sorting), i.e. count of non-padded rows.
            const int sorted_M_valid = num_valid_ids.data_ptr<int32_t>()[0];
            const int num_tiles = (sorted_M + HK_TILE_SORT_M - 1) / HK_TILE_SORT_M;
            hk_moe_tile_sort_by_token_kernel<<<dim3(num_tiles), dim3(HK_TILE_SORT_M), 0, stream>>>(
                sorted_ids.data_ptr<int32_t>(),
                sorted_weights.data_ptr<float>(),
                sorted_M,
                sorted_M_valid
            );
        }
    }
    
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
