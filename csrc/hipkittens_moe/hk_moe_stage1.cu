// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Stage 1: Gate-Up Projection
 * 
 * Computes: intermediate = hidden_states @ W1^T
 * 
 * This is a simplified initial implementation for testing.
 * TODO: Add XCD-aware scheduling and optimizations.
 */

#include "hk_moe_kernel.cuh"

// Simple naive kernel for testing - no HipKittens optimizations yet
__global__ void hk_moe_stage1_kernel_naive(
    const bf16* __restrict__ hidden_states,  // [num_tokens, model_dim]
    const bf16* __restrict__ w1,             // [num_experts, inter_dim*2, model_dim]
    bf16* __restrict__ intermediate,         // [sorted_M, inter_dim*2]
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    int sorted_M,
    int num_tokens,
    int model_dim,
    int inter_dim,
    int num_experts,
    int block_m  // Block size from sorting - sorted_expert_ids is per block_m tile
) {
    // Simple 1D grid: each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_n = inter_dim * 2;
    
    if (idx >= sorted_M * total_n) return;
    
    int row = idx / total_n;  // Which sorted token
    int col = idx % total_n;  // Which output dimension
    
    if (row >= sorted_M) return;
    
    // Decode sorted_ids: upper 8 bits = topk_slot, lower 24 bits = token_id
    int packed_id = sorted_ids[row];
    int token_id = packed_id & 0xFFFFFF;
    
    // Get expert ID from sorted_expert_ids - indexed by block_m tiles
    int tile_id = row / block_m;
    int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) {
        intermediate[row * total_n + col] = __float2bfloat16(0.0f);
        return;
    }
    
    // Check if this is a padding slot (use actual num_tokens for bounds)
    if (token_id < 0 || token_id >= num_tokens) {
        intermediate[row * total_n + col] = __float2bfloat16(0.0f);
        return;
    }
    
    // Compute dot product: hidden_states[token_id, :] @ w1[expert_id, col, :]
    float sum = 0.0f;
    for (int k = 0; k < model_dim; k++) {
        float a = __bfloat162float(hidden_states[token_id * model_dim + k]);
        float b = __bfloat162float(w1[expert_id * total_n * model_dim + col * model_dim + k]);
        sum += a * b;
    }
    
    intermediate[row * total_n + col] = __float2bfloat16(sum);
}

void dispatch_hk_moe_stage1(const moe_stage1_globals& g) {
    // Use naive kernel for now
    int total_elements = g.sorted_M * g.inter_dim * 2;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    hk_moe_stage1_kernel_naive<<<num_blocks, threads_per_block, 0, g.stream>>>(
        g.hidden_states.raw_ptr,
        g.w1.raw_ptr,
        g.intermediate.raw_ptr,
        g.sorted_ids,
        g.sorted_expert_ids,
        g.sorted_M,
        g.num_tokens,
        g.model_dim,
        g.inter_dim,
        g.num_experts,
        g.block_m
    );
}
