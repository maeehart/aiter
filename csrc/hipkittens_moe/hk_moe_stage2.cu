// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Stage 2: Down Projection with Weighted Accumulation
 * 
 * Computes: output += topk_weight * (intermediate @ W2^T)
 * 
 * This is a simplified initial implementation for testing.
 * TODO: Add HipKittens optimizations once basic correctness is verified.
 */

#include "hk_moe_kernel.cuh"

// Simple naive kernel for testing - no HipKittens optimizations yet
// This properly handles arbitrary dimensions without the tile alignment requirements
__global__ void hk_moe_stage2_kernel_naive(
    const bf16* __restrict__ intermediate,   // [sorted_M, inter_dim]
    const bf16* __restrict__ w2,             // [num_experts, model_dim, inter_dim]
    float* __restrict__ output_fp32,         // [num_tokens, model_dim] - use fp32 for atomic adds
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,  // [sorted_M] - weights already in sorted order
    int sorted_M,
    int num_tokens,
    int model_dim,
    int inter_dim,
    int num_experts,
    int topk,
    int block_m  // Block size from sorting - sorted_expert_ids is per block_m tile
) {
    // Simple 1D grid: each thread handles one output element contribution
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= sorted_M * model_dim) return;
    
    int sorted_row = idx / model_dim;  // Which sorted token
    int col = idx % model_dim;         // Which output dimension
    
    if (sorted_row >= sorted_M) return;
    
    // Decode sorted_ids: upper 8 bits = topk_slot, lower 24 bits = token_id
    int packed_id = sorted_ids[sorted_row];
    int token_id = packed_id & 0xFFFFFF;
    
    // Check if this is a padding slot
    if (token_id < 0 || token_id >= num_tokens) return;
    
    // Get expert ID from sorted_expert_ids - indexed by block_m tiles
    int tile_id = sorted_row / block_m;
    int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    // Compute dot product: intermediate[sorted_row, :] @ w2[expert_id, col, :]
    float sum = 0.0f;
    for (int k = 0; k < inter_dim; k++) {
        float a = __bfloat162float(intermediate[sorted_row * inter_dim + k]);
        float b = __bfloat162float(w2[expert_id * model_dim * inter_dim + col * inter_dim + k]);
        sum += a * b;
    }
    
    // Get weight from sorted_weights - already in correct order!
    float weight = sorted_weights[sorted_row];
    
    // Atomic add to output (since multiple experts contribute to the same token)
    atomicAdd(&output_fp32[token_id * model_dim + col], sum * weight);
}

void dispatch_hk_moe_stage2(const moe_stage2_globals& g, float* output_fp32) {
    // Use naive kernel with fp32 output for atomic adds
    int total_elements = g.sorted_M * g.model_dim;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    hk_moe_stage2_kernel_naive<<<num_blocks, threads_per_block, 0, g.stream>>>(
        g.intermediate.raw_ptr,
        g.w2.raw_ptr,
        output_fp32,
        g.sorted_ids,
        g.sorted_expert_ids,
        g.sorted_weights,
        g.sorted_M,
        g.num_tokens,
        g.model_dim,
        g.inter_dim,
        g.num_experts,
        g.topk,
        g.block_m
    );
}
