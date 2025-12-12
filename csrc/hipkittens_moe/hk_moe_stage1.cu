// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Stage 1: Gate-Up Projection
 * 
 * Computes: intermediate = hidden_states @ W1^T
 * 
 * Current: Naive kernel for correctness
 * TODO: Implement tiled GEMM with HipKittens mma + XCD scheduling
 */

#include "hk_moe_kernel.cuh"

using namespace kittens;

// Naive kernel - correct but slow
__global__ void hk_moe_stage1_kernel_naive(
    const bf16* __restrict__ hidden_states,
    const bf16* __restrict__ w1,
    bf16* __restrict__ intermediate,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    int sorted_M,
    int num_tokens,
    int model_dim,
    int inter_dim,
    int num_experts,
    int block_m
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_n = inter_dim * 2;
    
    if (idx >= sorted_M * total_n) return;
    
    int row = idx / total_n;
    int col = idx % total_n;
    
    if (row >= sorted_M) return;
    
    int packed_id = sorted_ids[row];
    int token_id = packed_id & 0xFFFFFF;
    
    int tile_id = row / block_m;
    int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) {
        intermediate[row * total_n + col] = __float2bfloat16(0.0f);
        return;
    }
    
    if (token_id < 0 || token_id >= num_tokens) {
        intermediate[row * total_n + col] = __float2bfloat16(0.0f);
        return;
    }
    
    float sum = 0.0f;
    for (int k = 0; k < model_dim; k++) {
        float a = __bfloat162float(hidden_states[token_id * model_dim + k]);
        float b = __bfloat162float(w1[expert_id * total_n * model_dim + col * model_dim + k]);
        sum += a * b;
    }
    
    intermediate[row * total_n + col] = __float2bfloat16(sum);
}

void dispatch_hk_moe_stage1(const moe_stage1_globals& g) {
    int total_n = g.inter_dim * 2;
    int total_elements = g.sorted_M * total_n;
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
