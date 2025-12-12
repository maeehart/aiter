// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Stage 2: Down Projection with Weighted Accumulation
 * 
 * Computes: output += weight * (intermediate @ W2^T)
 * 
 * Current: Naive kernel for correctness
 * TODO: Implement tiled GEMM with HipKittens mma + XCD scheduling
 */

#include "hk_moe_kernel.cuh"

using namespace kittens;

// Naive kernel - correct but slow
__global__ void hk_moe_stage2_kernel_naive(
    const bf16* __restrict__ intermediate,
    const bf16* __restrict__ w2,
    float* __restrict__ output_fp32,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    int sorted_M,
    int num_tokens,
    int model_dim,
    int inter_dim,
    int num_experts,
    int topk,
    int block_m
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= sorted_M * model_dim) return;
    
    int sorted_row = idx / model_dim;
    int col = idx % model_dim;
    
    if (sorted_row >= sorted_M) return;
    
    int packed_id = sorted_ids[sorted_row];
    int token_id = packed_id & 0xFFFFFF;
    
    if (token_id < 0 || token_id >= num_tokens) return;
    
    int tile_id = sorted_row / block_m;
    int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    float sum = 0.0f;
    for (int k = 0; k < inter_dim; k++) {
        float a = __bfloat162float(intermediate[sorted_row * inter_dim + k]);
        float b = __bfloat162float(w2[expert_id * model_dim * inter_dim + col * inter_dim + k]);
        sum += a * b;
    }
    
    float weight = sorted_weights[sorted_row];
    atomicAdd(&output_fp32[token_id * model_dim + col], sum * weight);
}

void dispatch_hk_moe_stage2(const moe_stage2_globals& g, float* output_fp32) {
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
