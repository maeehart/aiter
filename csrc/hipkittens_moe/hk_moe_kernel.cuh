// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Kernel Header
 * 
 * Optimized Mixture of Experts kernel using HipKittens primitives
 * with XCD-aware scheduling for AMD MI300X/MI325X GPUs.
 * 
 * Requires: HipKittens cdna3 branch for MI300X/MI325X
 */
#pragma once

#include "kittens.cuh"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

using namespace kittens;

// Common configuration constants - used only in struct methods below
namespace hk_moe_common {
    constexpr int BLOCK_SIZE_DEFAULT = 256;
    constexpr int HK_NUM_WARPS = 8;
    constexpr int HK_NUM_THREADS = WARP_THREADS * HK_NUM_WARPS;  // 64 * 8 = 512
}

// Global memory layout types for MoE
// Using 4D layout with all runtime dimensions
using gl_bf16 = gl<bf16, -1, -1, -1, -1>;

// Helper to create gl with proper runtime dimension arguments
template<typename T>
gl<T, -1, -1, -1, -1> make_gl_4d(T* data, size_t batch, size_t depth, size_t rows, size_t cols) {
    return gl<T, -1, -1, -1, -1>(data, batch, depth, rows, cols);
}

// Group type for collective operations
using HKGroup = kittens::group<hk_moe_common::HK_NUM_WARPS>;

// MoE Stage 1 globals structure
struct moe_stage1_globals {
    gl_bf16 hidden_states;      // [1, 1, num_tokens, model_dim]
    gl_bf16 w1;                 // [1, num_experts, inter_dim*2, model_dim]
    gl_bf16 intermediate;       // [1, 1, sorted_M, inter_dim*2]
    const int32_t* sorted_ids;
    const int32_t* sorted_expert_ids;  // Indexed by block_m tiles, NOT BLOCK_SIZE tiles
    const int32_t* num_valid_ids;
    int sorted_M;
    int num_tokens;             // Number of input tokens (for bounds checking)
    int model_dim;
    int inter_dim;
    int num_experts;
    int topk;
    int block_m;                // Block size from sorting (32-128), used for sorted_expert_ids indexing
    hipStream_t stream;
    
    dim3 grid() const {
        int tiles_M = ceil_div(sorted_M, hk_moe_common::BLOCK_SIZE_DEFAULT);
        int tiles_N = ceil_div(inter_dim * 2, hk_moe_common::BLOCK_SIZE_DEFAULT);
        return dim3(tiles_N * tiles_M);
    }
    dim3 block() const { return dim3(hk_moe_common::HK_NUM_THREADS); }
    size_t dynamic_shared_memory() const { return 65536; }
};

// MoE Stage 2 globals structure
struct moe_stage2_globals {
    gl_bf16 intermediate;       // [1, 1, sorted_M, inter_dim]
    gl_bf16 w2;                 // [1, num_experts, model_dim, inter_dim]
    gl_bf16 output;             // [1, 1, num_tokens, model_dim]
    const int32_t* sorted_ids;
    const int32_t* sorted_expert_ids;  // Indexed by block_m tiles
    const int32_t* num_valid_ids;
    const float* sorted_weights;       // [sorted_M] - weights in sorted order, much simpler!
    int sorted_M;
    int num_tokens;
    int model_dim;
    int inter_dim;
    // Row stride (in elements) of the stage2 `intermediate` pointer. This lets Stage2 read the
    // activated first-half view from the Stage1 output buffer ([sorted_M, inter_dim*2])
    // without materializing `activated = intermediate.slice(...).contiguous()`.
    int inter_row_stride;
    int num_experts;
    int topk;
    int block_m;                // Block size from sorting, used for sorted_expert_ids indexing
    hipStream_t stream;
    
    dim3 grid() const {
        int tiles_M = ceil_div(sorted_M, hk_moe_common::BLOCK_SIZE_DEFAULT);
        int tiles_N = ceil_div(model_dim, hk_moe_common::BLOCK_SIZE_DEFAULT);
        return dim3(tiles_N * tiles_M);
    }
    dim3 block() const { return dim3(hk_moe_common::HK_NUM_THREADS); }
    size_t dynamic_shared_memory() const { return 65536; }
};

// Forward declarations
void dispatch_hk_moe_stage1(const moe_stage1_globals& g);
void dispatch_hk_moe_stage2(const moe_stage2_globals& g, float* output_fp32);

// SiLU activation
__device__ __forceinline__ float silu_activation(float x) {
    return x / (1.0f + expf(-x));
}
