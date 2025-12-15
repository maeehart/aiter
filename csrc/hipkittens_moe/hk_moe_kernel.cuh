// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Kernel Header
 * 
 * Optimized Mixture of Experts kernel using HipKittens primitives
 * with XCD-aware scheduling for AMD MI300X/MI325X GPUs.
 * 
 * Requires: HipKittens cdna3 branch for MI300X/MI325X
 * 
 * FP8 Blockscale Support:
 *   - Weights stored as FP8 (e4m3fnuz on MI300 series)
 *   - Per-block scales with configurable block size (default 128x128)
 *   - On-the-fly dequantization: bf16 = fp8 * scale
 */
#pragma once

#include "kittens.cuh"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>
#include <hip/hip_fp8.h>

using namespace kittens;

// FP8 type alias for AMD MI300 series (e4m3fnuz format)
using fp8_t = __hip_fp8_e4m3_fnuz;

// Blockscale configuration for FP8 quantization
// DeepSeek R1 uses 128x128 blocks (per_1x128 in aiter terminology)
namespace fp8_cfg {
    constexpr int SCALE_BLOCK_N = 128;  // Block size in N dimension for scales
    constexpr int SCALE_BLOCK_K = 128;  // Block size in K dimension for scales
}

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

// MoE Stage 1 FP8 Blockscale globals structure
// For FP8 weights with per-block dequantization scales
struct moe_stage1_fp8_globals {
    gl_bf16 hidden_states;      // [1, 1, num_tokens, model_dim] - still bf16 input
    // FP8 weights: [1, num_experts, inter_dim*2, model_dim]
    const fp8_t* w1_fp8;
    // Per-block scales: [num_experts, ceil(inter_dim*2/SCALE_BLOCK_N), ceil(model_dim/SCALE_BLOCK_K)]
    // Flattened to [num_experts, num_scale_blocks]
    const float* w1_scale;
    gl_bf16 intermediate;       // [1, 1, sorted_M, inter_dim*2]
    const int32_t* sorted_ids;
    const int32_t* sorted_expert_ids;
    const int32_t* num_valid_ids;
    int sorted_M;
    int num_tokens;
    int model_dim;
    int inter_dim;
    int num_experts;
    int topk;
    int block_m;
    // Scale tensor dimensions for proper indexing
    int num_scale_n;           // ceil(inter_dim*2 / SCALE_BLOCK_N)
    int num_scale_k;           // ceil(model_dim / SCALE_BLOCK_K)
    hipStream_t stream;
    
    dim3 grid() const {
        int tiles_M = ceil_div(sorted_M, hk_moe_common::BLOCK_SIZE_DEFAULT);
        int tiles_N = ceil_div(inter_dim * 2, hk_moe_common::BLOCK_SIZE_DEFAULT);
        return dim3(tiles_N * tiles_M);
    }
    dim3 block() const { return dim3(hk_moe_common::HK_NUM_THREADS); }
    size_t dynamic_shared_memory() const { return 65536; }
};

// MoE Stage 2 FP8 Blockscale globals structure
struct moe_stage2_fp8_globals {
    gl_bf16 intermediate;       // [1, 1, sorted_M, inter_dim]
    // FP8 weights: [1, num_experts, model_dim, inter_dim]
    const fp8_t* w2_fp8;
    // Per-block scales: [num_experts, ceil(model_dim/SCALE_BLOCK_N), ceil(inter_dim/SCALE_BLOCK_K)]
    const float* w2_scale;
    gl_bf16 output;             // [1, 1, num_tokens, model_dim]
    const int32_t* sorted_ids;
    const int32_t* sorted_expert_ids;
    const int32_t* num_valid_ids;
    const float* sorted_weights;
    int sorted_M;
    int num_tokens;
    int model_dim;
    int inter_dim;
    int inter_row_stride;
    int num_experts;
    int topk;
    int block_m;
    // Scale tensor dimensions
    int num_scale_n;           // ceil(model_dim / SCALE_BLOCK_N)
    int num_scale_k;           // ceil(inter_dim / SCALE_BLOCK_K)
    hipStream_t stream;
    
    dim3 grid() const {
        int tiles_M = ceil_div(sorted_M, hk_moe_common::BLOCK_SIZE_DEFAULT);
        int tiles_N = ceil_div(model_dim, hk_moe_common::BLOCK_SIZE_DEFAULT);
        return dim3(tiles_N * tiles_M);
    }
    dim3 block() const { return dim3(hk_moe_common::HK_NUM_THREADS); }
    size_t dynamic_shared_memory() const { return 65536; }
};

// Forward declarations - BF16 weights
void dispatch_hk_moe_stage1(const moe_stage1_globals& g);
void dispatch_hk_moe_stage2(const moe_stage2_globals& g, float* output_fp32);

// Forward declarations - FP8 weights with blockscale dequantization
void dispatch_hk_moe_stage1_fp8(const moe_stage1_fp8_globals& g);
void dispatch_hk_moe_stage2_fp8(const moe_stage2_fp8_globals& g, float* output_fp32);

// SiLU activation
__device__ __forceinline__ float silu_activation(float x) {
    return x / (1.0f + expf(-x));
}

// FP8 to BF16 dequantization with scale
// Converts a single FP8 value to float, multiplies by scale, then stores as bf16
__device__ __forceinline__ bf16 fp8_to_bf16_scaled(fp8_t val, float scale) {
    float f = static_cast<float>(val);
    return __float2bfloat16(f * scale);
}

// Vectorized FP8 dequantization: load 8 FP8 values, dequantize to 8 bf16 with single scale
// Returns two float2 (representing 8 bf16 values for store_shared_vec)
__device__ __forceinline__ void fp8x8_to_bf16x8_scaled(
    const fp8_t* src,
    float scale,
    float2& out_lo,  // First 4 bf16 as float2 (bit-cast)
    float2& out_hi   // Last 4 bf16 as float2 (bit-cast)
) {
    // Load 8 FP8 values (8 bytes = 2 uint32)
    const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(src);
    uint32_t v0 = src_u32[0];
    uint32_t v1 = src_u32[1];
    
    // Extract individual FP8 bytes and convert to bf16
    bf16 b[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t byte0 = (v0 >> (i * 8)) & 0xFF;
        uint8_t byte1 = (v1 >> (i * 8)) & 0xFF;
        fp8_t f0, f1;
        memcpy(&f0, &byte0, 1);
        memcpy(&f1, &byte1, 1);
        b[i] = __float2bfloat16(static_cast<float>(f0) * scale);
        b[i + 4] = __float2bfloat16(static_cast<float>(f1) * scale);
    }
    
    // Pack bf16 values into float2 for store_shared_vec
    // Each float2 holds 4 bf16 values (8 bytes)
    memcpy(&out_lo, &b[0], sizeof(float2));
    memcpy(&out_hi, &b[4], sizeof(float2));
}
