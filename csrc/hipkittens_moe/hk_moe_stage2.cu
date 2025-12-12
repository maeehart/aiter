// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Stage 2: Down Projection with Weighted Accumulation
 * 
 * Optimized implementation with:
 * - Tiled computation
 * - Shared memory for weight reuse
 * - fp32 atomic accumulation
 * - XCD-aware scheduling for L2 cache optimization
 */

#include "hk_moe_kernel.cuh"

using namespace kittens;

// Configuration
namespace s2_cfg {
    constexpr int BLOCK_M = 32;      // Sorted rows per block
    constexpr int BLOCK_N = 128;     // Output cols per block (reduced for shared memory)
    constexpr int BLOCK_K = 64;      // K per iteration
    constexpr int NUM_THREADS = 256; // 2 threads per output column
    constexpr int VEC_SIZE = 8;      // bf16 elements per float4 load
    
    // XCD-aware scheduling parameters
    constexpr int WGM = 4;           // Workgroup grouping factor for L2 locality
}

// XCD-aware block ID transformation
__device__ __forceinline__ int xcd_transform_chunked_s2(
    int workgroup_id, 
    int num_workgroups,
    int num_xcds,
    int chunk_size 
) {
    int xcd = workgroup_id % num_xcds;
    int block = num_xcds * chunk_size;
    int limit = (num_workgroups / block) * block;
    if (workgroup_id > limit) return workgroup_id;
    int local_pid = workgroup_id / num_xcds;
    int chunk_idx = local_pid / chunk_size;
    int pos_in_chunk = local_pid % chunk_size;
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}

// Tiled kernel with shared memory and XCD-aware scheduling
__global__ __launch_bounds__(s2_cfg::NUM_THREADS, 2)
void hk_moe_stage2_kernel_tiled(
    const bf16* __restrict__ intermediate,   // [sorted_M, inter_dim]
    const bf16* __restrict__ w2,             // [num_experts, model_dim, inter_dim]
    float* __restrict__ output_fp32,         // [num_tokens, model_dim]
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    const int sorted_M,
    const int num_tokens,
    const int model_dim,
    const int inter_dim,
    const int num_experts,
    const int topk,
    const int block_m_sorting,
    const int num_m_blocks,
    const int num_n_blocks
) {
    using namespace s2_cfg;
    
    __shared__ float s_input[BLOCK_M][BLOCK_K + 8];
    __shared__ float s_weight[BLOCK_N][BLOCK_K + 8];
    
    // XCD-aware block scheduling
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    
    // Apply XCD-aware transformation (use kittens::NUM_XCDS = 8 for MI300X/MI325X)
    wgid = xcd_transform_chunked_s2(wgid, NUM_WGS, kittens::NUM_XCDS, WGM * WGM);
    
    // Swizzle for better L2 within the same XCD
    int num_wgid_in_group = WGM * num_n_blocks;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_m_blocks - first_pid_m, WGM);
    
    // Compute block indices with swizzling
    int block_m_idx, block_n_idx;
    if (group_size_m > 0) {
        block_m_idx = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
        block_n_idx = (wgid % num_wgid_in_group) / group_size_m;
    } else {
        block_m_idx = blockIdx.x;
        block_n_idx = blockIdx.y;
    }
    
    const int row_start = block_m_idx * BLOCK_M;
    const int col_start = block_n_idx * BLOCK_N;
    
    if (row_start >= sorted_M) return;
    
    const int tile_id = row_start / block_m_sorting;
    const int expert_id = sorted_expert_ids[tile_id];
    
    // Thread mapping: first BLOCK_N threads each handle one output column
    const int thread_col = threadIdx.x % BLOCK_N;
    const int col = col_start + thread_col;
    const bool col_valid = (col < model_dim) && (threadIdx.x < BLOCK_N);
    
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    float acc[BLOCK_M];
    #pragma unroll
    for (int m = 0; m < BLOCK_M; m++) {
        acc[m] = 0.0f;
    }
    
    const int num_k_tiles = (inter_dim + BLOCK_K - 1) / BLOCK_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * BLOCK_K;
        
        __syncthreads();
        
        // Cooperative vectorized load of input tile [BLOCK_M x BLOCK_K]
        {
            constexpr int TOTAL_ELEMS = BLOCK_M * BLOCK_K;
            constexpr int ELEMS_PER_THREAD = TOTAL_ELEMS / NUM_THREADS;
            
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i += VEC_SIZE) {
                int idx = threadIdx.x * ELEMS_PER_THREAD + i;
                int m = idx / BLOCK_K;
                int k = idx % BLOCK_K;
                
                if (m < BLOCK_M) {
                    int row = row_start + m;
                    float vals[VEC_SIZE] = {0.0f};
                    
                    if (row < sorted_M && (k_start + k + VEC_SIZE - 1) < inter_dim) {
                        // Vectorized load
                        float4 vec = *reinterpret_cast<const float4*>(&intermediate[row * inter_dim + k_start + k]);
                        const bf16* bf = reinterpret_cast<const bf16*>(&vec);
                        #pragma unroll
                        for (int j = 0; j < VEC_SIZE; j++) {
                            vals[j] = __bfloat162float(bf[j]);
                        }
                    } else if (row < sorted_M) {
                        // Scalar fallback for boundary
                        for (int j = 0; j < VEC_SIZE && (k_start + k + j) < inter_dim; j++) {
                            vals[j] = __bfloat162float(intermediate[row * inter_dim + k_start + k + j]);
                        }
                    }
                    
                    #pragma unroll
                    for (int j = 0; j < VEC_SIZE; j++) {
                        s_input[m][k + j] = vals[j];
                    }
                }
            }
        }
        
        // Cooperative vectorized load of weight tile [BLOCK_N x BLOCK_K]
        {
            constexpr int TOTAL_ELEMS = BLOCK_N * BLOCK_K;
            constexpr int ELEMS_PER_THREAD = TOTAL_ELEMS / NUM_THREADS;
            
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i += VEC_SIZE) {
                int idx = threadIdx.x * ELEMS_PER_THREAD + i;
                int n = idx / BLOCK_K;
                int k = idx % BLOCK_K;
                
                if (n < BLOCK_N) {
                    int out_col = col_start + n;
                    float vals[VEC_SIZE] = {0.0f};
                    
                    if (out_col < model_dim && (k_start + k + VEC_SIZE - 1) < inter_dim) {
                        // Vectorized load
                        float4 vec = *reinterpret_cast<const float4*>(&w2[expert_id * model_dim * inter_dim + out_col * inter_dim + k_start + k]);
                        const bf16* bf = reinterpret_cast<const bf16*>(&vec);
                        #pragma unroll
                        for (int j = 0; j < VEC_SIZE; j++) {
                            vals[j] = __bfloat162float(bf[j]);
                        }
                    } else if (out_col < model_dim) {
                        // Scalar fallback for boundary
                        for (int j = 0; j < VEC_SIZE && (k_start + k + j) < inter_dim; j++) {
                            vals[j] = __bfloat162float(w2[expert_id * model_dim * inter_dim + out_col * inter_dim + k_start + k + j]);
                        }
                    }
                    
                    #pragma unroll
                    for (int j = 0; j < VEC_SIZE; j++) {
                        s_weight[n][k + j] = vals[j];
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute - threads < BLOCK_N each compute one column for all rows
        if (col_valid) {
            int k_end = min(BLOCK_K, inter_dim - k_start);
            #pragma unroll
            for (int m = 0; m < BLOCK_M; m++) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < k_end; k++) {
                    sum += s_input[m][k] * s_weight[thread_col][k];
                }
                acc[m] += sum;
            }
        }
    }
    
    // Scatter-add with weights
    if (col_valid) {
        #pragma unroll
        for (int m = 0; m < BLOCK_M; m++) {
            int row = row_start + m;
            if (row < sorted_M) {
                int packed_id = sorted_ids[row];
                int token_id = packed_id & 0xFFFFFF;
                
                if (token_id >= 0 && token_id < num_tokens) {
                    float weight = sorted_weights[row];
                    atomicAdd(&output_fp32[token_id * model_dim + col], acc[m] * weight);
                }
            }
        }
    }
}

void dispatch_hk_moe_stage2(const moe_stage2_globals& g, float* output_fp32) {
    using namespace s2_cfg;
    
    const int num_m_blocks = (g.sorted_M + BLOCK_M - 1) / BLOCK_M;
    const int num_n_blocks = (g.model_dim + BLOCK_N - 1) / BLOCK_N;
    
    dim3 grid(num_m_blocks, num_n_blocks);
    dim3 block(NUM_THREADS);
    
    hk_moe_stage2_kernel_tiled<<<grid, block, 0, g.stream>>>(
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
        g.block_m,
        num_m_blocks,
        num_n_blocks
    );
}
