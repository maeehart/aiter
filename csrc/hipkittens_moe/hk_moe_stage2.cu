// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Stage 2: Down Projection with Weighted Accumulation
 * 
 * Based on HipKittens GEMM patterns:
 * - Uses st_bf shared tiles + rt_bf/rt_fl register tiles  
 * - mma_ABt for efficient matrix multiplication
 * - XCD-aware scheduling for L2 cache optimization
 * - 8-wave kernel pattern (512 threads)
 * - Weighted scatter-add to output
 * 
 * Reference: https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr
 */

#include "hk_moe_kernel.cuh"

using namespace kittens;

// Configuration following HipKittens GEMM pattern
namespace s2_cfg {
    constexpr int BLOCK_SIZE = 128;   // Output tile size
    constexpr int K_STEP = 64;        // K dimension per iteration
    constexpr int REG_BLOCK = BLOCK_SIZE / 4;  // 32
    constexpr int DOT_SLICE = 16;     // MMA native dimension
    
    constexpr int NUM_WARPS = 8;
    constexpr int NUM_THREADS = WARP_THREADS * NUM_WARPS;  // 512
    
    constexpr int WGM = 4;
}

// Shared tile types
using s2_st_tile = st_bf<s2_cfg::BLOCK_SIZE, s2_cfg::K_STEP>;

__global__ __launch_bounds__(s2_cfg::NUM_THREADS, 2)
void hk_moe_stage2_kernel_mma(
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
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    s2_st_tile (&As) = al.allocate<s2_st_tile>();  // Input tile
    s2_st_tile (&Bs) = al.allocate<s2_st_tile>();  // Weight tile
    
    // Register tiles for MMA
    rt_bf<REG_BLOCK, DOT_SLICE> a_tile, b_tile;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];
    zero(C_accum[0]);
    zero(C_accum[1]);
    
    // XCD-aware block scheduling
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);
    
    int num_wgid_in_group = WGM * num_n_blocks;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_m_blocks - first_pid_m, WGM);
    
    int pid_m, pid_n;
    if (group_size_m > 0) {
        pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
        pid_n = (wgid % num_wgid_in_group) / group_size_m;
    } else {
        return;
    }
    
    const int row_start = pid_m * BLOCK_SIZE;
    const int col_start = pid_n * BLOCK_SIZE;
    
    if (row_start >= sorted_M || col_start >= model_dim) return;
    
    const int tile_id = row_start / block_m_sorting;
    const int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    const int warp_id = warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    
    const int num_k_tiles = (inter_dim + K_STEP - 1) / K_STEP;
    
    const int lane = threadIdx.x;
    constexpr int VEC_SIZE = 8;  // bf16 elements per float4
    constexpr int TOTAL_VECS = (BLOCK_SIZE * K_STEP) / VEC_SIZE;
    constexpr int VECS_PER_THREAD = TOTAL_VECS / NUM_THREADS;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * K_STEP;
        
        // === Cooperative vectorized load input tile [BLOCK_SIZE x K_STEP] ===
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_idx = lane * VECS_PER_THREAD + v;
            int flat_idx = vec_idx * VEC_SIZE;
            int m = flat_idx / K_STEP;
            int k = flat_idx % K_STEP;
            
            int row = row_start + m;
            bf16 vals[VEC_SIZE];
            
            if (row < sorted_M && (k_start + k + VEC_SIZE - 1) < inter_dim) {
                float4 vec = *reinterpret_cast<const float4*>(&intermediate[row * inter_dim + k_start + k]);
                *reinterpret_cast<float4*>(vals) = vec;
            } else {
                #pragma unroll
                for (int j = 0; j < VEC_SIZE; j++) {
                    vals[j] = (row < sorted_M && (k_start + k + j) < inter_dim) 
                        ? intermediate[row * inter_dim + k_start + k + j]
                        : __float2bfloat16(0.0f);
                }
            }
            
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                As[{m, k + j}] = vals[j];
            }
        }
        
        // === Cooperative vectorized load weight tile [BLOCK_SIZE x K_STEP] ===
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_idx = lane * VECS_PER_THREAD + v;
            int flat_idx = vec_idx * VEC_SIZE;
            int n = flat_idx / K_STEP;
            int k = flat_idx % K_STEP;
            
            int col = col_start + n;
            bf16 vals[VEC_SIZE];
            
            if (col < model_dim && (k_start + k + VEC_SIZE - 1) < inter_dim) {
                float4 vec = *reinterpret_cast<const float4*>(&w2[expert_id * model_dim * inter_dim + col * inter_dim + k_start + k]);
                *reinterpret_cast<float4*>(vals) = vec;
            } else {
                #pragma unroll
                for (int j = 0; j < VEC_SIZE; j++) {
                    vals[j] = (col < model_dim && (k_start + k + j) < inter_dim)
                        ? w2[expert_id * model_dim * inter_dim + col * inter_dim + k_start + k + j]
                        : __float2bfloat16(0.0f);
                }
            }
            
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                Bs[{n, k + j}] = vals[j];
            }
        }
        
        __syncthreads();
        
        // === Compute MMA ===
        #pragma unroll
        for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row * 2, kk}));
            load(b_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0], a_tile, b_tile, C_accum[0]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
            
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row * 2 + 1, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1], a_tile, b_tile, C_accum[1]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
        }
        
        __syncthreads();
    }
    
    // === Weighted scatter-add to output ===
    // Each warp handles its portion of the 128x128 output tile
    // C_accum[0]: rows [warp_row*64, warp_row*64+32), cols [warp_col*32, warp_col*32+32)
    // C_accum[1]: rows [warp_row*64+32, warp_row*64+64), cols [...]
    
    const int out_row_base_0 = row_start + warp_row * 64;
    const int out_row_base_1 = row_start + warp_row * 64 + 32;
    const int out_col_base = col_start + warp_col * 32;
    const int lane_id = laneid();
    const int row_off = 4 * (lane_id / 16);
    const int col_off = lane_id % 16;
    
    // C_accum[0] scatter-add
    #pragma unroll
    for (int tile_row = 0; tile_row < 2; tile_row++) {
        #pragma unroll
        for (int tile_col = 0; tile_col < 2; tile_col++) {
            const auto& tile = C_accum[0].tiles[tile_row][tile_col];
            int base_row = out_row_base_0 + tile_row * 16 + row_off;
            int col = out_col_base + tile_col * 16 + col_off;
            
            if (col < model_dim) {
                float vals[4] = {tile.data[0].x, tile.data[0].y, tile.data[1].x, tile.data[1].y};
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int row = base_row + r;
                    if (row < sorted_M) {
                        int packed_id = sorted_ids[row];
                        int token_id = packed_id & 0xFFFFFF;
                        if (token_id >= 0 && token_id < num_tokens) {
                            float weight = sorted_weights[row];
                            atomicAdd(&output_fp32[token_id * model_dim + col], vals[r] * weight);
                        }
                    }
                }
            }
        }
    }
    
    // C_accum[1] scatter-add
    #pragma unroll
    for (int tile_row = 0; tile_row < 2; tile_row++) {
        #pragma unroll
        for (int tile_col = 0; tile_col < 2; tile_col++) {
            const auto& tile = C_accum[1].tiles[tile_row][tile_col];
            int base_row = out_row_base_1 + tile_row * 16 + row_off;
            int col = out_col_base + tile_col * 16 + col_off;
            
            if (col < model_dim) {
                float vals[4] = {tile.data[0].x, tile.data[0].y, tile.data[1].x, tile.data[1].y};
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int row = base_row + r;
                    if (row < sorted_M) {
                        int packed_id = sorted_ids[row];
                        int token_id = packed_id & 0xFFFFFF;
                        if (token_id >= 0 && token_id < num_tokens) {
                            float weight = sorted_weights[row];
                            atomicAdd(&output_fp32[token_id * model_dim + col], vals[r] * weight);
                        }
                    }
                }
            }
        }
    }
}

void dispatch_hk_moe_stage2(const moe_stage2_globals& g, float* output_fp32) {
    using namespace s2_cfg;
    
    const int num_m_blocks = (g.sorted_M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_n_blocks = (g.model_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 grid(num_m_blocks, num_n_blocks);
    dim3 block(NUM_THREADS);
    
    size_t smem_size = 65536;
    hipFuncSetAttribute((void*)hk_moe_stage2_kernel_mma, 
                        hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    hk_moe_stage2_kernel_mma<<<grid, block, smem_size, g.stream>>>(
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
