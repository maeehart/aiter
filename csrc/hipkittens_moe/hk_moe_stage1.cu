// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Stage 1: Gate-Up Projection with MMA Instructions
 * 
 * Based on HipKittens GEMM patterns:
 * - Uses st_bf shared tiles + rt_bf/rt_fl register tiles
 * - mma_ABt for efficient matrix multiplication
 * - XCD-aware scheduling for L2 cache optimization
 * - 8-wave kernel pattern (512 threads)
 * 
 * Reference: https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr
 */

#include "hk_moe_kernel.cuh"

using namespace kittens;

// Configuration following HipKittens GEMM pattern
namespace s1_cfg {
    constexpr int BLOCK_SIZE = 128;   // Output tile size (M and N dimension of output tile)
    constexpr int K_STEP = 64;        // K dimension per iteration
    constexpr int REG_BLOCK = BLOCK_SIZE / 4;  // 32 - register tile dimension per warp
    constexpr int DOT_SLICE = 16;     // MMA native dimension (16x16x16)
    
    constexpr int NUM_WARPS = 8;
    constexpr int NUM_THREADS = WARP_THREADS * NUM_WARPS;  // 64 * 8 = 512
    
    // XCD-aware scheduling parameters
    constexpr int WGM = 4;            // Workgroup grouping factor for L2 locality
}

// Shared tile types for input and weight
using s1_st_tile = st_bf<s1_cfg::BLOCK_SIZE, s1_cfg::K_STEP>;  // [128, 64] shared bf16 tile

// Group for cooperative loading
using s1_group = group<s1_cfg::NUM_WARPS>;

__global__ __launch_bounds__(s1_cfg::NUM_THREADS, 2)
void hk_moe_stage1_kernel_mma(
    const bf16* __restrict__ hidden_states,  // [num_tokens, model_dim]
    const bf16* __restrict__ w1,             // [num_experts, inter_dim*2, model_dim]
    bf16* __restrict__ intermediate,         // [sorted_M, inter_dim*2]
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const int sorted_M,
    const int num_tokens,
    const int model_dim,
    const int inter_dim,
    const int num_experts,
    const int block_m_sorting,
    const int num_m_blocks,
    const int num_n_blocks
) {
    using namespace s1_cfg;
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    // Allocate shared memory tiles
    s1_st_tile (&As) = al.allocate<s1_st_tile>();   // Input tile [BLOCK_SIZE, K_STEP]
    s1_st_tile (&Bs) = al.allocate<s1_st_tile>();   // Weight tile [BLOCK_SIZE, K_STEP]
    
    const int total_n = inter_dim * 2;
    
    // Register tiles for MMA - 32x16 bf16 tiles and 32x32 fp32 accumulators
    rt_bf<REG_BLOCK, DOT_SLICE> a_tile, b_tile;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];  // Two 32x32 accumulators
    zero(C_accum[0]);
    zero(C_accum[1]);
    
    // XCD-aware block scheduling
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    
    // Apply XCD-aware transformation
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);
    
    // Swizzle for better L2 within the same XCD
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
    
    if (row_start >= sorted_M || col_start >= total_n) return;
    
    // Get expert for this block (using first row in tile)
    const int tile_id = row_start / block_m_sorting;
    const int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    // Warp mapping: 8 warps arranged in a 2x4 grid
    // Warps 0-3 are row 0, warps 4-7 are row 1
    const int warp_id = warpid();
    const int warp_row = warp_id / 4;  // 0 or 1
    const int warp_col = warp_id % 4;  // 0, 1, 2, or 3
    
    const int num_k_tiles = (model_dim + K_STEP - 1) / K_STEP;
    
    const int lane = threadIdx.x;
    constexpr int VEC_SIZE = 8;  // bf16 elements per float4
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * K_STEP;
        
        // === Cooperative vectorized load input tile with gather via sorted_ids ===
        // For gather: we load K values per row with vectorization
        // Each thread handles some complete rows (to enable vec loads within a row)
        constexpr int ROWS_PER_BLOCK = BLOCK_SIZE;
        constexpr int VECS_PER_ROW = K_STEP / VEC_SIZE;  // 64/8 = 8 vectors per row
        constexpr int TOTAL_VECS = ROWS_PER_BLOCK * VECS_PER_ROW;  // 128 * 8 = 1024
        constexpr int VECS_PER_THREAD = TOTAL_VECS / NUM_THREADS;  // 1024/512 = 2
        
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_idx = lane * VECS_PER_THREAD + v;
            int m = vec_idx / VECS_PER_ROW;
            int k = (vec_idx % VECS_PER_ROW) * VEC_SIZE;
            
            int row = row_start + m;
            bf16 vals[VEC_SIZE];
            
            if (row < sorted_M && (k_start + k + VEC_SIZE - 1) < model_dim) {
                int packed_id = sorted_ids[row];
                int token_id = packed_id & 0xFFFFFF;
                if (token_id >= 0 && token_id < num_tokens) {
                    // Vectorized load from hidden_states
                    float4 vec = *reinterpret_cast<const float4*>(&hidden_states[token_id * model_dim + k_start + k]);
                    *reinterpret_cast<float4*>(vals) = vec;
                } else {
                    #pragma unroll
                    for (int j = 0; j < VEC_SIZE; j++) vals[j] = __float2bfloat16(0.0f);
                }
            } else {
                // Boundary case - scalar loads
                int packed_id = (row < sorted_M) ? sorted_ids[row] : -1;
                int token_id = (packed_id >= 0) ? (packed_id & 0xFFFFFF) : -1;
                #pragma unroll
                for (int j = 0; j < VEC_SIZE; j++) {
                    vals[j] = (token_id >= 0 && token_id < num_tokens && (k_start + k + j) < model_dim)
                        ? hidden_states[token_id * model_dim + k_start + k + j]
                        : __float2bfloat16(0.0f);
                }
            }
            
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                As[{m, k + j}] = vals[j];
            }
        }
        
        // === Cooperative vectorized load weight tile (contiguous) ===
        constexpr int TOTAL_VECS_W = (BLOCK_SIZE * K_STEP) / VEC_SIZE;
        constexpr int VECS_PER_THREAD_W = TOTAL_VECS_W / NUM_THREADS;
        
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD_W; v++) {
            int vec_idx = lane * VECS_PER_THREAD_W + v;
            int flat_idx = vec_idx * VEC_SIZE;
            int n = flat_idx / K_STEP;
            int k = flat_idx % K_STEP;
            
            int col = col_start + n;
            bf16 vals[VEC_SIZE];
            
            if (col < total_n && (k_start + k + VEC_SIZE - 1) < model_dim) {
                // w1 layout: [num_experts, inter_dim*2, model_dim]
                float4 vec = *reinterpret_cast<const float4*>(&w1[expert_id * total_n * model_dim + col * model_dim + k_start + k]);
                *reinterpret_cast<float4*>(vals) = vec;
            } else {
                #pragma unroll
                for (int j = 0; j < VEC_SIZE; j++) {
                    vals[j] = (col < total_n && (k_start + k + j) < model_dim)
                        ? w1[expert_id * total_n * model_dim + col * model_dim + k_start + k + j]
                        : __float2bfloat16(0.0f);
                }
            }
            
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                Bs[{n, k + j}] = vals[j];
            }
        }
        
        __syncthreads();
        
        // === Compute MMA for this K tile ===
        // Process K_STEP in DOT_SLICE (16) chunks
        // Each K chunk: As[warp_row*64 + offset, kk:kk+16] @ Bs[warp_col*32, kk:kk+16]^T
        
        #pragma unroll
        for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
            // Load subtiles from shared memory to registers
            // Warp (warp_row, warp_col) computes output tile at:
            //   rows: [warp_row*64, warp_row*64+64)
            //   cols: [warp_col*32, warp_col*32+32)
            
            // For MMA, we need:
            // C_accum[0] = As[warp_row*64:warp_row*64+32, :] @ Bs[warp_col*32:warp_col*32+32, :]^T
            // C_accum[1] = As[warp_row*64+32:warp_row*64+64, :] @ Bs[...]^T
            
            // Load A subtile for first 32 rows
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row * 2, kk}));
            // Load B subtile 
            load(b_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            
            // MMA: C_accum[0] += a_tile @ b_tile^T
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0], a_tile, b_tile, C_accum[0]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
            
            // Load A subtile for second 32 rows
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row * 2 + 1, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            
            // MMA: C_accum[1] += a_tile @ b_tile^T
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1], a_tile, b_tile, C_accum[1]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
        }
        
        __syncthreads();
    }
    
    // === Store results ===
    // C_accum[0] is 32x32 at output rows [row_start + warp_row*64, row_start + warp_row*64 + 32)
    // C_accum[1] is 32x32 at output rows [row_start + warp_row*64 + 32, row_start + warp_row*64 + 64)
    // Both at cols [col_start + warp_col*32, col_start + warp_col*32 + 32)
    
    // rt_fl<32, 32, col> layout: tiles[height][width] where height=width=2 (each 16x16 tile)
    // For col layout, each lane's data layout is:
    //   row_offset = 4 * (laneid / 16)  -- 4 consecutive rows
    //   col_offset = laneid % 16        -- one column
    //   data[0].x, data[0].y, data[1].x, data[1].y for rows row_offset+0,+1,+2,+3
    
    const int out_row_base_0 = row_start + warp_row * 64;
    const int out_row_base_1 = row_start + warp_row * 64 + 32;
    const int out_col_base = col_start + warp_col * 32;
    const int lane_id = laneid();
    const int row_off = 4 * (lane_id / 16);
    const int col_off = lane_id % 16;
    
    // C_accum[0] - first 32x32 output tile
    #pragma unroll
    for (int tile_row = 0; tile_row < 2; tile_row++) {
        #pragma unroll
        for (int tile_col = 0; tile_col < 2; tile_col++) {
            const auto& tile = C_accum[0].tiles[tile_row][tile_col];
            int base_row = out_row_base_0 + tile_row * 16 + row_off;
            int col = out_col_base + tile_col * 16 + col_off;
            
            if (col < total_n) {
                if (base_row + 0 < sorted_M) intermediate[(base_row + 0) * total_n + col] = __float2bfloat16(tile.data[0].x);
                if (base_row + 1 < sorted_M) intermediate[(base_row + 1) * total_n + col] = __float2bfloat16(tile.data[0].y);
                if (base_row + 2 < sorted_M) intermediate[(base_row + 2) * total_n + col] = __float2bfloat16(tile.data[1].x);
                if (base_row + 3 < sorted_M) intermediate[(base_row + 3) * total_n + col] = __float2bfloat16(tile.data[1].y);
            }
        }
    }
    
    // C_accum[1] - second 32x32 output tile
    #pragma unroll
    for (int tile_row = 0; tile_row < 2; tile_row++) {
        #pragma unroll
        for (int tile_col = 0; tile_col < 2; tile_col++) {
            const auto& tile = C_accum[1].tiles[tile_row][tile_col];
            int base_row = out_row_base_1 + tile_row * 16 + row_off;
            int col = out_col_base + tile_col * 16 + col_off;
            
            if (col < total_n) {
                if (base_row + 0 < sorted_M) intermediate[(base_row + 0) * total_n + col] = __float2bfloat16(tile.data[0].x);
                if (base_row + 1 < sorted_M) intermediate[(base_row + 1) * total_n + col] = __float2bfloat16(tile.data[0].y);
                if (base_row + 2 < sorted_M) intermediate[(base_row + 2) * total_n + col] = __float2bfloat16(tile.data[1].x);
                if (base_row + 3 < sorted_M) intermediate[(base_row + 3) * total_n + col] = __float2bfloat16(tile.data[1].y);
            }
        }
    }
}

// Dispatch function
void dispatch_hk_moe_stage1(const moe_stage1_globals& g) {
    using namespace s1_cfg;
    
    const int total_n = g.inter_dim * 2;
    
    const int num_m_blocks = (g.sorted_M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_n_blocks = (total_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 grid(num_m_blocks, num_n_blocks);
    dim3 block(NUM_THREADS);
    
    // Set dynamic shared memory size
    size_t smem_size = 65536;  // 64KB
    hipFuncSetAttribute((void*)hk_moe_stage1_kernel_mma, 
                        hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    hk_moe_stage1_kernel_mma<<<grid, block, smem_size, g.stream>>>(
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
        g.block_m,
        num_m_blocks,
        num_n_blocks
    );
}
