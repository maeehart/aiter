// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * ============================================================================
 * HipKittens MoE Fused Kernel - Stage1+Stage2 in single kernel
 * ============================================================================
 * 
 * ## Computation
 *   output[token] += routing_weight * silu(hidden @ W1_gate) * (hidden @ W1_up) @ W2
 * 
 * ## Tile Strategy
 *   - M_TILE = 32 rows per workgroup
 *   - Intermediate [32, 256] stored in LDS (16KB)
 *   - N_TILE = 128 output columns per workgroup
 *   - 4 warps: 2 row halves × 2 column halves
 *   - Each warp computes [16, 64] output tiles
 * 
 * ## Warp Layout (4 warps for [32, 128] output)
 *   warp 0: rows [0,16),  cols [0,64)
 *   warp 1: rows [0,16),  cols [64,128)
 *   warp 2: rows [16,32), cols [0,64)
 *   warp 3: rows [16,32), cols [64,128)
 * 
 * ## Memory Savings
 *   Without fusion: 66MB intermediate traffic (write 33MB + read 33MB)
 *   With fusion: 16KB intermediate in LDS per workgroup
 */
#include "hk_moe_kernel.cuh"

using namespace kittens;

namespace fused_cfg {
    constexpr int M_TILE = 32;         // Rows per workgroup
    constexpr int N_TILE = 128;        // Output columns per workgroup (for Stage 2)
    constexpr int K_STEP = 32;         // K dimension per iteration
    
    // Register tile dimensions for MFMA 16x16x16
    constexpr int REG_M = 16;          // Each warp handles 16 rows
    constexpr int REG_N = 64;          // Each warp handles 64 columns (4× 16x16 subtiles)
    constexpr int DOT_SLICE = 16;      // MFMA native K dimension
    
    // Thread organization: 4 warps = 256 threads
    constexpr int NUM_WARPS = 4;
    constexpr int NUM_THREADS = WARP_THREADS * NUM_WARPS;  // 64 * 4 = 256
    
    // DeepSeek R1 dimensions
    constexpr int MODEL_DIM = 7168;
    constexpr int INTER_DIM = 256;
    
    // Intermediate buffer in LDS: M_TILE × INTER_DIM × sizeof(bf16)
    constexpr int INTER_LDS_SIZE = M_TILE * INTER_DIM * 2;  // 32 × 256 × 2 = 16KB
    
    // XCD-aware scheduling
    constexpr int NUM_XCDS = 8;
    constexpr int WGM = 4;
}

// LDS tile types
using fused_input_tile = st_bf<fused_cfg::M_TILE, fused_cfg::K_STEP>;     // [32, 32] bf16
using fused_weight_tile = st_bf<fused_cfg::N_TILE, fused_cfg::K_STEP>;    // [128, 32] bf16

// XCD transform
__device__ __forceinline__ int fused_xcd_transform(int wgid, int num_wgs) {
    constexpr int NUM_XCDS = fused_cfg::NUM_XCDS;
    constexpr int WGM = fused_cfg::WGM;
    constexpr int CHUNK = WGM * WGM;
    
    int xcd = wgid % NUM_XCDS;
    int chunk_id = wgid / NUM_XCDS;
    int chunk_group = chunk_id / CHUNK;
    int chunk_local = chunk_id % CHUNK;
    
    int chunks_per_xcd = (num_wgs / NUM_XCDS + CHUNK - 1) / CHUNK;
    int new_chunk_id = xcd * chunks_per_xcd + chunk_group;
    
    return new_chunk_id * CHUNK + chunk_local;
}

/**
 * Fused MoE FP8 Kernel
 */
__global__ __launch_bounds__(fused_cfg::NUM_THREADS, 2)
void hk_moe_fused_fp8_kernel(
    const bf16* __restrict__ hidden_states,
    const fp8_t* __restrict__ w1_fp8,
    const fp8_t* __restrict__ w2_fp8,
    const float* __restrict__ w1_scale,
    const float* __restrict__ w2_scale,
    float* __restrict__ output_fp32,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const int32_t* __restrict__ num_valid_ids,
    const float* __restrict__ sorted_weights,
    const int sorted_M,
    const int num_tokens,
    const int model_dim,
    const int inter_dim,
    const int num_experts,
    const int block_m_sorting,
    const int num_m_blocks,
    const int num_n_blocks,
    const int num_scale_n_w1,
    const int num_scale_k_w1,
    const int num_scale_n_w2,
    const int num_scale_k_w2
) {
    using namespace fused_cfg;
    using namespace fp8_cfg;
    
    const int sorted_M_valid = num_valid_ids[0];
    const int total_n_w1 = inter_dim * 2;  // 512 (gate + up)
    
    // ========================================================================
    // LDS Allocation
    // ========================================================================
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    fused_input_tile (&As) = al.allocate<fused_input_tile>();           // [32, 32] input tile
    fused_weight_tile (&Bs) = al.allocate<fused_weight_tile>();         // [128, 32] weight tile
    bf16 (&inter_lds)[M_TILE][INTER_DIM] = al.allocate<bf16, M_TILE, INTER_DIM>();  // [32, 256] intermediate
    int (&token_row_offsets)[M_TILE] = al.allocate<int, M_TILE>();
    
    // ========================================================================
    // Workgroup Assignment
    // ========================================================================
    int wgid = blockIdx.y * gridDim.x + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    wgid = fused_xcd_transform(wgid, NUM_WGS);
    
    if (wgid >= NUM_WGS) return;
    
    const int pid_m = wgid / num_n_blocks;
    const int pid_n = wgid % num_n_blocks;
    
    if (pid_m >= num_m_blocks || pid_n >= num_n_blocks) return;
    
    const int row_start = pid_m * M_TILE;
    const int col_start = pid_n * N_TILE;
    
    if (row_start >= sorted_M_valid || col_start >= model_dim) return;
    
    // Expert assignment
    const int tile_id = row_start / block_m_sorting;
    const int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    // Expert weight pointers
    const fp8_t* w1_expert = w1_fp8 + expert_id * (size_t)total_n_w1 * model_dim;
    const fp8_t* w2_expert = w2_fp8 + expert_id * (size_t)model_dim * inter_dim;
    const float* w1_expert_scale = w1_scale + expert_id * (size_t)num_scale_n_w1 * num_scale_k_w1;
    const float* w2_expert_scale = w2_scale + expert_id * (size_t)num_scale_n_w2 * num_scale_k_w2;
    
    const int lane = threadIdx.x;
    const int warp_id = lane / WARP_THREADS;         // 0-3
    const int lane_in_warp = lane % WARP_THREADS;    // 0-63
    
    // Warp position in output tile
    // warp_row: 0 for warps 0,1; 16 for warps 2,3
    // warp_col: 0 for warps 0,2; 64 for warps 1,3
    const int warp_row_idx = warp_id / 2;            // 0 or 1
    const int warp_col_idx = warp_id % 2;            // 0 or 1
    const int warp_row_base = warp_row_idx * REG_M;  // 0 or 16
    const int warp_col_base = warp_col_idx * REG_N;  // 0 or 64
    
    // Precompute token row offsets
    if (lane < M_TILE) {
        const int row = row_start + lane;
        int base = -1;
        if (row < sorted_M_valid) {
            const int packed_id = sorted_ids[row];
            const int token_id = packed_id & 0xFFFFFF;
            if (token_id >= 0 && token_id < num_tokens) {
                base = token_id * model_dim;
            }
        }
        token_row_offsets[lane] = base;
    }
    __syncthreads();
    
    // ========================================================================
    // STAGE 1: Gate-Up Projection + SiLU
    // ========================================================================
    // Compute: inter[m, n] = silu(input[m] @ W1_gate[n]) * (input[m] @ W1_up[n])
    // 
    // We process inter_dim=256 in 2 chunks of 128 columns each (INTER_CHUNKS=2)
    // Each chunk: each warp computes [16, 64], total 4 warps = [32, 128]
    
    constexpr int INTER_CHUNKS = INTER_DIM / N_TILE;  // 256/128 = 2
    constexpr int COL_SUBTILES = REG_N / DOT_SLICE;   // 64/16 = 4 column subtiles per warp
    
    for (int inter_chunk = 0; inter_chunk < INTER_CHUNKS; inter_chunk++) {
        const int inter_col_start = inter_chunk * N_TILE;  // 0 or 128
        
        // Register accumulators: Use array of 4× [16, 16] tiles instead of one [16, 64]
        // This is required because mma_ABt expects full rt_fl types, not rt_base subtiles
        rt_fl<REG_M, DOT_SLICE, ducks::rt_layout::col> C_gate[COL_SUBTILES], C_up[COL_SUBTILES];
        #pragma unroll
        for (int i = 0; i < COL_SUBTILES; i++) {
            zero(C_gate[i]);
            zero(C_up[i]);
        }
        
        // K-loop for Stage 1 (K = model_dim = 7168)
        const int num_k_tiles_s1 = (model_dim + K_STEP - 1) / K_STEP;  // 224
        
        for (int k_tile = 0; k_tile < num_k_tiles_s1; k_tile++) {
            const int k_start = k_tile * K_STEP;
            
            // Load input tile [32, 32] from hidden_states (gathered via token_row_offsets)
            uint32_t As_ptr = reinterpret_cast<uintptr_t>(&As.data[0]);
            constexpr int VEC_SIZE = 8;
            constexpr int VECS_PER_ROW = K_STEP / VEC_SIZE;              // 4
            constexpr int TOTAL_VECS_IN = M_TILE * VECS_PER_ROW;         // 128
            constexpr int VECS_PER_THREAD_IN = (TOTAL_VECS_IN + NUM_THREADS - 1) / NUM_THREADS;  // 1
            
            #pragma unroll
            for (int v = 0; v < VECS_PER_THREAD_IN; v++) {
                int vec_idx = lane * VECS_PER_THREAD_IN + v;
                if (vec_idx < TOTAL_VECS_IN) {
                    int m = vec_idx / VECS_PER_ROW;
                    int k = (vec_idx % VECS_PER_ROW) * VEC_SIZE;
                    
                    float4 buf = {0.f, 0.f, 0.f, 0.f};
                    int base = token_row_offsets[m];
                    if (base >= 0 && (k_start + k + VEC_SIZE - 1) < model_dim) {
                        buf = *reinterpret_cast<const float4*>(&hidden_states[base + k_start + k]);
                    }
                    store_shared_vec(As.idx(As_ptr, {m, k}), {buf.x, buf.y});
                    store_shared_vec(As.idx(As_ptr, {m, k + 4}), {buf.z, buf.w});
                }
            }
            
            // Load gate weight tile [128, 32] with FP8 dequant
            // Gate weights are at W1[0:inter_dim, :]
            uint32_t Bs_ptr = reinterpret_cast<uintptr_t>(&Bs.data[0]);
            constexpr int TOTAL_VECS_W = N_TILE * K_STEP / VEC_SIZE;     // 512
            constexpr int VECS_PER_THREAD_W = (TOTAL_VECS_W + NUM_THREADS - 1) / NUM_THREADS;  // 2
            
            int gate_col_global = inter_col_start;  // 0 or 128 (for gate weights)
            int k_block = k_start / SCALE_BLOCK_K;
            int n_block_gate = gate_col_global / SCALE_BLOCK_N;
            float scale_gate = w1_expert_scale[n_block_gate * num_scale_k_w1 + k_block];
            
            #pragma unroll
            for (int v = 0; v < VECS_PER_THREAD_W; v++) {
                int vec_idx = lane * VECS_PER_THREAD_W + v;
                if (vec_idx < TOTAL_VECS_W) {
                    int n = vec_idx / (K_STEP / VEC_SIZE);
                    int k = (vec_idx % (K_STEP / VEC_SIZE)) * VEC_SIZE;
                    int col = gate_col_global + n;
                    int k_global = k_start + k;
                    
                    float2 buf_lo = {0.f, 0.f}, buf_hi = {0.f, 0.f};
                    if (col < inter_dim && (k_global + VEC_SIZE - 1) < model_dim) {
                        const fp8_t* src = &w1_expert[col * model_dim + k_global];
                        fp8x8_to_bf16x8_scaled(src, scale_gate, buf_lo, buf_hi);
                    }
                    store_shared_vec(Bs.idx(Bs_ptr, {n, k}), {buf_lo.x, buf_lo.y});
                    store_shared_vec(Bs.idx(Bs_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
                }
            }
            
            __syncthreads();
            
            // MFMA for gate: Each warp computes [16, 64] output
            // We use 4 separate [16,16] accumulators instead of one [16,64] with subtiles
            rt_bf<REG_M, DOT_SLICE> a_tile;  // [16, 16] input subtile
            
            #pragma unroll
            for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {  // 2 iterations (K_STEP=32, DOT_SLICE=16)
                // Load a_tile: [16, 16] from As at (warp_row_idx, kk)
                load(a_tile, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row_idx, kk}));
                
                // Process 4 column subtiles of the [16, 64] output
                #pragma unroll
                for (int col_subtile = 0; col_subtile < COL_SUBTILES; col_subtile++) {
                    rt_bf<DOT_SLICE, DOT_SLICE> b_tile;  // [16, 16] weight subtile
                    
                    // Bs row index: (warp_col_base + col_subtile * 16) / 16
                    int bs_row_tile = (warp_col_base + col_subtile * DOT_SLICE) / DOT_SLICE;
                    
                    load(b_tile, subtile_inplace<DOT_SLICE, DOT_SLICE>(Bs, {bs_row_tile, kk}));
                    
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    
                    // Accumulate to separate accumulator tile
                    mma_ABt(C_gate[col_subtile], a_tile, b_tile, C_gate[col_subtile]);
                }
            }
            
            __syncthreads();
            
            // Load up weight tile (second half of W1: inter_dim to inter_dim*2)
            int up_col_global = inter_dim + inter_col_start;  // 256+0=256 or 256+128=384
            int n_block_up = up_col_global / SCALE_BLOCK_N;
            float scale_up = w1_expert_scale[n_block_up * num_scale_k_w1 + k_block];
            
            #pragma unroll
            for (int v = 0; v < VECS_PER_THREAD_W; v++) {
                int vec_idx = lane * VECS_PER_THREAD_W + v;
                if (vec_idx < TOTAL_VECS_W) {
                    int n = vec_idx / (K_STEP / VEC_SIZE);
                    int k = (vec_idx % (K_STEP / VEC_SIZE)) * VEC_SIZE;
                    int col = up_col_global + n;
                    int k_global = k_start + k;
                    
                    float2 buf_lo = {0.f, 0.f}, buf_hi = {0.f, 0.f};
                    if (col < inter_dim * 2 && (k_global + VEC_SIZE - 1) < model_dim) {
                        const fp8_t* src = &w1_expert[col * model_dim + k_global];
                        fp8x8_to_bf16x8_scaled(src, scale_up, buf_lo, buf_hi);
                    }
                    store_shared_vec(Bs.idx(Bs_ptr, {n, k}), {buf_lo.x, buf_lo.y});
                    store_shared_vec(Bs.idx(Bs_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
                }
            }
            
            __syncthreads();
            
            // MFMA for up (same structure as gate)
            #pragma unroll
            for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
                load(a_tile, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row_idx, kk}));
                
                #pragma unroll
                for (int col_subtile = 0; col_subtile < COL_SUBTILES; col_subtile++) {
                    rt_bf<DOT_SLICE, DOT_SLICE> b_tile;
                    int bs_row_tile = (warp_col_base + col_subtile * DOT_SLICE) / DOT_SLICE;
                    
                    load(b_tile, subtile_inplace<DOT_SLICE, DOT_SLICE>(Bs, {bs_row_tile, kk}));
                    
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    
                    mma_ABt(C_up[col_subtile], a_tile, b_tile, C_up[col_subtile]);
                }
            }
            
            __syncthreads();
        }
        // End of K-loop for Stage 1
        
        // ====================================================================
        // Apply SiLU(gate) * up and store to intermediate LDS
        // ====================================================================
        // C_gate and C_up are [16, 64] per warp = tiles[1][4]
        // Each tiles[0][i] is a 16×16 subtile (rt_base)
        // 
        // Register tile data layout (AMD CDNA3):
        // For col-major layout, each lane holds 4 values across 4 rows at a specific column.
        // Lane mapping within 16×16 subtile:
        //   - lanes 0-15:  columns 0-15, rows determined by data[0..1].x/y
        //   - lanes 16-31: same columns, different rows
        //   - lanes 32-47: same
        //   - lanes 48-63: same
        //
        // For rt_base<float, col>: packed_per_thread = 4/2 = 2 (float2)
        // data[0].x, data[0].y = 2 values (rows 0, 1 for lanes 0-15; rows 8, 9 for lanes 16-31; etc.)
        // Actually for col layout: each lane covers certain (row, col) positions
        
        // Simpler approach: iterate over all positions in the register tile
        // and write to LDS directly
        
        const int row_offset_in_tile = (lane_in_warp / 16) * 4;  // 0, 4, 8, 12 (every 16 lanes)
        const int col_offset_in_tile = lane_in_warp % 16;        // 0-15
        
        #pragma unroll
        for (int tile_col = 0; tile_col < COL_SUBTILES; tile_col++) {  // 4 subtiles
            // Access individual rt_fl<16,16> accumulators, not subtiles
            const auto& gate_subtile = C_gate[tile_col].tiles[0][0];
            const auto& up_subtile = C_up[tile_col].tiles[0][0];
            
            // Column in inter_lds
            int lds_col = inter_col_start + warp_col_base + tile_col * DOT_SLICE + col_offset_in_tile;
            
            if (lds_col < INTER_DIM) {
                // Write 4 rows per lane
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int lds_row = warp_row_base + row_offset_in_tile + r;
                    if (lds_row < M_TILE) {
                        float g, u;
                        if (r == 0) { g = gate_subtile.data[0].x; u = up_subtile.data[0].x; }
                        else if (r == 1) { g = gate_subtile.data[0].y; u = up_subtile.data[0].y; }
                        else if (r == 2) { g = gate_subtile.data[1].x; u = up_subtile.data[1].x; }
                        else { g = gate_subtile.data[1].y; u = up_subtile.data[1].y; }
                        
                        float silu = g / (1.0f + expf(-g));
                        inter_lds[lds_row][lds_col] = __float2bfloat16(silu * u);
                    }
                }
            }
        }
    }
    // End of inter_chunk loop
    
    __syncthreads();  // All threads must finish writing inter_lds before Stage 2
    
    // ========================================================================
    // STAGE 2: Down Projection
    // ========================================================================
    // Compute: output[m, n] = inter[m] @ W2[n]
    // inter: [32, 256] in LDS
    // W2: [model_dim, inter_dim] = [7168, 256] (for this expert)
    // output: [32, 128] (N_TILE columns at col_start)
    
    // Register accumulator: 4× [16, 16] per warp (like Stage 1)
    constexpr int OUT_COL_SUBTILES = REG_N / DOT_SLICE;   // 64/16 = 4
    rt_fl<REG_M, DOT_SLICE, ducks::rt_layout::col> C_out[OUT_COL_SUBTILES];
    #pragma unroll
    for (int i = 0; i < OUT_COL_SUBTILES; i++) {
        zero(C_out[i]);
    }
    
    // K-loop for Stage 2 (K = inter_dim = 256)
    const int num_k_tiles_s2 = (inter_dim + K_STEP - 1) / K_STEP;  // 8
    
    for (int k_tile = 0; k_tile < num_k_tiles_s2; k_tile++) {
        const int k_start = k_tile * K_STEP;  // 0, 32, 64, ..., 224
        
        // Load intermediate from LDS to As [32, 32]
        uint32_t As_ptr = reinterpret_cast<uintptr_t>(&As.data[0]);
        constexpr int VEC_SIZE = 8;
        constexpr int VECS_PER_ROW = K_STEP / VEC_SIZE;              // 4
        constexpr int TOTAL_VECS_IN = M_TILE * VECS_PER_ROW;         // 128
        constexpr int VECS_PER_THREAD_IN = (TOTAL_VECS_IN + NUM_THREADS - 1) / NUM_THREADS;
        
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD_IN; v++) {
            int vec_idx = lane * VECS_PER_THREAD_IN + v;
            if (vec_idx < TOTAL_VECS_IN) {
                int m = vec_idx / VECS_PER_ROW;
                int k = (vec_idx % VECS_PER_ROW) * VEC_SIZE;
                
                // Read 8 bf16 values from inter_lds
                bf16* src = &inter_lds[m][k_start + k];
                
                // Pack into 2× float2 (each float2 holds 4 bf16 as 2× bf16_2)
                // store_shared_vec expects {float, float} = 4 bf16 values
                __hip_bfloat162 v01 = *reinterpret_cast<const __hip_bfloat162*>(&src[0]);
                __hip_bfloat162 v23 = *reinterpret_cast<const __hip_bfloat162*>(&src[2]);
                __hip_bfloat162 v45 = *reinterpret_cast<const __hip_bfloat162*>(&src[4]);
                __hip_bfloat162 v67 = *reinterpret_cast<const __hip_bfloat162*>(&src[6]);
                
                // Combine pairs: v01+v23 → packed_lo (4 bf16), v45+v67 → packed_hi (4 bf16)
                float2 packed_lo, packed_hi;
                memcpy(&packed_lo.x, &v01, sizeof(float));
                memcpy(&packed_lo.y, &v23, sizeof(float));
                memcpy(&packed_hi.x, &v45, sizeof(float));
                memcpy(&packed_hi.y, &v67, sizeof(float));
                
                store_shared_vec(As.idx(As_ptr, {m, k}), packed_lo);
                store_shared_vec(As.idx(As_ptr, {m, k + 4}), packed_hi);
            }
        }
        
        // Load W2 weight tile [128, 32] with FP8 dequant
        // W2 layout: [model_dim, inter_dim] = [7168, 256]
        // This workgroup computes output columns [col_start, col_start+128)
        uint32_t Bs_ptr = reinterpret_cast<uintptr_t>(&Bs.data[0]);
        constexpr int TOTAL_VECS_W = N_TILE * K_STEP / VEC_SIZE;     // 512
        constexpr int VECS_PER_THREAD_W = (TOTAL_VECS_W + NUM_THREADS - 1) / NUM_THREADS;
        
        int n_block = col_start / SCALE_BLOCK_N;
        int k_block = k_start / SCALE_BLOCK_K;
        float scale = w2_expert_scale[n_block * num_scale_k_w2 + k_block];
        
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD_W; v++) {
            int vec_idx = lane * VECS_PER_THREAD_W + v;
            if (vec_idx < TOTAL_VECS_W) {
                int n = vec_idx / (K_STEP / VEC_SIZE);
                int k = (vec_idx % (K_STEP / VEC_SIZE)) * VEC_SIZE;
                int col = col_start + n;      // Output column (0 to model_dim)
                int k_global = k_start + k;   // K index in inter_dim
                
                float2 buf_lo = {0.f, 0.f}, buf_hi = {0.f, 0.f};
                if (col < model_dim && (k_global + VEC_SIZE - 1) < inter_dim) {
                    const fp8_t* src = &w2_expert[col * inter_dim + k_global];
                    fp8x8_to_bf16x8_scaled(src, scale, buf_lo, buf_hi);
                }
                store_shared_vec(Bs.idx(Bs_ptr, {n, k}), {buf_lo.x, buf_lo.y});
                store_shared_vec(Bs.idx(Bs_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
            }
        }
        
        __syncthreads();
        
        // MFMA for down projection
        rt_bf<REG_M, DOT_SLICE> a_tile;
        
        #pragma unroll
        for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
            load(a_tile, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row_idx, kk}));
            
            #pragma unroll
            for (int col_subtile = 0; col_subtile < OUT_COL_SUBTILES; col_subtile++) {
                rt_bf<DOT_SLICE, DOT_SLICE> b_tile;
                int bs_row_tile = (warp_col_base + col_subtile * DOT_SLICE) / DOT_SLICE;
                
                load(b_tile, subtile_inplace<DOT_SLICE, DOT_SLICE>(Bs, {bs_row_tile, kk}));
                
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(0)");
                
                mma_ABt(C_out[col_subtile], a_tile, b_tile, C_out[col_subtile]);
            }
        }
        
        __syncthreads();
    }
    // End of K-loop for Stage 2
    
    // ========================================================================
    // EPILOGUE: Weighted atomic scatter to output
    // ========================================================================
    const int row_offset_out = (lane_in_warp / 16) * 4;
    const int col_offset_out = lane_in_warp % 16;
    
    #pragma unroll
    for (int tile_col = 0; tile_col < OUT_COL_SUBTILES; tile_col++) {
        // Access the individual rt_fl<16,16> accumulator
        const auto& out_subtile = C_out[tile_col].tiles[0][0];
        
        int out_col = col_start + warp_col_base + tile_col * DOT_SLICE + col_offset_out;
        
        if (out_col < model_dim) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                int m = warp_row_base + row_offset_out + r;
                if (m < M_TILE) {
                    int row = row_start + m;
                    if (row < sorted_M_valid) {
                        int packed_id = sorted_ids[row];
                        int token_id = packed_id & 0xFFFFFF;
                        float weight = sorted_weights[row];
                        
                        if (token_id >= 0 && token_id < num_tokens) {
                            float val;
                            if (r == 0) val = out_subtile.data[0].x;
                            else if (r == 1) val = out_subtile.data[0].y;
                            else if (r == 2) val = out_subtile.data[1].x;
                            else val = out_subtile.data[1].y;
                            
                            atomicAdd(&output_fp32[token_id * model_dim + out_col], val * weight);
                        }
                    }
                }
            }
        }
    }
}


// ============================================================================
// STREAMING FUSION KERNEL
// ============================================================================
// Key insight: Process inter_dim in chunks, keeping intermediate in registers.
// For each inter_k chunk:
//   1. Compute partial intermediate by iterating over model_dim
//   2. Apply SiLU immediately
//   3. Multiply by W2 chunk and accumulate to output
// Output accumulator stays in registers throughout - no LDS for intermediate!

namespace stream_cfg {
    constexpr int M_TILE = 32;         // Rows per workgroup  
    constexpr int N_TILE = 128;        // Output columns per workgroup
    constexpr int K_STEP = 32;         // K dimension per iteration (for model_dim)
    constexpr int K_INTER = 32;        // Inter_dim chunk size (K of Stage 2)
    
    constexpr int REG_M = 16;          // Each warp handles 16 rows
    constexpr int REG_N = 64;          // Each warp handles 64 output columns
    constexpr int DOT_SLICE = 16;      // MFMA native dimension
    
    constexpr int NUM_WARPS = 4;
    constexpr int NUM_THREADS = WARP_THREADS * NUM_WARPS;  // 256
    
    constexpr int NUM_XCDS = 8;
    constexpr int WGM = 4;
}

// Shared tiles for streaming kernel
using stream_input_tile = st_bf<stream_cfg::M_TILE, stream_cfg::K_STEP>;      // [32, 32]
using stream_w1_tile = st_bf<stream_cfg::K_INTER, stream_cfg::K_STEP>;        // [32, 32] for W1 slice
using stream_w2_tile = st_bf<stream_cfg::N_TILE, stream_cfg::K_INTER>;        // [128, 32] for W2 slice
using stream_inter_tile = st_bf<stream_cfg::M_TILE, stream_cfg::K_INTER>;     // [32, 32] for intermediate

__global__ __launch_bounds__(stream_cfg::NUM_THREADS, 2)
void hk_moe_streaming_fp8_kernel(
    const bf16* __restrict__ hidden_states,
    const fp8_t* __restrict__ w1_fp8,
    const fp8_t* __restrict__ w2_fp8,
    const float* __restrict__ w1_scale,
    const float* __restrict__ w2_scale,
    float* __restrict__ output_fp32,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const int32_t* __restrict__ num_valid_ids,
    const float* __restrict__ sorted_weights,
    const int sorted_M,
    const int num_tokens,
    const int model_dim,
    const int inter_dim,
    const int num_experts,
    const int block_m_sorting,
    const int num_m_blocks,
    const int num_n_blocks,
    const int num_scale_n_w1,
    const int num_scale_k_w1,
    const int num_scale_n_w2,
    const int num_scale_k_w2
) {
    using namespace stream_cfg;
    using namespace fp8_cfg;
    
    const int sorted_M_valid = num_valid_ids[0];
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    // LDS tiles
    stream_input_tile (&As) = al.allocate<stream_input_tile>();      // [32, 32] for hidden
    stream_w1_tile (&W1s) = al.allocate<stream_w1_tile>();           // [32, 32] for W1 slice
    // Raw 2D array for intermediate (compatible with simple indexing)
    bf16 (*inter_lds)[K_INTER] = reinterpret_cast<bf16(*)[K_INTER]>(al.ptr);
    al.ptr += M_TILE * K_INTER * sizeof(bf16);                       // [32, 32] for intermediate
    stream_w2_tile (&W2s) = al.allocate<stream_w2_tile>();           // [128, 32] for W2 slice
    int* token_row_offsets = (int*)al.allocate<int[M_TILE]>();
    
    // Workgroup position
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    wgid = fused_xcd_transform(wgid, NUM_WGS);
    if (wgid >= NUM_WGS) return;
    
    const int pid_m = wgid / num_n_blocks;
    const int pid_n = wgid % num_n_blocks;
    if (pid_m >= num_m_blocks || pid_n >= num_n_blocks) return;
    
    const int row_start = pid_m * M_TILE;
    const int col_start = pid_n * N_TILE;  // Output column start
    if (row_start >= sorted_M_valid || col_start >= model_dim) return;
    
    const int tile_id = row_start / block_m_sorting;
    const int expert_id = sorted_expert_ids[tile_id];
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    // Expert weight pointers
    const fp8_t* w1_expert = w1_fp8 + expert_id * (size_t)(inter_dim * 2) * model_dim;
    const float* w1_expert_scale = w1_scale + expert_id * (size_t)num_scale_n_w1 * num_scale_k_w1;
    const fp8_t* w2_expert = w2_fp8 + expert_id * (size_t)model_dim * inter_dim;
    const float* w2_expert_scale = w2_scale + expert_id * (size_t)num_scale_n_w2 * num_scale_k_w2;
    
    const int warp_id = warpid();
    const int lane = threadIdx.x;
    const int lane_in_warp = lane % WARP_THREADS;
    
    // Warp mapping: 2×2 for [32, 128] output
    const int warp_row_idx = warp_id / 2;      // 0 or 1
    const int warp_col_idx = warp_id % 2;      // 0 or 1
    const int warp_row_base = warp_row_idx * REG_M;   // 0 or 16
    const int warp_col_base = warp_col_idx * REG_N;   // 0 or 64
    
    // Precompute token offsets
    for (int i = lane; i < M_TILE; i += NUM_THREADS) {
        int row = row_start + i;
        int base = -1;
        if (row < sorted_M_valid) {
            int packed_id = sorted_ids[row];
            int token_id = packed_id & 0xFFFFFF;
            if (token_id >= 0 && token_id < num_tokens) {
                base = token_id * model_dim;
            }
        }
        token_row_offsets[i] = base;
    }
    __syncthreads();
    
    // Output accumulators - stay in registers throughout!
    constexpr int OUT_SUBTILES = REG_N / DOT_SLICE;  // 4
    rt_fl<REG_M, DOT_SLICE, ducks::rt_layout::col> C_out[OUT_SUBTILES];
    #pragma unroll
    for (int i = 0; i < OUT_SUBTILES; i++) {
        zero(C_out[i]);
    }
    
    // ==========================================================================
    // MAIN LOOP: Iterate over inter_dim in K_INTER chunks
    // ==========================================================================
    const int num_inter_chunks = (inter_dim + K_INTER - 1) / K_INTER;
    
    for (int inter_chunk = 0; inter_chunk < num_inter_chunks; inter_chunk++) {
        const int inter_k_start = inter_chunk * K_INTER;  // Start of this inter_dim slice
        
        // ------------------------------------------------------------------
        // STAGE 1: Compute partial intermediate [M_TILE, K_INTER]
        // ------------------------------------------------------------------
        // inter[:, inter_k_start:inter_k_start+K_INTER] = 
        //   silu(hidden × W1_gate[inter_k_start:...]^T) * (hidden × W1_up[inter_k_start:...]^T)
        
        // Intermediate accumulators for this chunk
        constexpr int INTER_SUBTILES = K_INTER / DOT_SLICE;  // 2 for K_INTER=32
        rt_fl<REG_M, DOT_SLICE, ducks::rt_layout::col> C_gate[INTER_SUBTILES];
        rt_fl<REG_M, DOT_SLICE, ducks::rt_layout::col> C_up[INTER_SUBTILES];
        #pragma unroll
        for (int i = 0; i < INTER_SUBTILES; i++) {
            zero(C_gate[i]);
            zero(C_up[i]);
        }
        
        // Inner K-loop over model_dim
        const int num_k1_tiles = (model_dim + K_STEP - 1) / K_STEP;
        
        for (int k1_tile = 0; k1_tile < num_k1_tiles; k1_tile++) {
            const int k1_start = k1_tile * K_STEP;
            
            // Load hidden tile [32, 32]
            uint32_t As_ptr = reinterpret_cast<uintptr_t>(&As.data[0]);
            constexpr int VEC_SIZE = 8;
            constexpr int VECS_PER_ROW = K_STEP / VEC_SIZE;
            constexpr int TOTAL_VECS_IN = M_TILE * VECS_PER_ROW;
            
            #pragma unroll
            for (int v = 0; v < (TOTAL_VECS_IN + NUM_THREADS - 1) / NUM_THREADS; v++) {
                int vec_idx = lane + v * NUM_THREADS;
                if (vec_idx < TOTAL_VECS_IN) {
                    int m = vec_idx / VECS_PER_ROW;
                    int k = (vec_idx % VECS_PER_ROW) * VEC_SIZE;
                    
                    float4 buf = {0.f, 0.f, 0.f, 0.f};
                    int base = token_row_offsets[m];
                    if (base >= 0 && (k1_start + k + VEC_SIZE - 1) < model_dim) {
                        buf = *reinterpret_cast<const float4*>(&hidden_states[base + k1_start + k]);
                    }
                    store_shared_vec(As.idx(As_ptr, {m, k}), {buf.x, buf.y});
                    store_shared_vec(As.idx(As_ptr, {m, k + 4}), {buf.z, buf.w});
                }
            }
            
            // Load W1_gate tile [K_INTER, K_STEP] with FP8 dequant
            uint32_t W1s_ptr = reinterpret_cast<uintptr_t>(&W1s.data[0]);
            constexpr int TOTAL_VECS_W1 = K_INTER * K_STEP / VEC_SIZE;
            
            int k_block = k1_start / SCALE_BLOCK_K;
            int n_block_gate = inter_k_start / SCALE_BLOCK_N;
            float scale_gate = w1_expert_scale[n_block_gate * num_scale_k_w1 + k_block];
            
            #pragma unroll
            for (int v = 0; v < (TOTAL_VECS_W1 + NUM_THREADS - 1) / NUM_THREADS; v++) {
                int vec_idx = lane + v * NUM_THREADS;
                if (vec_idx < TOTAL_VECS_W1) {
                    int n = vec_idx / (K_STEP / VEC_SIZE);
                    int k = (vec_idx % (K_STEP / VEC_SIZE)) * VEC_SIZE;
                    int col = inter_k_start + n;
                    int k_global = k1_start + k;
                    
                    float2 buf_lo = {0.f, 0.f}, buf_hi = {0.f, 0.f};
                    if (col < inter_dim && (k_global + VEC_SIZE - 1) < model_dim) {
                        const fp8_t* src = &w1_expert[col * model_dim + k_global];
                        fp8x8_to_bf16x8_scaled(src, scale_gate, buf_lo, buf_hi);
                    }
                    store_shared_vec(W1s.idx(W1s_ptr, {n, k}), {buf_lo.x, buf_lo.y});
                    store_shared_vec(W1s.idx(W1s_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
                }
            }
            
            __syncthreads();
            
            // MFMA for gate: compute partial intermediate
            rt_bf<REG_M, DOT_SLICE> a_tile;
            
            #pragma unroll
            for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
                load(a_tile, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row_idx, kk}));
                
                #pragma unroll
                for (int inter_sub = 0; inter_sub < INTER_SUBTILES; inter_sub++) {
                    rt_bf<DOT_SLICE, DOT_SLICE> b_tile;
                    load(b_tile, subtile_inplace<DOT_SLICE, DOT_SLICE>(W1s, {inter_sub, kk}));
                    
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    
                    mma_ABt(C_gate[inter_sub], a_tile, b_tile, C_gate[inter_sub]);
                }
            }
            
            __syncthreads();
            
            // Load W1_up tile and compute
            int up_col_start = inter_dim + inter_k_start;
            int n_block_up = up_col_start / SCALE_BLOCK_N;
            float scale_up = w1_expert_scale[n_block_up * num_scale_k_w1 + k_block];
            
            #pragma unroll
            for (int v = 0; v < (TOTAL_VECS_W1 + NUM_THREADS - 1) / NUM_THREADS; v++) {
                int vec_idx = lane + v * NUM_THREADS;
                if (vec_idx < TOTAL_VECS_W1) {
                    int n = vec_idx / (K_STEP / VEC_SIZE);
                    int k = (vec_idx % (K_STEP / VEC_SIZE)) * VEC_SIZE;
                    int col = up_col_start + n;
                    int k_global = k1_start + k;
                    
                    float2 buf_lo = {0.f, 0.f}, buf_hi = {0.f, 0.f};
                    if (col < inter_dim * 2 && (k_global + VEC_SIZE - 1) < model_dim) {
                        const fp8_t* src = &w1_expert[col * model_dim + k_global];
                        fp8x8_to_bf16x8_scaled(src, scale_up, buf_lo, buf_hi);
                    }
                    store_shared_vec(W1s.idx(W1s_ptr, {n, k}), {buf_lo.x, buf_lo.y});
                    store_shared_vec(W1s.idx(W1s_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
                }
            }
            
            __syncthreads();
            
            // MFMA for up
            #pragma unroll
            for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
                load(a_tile, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row_idx, kk}));
                
                #pragma unroll
                for (int inter_sub = 0; inter_sub < INTER_SUBTILES; inter_sub++) {
                    rt_bf<DOT_SLICE, DOT_SLICE> b_tile;
                    load(b_tile, subtile_inplace<DOT_SLICE, DOT_SLICE>(W1s, {inter_sub, kk}));
                    
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    
                    mma_ABt(C_up[inter_sub], a_tile, b_tile, C_up[inter_sub]);
                }
            }
            
            __syncthreads();
        }
        // End of K1 loop for Stage 1 partial
        
        // ------------------------------------------------------------------
        // Apply SiLU to C_gate, multiply by C_up -> C_inter in registers
        // ------------------------------------------------------------------
        // C_inter[i] = silu(C_gate[i]) * C_up[i] for each subtile
        // We'll keep the result in C_gate (reuse the registers)
        #pragma unroll
        for (int inter_sub = 0; inter_sub < INTER_SUBTILES; inter_sub++) {
            auto& gate_tile = C_gate[inter_sub].tiles[0][0];
            auto& up_tile = C_up[inter_sub].tiles[0][0];
            
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                float g_x = gate_tile.data[i].x;
                float g_y = gate_tile.data[i].y;
                float u_x = up_tile.data[i].x;
                float u_y = up_tile.data[i].y;
                
                float silu_x = g_x / (1.0f + expf(-g_x));
                float silu_y = g_y / (1.0f + expf(-g_y));
                
                gate_tile.data[i].x = silu_x * u_x;
                gate_tile.data[i].y = silu_y * u_y;
            }
        }
        // Now C_gate contains the intermediate values for this chunk
        
        // ------------------------------------------------------------------
        // STAGE 2: Multiply intermediate by W2 chunk and accumulate to output
        // ------------------------------------------------------------------
        // C_out += C_inter × W2[:, inter_k_start:inter_k_start+K_INTER]^T
        // 
        // C_inter is [16, K_INTER] per warp (INTER_SUBTILES × [16,16] tiles)
        // W2 is [N_TILE, K_INTER] = [128, 32]
        // C_out is [16, 64] per warp
        
        // Load W2 slice [128, K_INTER] to LDS
        uint32_t W2s_ptr = reinterpret_cast<uintptr_t>(&W2s.data[0]);
        constexpr int VEC_SIZE_W2 = 8;  // bf16 elements per float4
        constexpr int TOTAL_VECS_W2 = N_TILE * K_INTER / VEC_SIZE_W2;
        
        int n_block_w2 = col_start / SCALE_BLOCK_N;
        int k_block_w2 = inter_k_start / SCALE_BLOCK_K;
        float scale_w2 = w2_expert_scale[n_block_w2 * num_scale_k_w2 + k_block_w2];
        
        #pragma unroll
        for (int v = 0; v < (TOTAL_VECS_W2 + NUM_THREADS - 1) / NUM_THREADS; v++) {
            int vec_idx = lane + v * NUM_THREADS;
            if (vec_idx < TOTAL_VECS_W2) {
                int n = vec_idx / (K_INTER / VEC_SIZE_W2);
                int k = (vec_idx % (K_INTER / VEC_SIZE_W2)) * VEC_SIZE_W2;
                int out_col = col_start + n;
                int k_global = inter_k_start + k;
                
                float2 buf_lo = {0.f, 0.f}, buf_hi = {0.f, 0.f};
                if (out_col < model_dim && (k_global + VEC_SIZE_W2 - 1) < inter_dim) {
                    const fp8_t* src = &w2_expert[out_col * inter_dim + k_global];
                    fp8x8_to_bf16x8_scaled(src, scale_w2, buf_lo, buf_hi);
                }
                store_shared_vec(W2s.idx(W2s_ptr, {n, k}), {buf_lo.x, buf_lo.y});
                store_shared_vec(W2s.idx(W2s_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
            }
        }
        
        __syncthreads();
        
        // ------------------------------------------------------------------
        // Store intermediate to LDS for Stage 2 MFMA
        // ------------------------------------------------------------------
        // C_gate now contains silu(gate) * up in [16, K_INTER] per warp
        // Store to raw 2D LDS array using the same pattern as original LDS fusion
        // Note: warps 0 and 1 compute same values (same warp_row_idx=0)
        //       warps 2 and 3 compute same values (same warp_row_idx=1)
        // Only warps with warp_col_idx=0 write to avoid duplicate writes
        
        const int row_off_s2 = (lane_in_warp / 16) * 4;  // 0, 4, 8, 12
        const int col_off_s2 = lane_in_warp % 16;        // 0-15
        
        if (warp_col_idx == 0) {  // Only warps 0,2 write (warps 1,3 have same data)
            #pragma unroll
            for (int inter_sub = 0; inter_sub < INTER_SUBTILES; inter_sub++) {
                const auto& inter_tile = C_gate[inter_sub].tiles[0][0];
                int lds_col = inter_sub * DOT_SLICE + col_off_s2;  // Column in [0, K_INTER)
                
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int lds_row = warp_row_base + row_off_s2 + r;  // Row in [0, M_TILE)
                    if (lds_row < M_TILE && lds_col < K_INTER) {
                        float val;
                        if (r == 0) val = inter_tile.data[0].x;
                        else if (r == 1) val = inter_tile.data[0].y;
                        else if (r == 2) val = inter_tile.data[1].x;
                        else val = inter_tile.data[1].y;
                        
                        inter_lds[lds_row][lds_col] = __float2bfloat16(val);
                    }
                }
            }
        }
        
        __syncthreads();
        
        // ------------------------------------------------------------------
        // STAGE 2 MFMA: C_out += C_inter × W2^T
        // ------------------------------------------------------------------
        // C_inter is now in inter_lds as [M_TILE, K_INTER] = [32, 32]
        // W2 is in W2s as [N_TILE, K_INTER] = [128, 32]
        
        // Load intermediate from LDS to shared tile As for MFMA
        uint32_t As_ptr_s2 = reinterpret_cast<uintptr_t>(&As.data[0]);
        constexpr int VEC_SIZE_S2 = 8;
        constexpr int VECS_PER_ROW_S2 = K_INTER / VEC_SIZE_S2;  // 4
        constexpr int TOTAL_VECS_S2 = M_TILE * VECS_PER_ROW_S2;  // 128
        
        #pragma unroll
        for (int v = 0; v < (TOTAL_VECS_S2 + NUM_THREADS - 1) / NUM_THREADS; v++) {
            int vec_idx = lane + v * NUM_THREADS;
            if (vec_idx < TOTAL_VECS_S2) {
                int m = vec_idx / VECS_PER_ROW_S2;
                int k = (vec_idx % VECS_PER_ROW_S2) * VEC_SIZE_S2;
                
                // Read 8 bf16 values from inter_lds
                bf16* src = &inter_lds[m][k];
                
                // Pack into 2× float2
                __hip_bfloat162 v01 = *reinterpret_cast<const __hip_bfloat162*>(&src[0]);
                __hip_bfloat162 v23 = *reinterpret_cast<const __hip_bfloat162*>(&src[2]);
                __hip_bfloat162 v45 = *reinterpret_cast<const __hip_bfloat162*>(&src[4]);
                __hip_bfloat162 v67 = *reinterpret_cast<const __hip_bfloat162*>(&src[6]);
                
                float2 packed_lo, packed_hi;
                memcpy(&packed_lo.x, &v01, sizeof(float));
                memcpy(&packed_lo.y, &v23, sizeof(float));
                memcpy(&packed_hi.x, &v45, sizeof(float));
                memcpy(&packed_hi.y, &v67, sizeof(float));
                
                store_shared_vec(As.idx(As_ptr_s2, {m, k}), packed_lo);
                store_shared_vec(As.idx(As_ptr_s2, {m, k + 4}), packed_hi);
            }
        }
        
        __syncthreads();
        
        // MFMA for Stage 2
        rt_bf<REG_M, DOT_SLICE> a_tile_s2;
        
        #pragma unroll
        for (int kk = 0; kk < K_INTER / DOT_SLICE; kk++) {  // 2 iterations for K_INTER=32
            // Load A (intermediate) from As
            load(a_tile_s2, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row_idx, kk}));
            
            #pragma unroll
            for (int out_sub = 0; out_sub < OUT_SUBTILES; out_sub++) {
                rt_bf<DOT_SLICE, DOT_SLICE> b_tile_s2;
                int w2_row_tile = (warp_col_base + out_sub * DOT_SLICE) / DOT_SLICE;
                load(b_tile_s2, subtile_inplace<DOT_SLICE, DOT_SLICE>(W2s, {w2_row_tile, kk}));
                
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(0)");
                
                mma_ABt(C_out[out_sub], a_tile_s2, b_tile_s2, C_out[out_sub]);
            }
        }
        
        __syncthreads();
    }
    // End of inter_chunk loop
    
    // ==========================================================================
    // EPILOGUE: Weighted atomic scatter to output
    // ==========================================================================
    const int row_offset_out = (lane_in_warp / 16) * 4;
    const int col_offset_out = lane_in_warp % 16;
    
    #pragma unroll
    for (int out_sub = 0; out_sub < OUT_SUBTILES; out_sub++) {
        const auto& out_tile = C_out[out_sub].tiles[0][0];
        int out_col = col_start + warp_col_base + out_sub * DOT_SLICE + col_offset_out;
        
        if (out_col < model_dim) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                int m = warp_row_base + row_offset_out + r;
                if (m < M_TILE) {
                    int row = row_start + m;
                    if (row < sorted_M_valid) {
                        int packed_id = sorted_ids[row];
                        int token_id = packed_id & 0xFFFFFF;
                        float weight = sorted_weights[row];
                        
                        if (token_id >= 0 && token_id < num_tokens) {
                            float val;
                            if (r == 0) val = out_tile.data[0].x;
                            else if (r == 1) val = out_tile.data[0].y;
                            else if (r == 2) val = out_tile.data[1].x;
                            else val = out_tile.data[1].y;
                            
                            atomicAdd(&output_fp32[token_id * model_dim + out_col], val * weight);
                        }
                    }
                }
            }
        }
    }
}

// Dispatch function for streaming kernel
void dispatch_hk_moe_streaming_fp8(
    const bf16* hidden_states,
    const fp8_t* w1_fp8,
    const fp8_t* w2_fp8,
    const float* w1_scale,
    const float* w2_scale,
    float* output_fp32,
    const int32_t* sorted_ids,
    const int32_t* sorted_expert_ids,
    const int32_t* num_valid_ids,
    const float* sorted_weights,
    const int sorted_M,
    const int num_tokens,
    const int model_dim,
    const int inter_dim,
    const int num_experts,
    const int block_m_sorting,
    hipStream_t stream
) {
    using namespace stream_cfg;
    
    const int num_m_blocks = (sorted_M + M_TILE - 1) / M_TILE;
    const int num_n_blocks = (model_dim + N_TILE - 1) / N_TILE;
    
    const int num_scale_n_w1 = (inter_dim * 2 + fp8_cfg::SCALE_BLOCK_N - 1) / fp8_cfg::SCALE_BLOCK_N;
    const int num_scale_k_w1 = (model_dim + fp8_cfg::SCALE_BLOCK_K - 1) / fp8_cfg::SCALE_BLOCK_K;
    const int num_scale_n_w2 = (model_dim + fp8_cfg::SCALE_BLOCK_N - 1) / fp8_cfg::SCALE_BLOCK_N;
    const int num_scale_k_w2 = (inter_dim + fp8_cfg::SCALE_BLOCK_K - 1) / fp8_cfg::SCALE_BLOCK_K;
    
    dim3 grid(num_n_blocks, num_m_blocks);
    dim3 block(NUM_THREADS);
    
    // Shared memory: input tile + W1 tile + raw intermediate + W2 tile + token offsets
    size_t shm_size = sizeof(stream_input_tile) + sizeof(stream_w1_tile) 
                    + M_TILE * K_INTER * sizeof(bf16) + sizeof(stream_w2_tile) + M_TILE * sizeof(int);
    
    hk_moe_streaming_fp8_kernel<<<grid, block, shm_size, stream>>>(
        hidden_states, w1_fp8, w2_fp8, w1_scale, w2_scale,
        output_fp32, sorted_ids, sorted_expert_ids, num_valid_ids, sorted_weights,
        sorted_M, num_tokens, model_dim, inter_dim, num_experts,
        block_m_sorting, num_m_blocks, num_n_blocks,
        num_scale_n_w1, num_scale_k_w1, num_scale_n_w2, num_scale_k_w2
    );
}


// ============================================================================
// Dispatch Function (Original LDS-based fusion)
// ============================================================================
void dispatch_hk_moe_fused_fp8(
    const bf16* hidden_states,
    const fp8_t* w1_fp8,
    const fp8_t* w2_fp8,
    const float* w1_scale,
    const float* w2_scale,
    float* output_fp32,
    const int32_t* sorted_ids,
    const int32_t* sorted_expert_ids,
    const int32_t* num_valid_ids,
    const float* sorted_weights,
    const int sorted_M,
    const int num_tokens,
    const int model_dim,
    const int inter_dim,
    const int num_experts,
    const int block_m_sorting,
    hipStream_t stream
) {
    using namespace fused_cfg;
    
    const int num_m_blocks = (sorted_M + M_TILE - 1) / M_TILE;
    const int num_n_blocks = (model_dim + N_TILE - 1) / N_TILE;
    
    const int num_scale_n_w1 = (inter_dim * 2 + fp8_cfg::SCALE_BLOCK_N - 1) / fp8_cfg::SCALE_BLOCK_N;
    const int num_scale_k_w1 = (model_dim + fp8_cfg::SCALE_BLOCK_K - 1) / fp8_cfg::SCALE_BLOCK_K;
    const int num_scale_n_w2 = (model_dim + fp8_cfg::SCALE_BLOCK_N - 1) / fp8_cfg::SCALE_BLOCK_N;
    const int num_scale_k_w2 = (inter_dim + fp8_cfg::SCALE_BLOCK_K - 1) / fp8_cfg::SCALE_BLOCK_K;
    
    dim3 grid(num_n_blocks, num_m_blocks);
    dim3 block(NUM_THREADS);
    
    size_t shm_size = sizeof(fused_input_tile) + sizeof(fused_weight_tile) 
                    + INTER_LDS_SIZE + M_TILE * sizeof(int);
    
    hk_moe_fused_fp8_kernel<<<grid, block, shm_size, stream>>>(
        hidden_states, w1_fp8, w2_fp8, w1_scale, w2_scale,
        output_fp32, sorted_ids, sorted_expert_ids, num_valid_ids, sorted_weights,
        sorted_M, num_tokens, model_dim, inter_dim, num_experts,
        block_m_sorting, num_m_blocks, num_n_blocks,
        num_scale_n_w1, num_scale_k_w1, num_scale_n_w2, num_scale_k_w2
    );
}
