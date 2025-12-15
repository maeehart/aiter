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

/*
 * ============================================================================
 * STREAMING FUSION KERNEL - Heavily Commented Version for Debugging
 * ============================================================================
 * 
 * MATHEMATICAL FORMULA:
 *   output[token, col] = Σ_expert weight[token,expert] * 
 *                        (silu(hidden[token] @ W1_gate[expert]^T) * 
 *                         (hidden[token] @ W1_up[expert]^T)) @ W2[expert]^T
 * 
 * WHERE:
 *   hidden:  [num_tokens, model_dim]     = [N, 7168]
 *   W1_gate: [inter_dim, model_dim]      = [256, 7168]  (rows 0:256 of W1)
 *   W1_up:   [inter_dim, model_dim]      = [256, 7168]  (rows 256:512 of W1)
 *   W2:      [model_dim, inter_dim]      = [7168, 256]
 *   output:  [num_tokens, model_dim]     = [N, 7168]
 * 
 * STREAMING APPROACH:
 *   Process inter_dim in K_INTER=32 chunks:
 *   For c in [0, 8):  (8 = 256/32)
 *     inter_partial = silu(hidden @ W1_gate[c*32:(c+1)*32]^T) * 
 *                          (hidden @ W1_up[c*32:(c+1)*32]^T)
 *     output += inter_partial @ W2[:, c*32:(c+1)*32]^T
 * 
 * TILE SIZES:
 *   M_TILE = 32  (rows per workgroup)
 *   N_TILE = 128 (output columns per workgroup)
 *   K_STEP = 32  (model_dim chunk for Stage 1 inner loop)
 *   K_INTER = 32 (inter_dim chunk for streaming)
 * 
 * WARP LAYOUT (4 warps per workgroup):
 *   warp_id | warp_row_idx | warp_col_idx | Output region
 *   --------|--------------|--------------|------------------
 *      0    |      0       |      0       | rows[0:16], cols[0:64]
 *      1    |      0       |      1       | rows[0:16], cols[64:128]
 *      2    |      1       |      0       | rows[16:32], cols[0:64]
 *      3    |      1       |      1       | rows[16:32], cols[64:128]
 * 
 * REGISTER TILE rt_fl<16, 16> LAYOUT (col-major, 64 threads per warp):
 *   - Each thread holds 4 elements: data[0].x, data[0].y, data[1].x, data[1].y
 *   - Thread lane L owns: row = (L/16)*4 + {0,1,2,3}, col = L%16
 *   - Example: lane 0 → rows {0,1,2,3}, col 0
 *             lane 8 → rows {0,1,2,3}, col 8
 *             lane 16 → rows {4,5,6,7}, col 0
 * ============================================================================
 */
__global__ __launch_bounds__(stream_cfg::NUM_THREADS, 2)
void hk_moe_streaming_fp8_kernel(
    const bf16* __restrict__ hidden_states,   // [num_tokens, model_dim]
    const fp8_t* __restrict__ w1_fp8,         // [num_experts, inter_dim*2, model_dim]
    const fp8_t* __restrict__ w2_fp8,         // [num_experts, model_dim, inter_dim]
    const float* __restrict__ w1_scale,       // [num_experts, scale_n_w1, scale_k_w1]
    const float* __restrict__ w2_scale,       // [num_experts, scale_n_w2, scale_k_w2]
    float* __restrict__ output_fp32,          // [num_tokens, model_dim]
    const int32_t* __restrict__ sorted_ids,   // [sorted_M] packed (token_id | topk_slot<<24)
    const int32_t* __restrict__ sorted_expert_ids,  // [num_tiles]
    const int32_t* __restrict__ num_valid_ids,
    const float* __restrict__ sorted_weights, // [sorted_M]
    const int sorted_M,
    const int num_tokens,
    const int model_dim,   // 7168
    const int inter_dim,   // 256
    const int num_experts, // 256
    const int block_m_sorting,
    const int num_m_blocks,
    const int num_n_blocks,
    const int num_scale_n_w1,  // ceil(inter_dim*2 / 128) = 4
    const int num_scale_k_w1,  // ceil(model_dim / 128) = 56
    const int num_scale_n_w2,  // ceil(model_dim / 128) = 56
    const int num_scale_k_w2   // ceil(inter_dim / 128) = 2
) {
    using namespace stream_cfg;
    using namespace fp8_cfg;
    
    const int sorted_M_valid = num_valid_ids[0];
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    // =========================================================================
    // LDS ALLOCATION (sequential, non-overlapping)
    // =========================================================================
    // As:        st_bf<32, 32>  - for hidden input / intermediate copy
    // W1s:       st_bf<32, 32>  - for W1_gate or W1_up weights  
    // W2s:       st_bf<128, 32> - for W2 weights
    // inter_lds: bf16[32][32]   - raw 2D array for intermediate results
    // token_row_offsets: int[32] - precomputed hidden_states offsets
    
    stream_input_tile (&As) = al.allocate<stream_input_tile>();      // offset 0, size 2KB
    stream_w1_tile (&W1s) = al.allocate<stream_w1_tile>();           // offset 2KB, size 2KB
    stream_w2_tile (&W2s) = al.allocate<stream_w2_tile>();           // offset 4KB, size 8KB
    bf16 (*inter_lds)[K_INTER] = reinterpret_cast<bf16(*)[K_INTER]>(al.ptr);
    al.ptr += M_TILE * K_INTER * sizeof(bf16);                       // offset 12KB, size 2KB
    int* token_row_offsets = (int*)al.allocate<int[M_TILE]>();       // offset 14KB, size 128B
    
    // =========================================================================
    // WORKGROUP POSITION
    // =========================================================================
    // Grid: [num_n_blocks, num_m_blocks]
    // Each workgroup computes output[row_start:row_start+32, col_start:col_start+128]
    
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    
    // NOTE: XCD transform disabled for now - causes correctness issues
    // The transform reorders workgroups for better cache utilization but
    // interacts badly with the streaming kernel's data dependencies
    // TODO: Investigate proper XCD handling for streaming kernel
    // if (NUM_WGS >= 128) {
    //     wgid = fused_xcd_transform(wgid, NUM_WGS);
    // }
    if (wgid >= NUM_WGS) return;
    
    const int pid_m = wgid / num_n_blocks;      // Which M-tile (row block)
    const int pid_n = wgid % num_n_blocks;      // Which N-tile (col block)
    if (pid_m >= num_m_blocks || pid_n >= num_n_blocks) return;
    
    // FORMULA: Global row/col start for this workgroup
    const int row_start = pid_m * M_TILE;       // First sorted row for this WG
    const int col_start = pid_n * N_TILE;       // First output column for this WG
    if (row_start >= sorted_M_valid || col_start >= model_dim) return;
    
    const int tile_id = row_start / block_m_sorting;
    const int expert_id = sorted_expert_ids[tile_id];
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    // =========================================================================
    // EXPERT WEIGHT POINTERS
    // =========================================================================
    // W1 layout: [num_experts, inter_dim*2, model_dim] = [256, 512, 7168]
    //   - Gate weights: W1[expert, 0:256, :]
    //   - Up weights:   W1[expert, 256:512, :]
    // W2 layout: [num_experts, model_dim, inter_dim] = [256, 7168, 256]
    
    const fp8_t* w1_expert = w1_fp8 + expert_id * (size_t)(inter_dim * 2) * model_dim;
    const float* w1_expert_scale = w1_scale + expert_id * (size_t)num_scale_n_w1 * num_scale_k_w1;
    const fp8_t* w2_expert = w2_fp8 + expert_id * (size_t)model_dim * inter_dim;
    const float* w2_expert_scale = w2_scale + expert_id * (size_t)num_scale_n_w2 * num_scale_k_w2;
    
    // =========================================================================
    // THREAD/WARP INDICES
    // =========================================================================
    const int warp_id = warpid();               // 0, 1, 2, or 3
    const int lane = threadIdx.x;               // 0-255 (global thread in WG)
    const int lane_in_warp = lane % WARP_THREADS;  // 0-63 (lane within warp)
    
    // Warp position in 2×2 layout:
    // warp_row_idx: 0 for warps 0,1; 1 for warps 2,3
    // warp_col_idx: 0 for warps 0,2; 1 for warps 1,3
    const int warp_row_idx = warp_id / 2;       // 0 or 1
    const int warp_col_idx = warp_id % 2;       // 0 or 1
    
    // FORMULA: Base row/col for this warp's output
    const int warp_row_base = warp_row_idx * REG_M;   // 0 or 16
    const int warp_col_base = warp_col_idx * REG_N;   // 0 or 64
    
    // =========================================================================
    // PRECOMPUTE TOKEN OFFSETS
    // =========================================================================
    // For each sorted row in this M_TILE, compute the offset into hidden_states
    // FORMULA: token_row_offsets[i] = token_id * model_dim
    //          where token_id = sorted_ids[row_start + i] & 0xFFFFFF
    // This allows gathering hidden states for different tokens
    
    for (int i = lane; i < M_TILE; i += NUM_THREADS) {
        int row = row_start + i;          // Global sorted row index
        int base = -1;                    // -1 means invalid
        if (row < sorted_M_valid) {
            int packed_id = sorted_ids[row];
            int token_id = packed_id & 0xFFFFFF;  // Lower 24 bits = token ID
            if (token_id >= 0 && token_id < num_tokens) {
                base = token_id * model_dim;  // Offset into hidden_states
            }
        }
        token_row_offsets[i] = base;
    }
    __syncthreads();
    
    // =========================================================================
    // OUTPUT ACCUMULATORS (persist across all inter_chunks)
    // =========================================================================
    // C_out[out_sub] is a [16, 16] register tile (rt_fl)
    // OUT_SUBTILES = REG_N / DOT_SLICE = 64 / 16 = 4
    // 
    // Each warp accumulates output for its region:
    //   C_out[0] → output columns [warp_col_base + 0,  warp_col_base + 16)
    //   C_out[1] → output columns [warp_col_base + 16, warp_col_base + 32)
    //   C_out[2] → output columns [warp_col_base + 32, warp_col_base + 48)
    //   C_out[3] → output columns [warp_col_base + 48, warp_col_base + 64)
    // 
    // For warp 0 (warp_col_base=0): C_out covers cols 0-63
    // For warp 1 (warp_col_base=64): C_out covers cols 64-127
    
    constexpr int OUT_SUBTILES = REG_N / DOT_SLICE;  // 4
    rt_fl<REG_M, DOT_SLICE, ducks::rt_layout::col> C_out[OUT_SUBTILES];
    #pragma unroll
    for (int i = 0; i < OUT_SUBTILES; i++) {
        zero(C_out[i]);  // Initialize to 0 before accumulation
    }
    
    // =========================================================================
    // MAIN LOOP: Iterate over inter_dim in K_INTER=32 chunks
    // =========================================================================
    // FORMULA: output += Σ_c (inter_partial[c] @ W2[:, c*32:(c+1)*32]^T)
    // where inter_partial[c] = silu(hidden @ W1_gate[c*32:(c+1)*32]^T) * 
    //                               (hidden @ W1_up[c*32:(c+1)*32]^T)
    // 
    // num_inter_chunks = ceil(inter_dim / K_INTER) = ceil(256 / 32) = 8
    
    const int num_inter_chunks = (inter_dim + K_INTER - 1) / K_INTER;
    
    // DEBUG: Can set debug_max_chunks = 1 to test single chunk
    // const int debug_max_chunks = 1;
    const int debug_max_chunks = num_inter_chunks;
    
    for (int inter_chunk = 0; inter_chunk < debug_max_chunks; inter_chunk++) {
        // FORMULA: inter_k_start = inter_chunk * K_INTER
        // This is the starting column in inter_dim for this chunk
        // inter_chunk=0 → inter_k_start=0   (process inter cols 0-31)
        // inter_chunk=1 → inter_k_start=32  (process inter cols 32-63)
        // ...
        // inter_chunk=7 → inter_k_start=224 (process inter cols 224-255)
        const int inter_k_start = inter_chunk * K_INTER;
        
        // =====================================================================
        // STAGE 1: Compute partial intermediate [M_TILE, K_INTER] = [32, 32]
        // =====================================================================
        // FORMULA: 
        //   C_gate = hidden @ W1_gate[inter_k_start:inter_k_start+32]^T
        //   C_up   = hidden @ W1_up[inter_k_start:inter_k_start+32]^T
        //   inter  = silu(C_gate) * C_up
        //
        // Dimensions:
        //   hidden:  [M_TILE, model_dim] = [32, 7168]  (gathered via token offsets)
        //   W1_gate: [K_INTER, model_dim] = [32, 7168] (rows inter_k_start to inter_k_start+32)
        //   C_gate:  [M_TILE, K_INTER] = [32, 32]
        //   C_up:    [M_TILE, K_INTER] = [32, 32]
        //   inter:   [M_TILE, K_INTER] = [32, 32]
        
        // INTERMEDIATE ACCUMULATORS (for this chunk only, reset each chunk)
        // INTER_SUBTILES = K_INTER / DOT_SLICE = 32 / 16 = 2
        // C_gate[0] → inter columns [0, 16)   relative to inter_k_start
        // C_gate[1] → inter columns [16, 32)  relative to inter_k_start
        // 
        // Each warp computes [16, 32] of intermediate (rows determined by warp_row_idx)
        // Warp 0,1 (warp_row_idx=0): compute rows [0, 16)
        // Warp 2,3 (warp_row_idx=1): compute rows [16, 32)
        
        constexpr int INTER_SUBTILES = K_INTER / DOT_SLICE;  // 2
        rt_fl<REG_M, DOT_SLICE, ducks::rt_layout::col> C_gate[INTER_SUBTILES];
        rt_fl<REG_M, DOT_SLICE, ducks::rt_layout::col> C_up[INTER_SUBTILES];
        #pragma unroll
        for (int i = 0; i < INTER_SUBTILES; i++) {
            zero(C_gate[i]);
            zero(C_up[i]);
        }
        
        // INNER K-LOOP: Iterate over model_dim in K_STEP=32 chunks
        // FORMULA: C_gate += Σ_k (hidden[:, k*32:(k+1)*32] @ W1_gate[:, k*32:(k+1)*32]^T)
        // num_k1_tiles = ceil(model_dim / K_STEP) = ceil(7168 / 32) = 224
        const int num_k1_tiles = (model_dim + K_STEP - 1) / K_STEP;
        
        for (int k1_tile = 0; k1_tile < num_k1_tiles; k1_tile++) {
            // k1_start is the starting column in model_dim for this K-tile
            const int k1_start = k1_tile * K_STEP;
            
            // -----------------------------------------------------------------
            // LOAD HIDDEN TILE [M_TILE, K_STEP] = [32, 32] to As
            // -----------------------------------------------------------------
            // FORMULA: As[m, k] = hidden_states[token_row_offsets[m] + k1_start + k]
            // 
            // INDEXING:
            //   VEC_SIZE = 8 bf16 values per vector load
            //   VECS_PER_ROW = K_STEP / VEC_SIZE = 32 / 8 = 4 vectors per row
            //   TOTAL_VECS_IN = M_TILE * VECS_PER_ROW = 32 * 4 = 128 vectors total
            //   
            // THREAD ASSIGNMENT (strided):
            //   vec_idx = lane + v * NUM_THREADS
            //   m = vec_idx / 4   (which row, 0-31)
            //   k = (vec_idx % 4) * 8   (which col group: 0, 8, 16, or 24)
            //
            // STORES: As[m, k:k+8] via store_shared_vec
            
            uint32_t As_ptr = reinterpret_cast<uintptr_t>(&As.data[0]);
            constexpr int VEC_SIZE = 8;
            constexpr int VECS_PER_ROW = K_STEP / VEC_SIZE;        // 4
            constexpr int TOTAL_VECS_IN = M_TILE * VECS_PER_ROW;   // 128
            
            #pragma unroll
            for (int v = 0; v < (TOTAL_VECS_IN + NUM_THREADS - 1) / NUM_THREADS; v++) {
                int vec_idx = lane + v * NUM_THREADS;
                if (vec_idx < TOTAL_VECS_IN) {
                    int m = vec_idx / VECS_PER_ROW;              // Row in [0, 32)
                    int k = (vec_idx % VECS_PER_ROW) * VEC_SIZE; // Col: 0, 8, 16, or 24
                    
                    float4 buf = {0.f, 0.f, 0.f, 0.f};
                    int base = token_row_offsets[m];
                    // FORMULA: Read hidden[token_id, k1_start+k : k1_start+k+8]
                    if (base >= 0 && (k1_start + k + VEC_SIZE - 1) < model_dim) {
                        buf = *reinterpret_cast<const float4*>(&hidden_states[base + k1_start + k]);
                    }
                    // FORMULA: Store to As[m, k:k+4] and As[m, k+4:k+8]
                    store_shared_vec(As.idx(As_ptr, {m, k}), {buf.x, buf.y});
                    store_shared_vec(As.idx(As_ptr, {m, k + 4}), {buf.z, buf.w});
                }
            }
            
            // -----------------------------------------------------------------
            // LOAD W1_GATE TILE [K_INTER, K_STEP] = [32, 32] to W1s
            // -----------------------------------------------------------------
            // FORMULA: W1s[n, k] = W1_gate[inter_k_start + n, k1_start + k] * scale
            // 
            // W1 LAYOUT: W1[expert, row, col] where:
            //   row ∈ [0, inter_dim)        → gate weights
            //   row ∈ [inter_dim, inter_dim*2) → up weights
            //
            // INDEXING:
            //   n = vec_idx / 4   → row in W1s, which is (inter_k_start + n) in W1_gate
            //   k = (vec_idx % 4) * 8   → col in W1s, which is (k1_start + k) in model_dim
            //   
            // W1 ACCESS: w1_expert[(inter_k_start + n) * model_dim + (k1_start + k)]
            //          = W1_gate[inter_k_start + n, k1_start + k]
            
            uint32_t W1s_ptr = reinterpret_cast<uintptr_t>(&W1s.data[0]);
            constexpr int TOTAL_VECS_W1 = K_INTER * K_STEP / VEC_SIZE;  // 32 * 32 / 8 = 128
            
            // SCALE LOOKUP: scale_gate = w1_scale[n_block * num_scale_k + k_block]
            int k_block = k1_start / SCALE_BLOCK_K;           // Which 128-block in model_dim
            int n_block_gate = inter_k_start / SCALE_BLOCK_N; // Which 128-block in inter_dim
            float scale_gate = w1_expert_scale[n_block_gate * num_scale_k_w1 + k_block];
            
            #pragma unroll
            for (int v = 0; v < (TOTAL_VECS_W1 + NUM_THREADS - 1) / NUM_THREADS; v++) {
                int vec_idx = lane + v * NUM_THREADS;
                if (vec_idx < TOTAL_VECS_W1) {
                    int n = vec_idx / (K_STEP / VEC_SIZE);              // Row in W1s [0, 32)
                    int k = (vec_idx % (K_STEP / VEC_SIZE)) * VEC_SIZE; // Col: 0, 8, 16, 24
                    int col = inter_k_start + n;   // Absolute row in W1_gate (inter_dim index)
                    int k_global = k1_start + k;   // Absolute col in W1 (model_dim index)
                    
                    float2 buf_lo = {0.f, 0.f}, buf_hi = {0.f, 0.f};
                    // FORMULA: Load W1_gate[col, k_global:k_global+8]
                    if (col < inter_dim && (k_global + VEC_SIZE - 1) < model_dim) {
                        const fp8_t* src = &w1_expert[col * model_dim + k_global];
                        fp8x8_to_bf16x8_scaled(src, scale_gate, buf_lo, buf_hi);
                    }
                    // FORMULA: Store to W1s[n, k:k+8]
                    store_shared_vec(W1s.idx(W1s_ptr, {n, k}), {buf_lo.x, buf_lo.y});
                    store_shared_vec(W1s.idx(W1s_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
                }
            }
            
            __syncthreads();
            
            // -----------------------------------------------------------------
            // MFMA FOR GATE: C_gate += As @ W1s^T
            // -----------------------------------------------------------------
            // FORMULA: C_gate[inter_sub] += subtile(As) @ subtile(W1s)^T
            //
            // LOOP STRUCTURE:
            //   kk ∈ [0, 2): K_STEP / DOT_SLICE = 32 / 16 = 2 subtiles in K
            //   inter_sub ∈ [0, 2): INTER_SUBTILES = 2 subtiles in intermediate
            //
            // SUBTILE LOADS:
            //   a_tile = As[warp_row_idx*16 : (warp_row_idx+1)*16, kk*16 : (kk+1)*16]
            //          = [16, 16] subtile of hidden
            //   b_tile = W1s[inter_sub*16 : (inter_sub+1)*16, kk*16 : (kk+1)*16]
            //          = [16, 16] subtile of gate weights
            //
            // MFMA: C_gate[inter_sub] += a_tile @ b_tile^T
            //       [16, 16] = [16, 16] @ [16, 16]^T
            //       Result rows = hidden rows [warp_row_base, warp_row_base+16)
            //       Result cols = inter cols [inter_k_start + inter_sub*16, ...)
            
            rt_bf<REG_M, DOT_SLICE> a_tile;  // [16, 16] bf16 register tile
            
            #pragma unroll
            for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {  // kk = 0, 1
                // Load A subtile from As
                // FORMULA: a_tile = As[warp_row_idx*16:(warp_row_idx+1)*16, kk*16:(kk+1)*16]
                load(a_tile, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row_idx, kk}));
                
                #pragma unroll
                for (int inter_sub = 0; inter_sub < INTER_SUBTILES; inter_sub++) {  // inter_sub = 0, 1
                    rt_bf<DOT_SLICE, DOT_SLICE> b_tile;  // [16, 16] bf16 register tile
                    // Load B subtile from W1s
                    // FORMULA: b_tile = W1s[inter_sub*16:(inter_sub+1)*16, kk*16:(kk+1)*16]
                    load(b_tile, subtile_inplace<DOT_SLICE, DOT_SLICE>(W1s, {inter_sub, kk}));
                    
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    
                    // MFMA: C_gate[inter_sub] += a_tile @ b_tile^T
                    mma_ABt(C_gate[inter_sub], a_tile, b_tile, C_gate[inter_sub]);
                }
            }
            
            __syncthreads();
            
            // -----------------------------------------------------------------
            // LOAD W1_UP TILE [K_INTER, K_STEP] = [32, 32] to W1s
            // -----------------------------------------------------------------
            // FORMULA: W1s[n, k] = W1_up[inter_k_start + n, k1_start + k] * scale
            // 
            // W1_UP ROW OFFSET: up_col_start = inter_dim + inter_k_start
            //   W1_up rows are at W1[inter_dim:inter_dim*2, :] = W1[256:512, :]
            //   For inter_k_start=0: up_col_start = 256 (rows 256-287 of W1)
            //   For inter_k_start=32: up_col_start = 288 (rows 288-319 of W1)
            //
            // W1 ACCESS: w1_expert[(up_col_start + n) * model_dim + (k1_start + k)]
            
            int up_col_start = inter_dim + inter_k_start;  // Row offset for up weights
            int n_block_up = up_col_start / SCALE_BLOCK_N; // Scale block: 2 for rows 256-383, 3 for 384-511
            float scale_up = w1_expert_scale[n_block_up * num_scale_k_w1 + k_block];
            
            #pragma unroll
            for (int v = 0; v < (TOTAL_VECS_W1 + NUM_THREADS - 1) / NUM_THREADS; v++) {
                int vec_idx = lane + v * NUM_THREADS;
                if (vec_idx < TOTAL_VECS_W1) {
                    int n = vec_idx / (K_STEP / VEC_SIZE);              // Row in W1s [0, 32)
                    int k = (vec_idx % (K_STEP / VEC_SIZE)) * VEC_SIZE; // Col: 0, 8, 16, 24
                    int col = up_col_start + n;  // Absolute row in W1 (256 + inter_k_start + n)
                    int k_global = k1_start + k; // Absolute col in W1 (model_dim index)
                    
                    float2 buf_lo = {0.f, 0.f}, buf_hi = {0.f, 0.f};
                    // FORMULA: Load W1_up[col - inter_dim, k_global:k_global+8]
                    //        = W1[col, k_global:k_global+8]
                    if (col < inter_dim * 2 && (k_global + VEC_SIZE - 1) < model_dim) {
                        const fp8_t* src = &w1_expert[col * model_dim + k_global];
                        fp8x8_to_bf16x8_scaled(src, scale_up, buf_lo, buf_hi);
                    }
                    // FORMULA: Store to W1s[n, k:k+8] (overwrites gate weights)
                    store_shared_vec(W1s.idx(W1s_ptr, {n, k}), {buf_lo.x, buf_lo.y});
                    store_shared_vec(W1s.idx(W1s_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
                }
            }
            
            __syncthreads();
            
            // -----------------------------------------------------------------
            // MFMA FOR UP: C_up += As @ W1s^T (same structure as gate)
            // -----------------------------------------------------------------
            // FORMULA: C_up[inter_sub] += subtile(As) @ subtile(W1s)^T
            // Uses same As (hidden) but W1s now contains up weights
            
            #pragma unroll
            for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
                load(a_tile, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row_idx, kk}));
                
                #pragma unroll
                for (int inter_sub = 0; inter_sub < INTER_SUBTILES; inter_sub++) {
                    rt_bf<DOT_SLICE, DOT_SLICE> b_tile;
                    load(b_tile, subtile_inplace<DOT_SLICE, DOT_SLICE>(W1s, {inter_sub, kk}));
                    
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    
                    // MFMA: C_up[inter_sub] += a_tile @ b_tile^T
                    mma_ABt(C_up[inter_sub], a_tile, b_tile, C_up[inter_sub]);
                }
            }
            
            __syncthreads();
        }
        // End of K1 loop - C_gate and C_up now contain complete partial results
        
        // =====================================================================
        // APPLY SiLU: C_gate = silu(C_gate) * C_up
        // =====================================================================
        // FORMULA: inter[i,j] = silu(gate[i,j]) * up[i,j]
        //          where silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
        //
        // REGISTER TILE LAYOUT (rt_fl<16, 16> col-major):
        //   - Each thread in warp holds 4 elements of the 16×16 tile
        //   - tiles[0][0] is the single 16×16 subtile
        //   - data[0].x, data[0].y, data[1].x, data[1].y are the 4 elements
        //   
        // THREAD-TO-ELEMENT MAPPING (for rt_fl col-major):
        //   Thread lane L owns elements at:
        //     row = (L / 16) * 4 + r,  where r ∈ {0, 1, 2, 3}
        //     col = L % 16
        //   
        //   data[0].x → row = (L/16)*4 + 0, col = L%16
        //   data[0].y → row = (L/16)*4 + 1, col = L%16
        //   data[1].x → row = (L/16)*4 + 2, col = L%16
        //   data[1].y → row = (L/16)*4 + 3, col = L%16
        //
        // After this, C_gate contains intermediate = silu(gate) * up
        
        #pragma unroll
        for (int inter_sub = 0; inter_sub < INTER_SUBTILES; inter_sub++) {
            auto& gate_tile = C_gate[inter_sub].tiles[0][0];
            auto& up_tile = C_up[inter_sub].tiles[0][0];
            
            // Process all 4 elements owned by this thread (i=0,1 for data[0] and data[1])
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                float g_x = gate_tile.data[i].x;  // Element at row (L/16)*4 + 2*i
                float g_y = gate_tile.data[i].y;  // Element at row (L/16)*4 + 2*i + 1
                float u_x = up_tile.data[i].x;
                float u_y = up_tile.data[i].y;
                
                // FORMULA: silu(g) = g / (1 + exp(-g))
                float silu_x = g_x / (1.0f + expf(-g_x));
                float silu_y = g_y / (1.0f + expf(-g_y));
                
                // FORMULA: inter = silu(gate) * up
                gate_tile.data[i].x = silu_x * u_x;
                gate_tile.data[i].y = silu_y * u_y;
            }
        }
        // Now C_gate[inter_sub] contains intermediate for:
        //   rows: [warp_row_base, warp_row_base + 16) = [0,16) or [16,32)
        //   cols: [inter_sub * 16, (inter_sub + 1) * 16) = [0,16) or [16,32)
        //   relative to this chunk's inter_k_start
        
        // =====================================================================
        // STAGE 2: C_out += intermediate @ W2[:, inter_k_start:inter_k_start+32]^T
        // =====================================================================
        // FORMULA: output[:, col_start:col_start+128] += 
        //          inter[:, 0:32] @ W2[col_start:col_start+128, inter_k_start:inter_k_start+32]^T
        //
        // DIMENSIONS:
        //   inter:   [M_TILE, K_INTER] = [32, 32]    (stored in inter_lds after this)
        //   W2:      [N_TILE, K_INTER] = [128, 32]   (loaded to W2s)
        //   W2^T:    [K_INTER, N_TILE] = [32, 128]
        //   C_out:   [M_TILE, N_TILE] = [32, 128]   (accumulated in registers)
        //
        // W2 LAYOUT: W2[expert, output_col, inter_col] = [256, 7168, 256]
        //   For this workgroup: output_col ∈ [col_start, col_start+128)
        //   For this chunk:     inter_col ∈ [inter_k_start, inter_k_start+32)
        
        // -----------------------------------------------------------------
        // LOAD W2 SLICE [N_TILE, K_INTER] = [128, 32] to W2s
        // -----------------------------------------------------------------
        // FORMULA: W2s[n, k] = W2[col_start + n, inter_k_start + k] * scale
        //
        // INDEXING (INTERLEAVED PATTERN - matches original LDS fusion):
        //   VEC_SIZE_W2 = 8 bf16 per vector
        //   TOTAL_VECS_W2 = N_TILE * K_INTER / 8 = 128 * 32 / 8 = 512
        //   VECS_PER_THREAD_W2 = ceil(512 / 256) = 2
        //
        //   vec_idx = lane * 2 + v  (INTERLEAVED: lane determines base, v adds offset)
        //   n = vec_idx / (K_INTER / VEC_SIZE_W2) = vec_idx / 4  → row in W2s [0, 128)
        //   k = (vec_idx % 4) * 8  → col in W2s: 0, 8, 16, or 24
        //
        // THREAD ASSIGNMENT (interleaved):
        //   lane=0,  v=0,1: vec_idx=0,1   → n=0,0   k=0,8     → W2s row 0
        //   lane=1,  v=0,1: vec_idx=2,3   → n=0,0   k=16,24   → W2s row 0
        //   lane=2,  v=0,1: vec_idx=4,5   → n=1,1   k=0,8     → W2s row 1
        //   ...
        //   lane=80, v=0,1: vec_idx=160,161 → n=40,40 k=0,8   → W2s row 40  ★
        //   lane=81, v=0,1: vec_idx=162,163 → n=40,40 k=16,24 → W2s row 40
        //   ...
        //
        // W2 ACCESS: w2_expert[(col_start + n) * inter_dim + (inter_k_start + k)]
        
        uint32_t W2s_ptr = reinterpret_cast<uintptr_t>(&W2s.data[0]);
        constexpr int VEC_SIZE_W2 = 8;
        constexpr int TOTAL_VECS_W2 = N_TILE * K_INTER / VEC_SIZE_W2;        // 512
        constexpr int VECS_PER_THREAD_W2 = (TOTAL_VECS_W2 + NUM_THREADS - 1) / NUM_THREADS;  // 2
        
        // SCALE LOOKUP for W2
        int n_block_w2 = col_start / SCALE_BLOCK_N;      // Which 128-block in model_dim
        int k_block_w2 = inter_k_start / SCALE_BLOCK_K;  // Which 128-block in inter_dim (0 or 1)
        float scale_w2 = w2_expert_scale[n_block_w2 * num_scale_k_w2 + k_block_w2];
        
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD_W2; v++) {
            // INTERLEAVED pattern: vec_idx = lane * 2 + v
            int vec_idx = lane * VECS_PER_THREAD_W2 + v;
            if (vec_idx < TOTAL_VECS_W2) {
                // FORMULA: n = vec_idx / 4, k = (vec_idx % 4) * 8
                int n = vec_idx / (K_INTER / VEC_SIZE_W2);              // Row in W2s [0, 128)
                int k = (vec_idx % (K_INTER / VEC_SIZE_W2)) * VEC_SIZE_W2; // Col: 0, 8, 16, 24
                int out_col = col_start + n;      // Absolute output column [0, 7168)
                int k_global = inter_k_start + k; // Absolute inter_dim col [0, 256)
                
                float2 buf_lo = {0.f, 0.f}, buf_hi = {0.f, 0.f};
                // FORMULA: Load W2[out_col, k_global:k_global+8]
                if (out_col < model_dim && (k_global + VEC_SIZE_W2 - 1) < inter_dim) {
                    const fp8_t* src = &w2_expert[out_col * inter_dim + k_global];
                    fp8x8_to_bf16x8_scaled(src, scale_w2, buf_lo, buf_hi);
                }
                // FORMULA: Store to W2s[n, k:k+4] and W2s[n, k+4:k+8]
                store_shared_vec(W2s.idx(W2s_ptr, {n, k}), {buf_lo.x, buf_lo.y});
                store_shared_vec(W2s.idx(W2s_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
            }
        }
        
        __syncthreads();
        
        // -----------------------------------------------------------------
        // STORE INTERMEDIATE TO inter_lds [M_TILE, K_INTER] = [32, 32]
        // -----------------------------------------------------------------
        // C_gate[inter_sub] contains silu(gate) * up for this warp's rows
        // Need to write to inter_lds so it can be copied to As for Stage 2 MFMA
        //
        // WARP COVERAGE:
        //   Warps 0 and 1 both compute rows [0, 16) (same warp_row_idx=0)
        //   Warps 2 and 3 both compute rows [16, 32) (same warp_row_idx=1)
        //   → Only warps 0 and 2 need to write (warp_col_idx=0)
        //
        // REGISTER-TO-LDS MAPPING (rt_fl col-major layout):
        //   lane_in_warp = 0-63
        //   row_off_s2 = (lane_in_warp / 16) * 4  → {0, 4, 8, 12}
        //   col_off_s2 = lane_in_warp % 16        → {0, 1, ..., 15}
        //
        //   THREAD-TO-LDS MAPPING:
        //     lane 0-15:  row_off=0,  col_off=0-15   → rows 0-3,   cols 0-15
        //     lane 16-31: row_off=4,  col_off=0-15   → rows 4-7,   cols 0-15
        //     lane 32-47: row_off=8,  col_off=0-15   → rows 8-11,  cols 0-15
        //     lane 48-63: row_off=12, col_off=0-15   → rows 12-15, cols 0-15
        //
        //   For inter_sub=0: writes to cols [0, 16)
        //   For inter_sub=1: writes to cols [16, 32)
        //
        // FORMULA: inter_lds[warp_row_base + row_off_s2 + r][inter_sub*16 + col_off_s2]
        //          = C_gate[inter_sub].tiles[0][0].data[r/2].(x or y)
        
        const int row_off_s2 = (lane_in_warp / 16) * 4;  // 0, 4, 8, 12 (groups of 16 lanes)
        const int col_off_s2 = lane_in_warp % 16;        // 0-15 (column within subtile)
        
        if (warp_col_idx == 0) {  // Only warps 0 and 2 write
            #pragma unroll
            for (int inter_sub = 0; inter_sub < INTER_SUBTILES; inter_sub++) {
                const auto& inter_tile = C_gate[inter_sub].tiles[0][0];
                // FORMULA: lds_col = inter_sub * 16 + col_off_s2
                //          For inter_sub=0: cols 0-15
                //          For inter_sub=1: cols 16-31
                int lds_col = inter_sub * DOT_SLICE + col_off_s2;
                
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    // FORMULA: lds_row = warp_row_base + row_off_s2 + r
                    //   warp 0: warp_row_base=0,  rows 0-15
                    //   warp 2: warp_row_base=16, rows 16-31
                    int lds_row = warp_row_base + row_off_s2 + r;
                    if (lds_row < M_TILE && lds_col < K_INTER) {
                        // FORMULA: val = data[r/2].(x if r%2==0 else y)
                        //   r=0: data[0].x
                        //   r=1: data[0].y
                        //   r=2: data[1].x
                        //   r=3: data[1].y
                        float val;
                        if (r == 0) val = inter_tile.data[0].x;
                        else if (r == 1) val = inter_tile.data[0].y;
                        else if (r == 2) val = inter_tile.data[1].x;
                        else val = inter_tile.data[1].y;
                        
                        // STORE: inter_lds[row][col] = bf16(val)
                        inter_lds[lds_row][lds_col] = __float2bfloat16(val);
                    }
                }
            }
        }
        
        __syncthreads();
        // After sync: inter_lds[0:32, 0:32] contains full intermediate for this chunk
        
        // -----------------------------------------------------------------
        // COPY INTERMEDIATE: inter_lds[32,32] → As[32,32] for MFMA
        // -----------------------------------------------------------------
        // inter_lds is a raw bf16[32][32] array
        // As is a st_bf<32, 32> shared tile (HipKittens type)
        // We need to copy so subtile_inplace can load subtiles correctly
        //
        // INDEXING (STRIDED pattern):
        //   VEC_SIZE_S2 = 8 bf16 per vector
        //   VECS_PER_ROW_S2 = K_INTER / 8 = 32 / 8 = 4 vectors per row
        //   TOTAL_VECS_S2 = M_TILE * 4 = 32 * 4 = 128 vectors total
        //   
        //   vec_idx = lane + v * 256  (STRIDED: lane determines position)
        //   Loop runs once (v=0) since 128 < 256
        //   Only threads 0-127 do work
        //
        //   m = vec_idx / 4   → row in [0, 32)
        //   k = (vec_idx % 4) * 8   → col: 0, 8, 16, or 24
        //
        // COPIES: inter_lds[m][k:k+8] → As[m, k:k+8]
        
        uint32_t As_ptr_s2 = reinterpret_cast<uintptr_t>(&As.data[0]);
        constexpr int VEC_SIZE_S2 = 8;
        constexpr int VECS_PER_ROW_S2 = K_INTER / VEC_SIZE_S2;      // 4
        constexpr int TOTAL_VECS_S2 = M_TILE * VECS_PER_ROW_S2;      // 128
        
        #pragma unroll
        for (int v = 0; v < (TOTAL_VECS_S2 + NUM_THREADS - 1) / NUM_THREADS; v++) {
            int vec_idx = lane + v * NUM_THREADS;  // lane ∈ [0,255], v=0 → vec_idx = lane
            if (vec_idx < TOTAL_VECS_S2) {         // Only if vec_idx < 128
                int m = vec_idx / VECS_PER_ROW_S2;              // Row: 0-31
                int k = (vec_idx % VECS_PER_ROW_S2) * VEC_SIZE_S2; // Col: 0, 8, 16, 24
                
                // READ: 8 bf16 values from inter_lds[m][k:k+8]
                bf16* src = &inter_lds[m][k];
                
                // Pack bf16 pairs into float2 for store_shared_vec
                __hip_bfloat162 v01 = *reinterpret_cast<const __hip_bfloat162*>(&src[0]);
                __hip_bfloat162 v23 = *reinterpret_cast<const __hip_bfloat162*>(&src[2]);
                __hip_bfloat162 v45 = *reinterpret_cast<const __hip_bfloat162*>(&src[4]);
                __hip_bfloat162 v67 = *reinterpret_cast<const __hip_bfloat162*>(&src[6]);
                
                float2 packed_lo, packed_hi;
                memcpy(&packed_lo.x, &v01, sizeof(float));  // bf16[0:2] → float
                memcpy(&packed_lo.y, &v23, sizeof(float));  // bf16[2:4] → float
                memcpy(&packed_hi.x, &v45, sizeof(float));  // bf16[4:6] → float
                memcpy(&packed_hi.y, &v67, sizeof(float));  // bf16[6:8] → float
                
                // STORE: As[m, k:k+4] and As[m, k+4:k+8]
                store_shared_vec(As.idx(As_ptr_s2, {m, k}), packed_lo);
                store_shared_vec(As.idx(As_ptr_s2, {m, k + 4}), packed_hi);
            }
        }
        
        __syncthreads();
        // After sync: As[0:32, 0:32] contains intermediate from inter_lds
        
        // -----------------------------------------------------------------
        // STAGE 2 MFMA: C_out += As @ W2s^T
        // -----------------------------------------------------------------
        // FORMULA: C_out[out_sub] += subtile(As) @ subtile(W2s)^T
        //
        // DIMENSIONS:
        //   As:  [M_TILE, K_INTER] = [32, 32]  (intermediate)
        //   W2s: [N_TILE, K_INTER] = [128, 32] (W2 weights)
        //   C_out: [REG_M, REG_N] = [16, 64] per warp
        //
        // LOOP STRUCTURE:
        //   kk ∈ [0, 2): K_INTER / DOT_SLICE = 32 / 16 = 2 subtiles in K
        //   out_sub ∈ [0, 4): OUT_SUBTILES = 4 subtiles in output cols
        //
        // SUBTILE LOADS:
        //   a_tile_s2 = As[warp_row_idx*16:(warp_row_idx+1)*16, kk*16:(kk+1)*16]
        //             = [16, 16] subtile of intermediate
        //   b_tile_s2 = W2s[w2_row_tile*16:(w2_row_tile+1)*16, kk*16:(kk+1)*16]
        //             = [16, 16] subtile of W2 weights
        //
        // W2_ROW_TILE FORMULA:
        //   w2_row_tile = (warp_col_base + out_sub * 16) / 16
        //   
        //   For warp 0 (warp_col_base=0):
        //     out_sub=0: w2_row_tile = (0 + 0) / 16 = 0  → W2s rows [0, 16)
        //     out_sub=1: w2_row_tile = (0 + 16) / 16 = 1 → W2s rows [16, 32)
        //     out_sub=2: w2_row_tile = (0 + 32) / 16 = 2 → W2s rows [32, 48)  ★ BUG REGION
        //     out_sub=3: w2_row_tile = (0 + 48) / 16 = 3 → W2s rows [48, 64)
        //   
        //   For warp 1 (warp_col_base=64):
        //     out_sub=0: w2_row_tile = (64 + 0) / 16 = 4 → W2s rows [64, 80)
        //     out_sub=1: w2_row_tile = (64 + 16) / 16 = 5 → W2s rows [80, 96)
        //     out_sub=2: w2_row_tile = (64 + 32) / 16 = 6 → W2s rows [96, 112)
        //     out_sub=3: w2_row_tile = (64 + 48) / 16 = 7 → W2s rows [112, 128)
        //
        // MFMA: C_out[out_sub] += a_tile_s2 @ b_tile_s2^T
        //       [16, 16] = [16, 16] @ [16, 16]^T
        //       Result rows: [warp_row_base, warp_row_base+16)
        //       Result cols: [warp_col_base + out_sub*16, warp_col_base + (out_sub+1)*16)
        
        rt_bf<REG_M, DOT_SLICE> a_tile_s2;  // [16, 16] intermediate subtile
        
        #pragma unroll
        for (int kk = 0; kk < K_INTER / DOT_SLICE; kk++) {  // kk = 0, 1
            // LOAD A: As[warp_row_idx*16 : (warp_row_idx+1)*16, kk*16 : (kk+1)*16]
            load(a_tile_s2, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row_idx, kk}));
            
            #pragma unroll
            for (int out_sub = 0; out_sub < OUT_SUBTILES; out_sub++) {  // out_sub = 0, 1, 2, 3
                rt_bf<DOT_SLICE, DOT_SLICE> b_tile_s2;  // [16, 16] W2 subtile
                
                // FORMULA: w2_row_tile = (warp_col_base + out_sub * 16) / 16
                int w2_row_tile = (warp_col_base + out_sub * DOT_SLICE) / DOT_SLICE;
                
                // LOAD B: W2s[w2_row_tile*16 : (w2_row_tile+1)*16, kk*16 : (kk+1)*16]
                load(b_tile_s2, subtile_inplace<DOT_SLICE, DOT_SLICE>(W2s, {w2_row_tile, kk}));
                
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(0)");
                
                // MFMA: C_out[out_sub] += a_tile_s2 @ b_tile_s2^T
                mma_ABt(C_out[out_sub], a_tile_s2, b_tile_s2, C_out[out_sub]);
            }
        }
        
        __syncthreads();
    }
    // End of inter_chunk loop
    // C_out now contains accumulated output for all 8 inter_chunks
    
    // =========================================================================
    // EPILOGUE: Weighted Atomic Scatter to Output
    // =========================================================================
    // FORMULA: output[token_id, out_col] += val * weight
    //
    // C_out[out_sub] is a [16, 16] register tile containing accumulated output
    // Each thread owns 4 elements and must scatter them to the correct output location
    //
    // REGISTER-TO-OUTPUT MAPPING:
    //   Same as intermediate store (rt_fl col-major layout):
    //   row_offset_out = (lane_in_warp / 16) * 4   → {0, 4, 8, 12}
    //   col_offset_out = lane_in_warp % 16        → {0, 1, ..., 15}
    //
    // OUTPUT COLUMN FORMULA:
    //   out_col = col_start + warp_col_base + out_sub * 16 + col_offset_out
    //   
    //   For warp 0, out_sub=2, lane=8 (col_offset_out=8):
    //     out_col = col_start + 0 + 32 + 8 = col_start + 40  ★ First zero column!
    //   
    //   The zeros at column 40+ suggest C_out[2] has zeros in columns 8-15
    //   (i.e., lanes 8-15, 24-31, 40-47, 56-63 have zeros for out_sub=2)
    //
    // OUTPUT ROW FORMULA:
    //   m = warp_row_base + row_offset_out + r   (local row in tile)
    //   row = row_start + m                       (global sorted row)
    //   token_id = sorted_ids[row] & 0xFFFFFF    (actual token)
    
    const int row_offset_out = (lane_in_warp / 16) * 4;  // {0, 4, 8, 12}
    const int col_offset_out = lane_in_warp % 16;        // {0, ..., 15}
    
    #pragma unroll
    for (int out_sub = 0; out_sub < OUT_SUBTILES; out_sub++) {
        const auto& out_tile = C_out[out_sub].tiles[0][0];
        
        // FORMULA: Global output column
        // out_col = col_start + warp_col_base + out_sub * 16 + col_offset_out
        int out_col = col_start + warp_col_base + out_sub * DOT_SLICE + col_offset_out;
        
        if (out_col < model_dim) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                // FORMULA: Local row in M_TILE
                int m = warp_row_base + row_offset_out + r;
                if (m < M_TILE) {
                    int row = row_start + m;  // Global sorted row
                    if (row < sorted_M_valid) {
                        int packed_id = sorted_ids[row];
                        int token_id = packed_id & 0xFFFFFF;
                        float weight = sorted_weights[row];
                        
                        if (token_id >= 0 && token_id < num_tokens) {
                            // FORMULA: Extract value from register tile
                            //   r=0: data[0].x
                            //   r=1: data[0].y
                            //   r=2: data[1].x
                            //   r=3: data[1].y
                            float val;
                            if (r == 0) val = out_tile.data[0].x;
                            else if (r == 1) val = out_tile.data[0].y;
                            else if (r == 2) val = out_tile.data[1].x;
                            else val = out_tile.data[1].y;
                            
                            // ATOMIC ADD: output[token_id, out_col] += val * weight
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
