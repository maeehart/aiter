// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Fused Kernel
 * 
 * Fuses Stage 1 (gate-up + SiLU) and Stage 2 (down projection) into a single kernel
 * to eliminate intermediate global memory traffic (~66MB per forward pass).
 * 
 * Key design decisions:
 * - M_tile = 32 rows (matches AITER's 32x256 tile, keeps LDS usage reasonable)
 * - Intermediate stored in LDS: 32 × 256 × 2B = 16KB
 * - Each workgroup produces a [32, N_tile] output tile
 * 
 * Memory savings vs separate kernels:
 * - Eliminates intermediate write: sorted_M × inter_dim × 2B (~35MB at 8k batch)
 * - Eliminates intermediate read: sorted_M × inter_dim × 2B (~35MB at 8k batch)
 * - Total: ~66MB saved per forward pass
 */
#include "hk_moe_kernel.cuh"

using namespace kittens;

// Fused kernel configuration
namespace fused_cfg {
    // Tile sizes optimized for LDS capacity and occupancy
    constexpr int M_TILE = 32;         // Rows per workgroup (matches AITER 32x256)
    constexpr int N_TILE = 128;        // Output columns per workgroup
    constexpr int K_STEP = 32;         // K dimension per iteration
    
    // Register tile dimensions
    constexpr int REG_M = 16;          // Register tile M (for MFMA 16x16x16)
    constexpr int REG_N = 16;          // Register tile N
    constexpr int DOT_SLICE = 16;      // MMA native K dimension
    
    // Thread organization: 4 warps = 256 threads
    constexpr int NUM_WARPS = 4;
    constexpr int NUM_THREADS = WARP_THREADS * NUM_WARPS;  // 64 * 4 = 256
    
    // DeepSeek R1 dimensions (for specialization)
    constexpr int MODEL_DIM = 7168;
    constexpr int INTER_DIM = 256;
    
    // Intermediate buffer in LDS: M_TILE × INTER_DIM × sizeof(bf16)
    constexpr int INTER_LDS_SIZE = M_TILE * INTER_DIM * 2;  // 32 × 256 × 2 = 16KB
    
    // XCD-aware scheduling
    constexpr int NUM_XCDS = 8;
    constexpr int WGM = 4;
}

// Shared tile types for fused kernel
using fused_input_tile = st_bf<fused_cfg::M_TILE, fused_cfg::K_STEP>;     // [32, 32] bf16
using fused_weight_tile = st_bf<fused_cfg::N_TILE, fused_cfg::K_STEP>;    // [128, 32] bf16 (N×K layout)
using fused_inter_tile = st_bf<fused_cfg::M_TILE, fused_cfg::INTER_DIM>;  // [32, 256] bf16 for intermediate

// XCD transform for better L2 locality
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
 * 
 * Computes: output[token] += Σ_expert weight[token] * silu(x @ W1_gate) * (x @ W1_up) @ W2
 * 
 * For each workgroup:
 * 1. Load M_TILE rows of input (gathered via sorted_ids)
 * 2. Stage 1: Compute gate-up projection, apply SiLU, store intermediate to LDS
 * 3. Stage 2: Compute down projection from LDS intermediate
 * 4. Atomic accumulate to output
 */
__global__ __launch_bounds__(fused_cfg::NUM_THREADS, 2)
void hk_moe_fused_fp8_kernel(
    const bf16* __restrict__ hidden_states,    // [num_tokens, model_dim]
    const fp8_t* __restrict__ w1_fp8,          // [num_experts, inter_dim*2, model_dim]
    const fp8_t* __restrict__ w2_fp8,          // [num_experts, model_dim, inter_dim]
    const float* __restrict__ w1_scale,        // [num_experts, num_scale_n_w1, num_scale_k_w1]
    const float* __restrict__ w2_scale,        // [num_experts, num_scale_n_w2, num_scale_k_w2]
    float* __restrict__ output_fp32,           // [num_tokens, model_dim]
    const int32_t* __restrict__ sorted_ids,    // [sorted_M] packed (token_id << 8 | slot)
    const int32_t* __restrict__ sorted_expert_ids,  // [num_m_blocks] expert per block
    const int32_t* __restrict__ num_valid_ids, // [1]
    const float* __restrict__ sorted_weights,  // [sorted_M] routing weights
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
    const int total_n_w1 = inter_dim * 2;  // Gate + Up weights
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    // Allocate LDS: input tile + weight tile + intermediate buffer
    fused_input_tile (&As) = al.allocate<fused_input_tile>();           // [32, 32] for input
    fused_weight_tile (&Bs) = al.allocate<fused_weight_tile>();         // [128, 32] for weights
    bf16 (&inter_lds)[M_TILE][INTER_DIM] = al.allocate<bf16, M_TILE, INTER_DIM>(); // [32, 256] intermediate
    int (&token_row_offsets)[M_TILE] = al.allocate<int, M_TILE>();
    
    // XCD-aware workgroup scheduling
    int wgid = blockIdx.y * gridDim.x + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    wgid = fused_xcd_transform(wgid, NUM_WGS);
    
    if (wgid >= NUM_WGS) return;
    
    // Compute M and N tile indices
    const int pid_m = wgid / num_n_blocks;
    const int pid_n = wgid % num_n_blocks;
    
    if (pid_m >= num_m_blocks || pid_n >= num_n_blocks) return;
    
    const int row_start = pid_m * M_TILE;
    const int col_start = pid_n * N_TILE;  // Output column
    
    if (row_start >= sorted_M_valid || col_start >= model_dim) return;
    
    // Get expert for this M tile
    const int tile_id = row_start / block_m_sorting;
    const int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    // Expert weight bases
    const fp8_t* w1_expert = w1_fp8 + expert_id * (size_t)total_n_w1 * model_dim;
    const fp8_t* w2_expert = w2_fp8 + expert_id * (size_t)model_dim * inter_dim;
    const float* w1_expert_scale = w1_scale + expert_id * (size_t)num_scale_n_w1 * num_scale_k_w1;
    const float* w2_expert_scale = w2_scale + expert_id * (size_t)num_scale_n_w2 * num_scale_k_w2;
    
    const int lane = threadIdx.x;
    const int warp_id = lane / WARP_THREADS;
    const int lane_in_warp = lane % WARP_THREADS;
    
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
    
    // ===========================================================================
    // STAGE 1: Gate-Up Projection + SiLU
    // Compute: inter[m, n] = silu(input[m] @ W1_gate[n]) * (input[m] @ W1_up[n])
    // ===========================================================================
    
    // Register accumulators for gate and up (need full inter_dim)
    // We'll process inter_dim in chunks of N_TILE=128
    constexpr int INTER_CHUNKS = INTER_DIM / N_TILE;  // 256/128 = 2
    
    for (int inter_chunk = 0; inter_chunk < INTER_CHUNKS; inter_chunk++) {
        const int inter_col_start = inter_chunk * N_TILE;
        
        // Register tiles for gate and up accumulators
        rt_fl<REG_M, REG_N, ducks::rt_layout::col> C_gate, C_up;
        zero(C_gate);
        zero(C_up);
        
        // K-loop for Stage 1
        const int num_k_tiles_s1 = (model_dim + K_STEP - 1) / K_STEP;
        
        for (int k_tile = 0; k_tile < num_k_tiles_s1; k_tile++) {
            const int k_start = k_tile * K_STEP;
            
            // Load input tile [M_TILE, K_STEP]
            uint32_t As_ptr = reinterpret_cast<uintptr_t>(&As.data[0]);
            constexpr int VEC_SIZE = 8;
            constexpr int VECS_PER_ROW = K_STEP / VEC_SIZE;
            constexpr int TOTAL_VECS_IN = M_TILE * VECS_PER_ROW;
            constexpr int VECS_PER_THREAD_IN = (TOTAL_VECS_IN + NUM_THREADS - 1) / NUM_THREADS;
            
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
            
            // Load gate weight tile [N_TILE, K_STEP] with FP8 dequant
            uint32_t Bs_ptr = reinterpret_cast<uintptr_t>(&Bs.data[0]);
            constexpr int TOTAL_VECS_W = N_TILE * K_STEP / VEC_SIZE;
            constexpr int VECS_PER_THREAD_W = (TOTAL_VECS_W + NUM_THREADS - 1) / NUM_THREADS;
            
            // Gate weights (first half of w1)
            int gate_col_global = inter_col_start;  // 0 to inter_dim
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
            
            // MFMA for gate
            rt_bf<REG_M, DOT_SLICE> a_tile;
            rt_bf<REG_N, DOT_SLICE> b_tile;
            
            #pragma unroll
            for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
                int warp_row = (warp_id / 2) * REG_M;
                int warp_col = (warp_id % 2) * (N_TILE / 2);
                
                load(a_tile, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row / REG_M, kk}));
                load(b_tile, subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col / REG_N, kk}));
                
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(0)");
                
                mma_ABt(C_gate, a_tile, b_tile, C_gate);
            }
            
            __syncthreads();
            
            // Load up weight tile (second half of w1: inter_dim to inter_dim*2)
            int up_col_global = inter_dim + inter_col_start;
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
            
            // MFMA for up
            #pragma unroll
            for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
                int warp_row = (warp_id / 2) * REG_M;
                int warp_col = (warp_id % 2) * (N_TILE / 2);
                
                load(a_tile, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row / REG_M, kk}));
                load(b_tile, subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col / REG_N, kk}));
                
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(0)");
                
                mma_ABt(C_up, a_tile, b_tile, C_up);
            }
            
            __syncthreads();
        }
        
        // Apply SiLU(gate) * up and store to intermediate LDS
        // Each warp writes its portion of the result
        // HipKittens packs 4 rows of output per tile, data[0].x, data[0].y, data[1].x, data[1].y
        const int lane_in_warp_loc = lane % WARP_THREADS;
        const int row_off = 4 * (lane_in_warp_loc / 16);  // 0 or 4 rows offset
        const int col_off = lane_in_warp_loc % 16;        // 0-15 column offset
        
        int warp_row_base = (warp_id / 2) * REG_M;
        int warp_col_base = inter_col_start + (warp_id % 2) * (N_TILE / 2);
        
        #pragma unroll
        for (int tile_row = 0; tile_row < C_gate.height; tile_row++) {
            #pragma unroll
            for (int tile_col = 0; tile_col < C_gate.width; tile_col++) {
                const auto& gate_tile = C_gate.tiles[tile_row][tile_col];
                const auto& up_tile = C_up.tiles[tile_row][tile_col];
                
                int base_row = warp_row_base + tile_row * 16 + row_off;
                int col = warp_col_base + tile_col * 16 + col_off;
                
                if (col < INTER_DIM) {
                    // Row 0 from data[0].x
                    if (base_row + 0 < M_TILE) {
                        float g = gate_tile.data[0].x;
                        float u = up_tile.data[0].x;
                        float silu = g / (1.0f + expf(-g));
                        inter_lds[base_row + 0][col] = __float2bfloat16(silu * u);
                    }
                    // Row 1 from data[0].y
                    if (base_row + 1 < M_TILE) {
                        float g = gate_tile.data[0].y;
                        float u = up_tile.data[0].y;
                        float silu = g / (1.0f + expf(-g));
                        inter_lds[base_row + 1][col] = __float2bfloat16(silu * u);
                    }
                    // Row 2 from data[1].x
                    if (base_row + 2 < M_TILE) {
                        float g = gate_tile.data[1].x;
                        float u = up_tile.data[1].x;
                        float silu = g / (1.0f + expf(-g));
                        inter_lds[base_row + 2][col] = __float2bfloat16(silu * u);
                    }
                    // Row 3 from data[1].y
                    if (base_row + 3 < M_TILE) {
                        float g = gate_tile.data[1].y;
                        float u = up_tile.data[1].y;
                        float silu = g / (1.0f + expf(-g));
                        inter_lds[base_row + 3][col] = __float2bfloat16(silu * u);
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // ===========================================================================
    // STAGE 2: Down Projection
    // Compute: output[m, n] = inter[m] @ W2[n]
    // ===========================================================================
    
    // Register accumulator for output
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> C_out;
    zero(C_out);
    
    // K-loop for Stage 2 (K = inter_dim = 256)
    const int num_k_tiles_s2 = (inter_dim + K_STEP - 1) / K_STEP;
    
    for (int k_tile = 0; k_tile < num_k_tiles_s2; k_tile++) {
        const int k_start = k_tile * K_STEP;
        
        // Load intermediate from LDS to As
        uint32_t As_ptr = reinterpret_cast<uintptr_t>(&As.data[0]);
        constexpr int VEC_SIZE = 8;
        constexpr int VECS_PER_ROW = K_STEP / VEC_SIZE;
        constexpr int TOTAL_VECS_IN = M_TILE * VECS_PER_ROW;
        constexpr int VECS_PER_THREAD_IN = (TOTAL_VECS_IN + NUM_THREADS - 1) / NUM_THREADS;
        
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD_IN; v++) {
            int vec_idx = lane * VECS_PER_THREAD_IN + v;
            if (vec_idx < TOTAL_VECS_IN) {
                int m = vec_idx / VECS_PER_ROW;
                int k = (vec_idx % VECS_PER_ROW) * VEC_SIZE;
                
                // Read from intermediate LDS
                float4 buf;
                bf16* src = &inter_lds[m][k_start + k];
                buf.x = __bfloat162float(src[0]);
                buf.y = __bfloat162float(src[1]);
                buf.z = __bfloat162float(src[2]);
                buf.w = __bfloat162float(src[3]);
                
                // Pack to bf16 and store to As
                __hip_bfloat162 lo = __float22bfloat162_rn(make_float2(buf.x, buf.y));
                __hip_bfloat162 hi = __float22bfloat162_rn(make_float2(buf.z, buf.w));
                float2 packed_lo, packed_hi;
                memcpy(&packed_lo.x, &lo, sizeof(float));
                memcpy(&packed_hi.x, &hi, sizeof(float));
                
                store_shared_vec(As.idx(As_ptr, {m, k}), packed_lo);
                store_shared_vec(As.idx(As_ptr, {m, k + 4}), packed_hi);
            }
        }
        
        // Load w2 weight tile [N_TILE, K_STEP] with FP8 dequant
        uint32_t Bs_ptr = reinterpret_cast<uintptr_t>(&Bs.data[0]);
        constexpr int TOTAL_VECS_W = N_TILE * K_STEP / VEC_SIZE;
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
                int col = col_start + n;
                int k_global = k_start + k;
                
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
        rt_bf<REG_N, DOT_SLICE> b_tile;
        
        #pragma unroll
        for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
            int warp_row = (warp_id / 2) * REG_M;
            int warp_col = (warp_id % 2) * (N_TILE / 2);
            
            load(a_tile, subtile_inplace<REG_M, DOT_SLICE>(As, {warp_row / REG_M, kk}));
            load(b_tile, subtile_inplace<REG_N, DOT_SLICE>(Bs, {warp_col / REG_N, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            
            mma_ABt(C_out, a_tile, b_tile, C_out);
        }
        
        __syncthreads();
    }
    
    // ===========================================================================
    // EPILOGUE: Weighted atomic scatter to output
    // ===========================================================================
    
    const int lane_in_warp_out = lane % WARP_THREADS;
    const int row_off_out = 4 * (lane_in_warp_out / 16);
    const int col_off_out = lane_in_warp_out % 16;
    
    int warp_row_base_out = (warp_id / 2) * REG_M;
    int warp_col_base_out = col_start + (warp_id % 2) * (N_TILE / 2);
    
    #pragma unroll
    for (int tile_row = 0; tile_row < C_out.height; tile_row++) {
        #pragma unroll
        for (int tile_col = 0; tile_col < C_out.width; tile_col++) {
            const auto& out_tile = C_out.tiles[tile_row][tile_col];
            
            int base_row = warp_row_base_out + tile_row * 16 + row_off_out;
            int col = warp_col_base_out + tile_col * 16 + col_off_out;
            
            if (col < model_dim) {
                // Process 4 rows per thread
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int m = base_row + r;
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
                                
                                atomicAdd(&output_fp32[token_id * model_dim + col], val * weight);
                            }
                        }
                    }
                }
            }
        }
    }
}


// ===========================================================================
// Dispatch function
// ===========================================================================

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
    
    // Compute grid dimensions
    const int num_m_blocks = (sorted_M + M_TILE - 1) / M_TILE;
    const int num_n_blocks = (model_dim + N_TILE - 1) / N_TILE;
    
    // Scale dimensions
    const int num_scale_n_w1 = (inter_dim * 2 + fp8_cfg::SCALE_BLOCK_N - 1) / fp8_cfg::SCALE_BLOCK_N;
    const int num_scale_k_w1 = (model_dim + fp8_cfg::SCALE_BLOCK_K - 1) / fp8_cfg::SCALE_BLOCK_K;
    const int num_scale_n_w2 = (model_dim + fp8_cfg::SCALE_BLOCK_N - 1) / fp8_cfg::SCALE_BLOCK_N;
    const int num_scale_k_w2 = (inter_dim + fp8_cfg::SCALE_BLOCK_K - 1) / fp8_cfg::SCALE_BLOCK_K;
    
    dim3 grid(num_n_blocks, num_m_blocks);
    dim3 block(NUM_THREADS);
    
    // LDS size: input tile + weight tile + intermediate + offsets
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

