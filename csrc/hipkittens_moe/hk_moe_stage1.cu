// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Stage 1: Gate-Up projection (G1U1) implemented as a tiled GEMM with a gathered A operand.
 *
 * -----------------------------------------------------------------------------------------------
 * ## 1) High-level computation (math)
 *
 * Inputs / outputs:
 *   - hidden_states[token, k]   ∈ bf16, shape [num_tokens, model_dim]
 *   - w1[expert, n, k]          ∈ bf16, shape [num_experts, 2*inter_dim, model_dim]
 *   - intermediate[row, n]      ∈ bf16, shape [sorted_M, 2*inter_dim]
 *
 * Each `row` corresponds to one routed token instance (token_id + topk slot) after MoE sorting.
 *
 * For each valid sorted row and output column:
 *
 *   token_id(row) = sorted_ids[row] & 0x00FFFFFF
 *   expert(row)   = sorted_expert_ids[ row_start / block_m_sorting ]
 *
 *   intermediate[row, n] = Σ_{k=0..model_dim-1} hidden_states[token_id(row), k] * w1[expert(row), n, k]
 *                         for n ∈ [0, 2*inter_dim)
 *
 * This is a GEMM over (M, N, K) = (sorted_M_valid, 2*inter_dim, model_dim).
 *
 * -----------------------------------------------------------------------------------------------
 * ## 2) Tiling (what a workgroup computes)
 *
 * Workgroup output tile:
 *   - BLOCK_SIZE = 128
 *   - C_tile is 128×128 over (M,N)
 *
 * K is iterated in tiles:
 *   - K_STEP = 32
 *   - Each iteration accumulates over K_STEP elements.
 *
 * Therefore (per K iteration):
 *   - A_shmem tile is [128, 32] bf16
 *   - B_shmem tile is [128, 32] bf16 (B is logically [N,K], but stored as [N,K_STEP] in LDS)
 *
 * -----------------------------------------------------------------------------------------------
 * ## 3) Warp mapping (how the 8 warps cover the tile)
 *
 * Workgroup has 512 threads = 8 warps × 64 lanes (HipKittens “8-wave” pattern).
 * Warps are arranged as a 2×4 grid:
 *   warp_row = warpid()/4  ∈ {0,1}          selects which 64-row half of the 128 M rows
 *   warp_col = warpid()%4  ∈ {0,1,2,3}      selects which 32-col quarter of the 128 N cols
 *
 * Each warp computes a 64×32 slab via two 32×32 fp32 accumulators:
 *   C_accum[0] covers rows [warp_row*64 + 0 .. +31], cols [warp_col*32 .. +31]
 *   C_accum[1] covers rows [warp_row*64 +32 .. +63], cols [warp_col*32 .. +31]
 *
 * -----------------------------------------------------------------------------------------------
 * ## 4) Validity / padding (correctness invariant)
 *
 * The MoE sorting pass produces padded arrays. We must never read/write past valid rows:
 *   sorted_M_valid := num_valid_ids[0]
 * Rows >= sorted_M_valid are padding and must act as zeros.
 *
 * -----------------------------------------------------------------------------------------------
 * ## 5) Cache + LDS sizing (why these choices matter)
 *
 * Per workgroup, per K tile:
 *   A bytes  ≈ 128 * 32 * 2  = 8192 bytes  (gathered; often low coalescing)
 *   B bytes  ≈ 128 * 32 * 2  = 8192 bytes  (contiguous; reuse is the main locality target)
 *
 * LDS footprint per workgroup (approx, ignoring swizzle padding):
 *   sizeof(st_bf<128,32>) ≈ 128*32*2 = 8192 bytes
 *   As  : 1 × 8192
 *   Bs0 : 1 × 8192   (ping)
 *   Bs1 : 1 × 8192   (pong)
 *   token_row_offsets[128] : 128 × 4 = 512
 * Total ≈ 24.5 KB (+ allocator alignment/padding).
 *
 * Occupancy intuition:
 *   Lower LDS per block makes it easier to fit 2 blocks/CU (if VGPR allows), which is why K_STEP=32
 *   is an attractive baseline versus K_STEP=64 (which doubles tile bytes).
 *
 * -----------------------------------------------------------------------------------------------
 * ## 6) Synchronization and waitcnt (why they exist)
 *
 * - __syncthreads():
 *   Ensure all threads finished writing LDS tiles before any warp reads subtiles for MFMA.
 *
 * - s_waitcnt lgkmcnt(0):
 *   Ensure LDS->VGPR reads issued by HipKittens `load(...)` complete before MFMA consumes a_tile/b_tile.
 *
 * - s_waitcnt vmcnt(0):
 *   Ensure VMEM reads (raw buffer loads) complete before committing prefetched values into LDS.
 *
 * Reference: https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr
 */

#include "hk_moe_kernel.cuh"

using namespace kittens;

// Configuration following HipKittens GEMM pattern.
// NOTE: historical experiments used K_STEP=64; current stable baseline uses K_STEP=32.
namespace s1_cfg {
    constexpr int BLOCK_SIZE = 128;   // Output tile size (M and N dimension of output tile)
    constexpr int K_STEP = 32;        // K dimension per iteration (stable baseline)
    constexpr int REG_BLOCK = BLOCK_SIZE / 4;  // 32 - register tile dimension per warp
    constexpr int DOT_SLICE = 16;     // MMA native dimension (16x16x16)
    
    constexpr int NUM_WARPS = 8;      // 8 warps = 512 threads for 128x128 tile
    constexpr int NUM_THREADS = WARP_THREADS * NUM_WARPS;  // 64 * 8 = 512
    
    // XCD-aware scheduling parameters
    // WGM controls chunking for chiplet_transform_chunked
    // TODO: Use template programming to tune WGM per batch size
    constexpr int WGM = 4;            // Workgroup grouping factor for L2 locality
}

// Shared tile type for input and weight.
// st_bf is stored in LDS with a swizzled layout; write via store_shared_vec(tile.idx(...)) to preserve swizzle.
using s1_st_tile = st_bf<s1_cfg::BLOCK_SIZE, s1_cfg::K_STEP>;  // [128, 32] bf16 tile in LDS

// Group for cooperative loading
using s1_group = group<s1_cfg::NUM_WARPS>;

// === AMD buffer-load helper (enables pipelined global->register prefetch) ===
// We use raw buffer loads so we can overlap the next K-tile weight fetch with MFMA on the current tile.
// The returned float4 is treated as raw 16B bits (8xbf16) and stored to LDS via store_shared_vec().
using i32x4_t = int __attribute__((ext_vector_type(4)));
__device__ __forceinline__ float4 raw_buffer_load_b128_to_float4(const __amdgpu_buffer_rsrc_t& rsrc,
                                                                 const int byte_offset)
{
    const i32x4_t v = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
    return __builtin_bit_cast(float4, v);
}


__global__ __launch_bounds__(s1_cfg::NUM_THREADS, 2)
void hk_moe_stage1_kernel_mma(
    const bf16* __restrict__ hidden_states,  // [num_tokens, model_dim]
    const bf16* __restrict__ w1,             // [num_experts, inter_dim*2, model_dim]
    bf16* __restrict__ intermediate,         // [sorted_M, inter_dim*2]
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const int32_t* __restrict__ num_valid_ids,  // [2]; num_valid_ids[0] = sorted_size (valid rows)
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
    const int sorted_M_valid = num_valid_ids[0];
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    // Allocate shared memory tiles
    s1_st_tile (&As) = al.allocate<s1_st_tile>();   // Input tile [BLOCK_SIZE, K_STEP]
    // Double-buffered weight tiles (Bs0/Bs1): enables overlap of global weight loads with MFMA.
    s1_st_tile (&Bs0) = al.allocate<s1_st_tile>();  // Weight tile for even k_tile
    s1_st_tile (&Bs1) = al.allocate<s1_st_tile>();  // Weight tile for odd  k_tile
    // Precompute per-row base offsets into `hidden_states` for this 128-row tile.
    // This avoids re-reading `sorted_ids[row]` and recomputing `token_id * model_dim` for every K tile.
    int (&token_row_offsets)[BLOCK_SIZE] = al.allocate<int, BLOCK_SIZE>();
    // Hoist shared-tile base pointers (reduces per-iteration address arithmetic).
    const uint32_t As_ptr  = reinterpret_cast<uintptr_t>(&As.data[0]);
    const uint32_t Bs0_ptr = reinterpret_cast<uintptr_t>(&Bs0.data[0]);
    const uint32_t Bs1_ptr = reinterpret_cast<uintptr_t>(&Bs1.data[0]);
    
    const int total_n = inter_dim * 2;
    
    // Register tiles for MMA - 32x16 bf16 tiles and 32x32 fp32 accumulators
    rt_bf<REG_BLOCK, DOT_SLICE> a_tile, b_tile;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];  // Two 32x32 accumulators
    zero(C_accum[0]);
    zero(C_accum[1]);
    
    // Expert-aware block scheduling
    // Key insight: tokens are sorted by expert, so consecutive M tiles share the same expert
    // Strategy: Process all N tiles for a given M tile before moving to next M tile
    // This keeps expert weights in L2 cache longer
    
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    
    // Apply XCD-aware transformation to distribute work across chiplets
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);
    
    if (wgid >= NUM_WGS) return;
    
    // Expert-aware N-first ordering: all N tiles for M=0, then all N tiles for M=1, etc.
    // Block 0: (M=0, N=0), Block 1: (M=0, N=1), ..., Block num_n-1: (M=0, N=num_n-1)
    // Block num_n: (M=1, N=0), ...
    int pid_m = wgid / num_n_blocks;
    int pid_n = wgid % num_n_blocks;
    
    if (pid_m >= num_m_blocks || pid_n >= num_n_blocks) {
        return;
    }
    
    const int row_start = pid_m * BLOCK_SIZE;
    const int col_start = pid_n * BLOCK_SIZE;
    
    if (row_start >= sorted_M_valid || col_start >= total_n) return;
    
    // Get expert for this block (using first row in tile)
    const int tile_id = row_start / block_m_sorting;
    const int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) return;

    // Hoist expert weight base pointer (reduces repeated address arithmetic in the load loop).
    const bf16* __restrict__ w1_expert = w1 + expert_id * total_n * model_dim;
    // Buffer resource for raw buffer loads (byte addressing).
    const __amdgpu_buffer_rsrc_t w1_rsrc =
        __builtin_amdgcn_make_buffer_rsrc(const_cast<bf16*>(w1_expert), 0, 0xffffffff, 0x00020000);
    
    // Warp mapping: 8 warps arranged in a 2x4 grid
    // Warps 0-3 are row 0, warps 4-7 are row 1
    const int warp_id = warpid();
    const int warp_row = warp_id / 4;  // 0 or 1
    const int warp_col = warp_id % 4;  // 0, 1, 2, or 3
    
    const int num_k_tiles = (model_dim + K_STEP - 1) / K_STEP;
    
    const int lane = threadIdx.x;
    constexpr int VEC_SIZE = 8;  // bf16 elements per float4

    // === Precompute token row base offsets (token_id * model_dim) into LDS ===
    // token_row_offsets[m] is either a valid base offset (in bf16 elements) or -1 for invalid/padded rows.
    if (lane < BLOCK_SIZE) {
        const int row = row_start + lane;
        int base = -1;
        if (row < sorted_M_valid) {
            const int packed_id = sorted_ids[row];
            const int token_id = packed_id & 0xFFFFFF;
            // NOTE: token_id should be valid for non-padded rows; keep a guard for safety.
            if (token_id >= 0 && token_id < num_tokens) base = token_id * model_dim;
        }
        token_row_offsets[lane] = base;
    }
    __syncthreads();

    // === Preload first weight K-tile (k_tile = 0) into Bs0 ===
    // For subsequent tiles, we prefetch the next weights into registers during MFMA and then commit to the
    // alternate Bs buffer at the end of the iteration.
    {
        constexpr int VEC_SIZE_W = 8;  // bf16 elements per 16B vector
        constexpr int TOTAL_VECS_W = (BLOCK_SIZE * K_STEP) / VEC_SIZE_W;
        constexpr int VECS_PER_THREAD_W = TOTAL_VECS_W / NUM_THREADS;

        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD_W; v++) {
            (void)v;
            const int n = (lane >> 2);              // lane/4
            const int k = (lane & 3) << 3;          // (lane%4)*8

            int col = col_start + n;
            float4 buf = {0.f, 0.f, 0.f, 0.f};
            if (col < total_n && (k + VEC_SIZE_W - 1) < model_dim) {
                const int elem_off = col * model_dim + k;
                const int byte_off = elem_off * (int)sizeof(bf16);
                buf = raw_buffer_load_b128_to_float4(w1_rsrc, byte_off);
            }

            store_shared_vec(Bs0.idx(Bs0_ptr, {n, k}), {buf.x, buf.y});
            store_shared_vec(Bs0.idx(Bs0_ptr, {n, k + 4}), {buf.z, buf.w});
        }
    }

    __syncthreads();

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * K_STEP;
        // Select current weight buffer.
        s1_st_tile& Bs_cur = (k_tile & 1) ? Bs1 : Bs0;
        const uint32_t Bs_cur_ptr = (k_tile & 1) ? Bs1_ptr : Bs0_ptr;
        
        // === Cooperative vectorized load input tile with gather via sorted_ids ===
        // For gather: we load K values per row with vectorization
        // Each thread handles some complete rows (to enable vec loads within a row)
        constexpr int ROWS_PER_BLOCK = BLOCK_SIZE;
        constexpr int VECS_PER_ROW = K_STEP / VEC_SIZE;  // 64/8 = 8 vectors per row
        constexpr int TOTAL_VECS = ROWS_PER_BLOCK * VECS_PER_ROW;  // 128 * 8 = 1024
        constexpr int VECS_PER_THREAD = TOTAL_VECS / NUM_THREADS;  // 1024/512 = 2
        
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            // IMPORTANT: st_bf tiles use a swizzled shared-memory layout. Writing via
            // `As[{m,k}] = ...` bypasses the expected vectorized LDS stores and can corrupt the tile.
            // Use HipKittens' `store_shared_vec` + `tile.idx(...)` path, same as `global_to_shared.cuh`.
            int vec_idx = lane * VECS_PER_THREAD + v;
            int m = vec_idx / VECS_PER_ROW;
            int k = (vec_idx % VECS_PER_ROW) * VEC_SIZE;

            // Each float4 is a raw 16B copy (8 bf16). We treat it as bits and store to LDS in two 8B chunks.
            float4 buf = {0.f, 0.f, 0.f, 0.f};
            const int base = token_row_offsets[m];
            if (base >= 0 && (k_start + k + VEC_SIZE - 1) < model_dim) {
                buf = load_global_vec4(reinterpret_cast<const float4*>(
                    &hidden_states[base + k_start + k]
                ));
            }

            store_shared_vec(As.idx(As_ptr, {m, k}), {buf.x, buf.y});
            store_shared_vec(As.idx(As_ptr, {m, k + 4}), {buf.z, buf.w});
        }

        // Barrier: ensure As is ready; Bs_cur is already resident (preloaded or committed by previous iter).
        __syncthreads();

        // === Prefetch next weight K-tile (w1 only) into registers while we MFMA on the current tile ===
        constexpr int TOTAL_VECS_W = (BLOCK_SIZE * K_STEP) / VEC_SIZE;
        constexpr int VECS_PER_THREAD_W = TOTAL_VECS_W / NUM_THREADS;
        float4 w1_prefetch[VECS_PER_THREAD_W];
        if (k_tile + 1 < num_k_tiles) {
            const int k_start_next = (k_tile + 1) * K_STEP;
            #pragma unroll
            for (int v = 0; v < VECS_PER_THREAD_W; v++) {
                (void)v;
                const int n = (lane >> 2);
                const int k = (lane & 3) << 3;
                const int col = col_start + n;
                float4 buf = {0.f, 0.f, 0.f, 0.f};
                if (col < total_n && (k_start_next + k + VEC_SIZE - 1) < model_dim) {
                    const int elem_off = col * model_dim + k_start_next + k;
                    const int byte_off = elem_off * (int)sizeof(bf16);
                    buf = raw_buffer_load_b128_to_float4(w1_rsrc, byte_off);
                }
                w1_prefetch[v] = buf;
            }
        }

        
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
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, kk}));
            // Load B subtile 
            load(b_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs_cur, {warp_col, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            
            // MMA: C_accum[0] += a_tile @ b_tile^T
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0], a_tile, b_tile, C_accum[0]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
            
            // Load A subtile for second 32 rows
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            
            // MMA: C_accum[1] += a_tile @ b_tile^T
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1], a_tile, b_tile, C_accum[1]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
        }
        
        __syncthreads();

        // Commit prefetched weights into the alternate Bs buffer for the next iteration.
        if (k_tile + 1 < num_k_tiles) {
            s1_st_tile& Bs_next = (k_tile & 1) ? Bs0 : Bs1;
            const uint32_t Bs_next_ptr = (k_tile & 1) ? Bs0_ptr : Bs1_ptr;

            // Ensure the prefetch loads are complete before writing to LDS.
            asm volatile("s_waitcnt vmcnt(0)");

            #pragma unroll
            for (int v = 0; v < VECS_PER_THREAD_W; v++) {
                (void)v;
                const int n = (lane >> 2);
                const int k = (lane & 3) << 3;

                const float4 buf = w1_prefetch[v];
                store_shared_vec(Bs_next.idx(Bs_next_ptr, {n, k}), {buf.x, buf.y});
                store_shared_vec(Bs_next.idx(Bs_next_ptr, {n, k + 4}), {buf.z, buf.w});
            }
        }
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
    
    const int out_row_base_0 = row_start + warp_row * 32;
    const int out_row_base_1 = row_start + (warp_row + 2) * 32;
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
                if (base_row + 0 < sorted_M_valid) intermediate[(base_row + 0) * total_n + col] = __float2bfloat16(tile.data[0].x);
                if (base_row + 1 < sorted_M_valid) intermediate[(base_row + 1) * total_n + col] = __float2bfloat16(tile.data[0].y);
                if (base_row + 2 < sorted_M_valid) intermediate[(base_row + 2) * total_n + col] = __float2bfloat16(tile.data[1].x);
                if (base_row + 3 < sorted_M_valid) intermediate[(base_row + 3) * total_n + col] = __float2bfloat16(tile.data[1].y);
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
                if (base_row + 0 < sorted_M_valid) intermediate[(base_row + 0) * total_n + col] = __float2bfloat16(tile.data[0].x);
                if (base_row + 1 < sorted_M_valid) intermediate[(base_row + 1) * total_n + col] = __float2bfloat16(tile.data[0].y);
                if (base_row + 2 < sorted_M_valid) intermediate[(base_row + 2) * total_n + col] = __float2bfloat16(tile.data[1].x);
                if (base_row + 3 < sorted_M_valid) intermediate[(base_row + 3) * total_n + col] = __float2bfloat16(tile.data[1].y);
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
    
    dim3 grid(num_n_blocks, num_m_blocks);
    dim3 block(NUM_THREADS);
    
    // Set dynamic shared memory size (match AITER's K_STEP=64)
    // Dynamic shared memory: As + 2*Bs (double-buffer) + token_row_offsets (+ a little padding).
    size_t smem_size = sizeof(s1_st_tile) * 3 + sizeof(int) * BLOCK_SIZE + 256;
    hipFuncSetAttribute((void*)hk_moe_stage1_kernel_mma, 
                        hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    hk_moe_stage1_kernel_mma<<<grid, block, smem_size, g.stream>>>(
        g.hidden_states.raw_ptr,
        g.w1.raw_ptr,
        g.intermediate.raw_ptr,
        g.sorted_ids,
        g.sorted_expert_ids,
        g.num_valid_ids,
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
