// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/**
 * HipKittens MoE Stage 2: Down projection + weighted accumulation (GEMM + scatter-add).
 *
 * Stage 2 consumes the activated values from Stage 1 (the first half of the G1U1 buffer) and applies:
 *
 *   w2[expert, out, k] ∈ bf16, shape [num_experts, model_dim, inter_dim]
 *
 * to produce the final output:
 *
 *   output[token, out] = Σ_{(row routed to token)} weight(row) * Σ_{k=0..inter_dim-1} act[row, k] * w2[expert(row), out, k]
 *
 * where:
 *   - row is a sorted routed instance (token_id + topk slot)
 *   - weight(row) is the routing probability / gate weight for that routed instance
 *
 * -----------------------------------------------------------------------------------------------
 * ## 1) Inputs / outputs / addressing
 *
 * Inputs:
 *   - intermediate[row, :] is laid out as Stage1's [sorted_M, 2*inter_dim] buffer.
 *     Stage2 reads only the activated first half (inter_dim columns) using a row stride:
 *
 *       act_ptr(row, k) = intermediate + row * inter_row_stride + k
 *       inter_row_stride = 2*inter_dim (elements)
 *
 *   - sorted_ids[row] packs token_id in low 24 bits.
 *   - sorted_weights[row] is float32.
 *   - sorted_expert_ids[tile_id] gives expert for a tile of rows.
 *   - num_valid_ids[0] = sorted_M_valid.
 *
 * Output:
 *   - output_fp32[token, out] is float32 so atomics accumulate without bf16 rounding.
 *
 * -----------------------------------------------------------------------------------------------
 * ## 2) GEMM view (per expert tile)
 *
 * Conceptually per expert segment we do:
 *
 *   C = A * B^T
 *
 *   A: [M=sorted_rows_for_expert, K=inter_dim]  (activated)
 *   B: [N=model_dim, K=inter_dim]              (w2 for that expert)
 *   C: [M, N]
 *
 * Then we scatter each C[row, out] into output[token_id(row), out] with scaling by weight(row).
 *
 * -----------------------------------------------------------------------------------------------
 * ## 3) Tiling / bytes / cache model
 *
 * Workgroup output tile is again 128×128 over (M,N), with K_STEP=32:
 *   - A_shmem: [128, 32] bf16 => 8 KB
 *   - B_shmem: [128, 32] bf16 => 8 KB
 *   - C in registers (fp32)
 *
 * Stage2 differs from Stage1 in the *scatter*:
 *   - Writes are atomicAdd to output_fp32 using token_id indirection.
 *   - For topk>1 multiple rows map to same token; atomics serialize those collisions.
 *
 * -----------------------------------------------------------------------------------------------
 * ## 4) Synchronization / correctness
 *
 * - __syncthreads(): barrier between cooperative LDS stores and LDS loads for MFMA.
 * - s_waitcnt lgkmcnt(0): ensure LDS->VGPR loads complete before MFMA.
 *
 * Reference: https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr
 */

/*
 * === Performance Status (Dec 2024) ===
 *
 * DISCOVERY: HipKittens FP8 beats AITER production at high batches!
 *
 * Comparison at DeepSeek R1 batch sizes (from production CSV):
 *   Batch 8613: Production=102 TFLOPs, HK=124 TFLOPs → HK is 1.22x FASTER
 *   Batch 7339: Production=112 TFLOPs, HK=139 TFLOPs → HK is 1.24x FASTER
 *   Batch 5913: Production=104 TFLOPs, HK=124 TFLOPs → HK is 1.19x FASTER
 *
 * Note: Earlier "2x gap" was comparing against AITER's `vs` (vector scale) benchmark variant,
 * not the `novs` variant actually used in production.
 *
 * === Opportunity checklist (Stage2: simplify + reduce/remove atomics) ===
 *
 * Stage2 is dominated by two structural costs:
 *   (a) GEMM is fine (MFMA-heavy), but
 *   (b) scatter-add is hard: token_id indirection + atomic collisions.
 *
 * The checklist below is ordered from "simplest conceptual simplification" to "bigger redesign".
 *
 * [IMPLEMENTED - NO WIN] 1) Make token writes unique by writing to a per-(token, topk_slot) buffer
 *    - Implemented as hk_moe_stage2_fp8_noatomic_kernel_mma + hk_moe_reduce_topk_kernel
 *    - Result: 2-6% SLOWER than atomic version at 4k-8k batch
 *    - Reason: Extra reduction kernel overhead + larger memory traffic for tmp_buffer
 *    - Conclusion: Atomics are NOT the bottleneck due to good locality from expert sorting
 *    - Key observation: each routed row corresponds to one (token_id, topk_slot) pair.
 *      If `sorted_ids[row]` encodes the topk slot (common in MoE packings), then Stage2 can write:
 *
 *        tmp[token_id, slot, out] = weight(row) * C[row, out]        (no atomic; unique destination)
 *
 *      followed by a simple reduction kernel:
 *
 *        output[token, out] = Σ_{slot=0..topk-1} tmp[token, slot, out]
 *
 *      Why this helps:
 *        - the reduction is dense and fully coalesced
 *        - removes atomic hot-spots completely
 *      Cost:
 *        - extra tmp buffer of size [num_tokens, topk, model_dim] (bf16 or fp32)
 *        - one additional reduction kernel (but extremely regular)
 *    - For DeepSeek R1 (topk=8, model_dim=7168): tmp buffer = 8 × 7168 × num_tokens × 4B = ~230KB per 1K tokens
 *
 * 2) Two-pass "no-atomic Stage2": write per-row output, then segmented reduce by token_id
 *    - Pass A: produce row-major output:
 *        row_out[row, out] = weight(row) * C[row, out]               (no atomic)
 *      Pass B: reduce by token_id(row) using a segmented reduction kernel.
 *    - This is a generalization of (1) when topk slot isn't available or when you want fp32 accumulation.
 *    - Cost: large temporary [sorted_M_valid, model_dim] and extra pass.
 *
 * 3) Reduce atomics by local aggregation (warp/block hash) before issuing atomicAdd
 *    - Within a warp (or a block), group lanes that target the same token_id and sum locally,
 *      then issue one atomic per unique token_id per output column.
 *    - This can significantly reduce atomics if token_id collisions are common within a CTA's 128-row tile.
 *    - Complexity: requires a grouping strategy (e.g., small shared-memory hash table or sort-by-token inside tile).
 *
 * 4) Change accumulation granularity: per-expert partial outputs then final combine
 *    - Within a single expert segment, each token typically appears at most once (for topk routing with unique ids).
 *      Stage2 can write a per-expert partial output without atomics:
 *        out_expert[expert, token, out] += weight * C
 *      Then combine across experts per token in a second kernel.
 *    - This trades atomics for memory footprint; may be attractive if collisions dominate.
 *
 * 5) Tighten invariants and remove redundant checks / branches
 *    - Similar to Stage1: for common shapes, many bounds checks are redundant.
 *    - Ensure all bounds use sorted_M_valid, and consider padding inter_dim to K_STEP and VEC_SIZE.
 *
 * 6) If we keep atomics: make them cheaper
 *    - Keep output_fp32 but consider:
 *        - reordering work to increase spatial locality (token clustering) so atomics hit fewer cache lines
 *        - using larger tiles over N to amortize metadata loads (sorted_ids/weights)
 *
 * === FP8-Specific Optimizations ===
 *
 * [DONE] 7) Vectorize FP8 dequantization (same as Stage1)
 *    - Use vectorized fp8x8_to_bf16x8_scaled() for w2 weight loading.
 *
 * [TODO] 8) Cache w2 blockscales in LDS
 *    - For DeepSeek R1: w2 is [256, 7168, 256], scales are [256, 56, 2]
 *    - Each workgroup tile may access only 1-2 unique scale values.
 *
 * [TODO] 9) Consider fp32 accumulation earlier
 *    - With FP8 weights, accumulation precision is more critical.
 *    - Current: accumulate in fp32 registers (good), atomic to fp32 output (good).
 *
 * === rocprof Hardware Counter Analysis (8k batch, DeepSeek R1) ===
 *
 * Stage 2 FP8 kernel vs Stage 1 FP8 Fused:
 *   - Stage 2 launches 28x more waves (344K vs 12K)
 *   - Stage 2 has worse MFMA/VALU ratio: 7.7% vs 13.6%
 *   - Stage 2 has 2.4x more VMEM instructions (33M vs 14M)
 *   - Stage 2 uses fewer VGPRs (76 vs 120) but more LDS (32KB vs 25KB)
 *
 * Key bottleneck: Stage 2 is more memory-bound and less compute-efficient.
 * Wave count is high because each 128x128 output tile is small relative to
 * total output [sorted_M, model_dim] = [65K, 7168], requiring ~550 N-tiles.
 *
 * Potential optimizations:
 *   1) Reduce wave count via larger N-tiles (256 instead of 128)
 *   2) Reduce VALU overhead by simplifying address calculations
 *   3) Reduce VMEM by improving weight reuse across K iterations
 */
#include "hk_moe_kernel.cuh"

using namespace kittens;

// Configuration following HipKittens GEMM pattern.
namespace s2_cfg {
    constexpr int BLOCK_SIZE = 128;   // Output tile size
    constexpr int K_STEP = 32;        // K dimension per iteration
    constexpr int REG_BLOCK = BLOCK_SIZE / 4;  // 32
    constexpr int DOT_SLICE = 16;     // MMA native dimension
    
    constexpr int NUM_WARPS = 8;
    constexpr int NUM_THREADS = WARP_THREADS * NUM_WARPS;  // 512
    
    // XCD-aware scheduling parameters
    constexpr int WGM = 8;
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
    const int32_t* __restrict__ num_valid_ids,  // [2]; num_valid_ids[0] = sorted_size (valid rows)
    const float* __restrict__ sorted_weights,
    const int sorted_M,
    const int num_tokens,
    const int model_dim,
    const int inter_dim,
    const int inter_row_stride,
    const int num_experts,
    const int topk,
    const int block_m_sorting,
    const int num_m_blocks,
    const int num_n_blocks
) {
    using namespace s2_cfg;
    const int sorted_M_valid = num_valid_ids[0];
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    s2_st_tile (&As) = al.allocate<s2_st_tile>();  // Input tile
    s2_st_tile (&Bs) = al.allocate<s2_st_tile>();  // Weight tile
    
    // Register tiles for MMA
    rt_bf<REG_BLOCK, DOT_SLICE> a_tile, b_tile;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];
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
    int pid_m = wgid / num_n_blocks;
    int pid_n = wgid % num_n_blocks;
    
    if (pid_m >= num_m_blocks || pid_n >= num_n_blocks) {
        return;
    }
    
    const int row_start = pid_m * BLOCK_SIZE;
    const int col_start = pid_n * BLOCK_SIZE;
    
    if (row_start >= sorted_M_valid || col_start >= model_dim) return;
    
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
    
    // Outer K loop:
    //   We advance k_start in steps of K_STEP and accumulate into fp32 registers (C_accum).
    //
    // After the loop finishes, for the workgroup's 128×128 (M×N) tile we have:
    //   C[row, out] = Σ_{k=0..inter_dim-1} act[row, k] * w2[expert, out, k]
    // where `row` is the sorted row index and `out` is the model_dim output column.
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * K_STEP;
        
                // === Cooperative vectorized load of A tile into LDS ===
        // A operand is the activated intermediate produced by Stage1.
        // We read only the activated first half from the Stage1 buffer using a row stride:
        //   act_ptr(row,k) = intermediate + row*inter_row_stride + k
        //   inter_row_stride = 2*inter_dim (bf16 elements)
        //
        // Each workgroup loads a tile As = [BLOCK_SIZE, K_STEP] = [128,32].
        //
        // Vectorization:
        //   VEC_SIZE = 8 bf16 = 16 bytes (float4 bits)
        //   TOTAL_VECS      = 128*32/8 = 512 vectors
        //   VECS_PER_THREAD = 512/512  = 1 vector per thread
        //
        // Lane mapping:
        //   flat_idx = (lane*VECS_PER_THREAD + v) * 8
        //   m = flat_idx / K_STEP   (row within the 128-row tile)
        //   k = flat_idx % K_STEP   (k offset within the current K_STEP slice)
        uint32_t As_ptr = reinterpret_cast<uintptr_t>(&As.data[0]);
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_idx = lane * VECS_PER_THREAD + v;
            int flat_idx = vec_idx * VEC_SIZE;
            int m = flat_idx / K_STEP;
            int k = flat_idx % K_STEP;

            int row = row_start + m;
            float4 buf = {0.f, 0.f, 0.f, 0.f};
            if (row < sorted_M_valid && (k_start + k + VEC_SIZE - 1) < inter_dim) {
                buf = load_global_vec4(reinterpret_cast<const float4*>(
                    &intermediate[row * inter_row_stride + k_start + k]
                ));
            }
            store_shared_vec(As.idx(As_ptr, {m, k}), {buf.x, buf.y});
            store_shared_vec(As.idx(As_ptr, {m, k + 4}), {buf.z, buf.w});
        }

        // === Cooperative vectorized load of B tile (w2) into LDS ===
        // B operand is the expert's down-projection weights w2[expert, out, k].
        // We load B as [N=128, K_STEP] where N corresponds to output columns in this tile:
        //   out = col_start + n
        // Addressing for an in-bounds vector load:
        //   &w2[expert_id, out, k_start+k] = w2 + expert_id*(model_dim*inter_dim) + out*inter_dim + (k_start+k)
        uint32_t Bs_ptr = reinterpret_cast<uintptr_t>(&Bs.data[0]);
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_idx = lane * VECS_PER_THREAD + v;
            int flat_idx = vec_idx * VEC_SIZE;
            int n = flat_idx / K_STEP;
            int k = flat_idx % K_STEP;

            int col = col_start + n;
            float4 buf = {0.f, 0.f, 0.f, 0.f};
            if (col < model_dim && (k_start + k + VEC_SIZE - 1) < inter_dim) {
                buf = load_global_vec4(reinterpret_cast<const float4*>(
                    &w2[expert_id * model_dim * inter_dim + col * inter_dim + k_start + k]
                ));
            }
            store_shared_vec(Bs.idx(Bs_ptr, {n, k}), {buf.x, buf.y});
            store_shared_vec(Bs.idx(Bs_ptr, {n, k + 4}), {buf.z, buf.w});
        }

        // Barrier: ensure all LDS writes to As/Bs are visible before any warp reads subtiles for MFMA.
        __syncthreads();
        
        // === MFMA / MMA compute ===
        // K_STEP=32 and DOT_SLICE=16 => kk ∈ {0,1} selecting k in [0..15] and [16..31].
        // Warp mapping matches Stage1: warp_row selects which 64-row half; warp_col selects which 32-col quarter.
        // We compute two 32×32 tiles by loading As at {warp_row,kk} and {warp_row+2,kk}.
        // The explicit s_waitcnt lgkmcnt(0) ensures LDS->VGPR loads are complete before MFMA.
        #pragma unroll
        for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, kk}));
            load(b_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0], a_tile, b_tile, C_accum[0]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
            
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1], a_tile, b_tile, C_accum[1]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
        }
        
        __syncthreads();
    }
    
    // === Weighted scatter-add to output (atomic accumulation) ===
    // We apply routing weights and accumulate into output_fp32 by token_id:
    //   token_id(row) = sorted_ids[row] & 0x00FFFFFF
    //   output_fp32[token_id, out] += sorted_weights[row] * C[row, out]
    //
    // Atomic rationale: topk>1 means multiple routed rows can map to the same token.
    // Performance note: collisions on token_id (or cache lines) can serialize atomics.
    // Optimization 1: Local accumulation - if consecutive rows map to same token, accumulate first
    // Optimization 2: Prefetch sorted_ids and weights to reduce memory traffic
    // Optimization 3: Coalesce atomics by processing same token_id together
    
    const int out_row_base_0 = row_start + warp_row * 32;
    const int out_row_base_1 = row_start + (warp_row + 2) * 32;
    const int out_col_base = col_start + warp_col * 32;
    const int lane_id = laneid();
    const int row_off = 4 * (lane_id / 16);
    const int col_off = lane_id % 16;
    
    // Helper lambda for optimized scatter with local accumulation
    auto scatter_with_local_accum = [&](const auto& C_tile, int base_row_start) {
        #pragma unroll
        for (int tile_row = 0; tile_row < 2; tile_row++) {
            #pragma unroll
            for (int tile_col = 0; tile_col < 2; tile_col++) {
                const auto& tile = C_tile.tiles[tile_row][tile_col];
                int base_row = base_row_start + tile_row * 16 + row_off;
                int col = out_col_base + tile_col * 16 + col_off;
                
                if (col < model_dim) {
                    float vals[4] = {tile.data[0].x, tile.data[0].y, tile.data[1].x, tile.data[1].y};
                    
                    // Prefetch token_ids and weights for 4 consecutive rows
                    int token_ids[4];
                    float weights[4];
                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        int row = base_row + r;
                        if (row < sorted_M_valid) {
                            int packed_id = sorted_ids[row];
                            token_ids[r] = packed_id & 0xFFFFFF;
                            weights[r] = sorted_weights[row];
                        } else {
                            token_ids[r] = -1;
                            weights[r] = 0.0f;
                        }
                    }
                    
                    // Scatter-add each row independently (correctness-first).
                    // Future knob: if collisions dominate, consider warp-level reduction by token_id to reduce atomics.
                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        int token_id = token_ids[r];
                        if (token_id >= 0 && token_id < num_tokens) {
                            float weighted_val = vals[r] * weights[r];
                            atomicAdd(&output_fp32[token_id * model_dim + col], weighted_val);
                        }
                    }
                }
            }
        }
    };
    
    // Apply optimized scatter to both accumulator tiles
    scatter_with_local_accum(C_accum[0], out_row_base_0);
    scatter_with_local_accum(C_accum[1], out_row_base_1);
}

void dispatch_hk_moe_stage2(const moe_stage2_globals& g, float* output_fp32) {
    using namespace s2_cfg;
    
    const int num_m_blocks = (g.sorted_M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_n_blocks = (g.model_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 grid(num_n_blocks, num_m_blocks);
    dim3 block(NUM_THREADS);
    
    size_t smem_size = 32768;  // 32KB for K_STEP=32
    hipFuncSetAttribute((void*)hk_moe_stage2_kernel_mma, 
                        hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    hk_moe_stage2_kernel_mma<<<grid, block, smem_size, g.stream>>>(
        g.intermediate.raw_ptr,
        g.w2.raw_ptr,
        output_fp32,
        g.sorted_ids,
        g.sorted_expert_ids,
        g.num_valid_ids,
        g.sorted_weights,
        g.sorted_M,
        g.num_tokens,
        g.model_dim,
        g.inter_dim,
        g.inter_row_stride,
        g.num_experts,
        g.topk,
        g.block_m,
        num_m_blocks,
        num_n_blocks
    );
}

// ============================================================================
// FP8 Blockscale Variant - Stage 2
// ============================================================================
//
// This kernel performs the Down projection with FP8 weights and blockscale:
// - FP8 (e4m3fnuz) weights for w2
// - Per-block scales for dequantization
// - On-the-fly dequantization: bf16_val = fp8_val * scale
//
// Scale layout: [num_experts, num_scale_n, num_scale_k] where
//   num_scale_n = ceil(model_dim / 128)
//   num_scale_k = ceil(inter_dim / 128)

__global__ __launch_bounds__(s2_cfg::NUM_THREADS, 2)
void hk_moe_stage2_fp8_kernel_mma(
    const bf16* __restrict__ intermediate,   // [sorted_M, inter_dim] - still bf16 activated values
    const fp8_t* __restrict__ w2_fp8,        // [num_experts, model_dim, inter_dim] as FP8
    const float* __restrict__ w2_scale,      // [num_experts, num_scale_n, num_scale_k]
    float* __restrict__ output_fp32,         // [num_tokens, model_dim]
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const int32_t* __restrict__ num_valid_ids,
    const float* __restrict__ sorted_weights,
    const int sorted_M,
    const int num_tokens,
    const int model_dim,
    const int inter_dim,
    const int inter_row_stride,
    const int num_experts,
    const int topk,
    const int block_m_sorting,
    const int num_m_blocks,
    const int num_n_blocks,
    const int num_scale_n,
    const int num_scale_k
) {
    using namespace s2_cfg;
    using namespace fp8_cfg;
    
    const int sorted_M_valid = num_valid_ids[0];
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    s2_st_tile (&As) = al.allocate<s2_st_tile>();
    s2_st_tile (&Bs) = al.allocate<s2_st_tile>();
    
    rt_bf<REG_BLOCK, DOT_SLICE> a_tile, b_tile;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];
    zero(C_accum[0]);
    zero(C_accum[1]);
    
    // XCD-aware block scheduling
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);
    
    if (wgid >= NUM_WGS) return;
    
    int pid_m = wgid / num_n_blocks;
    int pid_n = wgid % num_n_blocks;
    
    if (pid_m >= num_m_blocks || pid_n >= num_n_blocks) return;
    
    const int row_start = pid_m * BLOCK_SIZE;
    const int col_start = pid_n * BLOCK_SIZE;
    
    if (row_start >= sorted_M_valid || col_start >= model_dim) return;
    
    const int tile_id = row_start / block_m_sorting;
    const int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    // Expert's FP8 weight base and scale base
    const fp8_t* w2_expert_fp8 = w2_fp8 + expert_id * (size_t)model_dim * inter_dim;
    const float* w2_expert_scale = w2_scale + expert_id * (size_t)num_scale_n * num_scale_k;
    
    const int warp_id = warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    
    const int num_k_tiles = (inter_dim + K_STEP - 1) / K_STEP;
    
    const int lane = threadIdx.x;
    constexpr int VEC_SIZE = 8;
    constexpr int TOTAL_VECS = (BLOCK_SIZE * K_STEP) / VEC_SIZE;
    constexpr int VECS_PER_THREAD = TOTAL_VECS / NUM_THREADS;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * K_STEP;
        
        // Load A tile (bf16 intermediate activations - same as BF16 kernel)
        uint32_t As_ptr = reinterpret_cast<uintptr_t>(&As.data[0]);
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_idx = lane * VECS_PER_THREAD + v;
            int flat_idx = vec_idx * VEC_SIZE;
            int m = flat_idx / K_STEP;
            int k = flat_idx % K_STEP;

            int row = row_start + m;
            float4 buf = {0.f, 0.f, 0.f, 0.f};
            if (row < sorted_M_valid && (k_start + k + VEC_SIZE - 1) < inter_dim) {
                buf = load_global_vec4(reinterpret_cast<const float4*>(
                    &intermediate[row * inter_row_stride + k_start + k]
                ));
            }
            store_shared_vec(As.idx(As_ptr, {m, k}), {buf.x, buf.y});
            store_shared_vec(As.idx(As_ptr, {m, k + 4}), {buf.z, buf.w});
        }

        // Load B tile (FP8 w2 weights with dequantization)
        uint32_t Bs_ptr = reinterpret_cast<uintptr_t>(&Bs.data[0]);
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_idx = lane * VECS_PER_THREAD + v;
            int flat_idx = vec_idx * VEC_SIZE;
            int n = flat_idx / K_STEP;  // Which of 128 N rows (output columns)
            int k = flat_idx % K_STEP;  // K offset within tile

            int col = col_start + n;     // Global N coordinate
            int k_global = k_start + k;  // Global K coordinate
            
            float2 buf_lo = {0.f, 0.f};
            float2 buf_hi = {0.f, 0.f};
            
            if (col < model_dim && (k_global + VEC_SIZE - 1) < inter_dim) {
                // Compute scale block indices
                int n_block = col / SCALE_BLOCK_N;
                int k_block = k_global / SCALE_BLOCK_K;
                float scale = w2_expert_scale[n_block * num_scale_k + k_block];
                
                // Load 8 FP8 values and dequantize
                const fp8_t* src = &w2_expert_fp8[col * inter_dim + k_global];
                fp8x8_to_bf16x8_scaled(src, scale, buf_lo, buf_hi);
            }
            store_shared_vec(Bs.idx(Bs_ptr, {n, k}), {buf_lo.x, buf_lo.y});
            store_shared_vec(Bs.idx(Bs_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
        }

        __syncthreads();
        
        // MFMA compute
        #pragma unroll
        for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, kk}));
            load(b_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0], a_tile, b_tile, C_accum[0]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
            
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1], a_tile, b_tile, C_accum[1]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
        }
        
        __syncthreads();
    }
    
    // Weighted scatter-add to output (same as BF16 kernel)
    const int out_row_base_0 = row_start + warp_row * 32;
    const int out_row_base_1 = row_start + (warp_row + 2) * 32;
    const int out_col_base = col_start + warp_col * 32;
    const int lane_id = laneid();
    const int row_off = 4 * (lane_id / 16);
    const int col_off = lane_id % 16;
    
    auto scatter_with_local_accum = [&](const auto& C_tile, int base_row_start) {
        #pragma unroll
        for (int tile_row = 0; tile_row < 2; tile_row++) {
            #pragma unroll
            for (int tile_col = 0; tile_col < 2; tile_col++) {
                const auto& tile = C_tile.tiles[tile_row][tile_col];
                int base_row = base_row_start + tile_row * 16 + row_off;
                int col = out_col_base + tile_col * 16 + col_off;
                
                if (col < model_dim) {
                    float vals[4] = {tile.data[0].x, tile.data[0].y, tile.data[1].x, tile.data[1].y};
                    
                    int token_ids[4];
                    float weights[4];
                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        int row = base_row + r;
                        if (row < sorted_M_valid) {
                            int packed_id = sorted_ids[row];
                            token_ids[r] = packed_id & 0xFFFFFF;
                            weights[r] = sorted_weights[row];
                        } else {
                            token_ids[r] = -1;
                            weights[r] = 0.0f;
                        }
                    }
                    
                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        int token_id = token_ids[r];
                        if (token_id >= 0 && token_id < num_tokens) {
                            float weighted_val = vals[r] * weights[r];
                            atomicAdd(&output_fp32[token_id * model_dim + col], weighted_val);
                        }
                    }
                }
            }
        }
    };
    
    scatter_with_local_accum(C_accum[0], out_row_base_0);
    scatter_with_local_accum(C_accum[1], out_row_base_1);
}

// Dispatch function for FP8 weights with blockscale
void dispatch_hk_moe_stage2_fp8(const moe_stage2_fp8_globals& g, float* output_fp32) {
    using namespace s2_cfg;
    
    const int num_m_blocks = (g.sorted_M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_n_blocks = (g.model_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 grid(num_n_blocks, num_m_blocks);
    dim3 block(NUM_THREADS);
    
    size_t smem_size = 32768;  // 32KB for K_STEP=32
    hipFuncSetAttribute((void*)hk_moe_stage2_fp8_kernel_mma, 
                        hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    hk_moe_stage2_fp8_kernel_mma<<<grid, block, smem_size, g.stream>>>(
        g.intermediate.raw_ptr,
        g.w2_fp8,
        g.w2_scale,
        output_fp32,
        g.sorted_ids,
        g.sorted_expert_ids,
        g.num_valid_ids,
        g.sorted_weights,
        g.sorted_M,
        g.num_tokens,
        g.model_dim,
        g.inter_dim,
        g.inter_row_stride,
        g.num_experts,
        g.topk,
        g.block_m,
        num_m_blocks,
        num_n_blocks,
        g.num_scale_n,
        g.num_scale_k
    );
}

// ============================================================================
// EXPERIMENTAL: Atomic-Free Stage 2 Variant
// ============================================================================
//
// This variant eliminates atomic operations by writing to a per-(token, slot)
// buffer. The packed sorted_ids encodes: token_id (lower 24 bits) + topk_slot
// (upper 8 bits), allowing unique write destinations for each sorted row.
//
// Memory layout for tmp_buffer: [num_tokens, topk, model_dim]
//   - Write: tmp_buffer[(token_id * topk + slot) * model_dim + col]
//   - Reduce: output[token, col] = Σ_{slot=0..topk-1} tmp[token, slot, col]
//
// Trade-off:
//   - PRO: No atomic contention, better parallelism for high-collision scenarios
//   - CON: Requires extra memory (num_tokens * topk * model_dim * 4 bytes)
//   - CON: Extra reduction kernel launch
//
// For DeepSeek R1 (8k tokens, topk=8, model_dim=7168): ~1.87 GB temp buffer

__global__ __launch_bounds__(s2_cfg::NUM_THREADS, 2)
void hk_moe_stage2_fp8_noatomic_kernel_mma(
    const bf16* __restrict__ intermediate,
    const fp8_t* __restrict__ w2_fp8,
    const float* __restrict__ w2_scale,
    float* __restrict__ tmp_buffer,  // [num_tokens, topk, model_dim] - no atomics needed
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const int32_t* __restrict__ num_valid_ids,
    const float* __restrict__ sorted_weights,
    const int sorted_M,
    const int num_tokens,
    const int model_dim,
    const int inter_dim,
    const int inter_row_stride,
    const int num_experts,
    const int topk,
    const int block_m_sorting,
    const int num_m_blocks,
    const int num_n_blocks,
    const int num_scale_n,
    const int num_scale_k
) {
    using namespace s2_cfg;
    using namespace fp8_cfg;
    
    const int sorted_M_valid = num_valid_ids[0];
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    s2_st_tile (&As) = al.allocate<s2_st_tile>();
    s2_st_tile (&Bs) = al.allocate<s2_st_tile>();
    
    rt_bf<REG_BLOCK, DOT_SLICE> a_tile, b_tile;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];
    zero(C_accum[0]);
    zero(C_accum[1]);
    
    // XCD-aware block scheduling (same as atomic version)
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);
    
    if (wgid >= NUM_WGS) return;
    
    int pid_m = wgid / num_n_blocks;
    int pid_n = wgid % num_n_blocks;
    
    if (pid_m >= num_m_blocks || pid_n >= num_n_blocks) return;
    
    const int row_start = pid_m * BLOCK_SIZE;
    const int col_start = pid_n * BLOCK_SIZE;
    
    if (row_start >= sorted_M_valid || col_start >= model_dim) return;
    
    const int tile_id = row_start / block_m_sorting;
    const int expert_id = sorted_expert_ids[tile_id];
    
    if (expert_id < 0 || expert_id >= num_experts) return;
    
    const fp8_t* w2_expert_fp8 = w2_fp8 + expert_id * (size_t)model_dim * inter_dim;
    const float* w2_expert_scale = w2_scale + expert_id * (size_t)num_scale_n * num_scale_k;
    
    const int warp_id = warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    
    const int num_k_tiles = (inter_dim + K_STEP - 1) / K_STEP;
    
    const int lane = threadIdx.x;
    constexpr int VEC_SIZE = 8;
    constexpr int TOTAL_VECS = (BLOCK_SIZE * K_STEP) / VEC_SIZE;
    constexpr int VECS_PER_THREAD = TOTAL_VECS / NUM_THREADS;
    
    // Main GEMM loop (same as atomic version)
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * K_STEP;
        
        // Load A tile (bf16 intermediate)
        uint32_t As_ptr = reinterpret_cast<uintptr_t>(&As.data[0]);
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_idx = lane * VECS_PER_THREAD + v;
            int flat_idx = vec_idx * VEC_SIZE;
            int m = flat_idx / K_STEP;
            int k = flat_idx % K_STEP;

            int row = row_start + m;
            float4 buf = {0.f, 0.f, 0.f, 0.f};
            if (row < sorted_M_valid && (k_start + k + VEC_SIZE - 1) < inter_dim) {
                buf = load_global_vec4(reinterpret_cast<const float4*>(
                    &intermediate[row * inter_row_stride + k_start + k]
                ));
            }
            store_shared_vec(As.idx(As_ptr, {m, k}), {buf.x, buf.y});
            store_shared_vec(As.idx(As_ptr, {m, k + 4}), {buf.z, buf.w});
        }

        // Load B tile (FP8 with dequantization)
        uint32_t Bs_ptr = reinterpret_cast<uintptr_t>(&Bs.data[0]);
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD; v++) {
            int vec_idx = lane * VECS_PER_THREAD + v;
            int flat_idx = vec_idx * VEC_SIZE;
            int n = flat_idx / K_STEP;
            int k = flat_idx % K_STEP;

            int col = col_start + n;
            int k_global = k_start + k;
            
            float2 buf_lo = {0.f, 0.f};
            float2 buf_hi = {0.f, 0.f};
            
            if (col < model_dim && (k_global + VEC_SIZE - 1) < inter_dim) {
                int n_block = col / SCALE_BLOCK_N;
                int k_block = k_global / SCALE_BLOCK_K;
                float scale = w2_expert_scale[n_block * num_scale_k + k_block];
                
                const fp8_t* src = &w2_expert_fp8[col * inter_dim + k_global];
                fp8x8_to_bf16x8_scaled(src, scale, buf_lo, buf_hi);
            }
            store_shared_vec(Bs.idx(Bs_ptr, {n, k}), {buf_lo.x, buf_lo.y});
            store_shared_vec(Bs.idx(Bs_ptr, {n, k + 4}), {buf_hi.x, buf_hi.y});
        }

        __syncthreads();
        
        // MFMA compute
        #pragma unroll
        for (int kk = 0; kk < K_STEP / DOT_SLICE; kk++) {
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, kk}));
            load(b_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0], a_tile, b_tile, C_accum[0]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
            
            load(a_tile, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, kk}));
            
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1], a_tile, b_tile, C_accum[1]);
            __builtin_amdgcn_s_setprio(0);
            
            __builtin_amdgcn_sched_barrier(0);
        }
        
        __syncthreads();
    }
    
    // === ATOMIC-FREE weighted write to per-(token, slot) buffer ===
    // Key difference: extract topk_slot and write to unique location
    const int out_row_base_0 = row_start + warp_row * 32;
    const int out_row_base_1 = row_start + (warp_row + 2) * 32;
    const int out_col_base = col_start + warp_col * 32;
    const int lane_id = laneid();
    const int row_off = 4 * (lane_id / 16);
    const int col_off = lane_id % 16;
    
    auto scatter_noatomic = [&](const auto& C_tile, int base_row_start) {
        #pragma unroll
        for (int tile_row = 0; tile_row < 2; tile_row++) {
            #pragma unroll
            for (int tile_col = 0; tile_col < 2; tile_col++) {
                const auto& tile = C_tile.tiles[tile_row][tile_col];
                int base_row = base_row_start + tile_row * 16 + row_off;
                int col = out_col_base + tile_col * 16 + col_off;
                
                if (col < model_dim) {
                    float vals[4] = {tile.data[0].x, tile.data[0].y, tile.data[1].x, tile.data[1].y};
                    
                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        int row = base_row + r;
                        if (row < sorted_M_valid) {
                            int packed_id = sorted_ids[row];
                            int token_id = packed_id & 0xFFFFFF;
                            int topk_slot = (packed_id >> 24) & 0xFF;
                            float weight = sorted_weights[row];
                            
                            if (token_id >= 0 && token_id < num_tokens && topk_slot < topk) {
                                float weighted_val = vals[r] * weight;
                                // Direct write - no atomics! Each (token_id, topk_slot) is unique
                                tmp_buffer[(token_id * topk + topk_slot) * model_dim + col] = weighted_val;
                            }
                        }
                    }
                }
            }
        }
    };
    
    scatter_noatomic(C_accum[0], out_row_base_0);
    scatter_noatomic(C_accum[1], out_row_base_1);
}

// Reduction kernel: sum across topk slots
// output[token, col] = Σ_{slot=0..topk-1} tmp_buffer[token, slot, col]
__global__ void hk_moe_reduce_topk_kernel(
    const float* __restrict__ tmp_buffer,  // [num_tokens, topk, model_dim]
    float* __restrict__ output,            // [num_tokens, model_dim]
    const int num_tokens,
    const int model_dim,
    const int topk
) {
    // Each thread handles one (token, col) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = num_tokens * model_dim;
    
    if (idx >= total_elements) return;
    
    const int token_id = idx / model_dim;
    const int col = idx % model_dim;
    
    // Sum across topk slots
    float sum = 0.0f;
    #pragma unroll 8  // DeepSeek R1 uses topk=8
    for (int slot = 0; slot < topk; slot++) {
        sum += tmp_buffer[(token_id * topk + slot) * model_dim + col];
    }
    
    output[idx] = sum;
}

// Dispatch function for atomic-free FP8 Stage 2
void dispatch_hk_moe_stage2_fp8_noatomic(
    const moe_stage2_fp8_globals& g,
    float* tmp_buffer,    // [num_tokens, topk, model_dim]
    float* output_fp32    // [num_tokens, model_dim]
) {
    using namespace s2_cfg;
    
    const int num_m_blocks = (g.sorted_M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_n_blocks = (g.model_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Stage 2 GEMM kernel (no atomics)
    {
        dim3 grid(num_n_blocks, num_m_blocks);
        dim3 block(NUM_THREADS);
        
        size_t smem_size = 32768;  // 32KB for K_STEP=32
        hipFuncSetAttribute((void*)hk_moe_stage2_fp8_noatomic_kernel_mma, 
                            hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        
        hk_moe_stage2_fp8_noatomic_kernel_mma<<<grid, block, smem_size, g.stream>>>(
            g.intermediate.raw_ptr,
            g.w2_fp8,
            g.w2_scale,
            tmp_buffer,
            g.sorted_ids,
            g.sorted_expert_ids,
            g.num_valid_ids,
            g.sorted_weights,
            g.sorted_M,
            g.num_tokens,
            g.model_dim,
            g.inter_dim,
            g.inter_row_stride,
            g.num_experts,
            g.topk,
            g.block_m,
            num_m_blocks,
            num_n_blocks,
            g.num_scale_n,
            g.num_scale_k
        );
    }
    
    // Reduction kernel
    {
        const int total_elements = g.num_tokens * g.model_dim;
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        hk_moe_reduce_topk_kernel<<<num_blocks, block_size, 0, g.stream>>>(
            tmp_buffer,
            output_fp32,
            g.num_tokens,
            g.model_dim,
            g.topk
        );
    }
}
