# HipKittens MoE Kernels

High-performance Mixture of Experts (MoE) kernels for AMD MI300X/MI325X GPUs using the HipKittens framework.

## Overview

This implementation provides optimized MoE kernels targeting high batch sizes (prefill) where L2 cache thrashing is a major performance issue. The kernels use XCD-aware scheduling to improve cache utilization across the MI300X/MI325X chiplets.

## Architecture

The MoE computation is split into three stages:

1. **Stage 1 (Gate-Up Projection)**: `hidden_states @ w1.T` → `[sorted_M, inter_dim*2]`
2. **Activation**: G1U1 activation (SiLU on gate, multiply with up)
3. **Stage 2 (Down Projection)**: `activated @ w2.T` with weighted atomic accumulation

### Key Files

- `hk_moe_stage1.cu` - Gate-up projection kernel
- `hk_moe_stage2.cu` - Down projection with weighted accumulation
- `hk_moe_torch.cu` - PyTorch interface and G1U1 activation kernel
- `hk_moe_kernel.cuh` - Common definitions and global structures

## Current Optimizations

### ✅ Completed

1. **XCD-aware Scheduling**
   - Uses `chiplet_transform_chunked` for MI300X/MI325X (8 XCDs)
   - L2-friendly block swizzling within each XCD
   - Improves cache hit rate for weight tiles

2. **MMA Instructions**
   - Uses HipKittens `mma_ABt` with MFMA 16x16x16 bf16 instructions
   - 8-wave kernel pattern (512 threads) for 128x128 output tiles
   - Register tiles (`rt_bf`, `rt_fl`) and shared tiles (`st_bf`)
   - Scheduling barriers and priority hints for instruction scheduling

3. **Tiled Computation**
   - BLOCK_SIZE=128, K_STEP=64 per HipKittens GEMM pattern
   - Shared memory staging with swizzled layout
   - Cooperative loading across all 512 threads

### ⏳ Pending (For Further Optimization)

4. **Ping-Pong Pipelining** (~1.5-2x potential gain)
   - Overlap compute with next tile's memory loads
   - Double-buffering in shared memory
   - Currently loads are sequential, not overlapped

5. **Gather/Scatter Optimization**
   - Pre-sort tokens to enable contiguous access
   - Reduce atomic operation overhead in Stage 2

## Current Performance

Tested on MI300X with model_dim=4096, inter_dim=4096, 8 experts, topk=2:

| Batch | AITER (TFLOPs) | HipKittens (TFLOPs) | Speedup |
|-------|----------------|---------------------|---------|
| 1024  | 533            | 35.6                | 0.09x   |
| 4096  | 474            | 50.0                | 0.11x   |
| 8192  | 555            | 52.6                | 0.09x   |

**Performance Analysis**: The MMA-based implementation achieves 35-52 TFLOPs, significantly improved from the scalar version (~7 TFLOPs). The remaining gap vs AITER (400-550 TFLOPs) is due to:
1. **Input gather overhead**: Stage 1 gathers input via `sorted_ids`, breaking coalesced memory access
2. **Scatter-add overhead**: Stage 2 uses atomic operations for weighted accumulation
3. **No pipelining**: Sequential load-compute instead of overlapped execution
4. **Tile size mismatch**: 128x128 tiles may not be optimal for MoE workloads

## Running the Benchmark

```bash
cd /workspace/dev/aiter_long_context_moe

# Quick test (3 batch sizes)
python op_tests/test_hipkittens_moe.py --quick

# Full benchmark (all high batch sizes from CSV analysis)
python op_tests/test_hipkittens_moe.py

# Custom batch sizes
python op_tests/test_hipkittens_moe.py --batch-sizes 1000 2000 4000
```

## Technical Details

### Sorted Token Layout

The MoE sorting produces:
- `sorted_ids`: Packed int32 where upper 8 bits = `topk_slot`, lower 24 bits = `token_id`
- `sorted_expert_ids`: Indexed by `block_m` tiles, not per-row
- `sorted_weights`: Per-row routing weights for weighted accumulation

### Memory Layout

- `hidden_states`: `[num_tokens, model_dim]` bf16
- `w1`: `[num_experts, inter_dim*2, model_dim]` bf16 (gate-up fused)
- `w2`: `[num_experts, model_dim, inter_dim]` bf16
- `intermediate`: `[sorted_M, inter_dim*2]` bf16 (after Stage 1)
- `output`: `[num_tokens, model_dim]` bf16

### Shared Memory Usage

Stage 1 and Stage 2:
- Input tile: `BLOCK_M × (BLOCK_K + 8)` floats = 32 × 72 × 4 = 9,216 bytes
- Weight tile: `BLOCK_N × (BLOCK_K + 8)` floats = 128 × 72 × 4 = 36,864 bytes
- Total: ~46 KB (within 64 KB LDS limit)

## Next Steps for MMA Implementation

To achieve competitive performance, implement MMA-based computation following the HipKittens GEMM pattern:

```cpp
// Example from 3rdparty/hipkittens/analysis/bf16_gemm/mi325x/
constexpr int BLOCK_SIZE = 128;
constexpr int K_STEP = 64;
constexpr int REG_BLOCK = 32;  // BLOCK_SIZE / 4
constexpr int DOT_SLICE = 16;  // MMA native dimension

// Shared tiles for staging
st_bf<BLOCK_SIZE, K_STEP> As, Bs;

// Register tiles for MMA
rt_bf<REG_BLOCK, DOT_SLICE> a_reg, b_reg;
rt_fl<REG_BLOCK, REG_BLOCK, col> C_accum;

// Main loop
for (int tile = 0; tile < num_tiles; tile++) {
    // Cooperative load to shared memory
    G::load(As, g.a, {0, 0, row, tile});
    G::load(Bs, g.b, {0, 0, col, tile});
    __builtin_amdgcn_s_barrier();
    
    // Load subtiles to registers and compute MMA
    load(a_reg, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, k}));
    load(b_reg, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, k}));
    mma_ABt(C_accum, a_reg, b_reg, C_accum);  // 16x16x16 MMA
}
```

### Challenges for MoE MMA Implementation

1. **Input Gathering**: Hidden states are indexed via `sorted_ids`, breaking simple tile loading
2. **Expert Routing**: Different blocks may use different experts
3. **Output Scatter**: Stage 2 requires weighted atomic accumulation

### Potential Solutions

1. Pre-gather inputs into contiguous buffer before MMA computation
2. Use expert-grouped block scheduling
3. Accumulate locally first, then scatter-add at tile boundaries

## References

- [HipKittens](https://github.com/HazyResearch/ThunderKittens) - CUDA/HIP high-performance primitives
- [AITER](https://github.com/ROCm/aiter) - AMD AI Tensor Engine for ROCm
- [MI300X Architecture](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) - 8 XCDs with shared L2 cache

