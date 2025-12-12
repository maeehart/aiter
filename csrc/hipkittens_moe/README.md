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

3. **Vectorized Memory Access**
   - float4 (8 bf16) vectorized loads from global memory
   - ~3.5x speedup over element-by-element loading
   - Applied to both input/intermediate and weight loading

4. **Occupancy Optimization**
   - Reduced K_STEP from 64 to 32
   - Halved shared memory: 64KB → 32KB per block
   - Enables 2 concurrent blocks per CU (was 1)
   - ~20% additional speedup

5. **Tiled Computation**
   - BLOCK_SIZE=128, K_STEP=32 per HipKittens GEMM pattern
   - Shared memory staging with swizzled layout
   - Cooperative loading across all 512 threads

### ⏳ Remaining Bottlenecks

5. **Async Pipelining** (would need AMD buffer load intrinsics)
   - HipKittens uses `load_global_to_register_buffer` for true async
   - Requires `gl` (global layout) abstraction for buffer operations
   - Current implementation loads sequentially

6. **Gather/Scatter Overhead** (inherent to MoE)
   - Stage 1 gather: `sorted_ids` lookup breaks coalesced access
   - Stage 2 scatter: Atomic operations for weighted accumulation
   - Would require pre-sorting or algorithm redesign to optimize

## Current Performance

Tested on MI300X with model_dim=4096, inter_dim=4096, 8 experts, topk=2:

| Batch | AITER (TFLOPs) | HipKittens (TFLOPs) | Speedup |
|-------|----------------|---------------------|---------|
| 1024  | ~633           | ~216                | 0.46x   |
| 4096  | ~469           | ~216                | 0.46x   |
| 8192  | ~557           | ~225                | 0.40x   |

### Detailed Profiling Analysis (rocprof)

**Per-kernel breakdown** (batch=8192):

| Kernel | HipKittens | AITER CK | Ratio |
|--------|-----------|----------|-------|
| Stage 1 | 4.5 ms | 2.0 ms | 2.25x slower |
| Stage 2 | 2.2 ms | 1.0 ms | 2.20x slower |
| Activation | 0.2 ms | (fused) | - |
| **Total** | **~7.0 ms** | **~3.0 ms** | **2.35x slower** |

**AITER CK Configuration** (from template analysis):
- Block: 256 threads (4 warps)
- Tile: 128×128×64 (M×N×K)
- MMA: 16×16×8 pattern with 2-stage pipeline
- Grid: 8192 × 136 blocks

**HipKittens Configuration**:
- Block: 512 threads (8 warps)  
- Tile: 128×128×64 (M×N×K)
- MMA: 16×16×16 pattern
- Grid: 128 × 64 blocks

### Performance Gap Analysis

The consistent 2.2-2.3x gap across both stages suggests fundamental differences in:

1. **Grid Structure**: AITER uses grid_y=136 (possibly 8 experts × 17 tiles), which may provide better expert batching
2. **Pipeline Scheduling**: CK uses `BlockGemmPipelineVersion=2` for sophisticated memory/compute overlap
3. **Thread Configuration**: AITER uses fewer threads (256 vs 512) but achieves better throughput

### Optimization Attempts

| Optimization | Result |
|-------------|--------|
| Local atomic accumulation | No improvement |
| K_STEP 32→64 | No improvement |
| XCD-aware scheduling | Already implemented |
| Vectorized loads | +3.5x (already applied) |
| Occupancy tuning | +20% (already applied) |
| **Expert-aware N-first scheduling** | **+8% at small batches (0.55x vs 0.51x)** |

### Expert-Aware Grid Scheduling

Implemented N-first ordering within M tiles: process all output columns (N tiles) for a given token tile (M tile) before moving to the next token tile. This keeps expert weights in L2 cache longer.

**Key insight**: Since tokens are already sorted by expert, consecutive M tiles share the same expert. By processing all N tiles for each M tile together, we maximize L2 cache hits for the expert's weight matrix.

```
Block ordering:
  Block 0: (M=0, N=0), Block 1: (M=0, N=1), ..., Block N-1: (M=0, N=num_n-1)
  Block N: (M=1, N=0), Block N+1: (M=1, N=1), ...
```

### Future Work: Template-Based Tuning

The remaining gap requires template programming to tune parameters per batch size:

```cpp
// TODO: Generate specialized kernels for different configurations
template<int BLOCK_SIZE, int K_STEP, int WGM>
__global__ void hk_moe_stage1_kernel();

// Dispatch based on batch size
if (batch_size < 2048) {
    hk_moe_stage1_kernel<128, 64, 8><<<grid, block>>>();
} else {
    hk_moe_stage1_kernel<128, 64, 4><<<grid, block>>>();
}
```

Parameters to tune per batch size:
- `WGM`: Workgroup grouping factor (4-16)
- `K_STEP`: K-tile size (32-64)
- `BLOCK_SIZE`: M/N tile size (64-256)
- Thread count: 256 vs 512

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

