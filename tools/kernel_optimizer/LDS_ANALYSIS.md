# LDS Usage Analysis for MoE Kernel

## Summary

The kernel declares 64KB LDS but analysis reveals significant optimization potential:

| Metric | Value |
|--------|-------|
| Declared LDS | 64 KB |
| Max offset used | 52.1 KB |
| Actual data stored | 6.9 KB |
| Waste | 86.7% |
| Safe reduction | 56 KB (tested, 2% speedup) |

## Data Flow to LDS

### Direct Global -> LDS Loads (buffer_load ... lds)
- **Up weights (U)**: 40 loads (160 bytes)
- **Input scale (XQ)**: 5 loads (20 bytes)
- Total: 45 instructions, 180 bytes directly loaded

### ds_write Operations
- 84 total writes
- 608 bytes per execution
- Sources: MFMA results, conversions, constants

### ds_read Operations  
- 268 total reads
- Heavy reuse of data (3.18x amplification factor)

## Why LDS is Sparse

The 6.9KB of actual data is spread across 52KB due to:

1. **Thread-dependent addressing**: Base addresses (v2, v3, v56) computed from thread ID
2. **Alignment padding**: Data aligned to prevent bank conflicts
3. **Double-buffering regions**: Upper/lower 32KB for pipelining

## L1 Cache Opportunity Analysis

### Could Move to L1 (Read-Only Data)
- Input scale factors (XQ) - 20 bytes
- BUT: Thread-dependent access patterns make LDS necessary

### Must Stay in LDS (Shared/Written Data)
- MFMA intermediate results - inter-thread sharing required
- Up weight broadcast data - already optimized for LDS broadcast

## Binary Patching Limitations

Cannot move data from LDS to L1 via simple binary patching because:

1. `buffer_load ... lds` uses VGPR field as LDS offset, not destination
2. Clearing LDS bit would require changing all consumer ds_read instructions
3. Address remapping requires understanding full thread indexing scheme

## Achievable Optimization

Successfully reduced LDS from 64KB to 56KB:
- Modify both kernel descriptor (0x1d00) AND ELF metadata (0x1a76)
- ~2% performance improvement
- No code changes required

Further reduction (to 48KB or below) causes data corruption because
the sparse addressing genuinely uses offsets up to 52KB.

## Recommendations for Further Optimization

1. **Source-level changes** needed to:
   - Use smaller tiles (reduce LDS per workgroup)
   - Restructure data layout for denser packing
   - Consider L1-cached loads for read-only data

2. **Alternative kernel variants**:
   - Smaller tile sizes (16x128 instead of 32x256)
   - Would use ~16KB LDS → 4 workgroups per CU possible
   - Trade-off: Lower compute intensity per memory access

## Bank Conflict Analysis

### Current Bank Distribution

According to [AMD's ROCm blog on LDS bank conflicts](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html):
- LDS has 32 banks, 4 bytes each
- Bank conflicts serialize accesses and reduce throughput
- XOR-based swizzle can eliminate conflicts without extra space

**Current kernel analysis:**
- Only uses 4 banks (0, 8, 16, 24) out of 32
- Stride of 34 bytes causes poor bank distribution
- 86.7% of LDS space is wasted due to sparse addressing

### Stride Analysis

```
v4 = 34 * thread_id (via v_mul_i32_i24_e32 v4, 34, v56)

Bank distribution with stride 34:
  34 bytes = 8 dwords + 2 bytes
  Thread 0: bank 0
  Thread 1: bank (34/4) % 32 = 8
  Thread 2: bank (68/4) % 32 = 17 -> wraps to 1? No...
  
Actually: (34 * N * 4) % 128 gives bank offset
  N=0: 0 -> bank 0
  N=1: 136 % 128 = 8 -> bank 2
  N=2: 272 % 128 = 16 -> bank 4
  N=3: 408 % 128 = 24 -> bank 6
  Pattern: 0, 2, 4, 6, 0, 2, 4, 6... (4 banks)
```

### Attempted Optimization

**Experiment:** Changed stride from 34 to 33
- 33 is coprime to 32, would distribute across all banks
- Modified bytes at 0x2914 and 0x2938 (0xA2 -> 0xA1)

**Result:** ❌ FAILED
- Output became NaN
- The stride is tightly coupled with static offsets in ds_read/ds_write
- Changing only the dynamic computation breaks data alignment

### Why XOR Swizzle Can't Be Applied via Binary Patching

1. **Coordinated changes required:**
   - Address computation (v_mul_i32_i24)
   - All static offsets (offset:XXXX in ds_read/ds_write)
   - Both read and write paths must use same transformation

2. **Need to insert instructions:**
   - XOR swizzle requires `v_xor_b32` operations
   - No free instruction slots available
   - Cannot easily expand binary size

3. **Complex dependency analysis:**
   - Must understand full thread indexing scheme
   - Data layout expectations across multiple computation stages

### Recommendation for AITER Team

The kernel would benefit significantly from implementing XOR-based LDS swizzle
at the source level. Based on the AMD blog, this could:
- Reduce LDS from 64KB to potentially <10KB
- Enable 2+ workgroups per CU
- Eliminate bank conflicts and improve bandwidth

The CK-Tile framework (Composable Kernel) already supports this:
```
// XOR transformation (from AMD blog)
K0' = K0 ^ (M % (KPerBlock / Kpack * MLdsLayer))
```

This is a source-level optimization that cannot be achieved via binary patching.

## Stride Optimization Attempt

### Attempted Changes

1. v_mul_i32_i24_e32 v4, **34**, v56 → v_mul_i32_i24_e32 v4, **33**, v56
2. v_mul_i32_i24_e32 v5, **34**, v56 → v_mul_i32_i24_e32 v5, **33**, v56  
3. s_mul_i32 s60, s7, **0x88** → s_mul_i32 s60, s7, **0x84**

### Result: Failed

Output became NaN because:
- Writes use v4 (modified stride)
- Reads use v56 and v2 (different formulas, NOT stride-34 based)
- Static offsets designed for stride 34 cause misalignment

### Root Cause

The kernel uses **multiple independent address computation paths**:
- v4 for writes (modified)
- v56 for some reads/writes (unmodified)
- v2 for other reads (different formula entirely)

Changing v4 and v5's stride without modifying v56 and v2's formulas 
(and their static offsets) breaks data coherence.

### Conclusion

LDS stride optimization requires source-level changes to:
1. Unify or coordinate all address computation paths
2. Update all static offsets consistently
3. Implement XOR swizzle for bank conflict avoidance

This cannot be achieved via binary patching alone.
