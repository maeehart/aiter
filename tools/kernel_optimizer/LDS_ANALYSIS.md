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
