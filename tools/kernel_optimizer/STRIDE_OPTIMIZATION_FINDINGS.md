# LDS Stride Optimization Findings

## Summary

Attempted to reduce LDS usage from 64KB to <32KB by changing the thread stride from 34 to 33.
**Result: Not feasible via binary patching** due to complex inter-dependencies.

## What We Learned

### Current Addressing Scheme

The kernel uses multiple address registers with different computation paths:

| Register | Computation | Usage |
|----------|-------------|-------|
| v4 | (34 * (tid>>4) + 2*(tid&15) + s7*136) << 2 | 64 writes |
| v5 | (34 * (tid>>1) + (tid&1) + s7*2) << 2 | reads |
| v56 | Various direct thread-based | 96 reads, 20 writes |
| v2 | Different formula (not stride-34 based) | 40 reads |

### Why Changing Stride Fails

1. **Multiple address paths**: Writes via v4, reads via v56/v2
2. **Different groupings**: v4 groups threads by 16, v5 by 2
3. **Static offsets**: Designed specifically for stride 34 layout
4. **Scalar constants**: s60 = s7 * 0x88 (136 = 34*4)

When we changed:
- v4: 34 → 33 ✓
- v5: 34 → 33 ✓
- s60: 136 → 132 ✓

But v56 and v2 still use their original formulas, causing write-read misalignment.

### Bank Conflict Analysis

| Metric | Current | With XOR Swizzle |
|--------|---------|------------------|
| Banks used | 4 of 32 | 30-32 of 32 |
| LDS footprint | 52KB | ~8KB potential |
| Bank utilization | 12.5% | 94-100% |

The current stride 34 causes addresses to only hit banks 0, 8, 16, 24.

### What Would Fix This (Source Level)

Per [AMD's CK-Tile blog](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html):

```python
# XOR transformation
K0' = K0 ^ (M % (KPerBlock / Kpack * MLdsLayer))
```

This requires:
1. Modifying ALL address computations consistently
2. Inserting v_xor_b32 instructions
3. Updating both read and write paths

## Binary Modifications Attempted

1. **v_mul_i32_i24 stride change**: 34 → 33 at 0x2914, 0x2938
2. **s_mul_i32 scalar change**: 0x88 → 0x84 at 0x2928
3. **Combined approach**: All three changes together

**All approaches failed** - output became NaN due to misaligned data access.

## Recommendations for AITER Team

1. **Implement XOR swizzle at source level**
   - Use CK-Tile framework's built-in support
   - Could reduce LDS to ~8KB
   - Enable 2+ workgroups per CU

2. **Consider smaller tile sizes**
   - 16x128 instead of 32x256
   - Would use ~16KB LDS
   - Better occupancy for large batches

3. **Alternative: Use existing 56KB reduction**
   - We successfully reduced declared LDS from 64KB to 56KB
   - ~2% performance improvement
   - Safe binary patch

## Files Created During Investigation

- `fmoe_*_stride33.co` - Failed stride modification attempts
- `fmoe_*_stride33_full.co` - Combined stride + scalar modification
- `fmoe_*_stride33_v2.co` - Alternative attempt

## References

- [AMD CK-Tile LDS Bank Conflict Blog](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
- [ROCm Workload Optimization Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
