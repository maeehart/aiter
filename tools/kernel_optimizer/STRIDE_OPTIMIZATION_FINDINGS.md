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

## Root Cause Analysis (Final)

### The Real Problem: v56 vs v5 Mismatch

The kernel has two separate data flows that share LDS space:

1. **v4-based flow**: Uses stride 34
   - Write: v4 = (34 * (tid>>4) + 2*(tid&15) + s7*136) << 2
   - Read: v5 = (34 * (tid>>1) + (tid&1) + s7*2) << 2
   - These stay aligned when stride changes (both shift proportionally)

2. **v56-based flow**: Does NOT use stride 34
   - Write: v56 = ((tid&15)<<1 + (tid>>5)<<5) << 2 + 256*s7
   - Read: v5 = (34 * (tid>>1) + (tid&1) + s7*2) << 2
   - **THESE BREAK** when stride changes!

### Communication Pairs Broken

| Writer (v56) | Reader (v5@34) | Reader (v5@33) | Status |
|--------------|----------------|----------------|--------|
| W0 → addr 0 | R0 → addr 0 | R0 → addr 0 | ✓ OK |
| W33 → addr 136 | R2 → addr 136 | R2 → addr 132 | ✗ BROKEN |
| W66 → addr 272 | R4 → addr 272 | R4 → addr 264 | ✗ BROKEN |
| W99 → addr 408 | R6 → addr 408 | R6 → addr 396 | ✗ BROKEN |

### Why NaN Occurs

When stride changes:
- v56 writes to address 136 (unchanged formula)
- v5 reads from address 132 (shifted by -4)
- Reader gets uninitialized/garbage data → NaN

### What Would Be Required

To properly change the stride, we need to modify v56's formula to be compatible.
Current v56: `((tid&15)<<1 + (tid>>5)<<5) << 2`

This formula has NO stride parameter - it's a completely different indexing scheme.
To make it compatible with stride 33, we'd need to:

1. Replace the entire computation chain (5+ instructions)
2. Derive a new formula that maintains the same thread communication pattern
3. Handle the s7*256 scalar addition separately

This is beyond practical binary patching.

## Final Conclusion

LDS stride optimization cannot be achieved via binary patching because:
1. The kernel uses multiple independent address formulas
2. v56 writes use a formula completely unrelated to stride 34
3. v5 reads depend on stride 34
4. Changing stride breaks v56-to-v5 data flow
5. Fixing v56's formula requires replacing entire computation chains

**Recommendation**: This optimization requires source-level changes to unify
the address computation schemes before any stride modification can work.

## Final Root Cause Analysis (Complete)

### The Three-Component System

The kernel's LDS addressing relies on THREE interrelated components:

1. **Write base formula**: `((tid>>5)*32 + extra + (tid&15)*2) * 4 + s7*256`
2. **Read base formula**: `((tid>>4)*64 + (tid&15)*2) * 4` (NO s7 term!)
3. **Static offsets**: Hardcoded in each instruction (20736, 20864, 21760, etc.)

### How They Work Together (3200 Mappings!)

The static offsets COMPENSATE for the different formulas:

| Write Thread | Write Base | Write Offset | Total | Read Thread | Read Base | Read Offset |
|--------------|------------|--------------|-------|-------------|-----------|-------------|
| W0 | 0 | 21760 | 21760 | R0 | 0 | 21760 |
| W0 | 0 | 21760 | 21760 | R64 | 1024 | 20736 |
| W0 | 0 | 22784 | 22784 | R128 | 2048 | 20736 |
| W33 | 136 | 20736 | 20872 | R1 | 8 | 20864 |

### Why Stride Change Breaks Everything

When stride changes (32→31, 64→62):
- Write base: 0 (unchanged for W0)
- Read base for R64: 1024 → 992 (changed!)
- Static offsets: UNCHANGED (hardcoded!)

Result: W0(0) + 21760 = 21760, but R64(992) + 20736 = 21728 ≠ 21760

### What Would Be Required

To change stride from 32 to 31:
1. Modify write base multiplier (32→31) ✓ We did this
2. Modify read base multiplier (64→62) ✓ We did this  
3. **Modify EVERY static offset in EVERY ds_read/ds_write instruction**
   - 96+ LDS operations
   - Each needs a DIFFERENT adjustment depending on which thread pairs it serves
   - Some threads use the same instruction but need different adjustments!

This is **mathematically impossible** to achieve with static offset values.
The offsets would need to be thread-dependent, which they are not.

### Conclusion

LDS stride optimization for bank conflict reduction is **NOT achievable via binary patching** because:

1. The addressing scheme uses three interrelated components
2. Static offsets are designed specifically for stride 32/64
3. Changing offsets would require thread-dependent values
4. Instruction encoding only supports constant immediate offsets

This optimization requires **source-level restructuring** of the entire LDS addressing scheme.
