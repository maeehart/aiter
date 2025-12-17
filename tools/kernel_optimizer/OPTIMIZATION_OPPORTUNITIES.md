# MoE Kernel Optimization Opportunities

## Summary

**Conclusion**: The ASM kernel is already highly optimized. After extensive testing of 
15 s_waitcnt variants across multiple batch sizes (2048, 4096, 8192, 16384), 
**no statistically significant performance improvements were found**.

All variants are **correct** (max_diff = 2.0, same as baseline's inherent variation).

---

## Kernel Profile Summary

| Resource | Value | Implication |
|----------|-------|-------------|
| VGPRs | 512 | 1 wave/SIMD (12.5% occupancy) |
| SGPRs | 112 | Not the bottleneck |
| LDS | 64 KB | Fully utilized |
| MFMA ops | 768 | Compute-heavy |
| Atomics | 112 | Output accumulation |
| Instruction mix | Excellent | MFMA interleaved with memory ops |

---

## Tested Optimizations

### s_waitcnt Tuning ✅ TESTED - NO IMPROVEMENT

Created 15 variants:
- `opt_vmcnt_zero` - All vmcnt to 0
- `opt_vmcnt_reduce25/50/75` - Reduce by percentage  
- `opt_vmcnt_cap2/4/6/8/12` - Cap at specific value
- `opt_lgkmcnt_zero/reduce50` - Modify lgkmcnt
- `opt_both_reduce50` - Both counters
- `opt_aggressive` - vmcnt≤2, lgkmcnt=0
- `opt_high_vmcnt_only` - Only reduce high values
- `opt_medium_vmcnt_only` - Only reduce medium values

**Results** (batch=8192, 100 iterations):
- All correct (max_diff ≤ 2.5)
- Best speedup: 1.004x (within noise)
- No statistically significant improvements

### Cache Policy Testing ✅ TESTED - MADE WORSE

- glc (bypass L1): 0.997x (neutral)
- slc (bypass L2): 0.858x (**14% slower**)

**Conclusion**: L2 cache is critical for performance.

---

## NOT Feasible via Binary Patching

| Target | Reason |
|--------|--------|
| VGPR reduction | 512→256 would require complete rewrite |
| LDS offsets | Already optimal stride pattern |
| Barriers | Would break synchronization |
| s_setvskip | Would break conditional execution |
| Instruction scheduling | Already excellent |

---

## Why No Improvements?

The kernel is already well-optimized:
1. **Memory/compute overlap**: MFMA interleaved with buffer_load
2. **Appropriate s_waitcnt values**: Tuned for the specific memory access pattern
3. **Good LDS bank access**: No significant conflicts
4. **Cache-friendly**: Relies on L2 cache effectively

The main limitation (512 VGPRs → 1 wave/SIMD) is a deliberate design choice 
for larger tiles and better compute efficiency.

---

## Tools Created

- `fast_benchmark.py` - Quick benchmark with correctness validation
- `create_new_variants.py` - Generate s_waitcnt variants
- `proper_correctness_test.py` - Max-diff validation (not p-values)
- `kernel_analysis.py` - Comprehensive kernel analysis
- `cache_optimizer.py` - Cache policy testing

