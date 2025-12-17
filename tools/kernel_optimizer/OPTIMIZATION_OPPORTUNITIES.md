# MoE Kernel Optimization Opportunities

## Summary

**⚠️ CRITICAL FINDINGS**:

1. **AITER caches loaded kernels** - swapping .co files requires fresh subprocess
2. **High variance (~3% std)** - results can vary ±6% from run-to-run noise
3. **Ordering effects** - first kernel tested often appears slower (cache warm-up)

**Results** (randomized ordering, 5 runs each at batch=8192):
- baseline: 3291 ± 92 μs
- opt_vmcnt_reduce25: 3307 ± 72 μs (0.995x - slightly slower)
- opt_vmcnt_cap8: 3286 ± 82 μs (1.002x - essentially same)

**Conclusion**: No statistically significant performance difference. The s_waitcnt 
values are already well-tuned. Earlier "improvements" were ordering artifacts.

All variants are **correct** (max_diff = 2.0, same as baseline's inherent variation).

---

## Kernel Profile Summary

| Resource | Value | Implication |
|----------|-------|-------------|
| VGPRs | 256 used (512 declared) | NOT the bottleneck |
| ACCVGPRs | 128 | For MFMA operations |
| SGPRs | 112 | Not the bottleneck |
| **LDS** | **64 KB** | **THE BOTTLENECK** - uses entire CU LDS |
| MFMA ops | 768 | Compute-heavy |
| Atomics | 112 | Output accumulation |
| Instruction mix | Excellent | MFMA interleaved with memory ops |

**Key Finding**: LDS is the occupancy limiter. The kernel uses 100% of CU LDS,
so only 1 workgroup can run per CU regardless of other resource usage.

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

## VGPR Reduction Testing ✅ TESTED - NO IMPROVEMENT

**Experiment:**
- Reduced VGPR count in COMPUTE_PGM_RSRC1 metadata
- Created variants: 384, 320, 256 VGPRs

**Results:**
- 384 VGPRs: ✅ Correct, 0.999x performance (no change)
- 320 VGPRs: ❌ Incorrect outputs
- 256 VGPRs: ❌ Produces NaNs

**Why no improvement with 384 VGPRs?**
The kernel uses **64KB LDS** (`group_segment_fixed_size: 65536`), which is the 
**entire LDS capacity** of an MI300X CU. This means:
- Only 1 workgroup can run per CU regardless of VGPR count
- VGPRs are NOT the occupancy bottleneck - LDS is!
- Reducing VGPRs cannot improve occupancy

---

## NOT Feasible via Binary Patching

| Target | Reason |
|--------|--------|
| VGPR reduction | Works but LDS is actual bottleneck |
| LDS reduction | Would require algorithm rewrite |
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

