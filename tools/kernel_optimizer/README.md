# MoE Kernel Optimization Tools

Tools for analyzing and optimizing AITER MoE ASM kernels for AMD MI300X GPUs.

## ⚠️ Critical Finding: s_waitcnt Optimization Fails Correctness

**Binary patching of `s_waitcnt` instructions produces incorrect results.**

All tested optimizations that modify `s_waitcnt` values fail correctness validation:

| Optimization | Mismatch Rate | Status |
|--------------|---------------|--------|
| vmcnt_zero | 2-11% | ❌ FAIL |
| vmcnt_reduce25 | 2-12% | ❌ FAIL |
| vmcnt_cap4 | 3-12% | ❌ FAIL |
| vmcnt_cap8 | 3-11% | ❌ FAIL |
| both_reduce50 | 3-10% | ❌ FAIL |

**Root Cause**: `s_waitcnt` instructions enforce memory synchronization. Reducing
wait counts creates race conditions where the kernel proceeds before data is ready.

The performance "improvements" from these modifications were false positives - the
kernel ran faster because it wasn't waiting for correct data.

## Files

- `benchmark_asm_variants.py` - Benchmark script with **correctness validation**
- `profile_moe_kernel.py` - rocprof profiling script
- `binary_patch_optimizer.py` - Binary patching tool (produces broken kernels)
- `annotate_asm.py` - Assembly annotation tool
- `benchmark_results/` - Results including correctness validation

## Lessons Learned

1. **Always validate correctness before trusting performance numbers**
2. `s_waitcnt` values are carefully tuned and cannot be arbitrarily reduced
3. Black-box binary optimization without understanding data dependencies is risky
4. Performance improvements without correctness checks are meaningless

## Safe Optimization Approaches

For future optimization attempts, consider:

1. **Source-level optimization** - Modify the actual kernel source code
2. **Algorithmic changes** - Different blocking/tiling strategies
3. **Higher-level optimization** - Batch scheduling, workload distribution
4. **Profile-guided optimization** - Use profiling to find actual bottlenecks
5. **Waiting for upstream improvements** - AITER team may release optimized kernels

## Running the Benchmark

```bash
# With correctness validation (recommended)
python benchmark_asm_variants.py --batch-sizes 1024 4096 8192 16000 24000

# Skip correctness (faster but dangerous)
python benchmark_asm_variants.py --skip-correctness
```

## Broken Kernels

The broken optimized kernels are archived in:
`hsa/gfx942/fmoe/silu/broken_optimizations/`

These should NOT be used in production.
