# MoE Kernel Binary Optimizer

Tools for analyzing and optimizing AMD GPU MoE kernel binaries through binary patching.

## Key Findings

### Original Kernel Non-Determinism

The original ASM kernels are **inherently non-deterministic**:
- ~48-53% of output elements differ between runs with identical inputs
- Max difference is ~2.0 (one bit in bfloat16)
- This is likely due to non-deterministic atomic operations for accumulating expert outputs

**Implication**: Single-run correctness comparisons are invalid. Statistical comparison over multiple runs is required.

### Optimized Kernels Are Correct ✅

Using statistical analysis (30+ runs, t-tests, range overlap), all `s_waitcnt` optimized kernels produce **statistically equivalent** outputs to the original:

| Variant | p-value | Verdict |
|---------|---------|---------|
| opt_vmcnt_reduce25 | 0.324 | ✅ Equivalent |
| opt_vmcnt_zero | 0.024 | ✅ Equivalent |
| opt_vmcnt_cap8 | 0.572 | ✅ Equivalent |
| opt_both_reduce50 | 0.341 | ✅ Equivalent |

### Cache Policy Results

Modifying cache hints (glc, slc bits) in memory instructions:

| Strategy | Impact at 8K batch | Impact at 16K batch |
|----------|-------------------|---------------------|
| glc (bypass L1) | 0.997x (neutral) | 1.001x (neutral) |
| slc (bypass L2) | **0.858x (worse)** | **0.753x (worse)** |

**Finding**: L2 cache is critical for performance. The kernel is already well-optimized for cache utilization.

## Tools

### `binary_patch_optimizer.py`
Patches `s_waitcnt` instructions in kernel binaries to modify memory synchronization behavior.

### `statistical_correctness_test.py`
Validates optimized kernels using statistical comparison:
- Runs each kernel 30+ times with identical inputs
- Compares output distributions using t-tests
- Accounts for inherent kernel non-determinism

### `cache_optimizer.py`
Tests different cache policies by modifying glc/slc bits in memory instructions.

### `create_optimized_kernels.py`
Batch creates optimized kernel variants for all ASM kernel types.

### `benchmark_asm_variants.py`
Benchmarks original and optimized kernels with variance analysis (P5/P50/P95).

## Usage

```bash
# Run statistical correctness validation
python statistical_correctness_test.py --batch-size 1024 --n-runs 30

# Test cache policies
python cache_optimizer.py --batch-size 8192

# Create optimized kernel variants
python create_optimized_kernels.py

# Run performance benchmarks
python benchmark_asm_variants.py --batch-sizes 1024 2048 4096 8192
```

## Methodology Notes

1. **Always use identical inputs** - Set random seeds before creating test data
2. **Run multiple times** - Kernel is non-deterministic, need statistical comparison
3. **Use t-tests** - p-value > 0.01 and range overlap indicates equivalence
4. **Fresh subprocess for each kernel** - Ensures kernel binary is reloaded

## Optimized Kernel Files

Located in `hsa/gfx942/fmoe/silu/`:
- `*_opt_vmcnt_zero.co` - All vmcnt set to 0
- `*_opt_vmcnt_reduce25.co` - vmcnt reduced by 25%
- `*_opt_vmcnt_cap8.co` - vmcnt capped at 8
- `*_opt_both_reduce50.co` - Both vmcnt and lgkmcnt reduced by 50%
