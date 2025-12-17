# MoE Kernel Binary Optimizer

Tools for analyzing and optimizing AMD GPU MoE kernel binaries through binary patching.

## Key Findings

### Original Kernel Non-Determinism

The original ASM kernels are **inherently non-deterministic**:
- Max element-wise difference between runs: **2.0** (exactly 1 bit in bfloat16)
- This is due to non-deterministic atomic operations for accumulating expert outputs

### Optimized Kernels Validation ✅

Using proper variation-based validation (NOT p-values), all optimized kernels produce outputs
within the acceptable variation threshold:

| Kernel | Max Diff to Reference | Self Variation | Status |
|--------|----------------------|----------------|--------|
| opt_vmcnt_reduce25 | 2.0 | 2.0 | ✅ CORRECT |
| opt_vmcnt_zero | 2.0 | 2.0 | ✅ CORRECT |
| opt_vmcnt_cap8 | 2.0 | 3.0 | ✅ CORRECT |
| opt_both_reduce50 | 2.0 | 2.0 | ✅ CORRECT |

**Validation criteria**: Max element-wise difference ≤ 2.5 (≈1 bit in bfloat16)

### Cache Policy Results

Modifying cache hints (glc, slc bits) in memory instructions:

| Strategy | Impact at 8K batch | Impact at 16K batch |
|----------|-------------------|---------------------|
| glc (bypass L1) | 0.997x (neutral) | 1.001x (neutral) |
| slc (bypass L2) | **0.858x (worse)** | **0.753x (worse)** |

**Finding**: L2 cache is critical for performance.

## Tools

### `proper_correctness_test.py`
Validates optimized kernels by measuring max element-wise variation:
- Compares against baseline (original kernel self-variation = 2.0)
- Accepts kernels with max_diff ≤ 2.5 (1 bit in bfloat16)
- Does NOT use p-values (which can be misleading)

### `binary_patch_optimizer.py`
Patches `s_waitcnt` instructions in kernel binaries.

### `cache_optimizer.py`
Tests different cache policies (glc/slc bits).

### `kernel_analysis.py`
Comprehensive kernel analysis (registers, LDS, instruction mix).

## Usage

```bash
# Run proper correctness validation
python proper_correctness_test.py --batch-size 1024 --n-runs 10

# Test cache policies
python cache_optimizer.py --batch-size 8192
```

## Optimized Kernel Files

Located in `hsa/gfx942/fmoe/silu/`:
- `*_opt_vmcnt_zero.co` - All vmcnt set to 0
- `*_opt_vmcnt_reduce25.co` - vmcnt reduced by 25%
- `*_opt_vmcnt_cap8.co` - vmcnt capped at 8
- `*_opt_both_reduce50.co` - Both vmcnt and lgkmcnt reduced by 50%
