# MoE ASM Kernel Optimizer

This directory contains tools for optimizing AMDGPU MoE (Mixture of Experts) kernels through binary patching.

## Overview

The ASM kernels in AITER are pre-compiled binaries that cannot be easily modified at the source level. This toolset enables **binary-level optimization** by:

1. Analyzing kernel binaries to identify optimization points
2. Applying targeted patches to improve performance
3. Benchmarking variants to find optimal configurations

## Key Finding: Cache Thrashing at High Batch Sizes

For large batch sizes (8K-24K tokens), we discovered that **reducing memory pipeline depth (`vmcnt`)** improves performance by reducing cache pressure:

| Batch Size | Best Strategy | Speedup |
|------------|---------------|---------|
| 8192       | vmcnt_reduce25 | 1.1% |
| 16000      | vmcnt_reduce25 | **2.7%** |
| 24000      | vmcnt_reduce25 | **2.6%** |

The `vmcnt` (vector memory count) instruction controls how many outstanding memory operations are allowed before synchronization. Lower values reduce parallelism but improve cache locality.

## Tools

### 1. `binary_patch_optimizer.py`

Black-box optimization of kernel binaries by modifying `s_waitcnt` instructions.

```bash
# Analyze kernel waitcnt instructions
python binary_patch_optimizer.py --analyze-only

# Run optimization for 8K batch size
python binary_patch_optimizer.py --batch-size 8192

# Run targeted optimization
python binary_patch_optimizer.py --targeted --batch-size 24000

# Restore original kernel
python binary_patch_optimizer.py --restore
```

### 2. `create_optimized_kernels.py`

Creates optimized kernel variants with different `vmcnt` strategies:

- `vmcnt_zero`: All vmcnt set to 0 (maximum synchronization)
- `vmcnt_reduce25`: vmcnt reduced by 25%
- `vmcnt_cap4`: vmcnt capped at 4
- `vmcnt_cap8`: vmcnt capped at 8

```bash
python create_optimized_kernels.py
```

### 3. `benchmark_asm_variants.py`

Benchmarks all kernel variants and generates comparison plots.

```bash
python benchmark_asm_variants.py \
    --batch-sizes 1024 2048 4096 8192 12000 16000 24000 \
    --output-dir ./benchmark_results
```

### 4. `annotate_asm.py`

Adds human-readable comments to disassembled kernel code.

```bash
python annotate_asm.py input.s output_annotated.s
```

## Optimization Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `vmcnt_zero` | Full memory synchronization | Maximum cache pressure reduction |
| `vmcnt_reduce25` | 25% reduction | Large batches (16K-24K) |
| `vmcnt_cap4` | Cap outstanding ops at 4 | Moderate batches |
| `vmcnt_cap8` | Cap outstanding ops at 8 | Balanced approach |

## Technical Details

### s_waitcnt Instruction Encoding

The `s_waitcnt` instruction (opcode `0xBF8C`) controls memory pipeline synchronization:

- `vmcnt[5:0]`: Vector memory operations to wait for
- `lgkmcnt[3:0]`: LDS/GDS/scalar memory operations
- `expcnt[2:0]`: Export operations

Higher values = more parallelism, more cache pressure
Lower values = less parallelism, better cache locality

### Files Modified

Optimized kernels are created in:
```
hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_*_opt_*.co
```

## Benchmark Results

See `benchmark_results/` for:
- `kernel_performance.png`: Latency vs batch size
- `kernel_speedup.png`: Speedup relative to original
- `kernel_bars.png`: Bar chart at key batch sizes
- `benchmark_results_*.json`: Raw data

## Configuration

These optimizations target:
- **Model**: DeepSeek R1 (TP8)
- **Quantization**: FP8 with blockscale [128, 128]
- **Experts**: 256
- **Hidden Size**: 7168
- **Intermediate Size**: 256 (per TP shard)
- **TopK**: 8

## License

Apache-2.0

