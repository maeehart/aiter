# FP8 GEMM Decode Kernel

Custom GCN assembly kernel optimized for LLaMA 70B decode workloads on AMD MI355X (gfx950) and MI300X (gfx942).

## Overview

This kernel targets the memory-bound decode phase of LLM inference where:
- **M is small**: Typically 128 (batch size)
- **N and K are large**: 1280-28672 (model dimensions)
- **Memory bandwidth is the bottleneck**, not compute

### Target Shapes (LLaMA 70B with TP=8)

| Operation | M | N | K | Notes |
|-----------|---|---|---|-------|
| QKV Projection | 128 | 1280 | 8192 | Main target |
| O Projection | 128 | 8192 | 1024 | |
| Gate/Up Projection | 128 | 3584 | 8192 | Each |
| Down Projection | 128 | 8192 | 28672 | |

## Building

The kernel must be compiled on a Linux system with ROCm installed:

```bash
# Make build script executable
chmod +x build.sh

# Build for both MI355X and MI300X
./build.sh all

# Or build for specific architecture
./build.sh gfx950  # MI355X only
./build.sh gfx942  # MI300X only
```

### Requirements

- ROCm 6.0+ (for FP8 support)
- LLVM/Clang with AMDGPU target

### Output

The build produces `.co` (Code Object) files:
- `fp8_gemm_decode_128x128_gfx950.co` - For MI355X
- `fp8_gemm_decode_128x128_gfx942.co` - For MI300X

## Usage

### Python API

```python
from aiter.ops.fp8_gemm_decode import fp8_gemm_decode, is_fp8_gemm_decode_available

# Check availability
if is_fp8_gemm_decode_available():
    output = fp8_gemm_decode(A, B, A_scale, B_scale)
```

### Environment Variables

The kernel is loaded via AITER's ASM kernel loader:

```bash
export AITER_ASM_DIR=/path/to/aiter/hsa/gfx950/fp8_gemm_decode/
```

## Kernel Details

### Tile Size
- **128x128x128** (M x N x K per workgroup)
- 4 waves per workgroup (256 threads)
- Uses 32KB LDS (double-buffered)

### Instructions
- **v_mfma_f32_16x16x32_fp8_fp8**: Main compute instruction
- 64 VGPRs for accumulator (4 VGPRs per 16x16 output block)

### Memory Access Pattern
- Coalesced global loads for A and B tiles
- LDS for tile staging and bank-conflict-free access
- Vectorized stores for output

### Split-K Support
For shapes where the base grid doesn't saturate CUs, split-K divides the K dimension across multiple workgroups with atomic accumulation.

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Memory Bandwidth Utilization | >85% | Peak is 6.5 TB/s on MI355X |
| MFMA Efficiency | 40-60% | Memory-bound, MFMA not the bottleneck |
| Speedup vs hipBLASLt | 1.3-1.5x | For decode shapes |

## Development Status

⚠️ **This is a work-in-progress kernel**. The current implementation is a starting point that needs:

1. **On-hardware profiling** to verify correctness
2. **Performance tuning** based on actual metrics
3. **Split-K implementation** completion
4. **Multiple tile size variants** for different shapes

## Files

| File | Description |
|------|-------------|
| `fp8_gemm_decode_128x128.s` | GCN assembly source |
| `build.sh` | Build script |
| `*.co` | Compiled code objects (generated) |

## References

- [AMD CDNA3 ISA](https://gpuopen.com/amd-cdna-3-isa/)
- [MFMA Instructions](https://rocm.docs.amd.com/en/latest/reference/rocmcc/rocmcc.html#amdgpu-mfma-instructions)
- [ROCm Assembly Guide](https://rocm.docs.amd.com/en/latest/reference/rocmcc/rocmcc.html)

