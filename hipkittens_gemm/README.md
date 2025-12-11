# HipKittens-based FP8 GEMM for Decode Workloads

This directory contains an **optimized memory-bound** FP8 GEMM implementation using [HipKittens](https://github.com/HazyResearch/HipKittens) principles from Stanford's Hazy Research.

## Why Memory-Bound Optimization?

Decode GEMMs have:
- **Small M** (1-256 tokens): Batch size during autoregressive generation
- **Large N and K** (1024-8192+): Model dimensions

This results in **low arithmetic intensity** (FLOPs/byte), making them memory-bandwidth limited:

```
Arithmetic Intensity = 2*M*N*K / (M*K + N*K + M*N*2)

For M=128, N=8192, K=8192:
  AI ≈ 127 FLOPs/byte (memory-bound on MI300X where roofline is ~200)

For M=1, N=8192, K=8192:
  AI ≈ 2 FLOPs/byte (extremely memory-bound)
```

## Optimization Strategies

Based on [ParallelKittens](https://hazyresearch.stanford.edu/blog/2025-11-17-pk) principles:

### 1. Vectorized Memory Access
- Use 128-bit loads (16 FP8 elements at once)
- Aligned memory access patterns
- Coalesced global memory reads

### 2. Double Buffering
- Overlap memory loads with computation
- Prefetch next tile while computing current
- Hide memory latency

### 3. Split-K for Very Small M
- For M ≤ 8: Split K dimension across blocks
- Increases parallelism to saturate memory bandwidth
- Uses atomic operations for reduction

### 4. Architecture-Specific Tuning

| Architecture | GPU | Memory BW | FP8 Type | Optimizations |
|--------------|-----|-----------|----------|---------------|
| gfx942 | MI300X, MI325X | 5.3 TB/s | FP8_E4M3FNUZ | Standard tiles |
| gfx950 | MI355X | 6.5 TB/s | FP8_E4M3 | Larger K tiles |

### 5. Tile Size Selection

```
Small M (1-8):    TILE_M=16,  TILE_N=128, TILE_K=256 (maximize K)
Medium M (8-256): TILE_M=64,  TILE_N=128, TILE_K=128 (balanced)
Large M (256+):   TILE_M=128, TILE_N=128, TILE_K=64  (compute-bound)
```

## Target Shapes (LLaMA 70B TP=8)

| Layer | Shape (M, N, K) | Memory Traffic | Arithmetic Intensity |
|-------|-----------------|----------------|---------------------|
| QKV proj | (128, 1280, 8192) | 10.8 MB | 127 |
| O proj | (128, 8192, 1024) | 9.3 MB | 127 |
| Gate/Up | (128, 3584, 8192) | 30.5 MB | 127 |
| Down | (128, 8192, 3584) | 30.5 MB | 127 |

## Benchmarking

```bash
# Run benchmark comparing HipKittens vs hipBLASLt
python benchmark_vs_hipblaslt.py

# Custom shapes
python benchmark_vs_hipblaslt.py --shapes 128,1280,8192 1,8192,8192

# Save results
python benchmark_vs_hipblaslt.py --output results.csv
```

### Expected Results

For memory-bound GEMMs, target **>70% bandwidth efficiency**:

```
MI300X (5.3 TB/s theoretical):
  Target: >3.7 TB/s achieved bandwidth
  
MI355X (6.5 TB/s theoretical):
  Target: >4.5 TB/s achieved bandwidth
```

## Building

### Prerequisites
```bash
# Clone HipKittens (optional, for advanced optimizations)
git clone --recursive https://github.com/HazyResearch/HipKittens.git
export HIPKITTENS_ROOT=$(pwd)/HipKittens
```

### Build with AITER
```bash
cd /path/to/aiter
pip install -e .

# The HipKittens GEMM module builds automatically via JIT
```

### Standalone Build
```bash
cd hipkittens_gemm
make              # Auto-detect GPU
make ARCH=gfx942  # MI300X/MI325X
make ARCH=gfx950  # MI355X
```

## Usage in vLLM

Enable via environment variables:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_HIPKITTENS_FP8_GEMM=1
```

The kernel automatically activates for decode-shaped GEMMs:
- M ≤ 256
- N ≥ 1024
- K ≥ 1024

## Performance Analysis

### Memory Traffic Breakdown

For decode GEMM C = A × B^T:
- **Read A**: M × K bytes (FP8)
- **Read B**: N × K bytes (FP8) ← **Dominant for small M**
- **Read scales**: (M + N) × 4 bytes
- **Write C**: M × N × 2 bytes (BF16)

### Bandwidth Utilization

```python
# Calculate achieved bandwidth
achieved_bw_gbps = total_bytes / time_seconds / 1e9

# Efficiency
efficiency = achieved_bw_gbps / theoretical_bw_gbps
```

## References

- [HipKittens Paper (arXiv)](https://arxiv.org/abs/2511.08083)
- [HipKittens Blog: Fast and Furious AMD Kernels](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)
- [ParallelKittens: Multi-GPU Kernels](https://hazyresearch.stanford.edu/blog/2025-11-17-pk)
- [AMD GPU Optimization Guide](https://rocm.docs.amd.com/en/latest/)
