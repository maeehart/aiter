# HipKittens-based FP8 GEMM for Decode

This directory contains an FP8 GEMM implementation using [HipKittens](https://github.com/HazyResearch/HipKittens) - a high-performance AMD kernel library from Stanford's Hazy Research.

## Why HipKittens?

HipKittens provides:
- **Tile primitives**: Efficient memory access patterns optimized for AMD GPUs
- **MFMA wrappers**: Easy-to-use matrix multiply-accumulate operations
- **Scheduling patterns**: Proven 8-wave ping-pong and 4-wave interleave patterns
- **Sustainability**: Active development and community support

## Prerequisites

```bash
# Clone HipKittens
git clone --recursive https://github.com/HazyResearch/HipKittens.git
cd HipKittens
source env.src
```

## Building

```bash
cd hipkittens_gemm
make clean && make
```

## Usage

```python
from hipkittens_gemm import fp8_gemm_decode

output = fp8_gemm_decode(A, B, A_scale, B_scale)
```

## Target Shapes

Optimized for LLaMA 70B decode with TP=8:
- QKV projection: (128, 1280, 8192)
- O projection: (128, 8192, 1024)
- Gate/Up projection: (128, 3584, 8192)

## References

- [HipKittens Paper (arXiv)](https://arxiv.org/abs/2511.08083)
- [HipKittens Blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)
- [Fast and Furious AMD Kernels Blog](https://hazyresearch.stanford.edu/blog/2025-11-14-hk-part2)


