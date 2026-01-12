#!/usr/bin/env python3
"""
Benchmark 64x256 MoE kernel vs baseline.
Based on production_benchmark.py with modifications for 64x256 testing.
"""

import os
import sys
import subprocess
import torch
import numpy as np
from scipy import stats
from typing import List, Dict
import pandas as pd
import gc

os.environ['VLLM_ROCM_USE_AITER'] = '1'

sys.path.insert(0, '/workspace')

from aiter.fused_moe import moe_sorting, fused_topk
from aiter.ops.shuffle import shuffle_weight
from aiter import pertoken_quant
from aiter import dtypes
from einops import rearrange
import aiter

# Production batch sizes from roofline analysis (subset for quick testing)
PRODUCTION_BATCH_SIZES = [
    64, 128, 256, 512, 1024, 2048, 4096
]

# Full production sizes
FULL_PRODUCTION_BATCH_SIZES = [
    480, 504, 561, 585, 737, 762, 775, 
    1104, 1192, 1239, 1294, 1348, 1478, 1496, 1547, 1606, 1622,
    2064, 2072, 2571, 2907,
    4132, 4341, 4604, 4644,
    5913, 7339, 8613
]

# Kernel parameters (from DeepSeek R1)
HIDDEN_DIM = 7168
INTERMEDIATE_DIM = 256
NUM_EXPERTS = 256
TOPK = 8
SCALE_BLKS = (128, 128)

torch.set_default_device("cuda")


def run_kernel_subprocess(batch_size: int, kernel_type: str, num_warmup: int = 3, num_iters: int = 20) -> Dict:
    """Run kernel benchmark in a subprocess to avoid memory issues."""
    script = f'''
import os
import sys
import torch
import numpy as np
import gc

os.environ['VLLM_ROCM_USE_AITER'] = '1'
sys.path.insert(0, '/workspace')

from aiter.fused_moe import moe_sorting, fused_topk
from aiter.ops.shuffle import shuffle_weight
from aiter import pertoken_quant
from aiter import dtypes
from einops import rearrange
import aiter

torch.set_default_device("cuda")

batch_size = {batch_size}
kernel_type = "{kernel_type}"
num_warmup = {num_warmup}
num_iters = {num_iters}

HIDDEN_DIM = {HIDDEN_DIM}
INTERMEDIATE_DIM = {INTERMEDIATE_DIM}
NUM_EXPERTS = {NUM_EXPERTS}
TOPK = {TOPK}
scale_blk_n, scale_blk_k = {SCALE_BLKS}

# Prepare inputs
torch.manual_seed(42)
input = torch.randn((batch_size, HIDDEN_DIM), dtype=dtypes.bf16)
w1 = torch.randn((NUM_EXPERTS, INTERMEDIATE_DIM * 2, HIDDEN_DIM), dtype=dtypes.bf16) / 10
w2 = torch.randn((NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM), dtype=dtypes.bf16) / 10
score = torch.randn((batch_size, NUM_EXPERTS), dtype=dtypes.bf16)

topk_weights, topk_ids = fused_topk(input, score, TOPK, True)

quant_dtype = dtypes.fp8

# Block quant w1
tmp = rearrange(
    w1.view(-1, w1.shape[1] // scale_blk_n, scale_blk_n, w1.shape[2] // scale_blk_k, scale_blk_k),
    "e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)",
).contiguous()
w1_q, w1_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
w1_q = rearrange(
    w1_q.view(-1, w1.shape[1] // scale_blk_n, w1.shape[2] // scale_blk_k, scale_blk_n, scale_blk_k),
    "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
).contiguous()
w1_scale = w1_scale.view(NUM_EXPERTS, -1)

# Block quant w2
tmp = rearrange(
    w2.view(-1, HIDDEN_DIM // scale_blk_n, scale_blk_n, INTERMEDIATE_DIM // scale_blk_k, scale_blk_k),
    "e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)",
).contiguous()
w2_q, w2_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
w2_q = rearrange(
    w2_q.view(-1, w2.shape[1] // scale_blk_n, w2.shape[2] // scale_blk_k, scale_blk_n, scale_blk_k),
    "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
).contiguous()
w2_scale = w2_scale.view(NUM_EXPERTS, -1)

# Block quant input
a1_q, a1_scale = pertoken_quant(
    input.view(-1, HIDDEN_DIM // scale_blk_k, scale_blk_k), quant_dtype=quant_dtype
)
a1_q = a1_q.view(-1, HIDDEN_DIM)
a1_scale = a1_scale.squeeze(-1)

# Shuffle weights
w1_shuffled = shuffle_weight(w1_q, (16, 16))
w2_shuffled = shuffle_weight(w2_q, (16, 16))
a1_scale_t = a1_scale.t().contiguous()

# Determine block size and kernel name
# KEY FIX: 64x256 kernel expects 32-block sorting internally!
if kernel_type == "64x256":
    block_size = 32  # Use 32-block sorting for 64x256 kernel
    kernel_name = "_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_64x256E"
else:
    block_size = 32
    kernel_name = ""  # Use heuristic selection

# Prepare sorted data
sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out = (
    moe_sorting(topk_ids, topk_weights, NUM_EXPERTS, HIDDEN_DIM, dtypes.bf16, block_size=block_size)
)

# Warmup
for _ in range(num_warmup):
    out.zero_()
    aiter.fmoe_fp8_blockscale_g1u1(
        out, a1_q, w1_shuffled, w2_shuffled,
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, TOPK,
        a1_scale_t, w1_scale, w2_scale, kernel_name, scale_blk_n, scale_blk_k, None
    )
torch.cuda.synchronize()

# Benchmark
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

times = []
for _ in range(num_iters):
    out.zero_()
    start_event.record()
    aiter.fmoe_fp8_blockscale_g1u1(
        out, a1_q, w1_shuffled, w2_shuffled,
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, TOPK,
        a1_scale_t, w1_scale, w2_scale, kernel_name, scale_blk_n, scale_blk_k, None
    )
    end_event.record()
    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event) * 1000)  # Convert to μs

mean_us = np.mean(times)
std_us = np.std(times)
min_us = np.min(times)
max_us = np.max(times)
output_sum = out.sum().item()

print(f"RESULT:{{mean_us:.2f}},{{std_us:.2f}},{{min_us:.2f}},{{max_us:.2f}},{{output_sum:.2f}}")
'''
    
    result = subprocess.run(
        ['python3', '-c', script],
        capture_output=True,
        text=True,
        env={**os.environ, 'HIP_VISIBLE_DEVICES': '0'}
    )
    
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:] if result.stderr else 'Unknown error'}")
        return None
    
    # Parse output
    for line in result.stdout.split('\n'):
        if line.startswith('RESULT:'):
            parts = line.replace('RESULT:', '').split(',')
            return {
                'mean_us': float(parts[0]),
                'std_us': float(parts[1]),
                'min_us': float(parts[2]),
                'max_us': float(parts[3]),
                'output_sum': float(parts[4]),
            }
    
    print(f"  PARSE ERROR: {result.stdout[-300:]}")
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark 64x256 MoE kernel")
    parser.add_argument('--full', action='store_true', help='Use full production batch sizes')
    parser.add_argument('--iters', type=int, default=20, help='Number of iterations per benchmark')
    args = parser.parse_args()
    
    batch_sizes = FULL_PRODUCTION_BATCH_SIZES if args.full else PRODUCTION_BATCH_SIZES
    
    print("="*80)
    print("64x256 MoE KERNEL BENCHMARK")
    print("="*80)
    print(f"Configuration:")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Intermediate dim: {INTERMEDIATE_DIM}")
    print(f"  Experts: {NUM_EXPERTS}")
    print(f"  Top-K: {TOPK}")
    print(f"  Batch sizes: {len(batch_sizes)}")
    print(f"  Iterations: {args.iters}")
    print("="*80)
    
    results = []
    
    for kernel_type in ["baseline", "64x256"]:
        if kernel_type == "baseline":
            print("\n\n🔵 TESTING BASELINE KERNEL (32x128/32x256)")
        else:
            print("\n\n🟢 TESTING 64x256 KERNEL")
        print("-"*80)
        
        for batch_size in batch_sizes:
            result = run_kernel_subprocess(batch_size, kernel_type, num_warmup=3, num_iters=args.iters)
            
            if result:
                results.append({
                    'kernel': kernel_type,
                    'batch_size': batch_size,
                    **result
                })
                print(f"  M={batch_size:5d}: {result['mean_us']:8.2f} ± {result['std_us']:6.2f} μs (sum={result['output_sum']:.0f})")
            else:
                results.append({
                    'kernel': kernel_type,
                    'batch_size': batch_size,
                    'mean_us': None,
                    'std_us': None,
                    'min_us': None,
                    'max_us': None,
                    'output_sum': None,
                })
                print(f"  M={batch_size:5d}: ❌ FAILED")
    
    # Create comparison table
    df = pd.DataFrame(results)
    
    # Pivot to compare kernels side by side
    baseline_df = df[df['kernel'] == 'baseline'].set_index('batch_size')
    optimized_df = df[df['kernel'] == '64x256'].set_index('batch_size')
    
    comparison = pd.DataFrame({
        'M': baseline_df.index,
        'Baseline (μs)': baseline_df['mean_us'].values,
        'Baseline Std': baseline_df['std_us'].values,
        '64x256 (μs)': optimized_df['mean_us'].values,
        '64x256 Std': optimized_df['std_us'].values,
    })
    
    # Calculate speedup
    comparison['Speedup'] = comparison['Baseline (μs)'] / comparison['64x256 (μs)']
    comparison['Δ (μs)'] = comparison['Baseline (μs)'] - comparison['64x256 (μs)']
    
    print("\n\n")
    print("="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print()
    print(comparison.to_string(index=False))
    
    # Summary statistics
    print("\n")
    print("="*80)
    print("SUMMARY")
    print("="*80)
    valid_speedups = comparison['Speedup'].dropna()
    if len(valid_speedups) > 0:
        print(f"Average speedup: {valid_speedups.mean():.3f}x")
        print(f"Median speedup: {valid_speedups.median():.3f}x")
        print(f"Min speedup: {valid_speedups.min():.3f}x")
        print(f"Max speedup: {valid_speedups.max():.3f}x")
        print(f"Cases where 64x256 is faster: {sum(valid_speedups > 1.0)}/{len(valid_speedups)}")
        print(f"Cases where 64x256 is slower: {sum(valid_speedups < 1.0)}/{len(valid_speedups)}")
    else:
        print("No valid results to compare")
    
    # Save to CSV
    output_path = "/workspace/tools/kernel_optimizer/benchmark_64x256_results.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()

