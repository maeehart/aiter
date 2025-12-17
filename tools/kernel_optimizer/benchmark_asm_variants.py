# SPDX-License-Identifier: Apache-2.0
"""
Benchmark ASM kernel variants for MoE FP8 blockscale.

Tests original and optimized ASM kernel variants by swapping kernel binaries.
Creates performance comparison plots.
"""

import argparse
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Enable AITER
os.environ['VLLM_ROCM_USE_AITER'] = '1'

AITER_PATH = "/workspace/dev/aiter_long_context2"
KERNEL_DIR = f"{AITER_PATH}/hsa/gfx942/fmoe/silu"
# AITER uses the ps (persistent scheduling) variant by default
ORIGINAL_KERNEL = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"

FP8_DTYPE = torch.float8_e4m3fnuz


def run_single_benchmark(batch_size: int, num_warmup: int = 10, num_iters: int = 50) -> float:
    """Run benchmark with current kernel configuration."""
    from aiter.fused_moe import fused_moe, QuantType, ActivationType
    
    num_experts = 256
    hidden_size = 7168
    intermediate_size = 256
    topk = 8
    device = "cuda"
    
    shard_intermediate_size = intermediate_size * 2
    
    # Create inputs
    hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
    
    # FP8 weights
    w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, 
                    device=device, dtype=torch.bfloat16).to(FP8_DTYPE)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, 
                    device=device, dtype=torch.bfloat16).to(FP8_DTYPE)
    
    # Scales
    scale_shape_w1 = (num_experts, shard_intermediate_size // 128, hidden_size // 128)
    scale_shape_w2 = (num_experts, hidden_size // 128, intermediate_size // 128)
    w1_scale = torch.ones(scale_shape_w1, device=device, dtype=torch.float32) * 0.1
    w2_scale = torch.ones(scale_shape_w2, device=device, dtype=torch.float32) * 0.1
    
    # Random routing
    topk_weights = torch.rand(batch_size, topk, device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(0, num_experts, (batch_size, topk), device=device, dtype=torch.int32)
    
    # Warmup
    for _ in range(num_warmup):
        _ = fused_moe(
            hidden_states, w1, w2,
            topk_weights, topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x128,
            w1_scale=w1_scale, w2_scale=w2_scale,
        )
        torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    
    for i in range(num_iters):
        start_events[i].record()
        _ = fused_moe(
            hidden_states, w1, w2,
            topk_weights, topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x128,
            w1_scale=w1_scale, w2_scale=w2_scale,
        )
        end_events[i].record()
    
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    
    return sum(times) / len(times)


def benchmark_kernel_variant(kernel_file: str, batch_sizes: list, 
                            num_warmup: int, num_iters: int) -> dict:
    """Benchmark a kernel variant by temporarily replacing the original."""
    original_path = f"{KERNEL_DIR}/{ORIGINAL_KERNEL}"
    variant_path = f"{KERNEL_DIR}/{kernel_file}"
    backup_path = f"{original_path}.backup"
    
    results = {}
    
    # Backup original
    if not os.path.exists(backup_path):
        shutil.copy(original_path, backup_path)
    
    # Replace with variant (if not original)
    if kernel_file != ORIGINAL_KERNEL:
        shutil.copy(variant_path, original_path)
    
    try:
        for bs in batch_sizes:
            try:
                avg_us = run_single_benchmark(bs, num_warmup, num_iters)
                results[bs] = avg_us
            except Exception as e:
                print(f"  Error at batch {bs}: {e}")
                results[bs] = None
    finally:
        # Restore original
        if os.path.exists(backup_path):
            shutil.copy(backup_path, original_path)
    
    return results


def create_plots(all_results: dict, output_dir: str):
    """Create performance comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    batch_sizes = sorted(list(all_results.values())[0].keys())
    kernel_names = list(all_results.keys())
    
    # Color scheme
    colors = {
        'original': '#3498db',
        'vmcnt_zero': '#e74c3c',
        'vmcnt_reduce25': '#2ecc71',
        'vmcnt_cap4': '#9b59b6',
        'vmcnt_cap8': '#f39c12',
    }
    
    # Plot 1: Performance vs Batch Size
    plt.figure(figsize=(12, 7))
    for kernel in kernel_names:
        times = [all_results[kernel].get(bs) for bs in batch_sizes]
        color = colors.get(kernel, '#95a5a6')
        linestyle = '-' if kernel == 'original' else '--'
        linewidth = 3 if kernel == 'original' else 2
        marker = 'o' if kernel == 'original' else 's'
        plt.plot(batch_sizes, times, marker=marker, linestyle=linestyle, 
                linewidth=linewidth, label=kernel, color=color, markersize=8)
    
    plt.xlabel('Batch Size (tokens)', fontsize=12)
    plt.ylabel('Latency (μs)', fontsize=12)
    plt.title('MoE Kernel Performance: Original vs Optimized ASM Kernels\n(DeepSeek R1 TP8, FP8 blockscale [128,128])', fontsize=14)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kernel_performance.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/kernel_performance.png")
    
    # Plot 2: Speedup vs Original
    plt.figure(figsize=(12, 7))
    baseline = all_results['original']
    
    for kernel in kernel_names:
        if kernel == 'original':
            continue
        speedups = []
        for bs in batch_sizes:
            if all_results[kernel].get(bs) and baseline.get(bs):
                speedups.append(baseline[bs] / all_results[kernel][bs])
            else:
                speedups.append(1.0)
        
        color = colors.get(kernel, '#95a5a6')
        plt.plot(batch_sizes, speedups, marker='s', linestyle='-', 
                linewidth=2.5, label=kernel, color=color, markersize=10)
    
    plt.axhline(y=1.0, color='#3498db', linestyle='--', linewidth=2, 
                label='original (baseline)', alpha=0.7)
    plt.xlabel('Batch Size (tokens)', fontsize=12)
    plt.ylabel('Speedup vs Original', fontsize=12)
    plt.title('Optimized Kernel Speedup vs Original ASM Kernel', fontsize=14)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kernel_speedup.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/kernel_speedup.png")
    
    # Plot 3: Bar chart at key batch sizes
    key_batches = [bs for bs in [1024, 4096, 8192, 16000, 24000] if bs in batch_sizes]
    if key_batches:
        fig, axes = plt.subplots(1, len(key_batches), figsize=(4*len(key_batches), 6))
        if len(key_batches) == 1:
            axes = [axes]
        
        for ax, bs in zip(axes, key_batches):
            kernel_names_sorted = sorted(kernel_names, 
                                        key=lambda k: all_results[k].get(bs, float('inf')))
            times = [all_results[k].get(bs, 0) for k in kernel_names_sorted]
            bar_colors = [colors.get(k, '#95a5a6') for k in kernel_names_sorted]
            
            bars = ax.barh(kernel_names_sorted, times, color=bar_colors)
            ax.set_xlabel('Latency (μs)')
            ax.set_title(f'Batch {bs}')
            ax.invert_yaxis()
            
            # Add value labels
            for bar, time in zip(bars, times):
                if time:
                    ax.text(time + max(t for t in times if t)*0.02, 
                           bar.get_y() + bar.get_height()/2,
                           f'{time:.0f}', va='center', fontsize=9)
        
        plt.suptitle('Kernel Performance at Key Batch Sizes', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/kernel_bars.png', dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/kernel_bars.png")


def main():
    parser = argparse.ArgumentParser(description='Benchmark ASM kernel variants')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[512, 1024, 2048, 4096, 8192, 12000, 16000, 20000, 24000])
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--num-iters', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='./benchmark_results')
    args = parser.parse_args()
    
    print("=" * 80)
    print("ASM Kernel Variants Benchmark")
    print("=" * 80)
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Warmup: {args.num_warmup}, Iterations: {args.num_iters}")
    print("=" * 80)
    
    # Kernel variants to test
    kernel_variants = {
        'original': ORIGINAL_KERNEL,
        'vmcnt_zero': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_zero.co",
        'vmcnt_reduce25': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_reduce25.co",
        'vmcnt_cap4': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_cap4.co",
        'vmcnt_cap8': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_cap8.co",
    }
    
    all_results = {}
    
    for name, kernel_file in kernel_variants.items():
        print(f"\n=== Benchmarking: {name} ===")
        
        # Check if kernel file exists
        kernel_path = f"{KERNEL_DIR}/{kernel_file}"
        if not os.path.exists(kernel_path):
            print(f"  Kernel not found: {kernel_path}")
            continue
        
        results = benchmark_kernel_variant(
            kernel_file, args.batch_sizes,
            args.num_warmup, args.num_iters
        )
        all_results[name] = results
        
        # Print results
        for bs, time in sorted(results.items()):
            if time:
                print(f"  Batch {bs:>6}: {time:>8.2f} μs")
            else:
                print(f"  Batch {bs:>6}: ERROR")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'{args.output_dir}/benchmark_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'model': 'DeepSeek R1 TP8',
                'quantization': 'FP8 blockscale [128,128]',
                'batch_sizes': args.batch_sizes,
                'warmup': args.num_warmup,
                'iterations': args.num_iters,
            },
            'results': {k: {str(bs): v for bs, v in r.items()} 
                       for k, r in all_results.items()},
            'timestamp': timestamp,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create plots
    if all_results:
        create_plots(all_results, args.output_dir)
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (μs)")
    print("=" * 100)
    
    # Header
    header = f"{'Batch':>8}"
    for kernel in all_results:
        header += f" | {kernel:>15}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for bs in args.batch_sizes:
        row = f"{bs:>8}"
        for kernel in all_results:
            time = all_results[kernel].get(bs, 0)
            if time:
                row += f" | {time:>15.2f}"
            else:
                row += f" | {'ERROR':>15}"
        print(row)
    
    # Speedup summary
    if 'original' in all_results:
        print("\n" + "=" * 100)
        print("SPEEDUP vs ORIGINAL")
        print("=" * 100)
        header = f"{'Batch':>8}"
        for kernel in all_results:
            if kernel != 'original':
                header += f" | {kernel:>15}"
        print(header)
        print("-" * len(header))
        
        for bs in args.batch_sizes:
            row = f"{bs:>8}"
            baseline = all_results['original'].get(bs)
            for kernel in all_results:
                if kernel != 'original':
                    time = all_results[kernel].get(bs)
                    if time and baseline:
                        speedup = baseline / time
                        row += f" | {speedup:>14.3f}x"
                    else:
                        row += f" | {'N/A':>15}"
            print(row)
    
    print("=" * 100)


if __name__ == '__main__':
    main()
