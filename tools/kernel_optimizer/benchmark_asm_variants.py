# SPDX-License-Identifier: Apache-2.0
"""
Benchmark ASM kernel variants for MoE FP8 blockscale.

Tests original and optimized ASM kernel variants by swapping kernel binaries.
Creates performance comparison plots with clear distinction between original
and optimized kernels.

Includes p5/p50/p95 percentile tracking to understand variance.
"""

import argparse
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Enable AITER
os.environ['VLLM_ROCM_USE_AITER'] = '1'

AITER_PATH = "/workspace/dev/aiter_long_context2"
KERNEL_DIR = f"{AITER_PATH}/hsa/gfx942/fmoe/silu"
# AITER uses the ps (persistent scheduling) variant by default
ORIGINAL_KERNEL = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"

FP8_DTYPE = torch.float8_e4m3fnuz


def run_single_benchmark(batch_size: int, num_warmup: int = 10, 
                         num_iters: int = 50) -> Dict[str, float]:
    """Run benchmark with current kernel configuration.
    
    Returns dict with keys: mean, p5, p50, p95, std, min, max
    """
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
    
    # Benchmark with timing
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
    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]  # μs
    
    times_arr = np.array(times)
    return {
        'mean': float(np.mean(times_arr)),
        'p5': float(np.percentile(times_arr, 5)),
        'p50': float(np.percentile(times_arr, 50)),
        'p95': float(np.percentile(times_arr, 95)),
        'std': float(np.std(times_arr)),
        'min': float(np.min(times_arr)),
        'max': float(np.max(times_arr)),
        'cv': float(np.std(times_arr) / np.mean(times_arr) * 100),  # Coefficient of variation %
    }


def benchmark_kernel_variant(kernel_file: str, batch_sizes: list, 
                            num_warmup: int, num_iters: int) -> Dict[int, Dict]:
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
                stats = run_single_benchmark(bs, num_warmup, num_iters)
                results[bs] = stats
            except Exception as e:
                print(f"  Error at batch {bs}: {e}")
                results[bs] = None
    finally:
        # Restore original
        if os.path.exists(backup_path):
            shutil.copy(backup_path, original_path)
    
    return results


def create_plots(all_results: dict, output_dir: str):
    """Create performance comparison plots with variance visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    batch_sizes = sorted(list(all_results.values())[0].keys())
    kernel_names = list(all_results.keys())
    
    # Categorize kernels
    original_kernels = [k for k in kernel_names if 'opt_' not in k]
    optimized_kernels = [k for k in kernel_names if 'opt_' in k]
    
    # Color palette
    COLORS = {
        'original': '#1e88e5',
        'vmcnt_zero': '#e53935',
        'vmcnt_reduce25': '#43a047',
        'vmcnt_cap4': '#8e24aa',
        'vmcnt_cap8': '#fb8c00',
        'both_reduce50': '#00acc1',
    }
    
    # =========================================================================
    # Plot 1: Performance with Error Bars (p5-p95 range)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 9))
    
    for kernel in kernel_names:
        times = [all_results[kernel].get(bs, {}).get('mean') for bs in batch_sizes]
        p5 = [all_results[kernel].get(bs, {}).get('p5') for bs in batch_sizes]
        p95 = [all_results[kernel].get(bs, {}).get('p95') for bs in batch_sizes]
        
        short_name = kernel.replace('opt_', '') if 'opt_' in kernel else kernel
        color = COLORS.get(short_name, '#757575')
        
        is_opt = 'opt_' in kernel
        linestyle = '--' if is_opt else '-'
        marker = 's' if is_opt else 'o'
        label = f'{short_name} {"(opt)" if is_opt else "(baseline)"}'
        
        # Plot with error bars showing p5-p95 range
        ax.errorbar(batch_sizes, times, 
                   yerr=[np.array(times) - np.array(p5), np.array(p95) - np.array(times)],
                   label=label, color=color, linestyle=linestyle, marker=marker,
                   linewidth=2.5, markersize=8, capsize=5, capthick=2, alpha=0.85)
    
    ax.set_xlabel('Batch Size (tokens)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency (μs)', fontsize=14, fontweight='bold')
    ax.set_title('MoE Kernel Performance with Variance (p5-p95)\n'
                 'DeepSeek R1 TP8, FP8 blockscale [128,128]', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kernel_performance_errorbars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/kernel_performance_errorbars.png")
    
    # =========================================================================
    # Plot 2: Coefficient of Variation Heatmap
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cv_matrix = []
    kernel_labels = []
    
    for kernel in kernel_names:
        short_name = kernel.replace('opt_', '') if 'opt_' in kernel else kernel
        kernel_labels.append(short_name)
        row = []
        for bs in batch_sizes:
            stats = all_results[kernel].get(bs, {})
            cv = stats.get('cv', 0) if stats else 0
            row.append(cv)
        cv_matrix.append(row)
    
    cv_array = np.array(cv_matrix)
    
    im = ax.imshow(cv_array, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10)
    
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=10)
    ax.set_yticks(range(len(kernel_labels)))
    ax.set_yticklabels(kernel_labels, fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Coefficient of Variation (%)', fontsize=11)
    
    # Add annotations
    for i in range(len(kernel_labels)):
        for j in range(len(batch_sizes)):
            value = cv_array[i, j]
            color = 'white' if value > 5 else 'black'
            ax.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                   color=color, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Kernel Variant', fontsize=12, fontweight='bold')
    ax.set_title('Run-to-Run Variance (Coefficient of Variation)\n'
                 'Lower is more stable', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/variance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/variance_heatmap.png")
    
    # =========================================================================
    # Plot 3: Speedup Heatmap with reliable comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    baseline = all_results['original']
    speedup_matrix = []
    kernel_labels = []
    
    for kernel in optimized_kernels:
        short_name = kernel.replace('opt_', '')
        kernel_labels.append(short_name)
        row = []
        for bs in batch_sizes:
            opt_stats = all_results[kernel].get(bs, {})
            base_stats = baseline.get(bs, {})
            if opt_stats and base_stats:
                # Use median for more robust comparison
                speedup = (base_stats['p50'] / opt_stats['p50'] - 1) * 100
                row.append(speedup)
            else:
                row.append(0)
        speedup_matrix.append(row)
    
    speedup_array = np.array(speedup_matrix)
    
    im = ax.imshow(speedup_array, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
    
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=10)
    ax.set_yticks(range(len(kernel_labels)))
    ax.set_yticklabels(kernel_labels, fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Improvement vs Original (%) [using p50]', fontsize=11)
    
    for i in range(len(kernel_labels)):
        for j in range(len(batch_sizes)):
            value = speedup_array[i, j]
            color = 'white' if abs(value) > 2.5 else 'black'
            ax.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                   color=color, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimization Strategy', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Heatmap (Median-based for stability)\n'
                 'Green = faster, Red = slower', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_heatmap_p50.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/speedup_heatmap_p50.png")
    
    # =========================================================================
    # Plot 4: Box plot of variance by batch size (for selected kernels)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    selected_batches = [bs for bs in [1024, 4096, 8192, 12000, 16000, 24000] if bs in batch_sizes][:6]
    
    for ax, bs in zip(axes, selected_batches):
        data_for_box = []
        labels = []
        for kernel in ['original', 'opt_both_reduce50', 'opt_vmcnt_cap8']:
            if kernel in all_results:
                stats = all_results[kernel].get(bs, {})
                if stats:
                    # Create synthetic data for boxplot from percentiles
                    data_for_box.append([stats['p5'], stats['p50'], stats['p95']])
                    labels.append(kernel.replace('opt_', ''))
        
        positions = range(len(labels))
        bp = ax.boxplot([[d[0], d[1], d[1], d[2]] for d in data_for_box], 
                        positions=positions, widths=0.6, patch_artist=True)
        
        colors = ['#1e88e5', '#00acc1', '#fb8c00'][:len(labels)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Latency (μs)', fontsize=10)
        ax.set_title(f'Batch {bs}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Latency Distribution (p5, p50, p95) by Batch Size', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/variance_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/variance_boxplots.png")


def main():
    parser = argparse.ArgumentParser(description='Benchmark ASM kernel variants with variance analysis')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[512, 1024, 2048, 4096, 8192, 12000, 16000, 20000, 24000])
    parser.add_argument('--num-warmup', type=int, default=15)
    parser.add_argument('--num-iters', type=int, default=100,
                        help='More iterations for better percentile accuracy')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results')
    args = parser.parse_args()
    
    print("=" * 90)
    print("ASM Kernel Variants Benchmark (with Variance Analysis)")
    print("=" * 90)
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Warmup: {args.num_warmup}, Iterations: {args.num_iters}")
    print("=" * 90)
    
    # Kernel variants to test
    kernel_variants = {
        'original': ORIGINAL_KERNEL,
        'opt_vmcnt_zero': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_zero.co",
        'opt_vmcnt_reduce25': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_reduce25.co",
        'opt_vmcnt_cap4': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_cap4.co",
        'opt_vmcnt_cap8': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_cap8.co",
        'opt_both_reduce50': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_both_reduce50.co",
    }
    
    all_results = {}
    
    for name, kernel_file in kernel_variants.items():
        print(f"\n=== Benchmarking: {name} ===")
        
        kernel_path = f"{KERNEL_DIR}/{kernel_file}"
        if not os.path.exists(kernel_path):
            print(f"  Kernel not found: {kernel_path}")
            continue
        
        results = benchmark_kernel_variant(
            kernel_file, args.batch_sizes,
            args.num_warmup, args.num_iters
        )
        all_results[name] = results
        
        # Print results with percentiles
        for bs, stats in sorted(results.items()):
            if stats:
                print(f"  Batch {bs:>6}: mean={stats['mean']:>8.2f} μs, "
                      f"p5={stats['p5']:>8.2f}, p50={stats['p50']:>8.2f}, "
                      f"p95={stats['p95']:>8.2f}, CV={stats['cv']:.1f}%")
            else:
                print(f"  Batch {bs:>6}: ERROR")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'{args.output_dir}/benchmark_results_detailed_{timestamp}.json'
    
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
    
    # Print comprehensive summary table
    print("\n" + "=" * 130)
    print("SUMMARY TABLE with PERCENTILES (μs)")
    print("=" * 130)
    
    header = f"{'Batch':>8} | {'Kernel':>20} | {'Mean':>10} | {'P5':>10} | {'P50':>10} | {'P95':>10} | {'CV%':>6}"
    print(header)
    print("-" * len(header))
    
    for bs in args.batch_sizes:
        for kernel in all_results:
            stats = all_results[kernel].get(bs)
            if stats:
                print(f"{bs:>8} | {kernel:>20} | {stats['mean']:>10.2f} | "
                      f"{stats['p5']:>10.2f} | {stats['p50']:>10.2f} | "
                      f"{stats['p95']:>10.2f} | {stats['cv']:>5.1f}%")
        print("-" * len(header))
    
    # Speedup summary using P50 (more stable)
    if 'original' in all_results:
        print("\n" + "=" * 100)
        print("SPEEDUP vs ORIGINAL (using P50 for stability)")
        print("=" * 100)
        header = f"{'Batch':>8}"
        for kernel in all_results:
            if kernel != 'original':
                short = kernel.replace('opt_', '')[:15]
                header += f" | {short:>12}"
        print(header)
        print("-" * len(header))
        
        for bs in args.batch_sizes:
            row = f"{bs:>8}"
            baseline = all_results['original'].get(bs, {}).get('p50')
            for kernel in all_results:
                if kernel != 'original':
                    opt = all_results[kernel].get(bs, {}).get('p50')
                    if opt and baseline:
                        speedup = baseline / opt
                        row += f" | {speedup:>11.3f}x"
                    else:
                        row += f" | {'N/A':>12}"
            print(row)
        
        print("=" * 100)
    
    # Check for high variance and warn
    print("\n" + "=" * 60)
    print("VARIANCE CHECK")
    print("=" * 60)
    high_variance = []
    for kernel in all_results:
        for bs, stats in all_results[kernel].items():
            if stats and stats['cv'] > 3:
                high_variance.append((kernel, bs, stats['cv']))
    
    if high_variance:
        print("⚠️  HIGH VARIANCE DETECTED (CV > 3%):")
        for kernel, bs, cv in sorted(high_variance, key=lambda x: -x[2])[:10]:
            print(f"   {kernel} @ batch {bs}: CV = {cv:.1f}%")
    else:
        print("✅ All measurements have acceptable variance (CV < 3%)")


if __name__ == '__main__':
    main()
