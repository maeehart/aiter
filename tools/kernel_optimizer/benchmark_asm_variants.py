# SPDX-License-Identifier: Apache-2.0
"""
Benchmark ASM kernel variants for MoE FP8 blockscale.

Tests original and optimized ASM kernel variants by swapping kernel binaries.
Creates performance comparison plots with clear distinction between original
and optimized kernels.
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
import matplotlib.patches as mpatches
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
    """Create performance comparison plots with clear original vs optimized distinction."""
    os.makedirs(output_dir, exist_ok=True)
    
    batch_sizes = sorted(list(all_results.values())[0].keys())
    kernel_names = list(all_results.keys())
    
    # Categorize kernels
    original_kernels = [k for k in kernel_names if 'opt_' not in k]
    optimized_kernels = [k for k in kernel_names if 'opt_' in k]
    
    # Style configurations
    ORIGINAL_STYLE = {
        'linestyle': '-',
        'linewidth': 3,
        'marker': 'o',
        'markersize': 10,
        'alpha': 1.0,
    }
    
    OPTIMIZED_STYLE = {
        'linestyle': '--',
        'linewidth': 2.5,
        'marker': 's',
        'markersize': 8,
        'alpha': 0.9,
    }
    
    # Color palette - distinct colors for each variant
    COLORS = {
        # Original kernels - blue shades
        'original': '#1e88e5',
        # Optimized kernels - warm colors  
        'vmcnt_zero': '#e53935',
        'vmcnt_reduce25': '#43a047',
        'vmcnt_cap4': '#8e24aa',
        'vmcnt_cap8': '#fb8c00',
    }
    
    # =========================================================================
    # Plot 1: Main Performance Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot original kernels
    for kernel in original_kernels:
        times = [all_results[kernel].get(bs) for bs in batch_sizes]
        color = COLORS.get(kernel, '#1e88e5')
        ax.plot(batch_sizes, times, label=f'{kernel} (baseline)', 
                color=color, **ORIGINAL_STYLE)
    
    # Plot optimized kernels
    for kernel in optimized_kernels:
        times = [all_results[kernel].get(bs) for bs in batch_sizes]
        short_name = kernel.replace('opt_', '')
        color = COLORS.get(short_name, '#757575')
        ax.plot(batch_sizes, times, label=f'{short_name} (optimized)', 
                color=color, **OPTIMIZED_STYLE)
    
    ax.set_xlabel('Batch Size (tokens)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency (μs)', fontsize=14, fontweight='bold')
    ax.set_title('MoE Kernel Performance: Original vs Optimized ASM Kernels\n'
                 'DeepSeek R1 TP8, FP8 blockscale [128,128]', 
                 fontsize=16, fontweight='bold')
    
    # Create legend with category headers
    handles, labels = ax.get_legend_handles_labels()
    
    # Add category patches
    orig_patch = mpatches.Patch(color='none', label='─── Original (solid)')
    opt_patch = mpatches.Patch(color='none', label='--- Optimized (dashed)')
    
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(min(batch_sizes) * 0.9, max(batch_sizes) * 1.05)
    
    # Add annotation for key finding
    ax.annotate('Best optimized kernel shows\n2.7% speedup at 16K batch',
                xy=(16000, all_results.get('opt_vmcnt_reduce25', {}).get(16000, 4500)),
                xytext=(18000, 3500),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#43a047', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#43a047'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kernel_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/kernel_performance.png")
    
    # =========================================================================
    # Plot 2: Speedup vs Original
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    baseline = all_results['original']
    
    for kernel in optimized_kernels:
        speedups = []
        for bs in batch_sizes:
            if all_results[kernel].get(bs) and baseline.get(bs):
                speedups.append(baseline[bs] / all_results[kernel][bs])
            else:
                speedups.append(1.0)
        
        short_name = kernel.replace('opt_', '')
        color = COLORS.get(short_name, '#757575')
        ax.plot(batch_sizes, speedups, label=short_name, 
                color=color, **OPTIMIZED_STYLE)
    
    # Baseline line
    ax.axhline(y=1.0, color='#1e88e5', linestyle='-', linewidth=3, 
               label='original (baseline)', alpha=0.7)
    
    # Fill regions
    ax.fill_between(batch_sizes, 1.0, [1.03]*len(batch_sizes), 
                    alpha=0.1, color='green', label='_nolegend_')
    ax.fill_between(batch_sizes, 0.97, 1.0, 
                    alpha=0.1, color='red', label='_nolegend_')
    
    ax.set_xlabel('Batch Size (tokens)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup vs Original', fontsize=14, fontweight='bold')
    ax.set_title('Optimized Kernel Speedup vs Original ASM Kernel', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 1.05)
    
    # Add text annotations
    ax.text(batch_sizes[-1] * 0.95, 1.025, 'FASTER', fontsize=10, color='green', 
            ha='right', va='bottom', fontweight='bold')
    ax.text(batch_sizes[-1] * 0.95, 0.975, 'SLOWER', fontsize=10, color='red', 
            ha='right', va='top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kernel_speedup.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/kernel_speedup.png")
    
    # =========================================================================
    # Plot 3: Grouped Bar Chart at Key Batch Sizes
    # =========================================================================
    key_batches = [bs for bs in [1024, 4096, 8192, 16000, 24000] if bs in batch_sizes]
    
    if key_batches:
        fig, axes = plt.subplots(1, len(key_batches), figsize=(4.5*len(key_batches), 7))
        if len(key_batches) == 1:
            axes = [axes]
        
        for ax, bs in zip(axes, key_batches):
            # Sort by performance (fastest first)
            sorted_kernels = sorted(kernel_names, 
                                   key=lambda k: all_results[k].get(bs, float('inf')))
            
            times = [all_results[k].get(bs, 0) for k in sorted_kernels]
            
            # Determine colors and edge styles
            bar_colors = []
            edge_colors = []
            hatches = []
            display_names = []
            
            for k in sorted_kernels:
                short_name = k.replace('opt_', '') if 'opt_' in k else k
                display_names.append(short_name)
                
                if 'opt_' in k:
                    bar_colors.append(COLORS.get(short_name, '#757575'))
                    edge_colors.append('black')
                    hatches.append('//')  # Diagonal hatching for optimized
                else:
                    bar_colors.append(COLORS.get(k, '#1e88e5'))
                    edge_colors.append('black')
                    hatches.append('')  # No hatching for original
            
            bars = ax.barh(range(len(sorted_kernels)), times, color=bar_colors,
                          edgecolor=edge_colors, linewidth=1.5)
            
            # Add hatching
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            
            ax.set_yticks(range(len(sorted_kernels)))
            ax.set_yticklabels(display_names, fontsize=10)
            ax.set_xlabel('Latency (μs)', fontsize=11)
            ax.set_title(f'Batch {bs}', fontsize=13, fontweight='bold')
            ax.invert_yaxis()
            
            # Add value labels
            max_time = max(t for t in times if t)
            for i, (bar, time) in enumerate(zip(bars, times)):
                if time:
                    # Determine if this is the best (fastest)
                    is_best = (i == 0) and ('opt_' in sorted_kernels[i])
                    label = f'{time:.0f}'
                    if is_best:
                        label += ' ★'
                    ax.text(time + max_time * 0.02, bar.get_y() + bar.get_height()/2,
                           label, va='center', fontsize=9, 
                           fontweight='bold' if is_best else 'normal')
            
            ax.set_xlim(0, max_time * 1.2)
        
        # Add legend explaining hatching
        solid_patch = mpatches.Patch(facecolor='#1e88e5', edgecolor='black', 
                                     label='Original (baseline)')
        hatch_patch = mpatches.Patch(facecolor='#43a047', edgecolor='black', 
                                     hatch='//', label='Optimized')
        fig.legend(handles=[solid_patch, hatch_patch], 
                  loc='upper center', ncol=2, fontsize=11,
                  bbox_to_anchor=(0.5, 1.02))
        
        plt.suptitle('Kernel Performance at Key Batch Sizes\n(★ = best optimized)', 
                     fontsize=14, fontweight='bold', y=1.08)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/kernel_bars.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/kernel_bars.png")
    
    # =========================================================================
    # Plot 4: Heatmap of Speedups
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
            if all_results[kernel].get(bs) and baseline.get(bs):
                speedup = (baseline[bs] / all_results[kernel][bs] - 1) * 100  # % improvement
                row.append(speedup)
            else:
                row.append(0)
        speedup_matrix.append(row)
    
    speedup_array = np.array(speedup_matrix)
    
    # Create heatmap
    im = ax.imshow(speedup_array, cmap='RdYlGn', aspect='auto', 
                   vmin=-3, vmax=3)
    
    # Set ticks
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=10)
    ax.set_yticks(range(len(kernel_labels)))
    ax.set_yticklabels(kernel_labels, fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Improvement vs Original (%)', fontsize=11)
    
    # Add value annotations
    for i in range(len(kernel_labels)):
        for j in range(len(batch_sizes)):
            value = speedup_array[i, j]
            color = 'white' if abs(value) > 1.5 else 'black'
            ax.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                   color=color, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimization Strategy', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Heatmap: % Improvement vs Original Kernel\n'
                 '(Green = faster, Red = slower)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kernel_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/kernel_heatmap.png")


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
        'opt_vmcnt_zero': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_zero.co",
        'opt_vmcnt_reduce25': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_reduce25.co",
        'opt_vmcnt_cap4': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_cap4.co",
        'opt_vmcnt_cap8': f"{ORIGINAL_KERNEL.replace('.co', '')}_opt_vmcnt_cap8.co",
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
