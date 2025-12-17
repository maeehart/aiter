# SPDX-License-Identifier: Apache-2.0
"""
Benchmark ASM kernel variants for MoE FP8 blockscale.

Tests original and optimized ASM kernel variants by swapping kernel binaries.
Creates performance comparison plots with clear distinction between original
and optimized kernels.

Includes:
- Correctness validation against original kernel
- p5/p50/p95 percentile tracking to understand variance
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


def create_test_inputs(batch_size: int, seed: int = 42):
    """Create deterministic test inputs for correctness validation."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
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
    
    # Deterministic routing
    topk_weights = torch.rand(batch_size, topk, device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(0, num_experts, (batch_size, topk), device=device, dtype=torch.int32)
    
    return {
        'hidden_states': hidden_states,
        'w1': w1,
        'w2': w2,
        'w1_scale': w1_scale,
        'w2_scale': w2_scale,
        'topk_weights': topk_weights,
        'topk_ids': topk_ids,
    }


def run_kernel_once(inputs: Dict) -> torch.Tensor:
    """Run the kernel once and return output."""
    from aiter.fused_moe import fused_moe, QuantType, ActivationType
    
    output = fused_moe(
        inputs['hidden_states'].clone(),
        inputs['w1'],
        inputs['w2'],
        inputs['topk_weights'],
        inputs['topk_ids'],
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,
        w1_scale=inputs['w1_scale'],
        w2_scale=inputs['w2_scale'],
    )
    torch.cuda.synchronize()
    return output


def validate_correctness(baseline_output: torch.Tensor, test_output: torch.Tensor,
                         atol: float = 1e-2, rtol: float = 1e-2) -> Dict:
    """
    Validate that test_output matches baseline_output within tolerance.
    
    Returns dict with:
    - max_abs_error: Maximum absolute error
    - max_rel_error: Maximum relative error  
    - mean_abs_error: Mean absolute error
    - pct_mismatch: Percentage of elements outside tolerance
    - passed: Boolean indicating if validation passed
    """
    # Compute errors
    abs_diff = torch.abs(baseline_output.float() - test_output.float())
    rel_diff = abs_diff / (torch.abs(baseline_output.float()) + 1e-8)
    
    max_abs_error = abs_diff.max().item()
    max_rel_error = rel_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    
    # Check tolerance
    within_atol = abs_diff <= atol
    within_rtol = rel_diff <= rtol
    within_tolerance = within_atol | within_rtol
    
    pct_mismatch = (1 - within_tolerance.float().mean().item()) * 100
    
    # Pass if <1% mismatch
    passed = pct_mismatch < 1.0
    
    return {
        'max_abs_error': max_abs_error,
        'max_rel_error': max_rel_error,
        'mean_abs_error': mean_abs_error,
        'pct_mismatch': pct_mismatch,
        'passed': passed,
    }


def run_correctness_test(kernel_file: str, batch_sizes: List[int], 
                        baseline_outputs: Dict[int, torch.Tensor]) -> Dict[int, Dict]:
    """Run correctness validation for a kernel variant."""
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
                # Create same inputs as baseline
                inputs = create_test_inputs(bs)
                
                # Run kernel
                output = run_kernel_once(inputs)
                
                if kernel_file == ORIGINAL_KERNEL:
                    # For original, just return the output (no comparison needed)
                    results[bs] = {
                        'max_abs_error': 0.0,
                        'max_rel_error': 0.0,
                        'mean_abs_error': 0.0,
                        'pct_mismatch': 0.0,
                        'passed': True,
                        'output': output,
                    }
                else:
                    # Compare with baseline
                    validation = validate_correctness(baseline_outputs[bs], output)
                    results[bs] = validation
                    
            except Exception as e:
                print(f"  Error at batch {bs}: {e}")
                results[bs] = {
                    'max_abs_error': float('inf'),
                    'max_rel_error': float('inf'),
                    'mean_abs_error': float('inf'),
                    'pct_mismatch': 100.0,
                    'passed': False,
                    'error': str(e),
                }
    finally:
        # Restore original
        if os.path.exists(backup_path):
            shutil.copy(backup_path, original_path)
    
    return results


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
    
    # Create inputs (random each time for performance testing)
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


def create_plots(all_results: dict, correctness_results: dict, output_dir: str):
    """Create performance comparison plots with variance visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    batch_sizes = sorted(list(all_results.values())[0].keys())
    kernel_names = list(all_results.keys())
    
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
        
        # Check if kernel passed correctness
        all_passed = all(correctness_results.get(kernel, {}).get(bs, {}).get('passed', False) 
                        for bs in batch_sizes)
        label = f'{short_name} {"(opt)" if is_opt else "(baseline)"}'
        if not all_passed and kernel != 'original':
            label += ' ⚠️'
        
        # Plot with error bars showing p5-p95 range
        ax.errorbar(batch_sizes, times, 
                   yerr=[np.array(times) - np.array(p5), np.array(p95) - np.array(times)],
                   label=label, color=color, linestyle=linestyle, marker=marker,
                   linewidth=2.5, markersize=8, capsize=5, capthick=2, alpha=0.85)
    
    ax.set_xlabel('Batch Size (tokens)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency (μs)', fontsize=14, fontweight='bold')
    ax.set_title('MoE Kernel Performance with Variance (p5-p95)\n'
                 'DeepSeek R1 TP8, FP8 blockscale [128,128]\n'
                 '⚠️ = correctness issues detected', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kernel_performance_errorbars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/kernel_performance_errorbars.png")
    
    # =========================================================================
    # Plot 2: Correctness Heatmap
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    correctness_matrix = []
    kernel_labels = []
    
    for kernel in kernel_names:
        if kernel == 'original':
            continue
        short_name = kernel.replace('opt_', '') if 'opt_' in kernel else kernel
        kernel_labels.append(short_name)
        row = []
        for bs in batch_sizes:
            corr = correctness_results.get(kernel, {}).get(bs, {})
            # Use max_abs_error for visualization
            error = corr.get('max_abs_error', float('inf'))
            if error == float('inf'):
                error = 100  # Cap for display
            row.append(min(error, 10))  # Cap at 10 for visualization
        correctness_matrix.append(row)
    
    if correctness_matrix:
        corr_array = np.array(correctness_matrix)
        
        im = ax.imshow(corr_array, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(batch_sizes)))
        ax.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=10)
        ax.set_yticks(range(len(kernel_labels)))
        ax.set_yticklabels(kernel_labels, fontsize=11)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Max Absolute Error (capped at 1.0)', fontsize=11)
        
        # Add pass/fail annotations
        for i, kernel in enumerate(kernel_labels):
            for j, bs in enumerate(batch_sizes):
                corr = correctness_results.get(f'opt_{kernel}', {}).get(bs, {})
                passed = corr.get('passed', False)
                error = corr.get('max_abs_error', float('inf'))
                
                symbol = '✓' if passed else '✗'
                color = 'green' if passed else 'red'
                ax.text(j, i, f'{symbol}\n{error:.3f}', ha='center', va='center', 
                       color=color, fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Kernel Variant', fontsize=12, fontweight='bold')
        ax.set_title('Correctness Validation vs Original Kernel\n'
                     '✓ = passed (< 1% mismatch), ✗ = failed', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correctness_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/correctness_heatmap.png")
    
    # =========================================================================
    # Plot 3: Speedup Heatmap (only for passing kernels)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    baseline = all_results['original']
    speedup_matrix = []
    kernel_labels = []
    
    optimized_kernels = [k for k in kernel_names if 'opt_' in k]
    
    for kernel in optimized_kernels:
        short_name = kernel.replace('opt_', '')
        kernel_labels.append(short_name)
        row = []
        for bs in batch_sizes:
            opt_stats = all_results[kernel].get(bs, {})
            base_stats = baseline.get(bs, {})
            corr = correctness_results.get(kernel, {}).get(bs, {})
            
            if opt_stats and base_stats and corr.get('passed', False):
                # Use median for more robust comparison
                speedup = (base_stats['p50'] / opt_stats['p50'] - 1) * 100
                row.append(speedup)
            else:
                row.append(float('nan'))  # Mark as invalid
        speedup_matrix.append(row)
    
    speedup_array = np.array(speedup_matrix)
    
    # Mask NaN values
    masked_array = np.ma.masked_invalid(speedup_array)
    
    im = ax.imshow(masked_array, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
    
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=10)
    ax.set_yticks(range(len(kernel_labels)))
    ax.set_yticklabels(kernel_labels, fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Improvement vs Original (%) [using p50]', fontsize=11)
    
    for i in range(len(kernel_labels)):
        for j in range(len(batch_sizes)):
            value = speedup_array[i, j]
            if np.isnan(value):
                ax.text(j, i, 'FAIL', ha='center', va='center', 
                       color='red', fontsize=9, fontweight='bold')
            else:
                color = 'white' if abs(value) > 2.5 else 'black'
                ax.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                       color=color, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimization Strategy', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Heatmap (only for validated kernels)\n'
                 'Green = faster, Red = slower, FAIL = correctness issue', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_heatmap_validated.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/speedup_heatmap_validated.png")


def main():
    parser = argparse.ArgumentParser(description='Benchmark ASM kernel variants with correctness validation')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[512, 1024, 2048, 4096, 8192, 12000, 16000, 20000, 24000])
    parser.add_argument('--num-warmup', type=int, default=15)
    parser.add_argument('--num-iters', type=int, default=100,
                        help='More iterations for better percentile accuracy')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results')
    parser.add_argument('--skip-correctness', action='store_true',
                        help='Skip correctness validation (faster but less safe)')
    args = parser.parse_args()
    
    print("=" * 90)
    print("ASM Kernel Variants Benchmark (with Correctness Validation)")
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
    
    # Filter to existing kernels
    existing_variants = {}
    for name, kernel_file in kernel_variants.items():
        kernel_path = f"{KERNEL_DIR}/{kernel_file}"
        if os.path.exists(kernel_path):
            existing_variants[name] = kernel_file
        else:
            print(f"⚠️  Kernel not found: {kernel_file}")
    
    # =========================================================================
    # Phase 1: Correctness Validation
    # =========================================================================
    correctness_results = {}
    baseline_outputs = {}
    
    if not args.skip_correctness:
        print("\n" + "=" * 90)
        print("PHASE 1: CORRECTNESS VALIDATION")
        print("=" * 90)
        
        # First, get baseline outputs from original kernel
        print("\n=== Getting baseline outputs from original kernel ===")
        original_results = run_correctness_test(ORIGINAL_KERNEL, args.batch_sizes, {})
        correctness_results['original'] = original_results
        
        # Extract baseline outputs
        for bs in args.batch_sizes:
            if 'output' in original_results.get(bs, {}):
                baseline_outputs[bs] = original_results[bs]['output']
                del original_results[bs]['output']  # Don't store in final results
        
        print(f"  Collected baseline outputs for {len(baseline_outputs)} batch sizes")
        
        # Validate each optimized kernel
        for name, kernel_file in existing_variants.items():
            if name == 'original':
                continue
            
            print(f"\n=== Validating: {name} ===")
            results = run_correctness_test(kernel_file, args.batch_sizes, baseline_outputs)
            correctness_results[name] = results
            
            # Print results
            all_passed = True
            for bs, corr in sorted(results.items()):
                status = "✓ PASS" if corr['passed'] else "✗ FAIL"
                all_passed = all_passed and corr['passed']
                print(f"  Batch {bs:>6}: {status} | "
                      f"max_abs={corr['max_abs_error']:.4f}, "
                      f"max_rel={corr['max_rel_error']:.4f}, "
                      f"mismatch={corr['pct_mismatch']:.2f}%")
            
            if all_passed:
                print(f"  ✓ {name} PASSED all correctness checks")
            else:
                print(f"  ✗ {name} FAILED some correctness checks!")
    else:
        print("\n⚠️  Skipping correctness validation (--skip-correctness)")
        # Assume all pass for plotting
        for name in existing_variants:
            correctness_results[name] = {
                bs: {'passed': True, 'max_abs_error': 0, 'max_rel_error': 0, 
                     'mean_abs_error': 0, 'pct_mismatch': 0}
                for bs in args.batch_sizes
            }
    
    # =========================================================================
    # Phase 2: Performance Benchmarking
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHASE 2: PERFORMANCE BENCHMARKING")
    print("=" * 90)
    
    all_results = {}
    
    for name, kernel_file in existing_variants.items():
        print(f"\n=== Benchmarking: {name} ===")
        
        results = benchmark_kernel_variant(
            kernel_file, args.batch_sizes,
            args.num_warmup, args.num_iters
        )
        all_results[name] = results
        
        # Print results with percentiles
        for bs, stats in sorted(results.items()):
            if stats:
                corr = correctness_results.get(name, {}).get(bs, {})
                corr_status = "✓" if corr.get('passed', True) else "✗"
                print(f"  {corr_status} Batch {bs:>6}: mean={stats['mean']:>8.2f} μs, "
                      f"p5={stats['p5']:>8.2f}, p50={stats['p50']:>8.2f}, "
                      f"p95={stats['p95']:>8.2f}, CV={stats['cv']:.1f}%")
            else:
                print(f"  Batch {bs:>6}: ERROR")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'{args.output_dir}/benchmark_results_validated_{timestamp}.json'
    
    # Serialize correctness results (remove non-serializable outputs)
    serializable_correctness = {}
    for k, v in correctness_results.items():
        serializable_correctness[k] = {}
        for bs, data in v.items():
            serializable_correctness[k][str(bs)] = {
                key: val for key, val in data.items() 
                if key != 'output' and not isinstance(val, torch.Tensor)
            }
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'model': 'DeepSeek R1 TP8',
                'quantization': 'FP8 blockscale [128,128]',
                'batch_sizes': args.batch_sizes,
                'warmup': args.num_warmup,
                'iterations': args.num_iters,
            },
            'correctness': serializable_correctness,
            'performance': {k: {str(bs): v for bs, v in r.items()} 
                          for k, r in all_results.items()},
            'timestamp': timestamp,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create plots
    if all_results:
        create_plots(all_results, correctness_results, args.output_dir)
    
    # =========================================================================
    # Summary Tables
    # =========================================================================
    print("\n" + "=" * 100)
    print("CORRECTNESS SUMMARY")
    print("=" * 100)
    
    header = f"{'Kernel':>20}"
    for bs in args.batch_sizes:
        header += f" | {bs:>7}"
    print(header)
    print("-" * len(header))
    
    for kernel in existing_variants:
        row = f"{kernel:>20}"
        for bs in args.batch_sizes:
            corr = correctness_results.get(kernel, {}).get(bs, {})
            if corr.get('passed', False):
                row += f" | {'✓':>7}"
            else:
                row += f" | {'✗':>7}"
        print(row)
    
    print("\n" + "=" * 100)
    print("SPEEDUP vs ORIGINAL (using P50, only validated kernels)")
    print("=" * 100)
    
    header = f"{'Batch':>8}"
    for kernel in existing_variants:
        if kernel != 'original':
            short = kernel.replace('opt_', '')[:12]
            header += f" | {short:>12}"
    print(header)
    print("-" * len(header))
    
    for bs in args.batch_sizes:
        row = f"{bs:>8}"
        baseline = all_results['original'].get(bs, {}).get('p50')
        for kernel in existing_variants:
            if kernel != 'original':
                opt = all_results[kernel].get(bs, {}).get('p50')
                corr = correctness_results.get(kernel, {}).get(bs, {})
                if opt and baseline and corr.get('passed', False):
                    speedup = baseline / opt
                    row += f" | {speedup:>11.3f}x"
                elif not corr.get('passed', False):
                    row += f" | {'FAIL':>12}"
                else:
                    row += f" | {'N/A':>12}"
        print(row)
    
    print("=" * 100)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    valid_speedups = []
    for kernel in existing_variants:
        if kernel == 'original':
            continue
        for bs in args.batch_sizes:
            corr = correctness_results.get(kernel, {}).get(bs, {})
            if corr.get('passed', False):
                baseline = all_results['original'].get(bs, {}).get('p50')
                opt = all_results[kernel].get(bs, {}).get('p50')
                if baseline and opt:
                    speedup = baseline / opt
                    if speedup > 1.01:  # >1% speedup
                        valid_speedups.append((kernel, bs, speedup))
    
    if valid_speedups:
        print("\n✓ Valid speedups (>1% improvement, correctness verified):")
        for kernel, bs, speedup in sorted(valid_speedups, key=lambda x: -x[2]):
            print(f"  {kernel} @ batch {bs}: {speedup:.3f}x ({(speedup-1)*100:.1f}% faster)")
    else:
        print("\n⚠️  No valid speedups found (>1% with correctness verified)")
    
    failed_kernels = set()
    for kernel in existing_variants:
        if kernel == 'original':
            continue
        for bs in args.batch_sizes:
            corr = correctness_results.get(kernel, {}).get(bs, {})
            if not corr.get('passed', False):
                failed_kernels.add(kernel)
    
    if failed_kernels:
        print(f"\n✗ Kernels with correctness issues: {', '.join(failed_kernels)}")


if __name__ == '__main__':
    main()
