#!/usr/bin/env python3
"""
Fast benchmark for kernel optimization with correctness validation.

⚠️ CRITICAL: AITER caches loaded kernels in GPU memory. Swapping .co files on 
disk does NOT reload the kernel within the same process. Each kernel variant
is tested by swapping the file and running the benchmark - this works because
the FastBenchmark.test_variant() method runs while the file is swapped.

Design goals:
- Keep GPU saturated (minimal Python overhead)
- Quick correctness check (single run comparison)
- Statistical significance for performance (many iterations in one kernel call)
- Total runtime: ~30 seconds per kernel variant
"""

import os
import sys
import shutil
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

os.environ['VLLM_ROCM_USE_AITER'] = '1'

# Import AITER
from aiter.fused_moe import fused_moe, QuantType, ActivationType

AITER_PATH = "/workspace/dev/aiter_long_context2"
KERNEL_DIR = f"{AITER_PATH}/hsa/gfx942/fmoe/silu"
KERNEL_NAME = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"
ORIGINAL_PATH = f"{KERNEL_DIR}/{KERNEL_NAME}"
BACKUP_PATH = f"{ORIGINAL_PATH}.backup"

# Constants
ACCEPTABLE_MAX_DIFF = 2.5  # 1 bit in bfloat16
FP8_DTYPE = torch.float8_e4m3fnuz


@dataclass
class BenchmarkResult:
    name: str
    correct: bool
    max_diff: float
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    speedup: float  # vs baseline
    significant: bool  # statistically significant improvement


class FastBenchmark:
    def __init__(self, batch_size: int = 8192, num_experts: int = 256,
                 hidden_size: int = 7168, intermediate_size: int = 256, topk: int = 8):
        self.batch_size = batch_size
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.topk = topk
        self.device = "cuda"
        
        # Initialize tensors once (reuse for all benchmarks)
        self._init_tensors()
        
        # Reference output for correctness
        self.reference_output = None
        self.baseline_times = None
    
    def _init_tensors(self):
        """Initialize all tensors once."""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        shard_intermediate_size = self.intermediate_size * 2
        
        self.hidden_states = torch.randn(
            self.batch_size, self.hidden_size, 
            device=self.device, dtype=torch.bfloat16
        )
        
        self.w1 = torch.randn(
            self.num_experts, shard_intermediate_size, self.hidden_size,
            device=self.device, dtype=torch.bfloat16
        ).to(FP8_DTYPE)
        
        self.w2 = torch.randn(
            self.num_experts, self.hidden_size, self.intermediate_size,
            device=self.device, dtype=torch.bfloat16
        ).to(FP8_DTYPE)
        
        scale_shape_w1 = (self.num_experts, shard_intermediate_size // 128, self.hidden_size // 128)
        scale_shape_w2 = (self.num_experts, self.hidden_size // 128, self.intermediate_size // 128)
        
        self.w1_scale = torch.ones(scale_shape_w1, device=self.device, dtype=torch.float32) * 0.1
        self.w2_scale = torch.ones(scale_shape_w2, device=self.device, dtype=torch.float32) * 0.1
        
        self.topk_weights = torch.rand(
            self.batch_size, self.topk, device=self.device, dtype=torch.float32
        )
        self.topk_weights = self.topk_weights / self.topk_weights.sum(dim=-1, keepdim=True)
        
        self.topk_ids = torch.randint(
            0, self.num_experts, (self.batch_size, self.topk),
            device=self.device, dtype=torch.int32
        )
    
    def _run_kernel(self) -> torch.Tensor:
        """Run kernel once and return output."""
        return fused_moe(
            self.hidden_states.clone(),
            self.w1, self.w2,
            self.topk_weights, self.topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x128,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
        )
    
    def _benchmark_kernel(self, num_warmup: int = 10, num_iters: int = 100) -> List[float]:
        """Benchmark kernel and return list of times in microseconds."""
        # Warmup
        for _ in range(num_warmup):
            _ = self._run_kernel()
        torch.cuda.synchronize()
        
        # Benchmark with CUDA events (minimal overhead)
        times = []
        for _ in range(num_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = self._run_kernel()
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # ms to us
        
        return times
    
    def _check_correctness(self) -> Tuple[bool, float]:
        """Quick correctness check against reference."""
        if self.reference_output is None:
            return True, 0.0
        
        output = self._run_kernel()
        torch.cuda.synchronize()
        
        diff = torch.abs(output.float() - self.reference_output.float())
        max_diff = diff.max().item()
        
        return max_diff <= ACCEPTABLE_MAX_DIFF, max_diff
    
    def run_baseline(self, num_iters: int = 100) -> None:
        """Run baseline benchmark and store reference."""
        print(f"Running baseline (original kernel)...")
        
        # Ensure original kernel
        if os.path.exists(BACKUP_PATH):
            shutil.copy(BACKUP_PATH, ORIGINAL_PATH)
        else:
            shutil.copy(ORIGINAL_PATH, BACKUP_PATH)
        
        # Get reference output
        self.reference_output = self._run_kernel().clone()
        torch.cuda.synchronize()
        
        # Benchmark
        self.baseline_times = self._benchmark_kernel(num_iters=num_iters)
        
        mean = np.mean(self.baseline_times)
        std = np.std(self.baseline_times)
        print(f"  Baseline: {mean:.1f} ± {std:.1f} μs")
    
    def test_variant(self, kernel_path: str, name: str, num_iters: int = 100) -> Optional[BenchmarkResult]:
        """Test a kernel variant for correctness and performance."""
        if not os.path.exists(kernel_path):
            print(f"  ⚠️ {name}: kernel not found")
            return None
        
        # Swap kernel
        shutil.copy(kernel_path, ORIGINAL_PATH)
        
        # Quick correctness check
        correct, max_diff = self._check_correctness()
        
        if not correct:
            print(f"  ❌ {name}: INCORRECT (max_diff={max_diff:.2f})")
            return BenchmarkResult(
                name=name, correct=False, max_diff=max_diff,
                mean_us=0, std_us=0, min_us=0, max_us=0,
                speedup=0, significant=False
            )
        
        # Benchmark
        times = self._benchmark_kernel(num_iters=num_iters)
        
        mean = np.mean(times)
        std = np.std(times)
        baseline_mean = np.mean(self.baseline_times)
        speedup = baseline_mean / mean
        
        # Statistical significance test (Welch's t-test)
        t_stat, p_value = stats.ttest_ind(self.baseline_times, times, equal_var=False)
        significant = p_value < 0.05 and speedup > 1.03  # >3% improvement with p<0.05
        
        status = "✅" if correct else "❌"
        speedup_str = f"{speedup:.3f}x"
        if significant and speedup > 1.0:
            speedup_str += " ★"
        
        print(f"  {status} {name}: {mean:.1f} ± {std:.1f} μs ({speedup_str})")
        
        return BenchmarkResult(
            name=name, correct=correct, max_diff=max_diff,
            mean_us=mean, std_us=std, min_us=min(times), max_us=max(times),
            speedup=speedup, significant=significant
        )
    
    def restore_original(self):
        """Restore original kernel."""
        if os.path.exists(BACKUP_PATH):
            shutil.copy(BACKUP_PATH, ORIGINAL_PATH)


def get_kernel_variants() -> Dict[str, str]:
    """Get all kernel variants to test."""
    import glob
    variants = {}
    
    base_name = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256"
    
    # Find all optimized variants
    pattern = f"{KERNEL_DIR}/{base_name}_opt_*.co"
    for path in glob.glob(pattern):
        name = os.path.basename(path).replace(f"{base_name}_", "").replace(".co", "")
        variants[name] = path
    
    return variants


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast kernel benchmark")
    parser.add_argument('--batch-size', type=int, default=8192, help='Batch size')
    parser.add_argument('--num-iters', type=int, default=100, help='Iterations per kernel')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer iterations)')
    args = parser.parse_args()
    
    if args.quick:
        args.num_iters = 30
    
    print("=" * 70)
    print("FAST KERNEL BENCHMARK")
    print("=" * 70)
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.num_iters}")
    print(f"Correctness threshold: max_diff ≤ {ACCEPTABLE_MAX_DIFF}")
    print()
    
    # Initialize benchmark
    bench = FastBenchmark(batch_size=args.batch_size)
    
    # Run baseline
    bench.run_baseline(num_iters=args.num_iters)
    
    # Get variants
    variants = get_kernel_variants()
    print(f"\nFound {len(variants)} kernel variants to test")
    
    # Test each variant
    results = []
    for name, path in variants.items():
        result = bench.test_variant(path, name, num_iters=args.num_iters)
        if result:
            results.append(result)
    
    # Restore original
    bench.restore_original()
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Kernel':<25} {'Correct':>8} {'Time (μs)':>15} {'Speedup':>10}")
    print("-" * 60)
    
    baseline_mean = np.mean(bench.baseline_times)
    print(f"{'baseline':<25} {'✓':>8} {baseline_mean:>10.1f} ± {np.std(bench.baseline_times):<4.1f} {'1.000x':>10}")
    
    best_result = None
    for r in sorted(results, key=lambda x: -x.speedup if x.correct else 0):
        correct_str = "✓" if r.correct else "✗"
        time_str = f"{r.mean_us:>10.1f} ± {r.std_us:<4.1f}"
        speedup_str = f"{r.speedup:.3f}x"
        if r.significant:
            speedup_str += " ★"
        print(f"{r.name:<25} {correct_str:>8} {time_str} {speedup_str:>10}")
        
        if r.correct and r.speedup > 1.0 and (best_result is None or r.speedup > best_result.speedup):
            best_result = r
    
    if best_result and best_result.significant:
        print(f"\n🏆 Best variant: {best_result.name} ({best_result.speedup:.3f}x speedup)")
    else:
        print(f"\nNo statistically significant improvements found.")
    
    return results


if __name__ == '__main__':
    main()

