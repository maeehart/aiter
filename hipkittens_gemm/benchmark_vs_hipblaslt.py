#!/usr/bin/env python3
"""
Comprehensive benchmark: HipKittens FP8 GEMM vs hipBLASLt

This script benchmarks FP8 GEMMs for LLaMA 70B decode workloads comparing:
- hipBLASLt (torch._scaled_mm)
- HipKittens (AITER integration)

Memory-bound decode GEMMs optimization considerations:
- Small M (1-256 tokens), large N and K (model dimensions)
- Memory bandwidth is the bottleneck, not compute
- Kernel launch overhead matters for small workloads
- Vectorized memory access patterns are critical

Usage:
    python benchmark_vs_hipblaslt.py [--shapes M,N,K ...] [--iters N] [--warmup N]
"""

import torch
import time
import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Callable
import csv
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    M: int
    N: int
    K: int
    backend: str
    time_ms: float
    tflops: float
    gbps: float  # Memory bandwidth (GB/s)
    correct: Optional[bool] = None
    max_diff: Optional[float] = None


def get_theoretical_bandwidth():
    """Get theoretical memory bandwidth for the current GPU."""
    device_name = torch.cuda.get_device_name()
    
    # Memory bandwidth in GB/s (theoretical peak)
    if "MI300X" in device_name or "MI325X" in device_name:
        return 5300  # HBM3 bandwidth
    elif "MI355X" in device_name or "MI350" in device_name:
        return 6500  # HBM3E bandwidth (approximate)
    else:
        return 5000  # Conservative default


def get_fp8_dtype():
    """Get the appropriate FP8 dtype for the current GPU."""
    try:
        from aiter.jit.utils.chip_info import get_gfx
        gfx = get_gfx()
        if gfx == "gfx942":
            return torch.float8_e4m3fnuz, "gfx942"
        elif gfx == "gfx950":
            return torch.float8_e4m3fn, "gfx950"
        else:
            return torch.float8_e4m3fnuz, gfx
    except Exception:
        # Default for MI300X
        return torch.float8_e4m3fnuz, "unknown"


def calculate_memory_traffic(M: int, N: int, K: int, fp8_dtype) -> float:
    """
    Calculate memory traffic for FP8 GEMM.
    
    For memory-bound decode GEMMs:
    - Read A: M * K bytes (FP8)
    - Read B: N * K bytes (FP8) 
    - Read scales: M + N floats = (M + N) * 4 bytes
    - Write C: M * N * 2 bytes (BF16)
    
    Total bytes = M*K + N*K + (M+N)*4 + M*N*2
    
    For small M, the dominant term is N*K (weight read).
    """
    bytes_A = M * K  # FP8
    bytes_B = N * K  # FP8
    bytes_scales = (M + N) * 4  # float32 scales
    bytes_C = M * N * 2  # BF16 output
    
    return bytes_A + bytes_B + bytes_scales + bytes_C


def benchmark_hipblaslt(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    warmup: int = 10,
    iters: int = 100
) -> tuple[torch.Tensor, float]:
    """Benchmark using torch._scaled_mm (hipBLASLt backend)."""
    M, K = A.shape
    N = B.shape[0]
    
    # torch._scaled_mm expects B in [K, N] format for column-major
    # Our B is [N, K], so we pass B.t() which gives [K, N] view
    B_for_mm = B.t()
    
    # Warmup
    for _ in range(warmup):
        out = torch._scaled_mm(
            A, B_for_mm,
            scale_a=A_scale,
            scale_b=B_scale,
            out_dtype=torch.bfloat16
        )
    
    torch.cuda.synchronize()
    
    # Timed iterations
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    
    for i in range(iters):
        start_events[i].record()
        out = torch._scaled_mm(
            A, B_for_mm,
            scale_a=A_scale,
            scale_b=B_scale,
            out_dtype=torch.bfloat16
        )
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]
    # Use median to reduce variance
    times.sort()
    median_time = times[len(times) // 2]
    
    return out, median_time


def benchmark_hipkittens(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    warmup: int = 10,
    iters: int = 100
) -> tuple[torch.Tensor, float]:
    """Benchmark HipKittens kernel."""
    from aiter import hipkittens_fp8_gemm_decode
    
    M, K = A.shape
    N = B.shape[0]
    
    # Expand scales for per-token/per-channel
    A_scale_hk = A_scale.expand(M).contiguous() if A_scale.numel() == 1 else A_scale.contiguous()
    B_scale_hk = B_scale.expand(N).contiguous() if B_scale.numel() == 1 else B_scale.contiguous()
    
    # Warmup
    for _ in range(warmup):
        out = hipkittens_fp8_gemm_decode(A, B, A_scale_hk, B_scale_hk)
    
    torch.cuda.synchronize()
    
    # Timed iterations with CUDA events for accuracy
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    
    for i in range(iters):
        start_events[i].record()
        out = hipkittens_fp8_gemm_decode(A, B, A_scale_hk, B_scale_hk)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]
    times.sort()
    median_time = times[len(times) // 2]
    
    return out, median_time


def check_correctness(out_test: torch.Tensor, out_ref: torch.Tensor, rtol=0.1, atol=0.1):
    """Check if outputs match within tolerance."""
    diff = (out_test.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # For FP8, allow larger tolerance due to quantization error
    close = torch.allclose(out_test.float(), out_ref.float(), rtol=rtol, atol=atol)
    return close, max_diff, mean_diff


def run_benchmark(
    M: int, N: int, K: int,
    fp8_dtype: torch.dtype,
    warmup: int = 10,
    iters: int = 100,
    hk_available: bool = True
) -> list[BenchmarkResult]:
    """Run benchmark for a single shape."""
    results = []
    
    # Create test data
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    
    A = A_bf16.to(fp8_dtype)
    B = B_bf16.to(fp8_dtype)
    
    # Per-tensor scales (typical for decode)
    A_scale = torch.ones(1, dtype=torch.float32, device='cuda')
    B_scale = torch.ones(1, dtype=torch.float32, device='cuda')
    
    # Calculate metrics
    flops = 2 * M * N * K
    mem_bytes = calculate_memory_traffic(M, N, K, fp8_dtype)
    theoretical_bw = get_theoretical_bandwidth()
    
    # Benchmark hipBLASLt
    try:
        out_ref, time_hipblaslt = benchmark_hipblaslt(A, B, A_scale, B_scale, warmup, iters)
        tflops_hipblaslt = flops / (time_hipblaslt * 1e-3) / 1e12
        gbps_hipblaslt = mem_bytes / (time_hipblaslt * 1e-3) / 1e9
        
        results.append(BenchmarkResult(
            M=M, N=N, K=K,
            backend="hipBLASLt",
            time_ms=time_hipblaslt,
            tflops=tflops_hipblaslt,
            gbps=gbps_hipblaslt,
            correct=True,
            max_diff=0.0
        ))
    except Exception as e:
        print(f"  hipBLASLt FAILED: {e}")
        out_ref = None
        results.append(BenchmarkResult(
            M=M, N=N, K=K,
            backend="hipBLASLt",
            time_ms=float('inf'),
            tflops=0,
            gbps=0
        ))
    
    # Benchmark HipKittens
    if hk_available:
        try:
            out_hk, time_hk = benchmark_hipkittens(A, B, A_scale, B_scale, warmup, iters)
            tflops_hk = flops / (time_hk * 1e-3) / 1e12
            gbps_hk = mem_bytes / (time_hk * 1e-3) / 1e9
            
            # Check correctness against hipBLASLt
            if out_ref is not None:
                correct, max_diff, _ = check_correctness(out_hk, out_ref)
            else:
                correct, max_diff = None, None
            
            results.append(BenchmarkResult(
                M=M, N=N, K=K,
                backend="HipKittens",
                time_ms=time_hk,
                tflops=tflops_hk,
                gbps=gbps_hk,
                correct=correct,
                max_diff=max_diff
            ))
        except Exception as e:
            print(f"  HipKittens FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(BenchmarkResult(
                M=M, N=N, K=K,
                backend="HipKittens",
                time_ms=float('inf'),
                tflops=0,
                gbps=0
            ))
    
    return results


def print_results(results: list[BenchmarkResult], theoretical_bw: float):
    """Print benchmark results in a formatted table."""
    # Group by shape
    shapes = {}
    for r in results:
        key = (r.M, r.N, r.K)
        if key not in shapes:
            shapes[key] = {}
        shapes[key][r.backend] = r
    
    print("\n" + "=" * 100)
    print(f"{'Shape':<25} {'Backend':<12} {'Time (ms)':<12} {'TFLOPs':<10} {'GB/s':<10} {'BW Eff':<10} {'Correct':<10}")
    print("=" * 100)
    
    for (M, N, K), backends in shapes.items():
        print(f"\nM={M}, N={N}, K={K}")
        flops = 2 * M * N * K
        
        for backend_name in ["hipBLASLt", "HipKittens"]:
            if backend_name in backends:
                r = backends[backend_name]
                bw_eff = f"{r.gbps / theoretical_bw * 100:.1f}%" if r.gbps > 0 else "N/A"
                correct_str = "✓" if r.correct else ("✗" if r.correct is False else "-")
                
                print(f"  {backend_name:<23} {r.time_ms:<12.3f} {r.tflops:<10.2f} {r.gbps:<10.1f} {bw_eff:<10} {correct_str:<10}")
        
        # Speedup comparison
        if "hipBLASLt" in backends and "HipKittens" in backends:
            hb = backends["hipBLASLt"]
            hk = backends["HipKittens"]
            if hb.time_ms > 0 and hk.time_ms > 0 and hk.time_ms != float('inf'):
                speedup = hb.time_ms / hk.time_ms
                print(f"  Speedup: {speedup:.2f}x")
    
    print("\n" + "=" * 100)


def save_results_csv(results: list[BenchmarkResult], filename: str):
    """Save results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['M', 'N', 'K', 'Backend', 'Time_ms', 'TFLOPs', 'GB/s', 'Correct', 'Max_Diff'])
        for r in results:
            writer.writerow([r.M, r.N, r.K, r.backend, r.time_ms, r.tflops, r.gbps, r.correct, r.max_diff])
    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark FP8 GEMM: HipKittens vs hipBLASLt')
    parser.add_argument('--shapes', nargs='+', type=str, default=None,
                        help='Shapes to benchmark as M,N,K (e.g., 128,1280,8192)')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    args = parser.parse_args()
    
    print("=" * 100)
    print("FP8 GEMM Benchmark: HipKittens vs hipBLASLt")
    print("=" * 100)
    
    # GPU info
    device_name = torch.cuda.get_device_name()
    fp8_dtype, arch = get_fp8_dtype()
    theoretical_bw = get_theoretical_bandwidth()
    
    print(f"\nDevice: {device_name}")
    print(f"Architecture: {arch}")
    print(f"FP8 dtype: {fp8_dtype}")
    print(f"Theoretical memory bandwidth: {theoretical_bw} GB/s")
    
    # Check HipKittens availability
    try:
        from aiter import hipkittens_fp8_gemm_decode, is_hipkittens_available
        hk_available = is_hipkittens_available()
        print(f"HipKittens available: {hk_available}")
    except ImportError as e:
        hk_available = False
        print(f"HipKittens not available: {e}")
    
    # Define shapes
    if args.shapes:
        shapes = [tuple(map(int, s.split(','))) for s in args.shapes]
    else:
        # LLaMA 70B decode shapes (TP=8)
        shapes = [
            # Single token (most memory-bound)
            (1, 1280, 8192),    # QKV projection
            (1, 8192, 1024),    # O projection
            (1, 3584, 8192),    # Gate/Up projection
            (1, 8192, 3584),    # Down projection
            
            # Small batch (typical decode)
            (8, 1280, 8192),
            (8, 8192, 1024),
            (8, 3584, 8192),
            (8, 8192, 3584),
            
            # Medium batch
            (32, 1280, 8192),
            (32, 8192, 1024),
            (32, 3584, 8192),
            (32, 8192, 3584),
            
            # Larger batch (still decode-like)
            (128, 1280, 8192),
            (128, 8192, 1024),
            (128, 3584, 8192),
            (128, 8192, 3584),
            
            # Edge of decode/prefill
            (256, 1280, 8192),
            (256, 8192, 1024),
        ]
    
    print(f"\nBenchmarking {len(shapes)} shapes...")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print("-" * 100)
    
    all_results = []
    
    for M, N, K in shapes:
        print(f"\nBenchmarking M={M}, N={N}, K={K}...")
        mem_bytes = calculate_memory_traffic(M, N, K, fp8_dtype)
        flops = 2 * M * N * K
        arithmetic_intensity = flops / mem_bytes
        print(f"  Memory traffic: {mem_bytes / 1e6:.2f} MB")
        print(f"  Arithmetic intensity: {arithmetic_intensity:.2f} FLOPs/byte")
        print(f"  (Memory-bound if AI < ~200 on MI300X)")
        
        results = run_benchmark(M, N, K, fp8_dtype, args.warmup, args.iters, hk_available)
        all_results.extend(results)
    
    # Print summary
    print_results(all_results, theoretical_bw)
    
    # Memory-boundedness analysis
    print("\n" + "=" * 100)
    print("Memory-Boundedness Analysis")
    print("=" * 100)
    print("\nFor decode GEMMs (small M), performance is limited by memory bandwidth.")
    print("Key optimization strategies for memory-bound GEMMs:")
    print("  1. Maximize memory bandwidth utilization")
    print("  2. Minimize kernel launch overhead")
    print("  3. Use vectorized loads (128-bit or 256-bit)")
    print("  4. Overlap memory access with compute")
    print("  5. Consider split-K for very small M to increase parallelism")
    print(f"\nTheoretical peak BW: {theoretical_bw} GB/s")
    
    # Calculate average efficiency
    hipblaslt_results = [r for r in all_results if r.backend == "hipBLASLt" and r.gbps > 0]
    if hipblaslt_results:
        avg_eff_hb = sum(r.gbps for r in hipblaslt_results) / len(hipblaslt_results) / theoretical_bw * 100
        print(f"Average hipBLASLt BW efficiency: {avg_eff_hb:.1f}%")
    
    hk_results = [r for r in all_results if r.backend == "HipKittens" and r.gbps > 0]
    if hk_results:
        avg_eff_hk = sum(r.gbps for r in hk_results) / len(hk_results) / theoretical_bw * 100
        print(f"Average HipKittens BW efficiency: {avg_eff_hk:.1f}%")
    
    # Save to CSV if requested
    if args.output:
        save_results_csv(all_results, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results_csv(all_results, f"benchmark_results_{arch}_{timestamp}.csv")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

