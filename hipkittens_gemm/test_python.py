#!/usr/bin/env python3
"""
Test and benchmark HipKittens FP8 GEMM kernel vs hipBLASLt

Usage:
    python test_python.py
"""

import torch
import time
import os

def benchmark_hipblaslt(A, B, A_scale, B_scale, warmup=5, iters=100):
    """Benchmark using torch._scaled_mm (hipBLASLt backend)"""
    M, K = A.shape
    N = B.shape[0]
    
    # Warmup
    for _ in range(warmup):
        out = torch._scaled_mm(
            A, B.t(),
            scale_a=A_scale,
            scale_b=B_scale,
            out_dtype=torch.bfloat16
        )
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iters):
        out = torch._scaled_mm(
            A, B.t(),
            scale_a=A_scale,
            scale_b=B_scale,
            out_dtype=torch.bfloat16
        )
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    
    return out, elapsed


def benchmark_hipkittens(A, B, A_scale, B_scale, warmup=5, iters=100):
    """Benchmark HipKittens kernel"""
    from aiter import hipkittens_fp8_gemm_decode
    
    M, K = A.shape
    N = B.shape[0]
    
    # Expand scales for HipKittens (per-token, per-channel)
    A_scale_hk = A_scale.expand(M) if A_scale.numel() == 1 else A_scale
    B_scale_hk = B_scale.expand(N) if B_scale.numel() == 1 else B_scale
    
    # Warmup
    for _ in range(warmup):
        out = hipkittens_fp8_gemm_decode(A, B, A_scale_hk, B_scale_hk)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iters):
        out = hipkittens_fp8_gemm_decode(A, B, A_scale_hk, B_scale_hk)
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    
    return out, elapsed


def check_correctness(out_hk, out_ref, rtol=0.1, atol=0.1):
    """Check if outputs match within tolerance"""
    diff = (out_hk.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    close = torch.allclose(out_hk.float(), out_ref.float(), rtol=rtol, atol=atol)
    return close, max_diff, mean_diff


def main():
    print("=" * 60)
    print("HipKittens FP8 GEMM Benchmark")
    print("=" * 60)
    
    # Try to import HipKittens from AITER
    try:
        from aiter import hipkittens_fp8_gemm_decode, is_hipkittens_available
        hk_available = is_hipkittens_available()
        print(f"✓ HipKittens available: {hk_available}")
    except ImportError as e:
        hk_available = False
        print(f"✗ HipKittens not available: {e}")
    
    # LLaMA 70B decode shapes (TP=8)
    shapes = [
        (128, 1280, 8192),   # QKV projection
        (128, 8192, 1024),   # O projection
        (128, 3584, 8192),   # Gate/Up projection
        (128, 8192, 3584),   # Down projection
        (1, 1280, 8192),     # Single token QKV
        (1, 8192, 1024),     # Single token O
        (256, 1280, 8192),   # Larger batch QKV
    ]
    
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print("-" * 60)
    
    # Determine FP8 dtype based on platform
    try:
        from aiter.jit.utils.chip_info import get_gfx
        gfx = get_gfx()
        if gfx == "gfx942":
            fp8_dtype = torch.float8_e4m3fnuz
        else:
            fp8_dtype = torch.float8_e4m3fn
        print(f"GPU Architecture: {gfx}")
        print(f"FP8 dtype: {fp8_dtype}")
    except Exception:
        fp8_dtype = torch.float8_e4m3fnuz  # Default for MI300X
        print(f"FP8 dtype: {fp8_dtype} (default)")
    
    print("-" * 60)
    
    for M, N, K in shapes:
        print(f"\nShape: M={M}, N={N}, K={K}")
        flops = 2 * M * N * K
        print(f"  FLOPs: {flops / 1e9:.2f} GFLOPs")
        
        # Create FP8 tensors
        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
        
        A = A_bf16.to(fp8_dtype)
        B = B_bf16.to(fp8_dtype)
        
        # Per-tensor scales
        A_scale = torch.ones(1, dtype=torch.float32, device='cuda')
        B_scale = torch.ones(1, dtype=torch.float32, device='cuda')
        
        # Benchmark hipBLASLt
        try:
            out_ref, time_hipblaslt = benchmark_hipblaslt(A, B, A_scale, B_scale)
            tflops_hipblaslt = flops / (time_hipblaslt * 1e-3) / 1e12
            print(f"  hipBLASLt: {time_hipblaslt:.3f} ms ({tflops_hipblaslt:.2f} TFLOPs)")
        except Exception as e:
            print(f"  hipBLASLt: FAILED - {e}")
            out_ref = None
            time_hipblaslt = float('inf')
        
        # Benchmark HipKittens
        if hk_available:
            try:
                out_hk, time_hk = benchmark_hipkittens(A, B, A_scale, B_scale)
                tflops_hk = flops / (time_hk * 1e-3) / 1e12
                print(f"  HipKittens: {time_hk:.3f} ms ({tflops_hk:.2f} TFLOPs)")
                
                # Check correctness
                if out_ref is not None:
                    correct, max_diff, mean_diff = check_correctness(out_hk, out_ref)
                    status = "✓" if correct else "✗"
                    print(f"  Correctness: {status} (max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f})")
                
                # Speedup
                if time_hipblaslt != float('inf'):
                    speedup = time_hipblaslt / time_hk
                    print(f"  Speedup: {speedup:.2f}x")
                    
            except Exception as e:
                print(f"  HipKittens: FAILED - {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
