#!/usr/bin/env python3
"""
Correctness test for HipKittens FP8 GEMM vs reference implementation.

Tests:
1. Correctness against torch matmul reference
2. Correctness against hipBLASLt (torch._scaled_mm)
3. Various shapes (decode workloads)
4. Edge cases (small M, boundary conditions)

Usage:
    python test_correctness.py
"""

import torch
import sys


def get_fp8_dtype():
    """Get the appropriate FP8 dtype for the current GPU."""
    try:
        from aiter.jit.utils.chip_info import get_gfx
        gfx = get_gfx()
        if gfx == "gfx942":
            return torch.float8_e4m3fnuz
        elif gfx == "gfx950":
            return torch.float8_e4m3fn
        else:
            return torch.float8_e4m3fnuz
    except Exception:
        return torch.float8_e4m3fnuz


def reference_fp8_gemm(A, B, A_scale, B_scale, bias=None):
    """
    Reference implementation using torch for correctness checking.
    
    Computes: C = (A * A_scale) @ (B * B_scale).T + bias
    """
    M, K = A.shape
    N = B.shape[0]
    
    # Dequantize to float32
    A_fp32 = A.float()
    B_fp32 = B.float()
    
    # Apply scales
    if A_scale.numel() == 1:
        A_scaled = A_fp32 * A_scale.item()
    else:
        A_scaled = A_fp32 * A_scale.view(M, 1)
    
    if B_scale.numel() == 1:
        B_scaled = B_fp32 * B_scale.item()
    else:
        B_scaled = B_fp32 * B_scale.view(N, 1)
    
    # GEMM: A @ B.T
    C = torch.matmul(A_scaled, B_scaled.t())
    
    # Add bias
    if bias is not None:
        C = C + bias
    
    return C.to(torch.bfloat16)


def test_hipblaslt(A, B, A_scale, B_scale):
    """Test using torch._scaled_mm (hipBLASLt backend)."""
    try:
        out = torch._scaled_mm(
            A, B.t(),
            scale_a=A_scale,
            scale_b=B_scale,
            out_dtype=torch.bfloat16
        )
        return out, None
    except Exception as e:
        return None, str(e)


def test_hipkittens(A, B, A_scale, B_scale):
    """Test HipKittens kernel."""
    try:
        from aiter import hipkittens_fp8_gemm_decode
        
        M, K = A.shape
        N = B.shape[0]
        
        # Expand scales
        A_scale_exp = A_scale.expand(M).contiguous() if A_scale.numel() == 1 else A_scale.contiguous()
        B_scale_exp = B_scale.expand(N).contiguous() if B_scale.numel() == 1 else B_scale.contiguous()
        
        out = hipkittens_fp8_gemm_decode(A, B, A_scale_exp, B_scale_exp)
        return out, None
    except Exception as e:
        import traceback
        return None, traceback.format_exc()


def check_close(out, ref, rtol=0.15, atol=0.15):
    """Check if outputs are close within tolerance."""
    if out is None:
        return False, float('inf'), float('inf')
    
    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # FP8 has significant quantization error, use relaxed tolerance
    close = torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol)
    return close, max_diff, mean_diff


def run_test(M, N, K, fp8_dtype, test_name=""):
    """Run correctness test for a single shape."""
    print(f"\n{'='*60}")
    print(f"Test: M={M}, N={N}, K={K} {test_name}")
    print(f"{'='*60}")
    
    # Create test data with controlled values for debugging
    torch.manual_seed(42)
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.1
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') * 0.1
    
    A = A_bf16.to(fp8_dtype)
    B = B_bf16.to(fp8_dtype)
    
    # Per-tensor scales
    A_scale = torch.tensor([1.0], dtype=torch.float32, device='cuda')
    B_scale = torch.tensor([1.0], dtype=torch.float32, device='cuda')
    
    # Reference
    ref = reference_fp8_gemm(A, B, A_scale, B_scale)
    print(f"Reference output shape: {ref.shape}")
    print(f"Reference output range: [{ref.min().item():.4f}, {ref.max().item():.4f}]")
    
    results = {}
    
    # Test hipBLASLt
    out_hb, err_hb = test_hipblaslt(A, B, A_scale, B_scale)
    if out_hb is not None:
        close, max_diff, mean_diff = check_close(out_hb, ref)
        status = "✓ PASS" if close else "✗ FAIL"
        print(f"\nhipBLASLt: {status}")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        results['hipBLASLt'] = close
    else:
        print(f"\nhipBLASLt: ✗ ERROR - {err_hb}")
        results['hipBLASLt'] = False
    
    # Test HipKittens
    out_hk, err_hk = test_hipkittens(A, B, A_scale, B_scale)
    if out_hk is not None:
        close, max_diff, mean_diff = check_close(out_hk, ref)
        status = "✓ PASS" if close else "✗ FAIL"
        print(f"\nHipKittens: {status}")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        results['HipKittens'] = close
        
        # Also compare against hipBLASLt
        if out_hb is not None:
            close_hb, max_diff_hb, _ = check_close(out_hk, out_hb)
            print(f"  vs hipBLASLt: max_diff={max_diff_hb:.6f}")
    else:
        print(f"\nHipKittens: ✗ ERROR")
        print(f"  {err_hk}")
        results['HipKittens'] = False
    
    return results


def main():
    print("=" * 60)
    print("HipKittens FP8 GEMM Correctness Tests")
    print("=" * 60)
    
    # GPU info
    device_name = torch.cuda.get_device_name()
    fp8_dtype = get_fp8_dtype()
    print(f"\nDevice: {device_name}")
    print(f"FP8 dtype: {fp8_dtype}")
    
    # Check HipKittens availability
    try:
        from aiter import hipkittens_fp8_gemm_decode, is_hipkittens_available
        hk_available = is_hipkittens_available()
        print(f"HipKittens available: {hk_available}")
    except ImportError as e:
        hk_available = False
        print(f"HipKittens not available: {e}")
    
    # Test shapes
    test_cases = [
        # Basic decode shapes
        (128, 128, 128, "Small square"),
        (128, 256, 512, "Basic decode"),
        
        # LLaMA 70B shapes
        (128, 1280, 8192, "QKV projection"),
        (128, 8192, 1024, "O projection"),
        (128, 3584, 8192, "Gate/Up projection"),
        
        # Edge cases
        (1, 1280, 8192, "Single token"),
        (8, 1280, 8192, "Tiny batch"),
        (256, 1280, 8192, "Large decode batch"),
        
        # Boundary conditions
        (127, 1280, 8192, "Non-aligned M"),
        (128, 1279, 8192, "Non-aligned N"),
        (128, 1280, 8191, "Non-aligned K"),
    ]
    
    all_results = {}
    
    for M, N, K, name in test_cases:
        results = run_test(M, N, K, fp8_dtype, name)
        all_results[(M, N, K)] = results
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    hipblaslt_pass = sum(1 for r in all_results.values() if r.get('hipBLASLt', False))
    hipkittens_pass = sum(1 for r in all_results.values() if r.get('HipKittens', False))
    total = len(all_results)
    
    print(f"\nhipBLASLt: {hipblaslt_pass}/{total} tests passed")
    print(f"HipKittens: {hipkittens_pass}/{total} tests passed")
    
    # Exit code
    all_pass = hipblaslt_pass == total and (not hk_available or hipkittens_pass == total)
    if all_pass:
        print("\n✓ All tests PASSED")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

