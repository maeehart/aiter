# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Custom FP8 GEMM kernel optimized for LLaMA 70B decode workloads.

Target: MI355X (gfx950) and MI300X (gfx942)
Optimized for M=128 (decode batch), memory-bound scenarios.

Usage:
    from aiter.ops.fp8_gemm_decode import fp8_gemm_decode, is_fp8_gemm_decode_available
    
    if is_fp8_gemm_decode_available():
        output = fp8_gemm_decode(A, B, A_scale, B_scale, out)
"""

from typing import Tuple
import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx


def gen_fp8_gemm_decode_fake_tensors(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    split_k: int = 1,
) -> Tensor:
    """Generate fake tensors for torch.compile compatibility."""
    return out


@compile_ops(
    "module_fp8_gemm_decode_asm",
    fc_name="fp8_gemm_decode_asm",
    gen_fake=gen_fp8_gemm_decode_fake_tensors,
)
def fp8_gemm_decode_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    split_k: int = 1,
) -> Tensor:
    """
    Low-level ASM kernel call for FP8 GEMM decode.
    
    Args:
        A: [M, K] FP8 input tensor
        B: [N, K] FP8 weight tensor (transposed)
        A_scale: [M] FP32 per-token scale
        B_scale: [N] FP32 per-channel scale  
        out: [M, N] BF16 output tensor
        split_k: Split-K factor for better CU utilization
    
    Returns:
        out tensor with GEMM result
    """
    ...


def gen_is_available_fake() -> bool:
    return True


@compile_ops(
    "module_fp8_gemm_decode_asm",
    fc_name="is_fp8_gemm_decode_available",
    gen_fake=gen_is_available_fake,
)
def is_fp8_gemm_decode_available() -> bool:
    """Check if FP8 decode GEMM kernel is available for current GPU."""
    ...


def gen_split_k_fake(M: int, N: int, K: int) -> int:
    return 1


@compile_ops(
    "module_fp8_gemm_decode_asm",
    fc_name="get_recommended_split_k",
    gen_fake=gen_split_k_fake,
)
def get_recommended_split_k(M: int, N: int, K: int) -> int:
    """Get recommended split_k for given shape."""
    ...


def fp8_gemm_decode(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor = None,
    split_k: int = -1,
) -> Tensor:
    """
    High-level interface for FP8 GEMM optimized for decode (small M).
    
    This kernel is specifically tuned for LLaMA 70B decode workloads where:
    - M is small (~128 for batch size)
    - N and K are large (1280-28672)
    - The operation is memory-bound
    
    Args:
        A: [M, K] FP8 input tensor
        B: [N, K] FP8 weight tensor (transposed, column-major)
        A_scale: [M] FP32 per-token scale for A
        B_scale: [N] FP32 per-channel scale for B
        out: Optional [M, N] BF16 output tensor (created if None)
        split_k: Split-K factor (-1 for auto-select)
    
    Returns:
        [M, N] BF16 tensor with result: out = (A * A_scale) @ (B * B_scale)^T
    
    Raises:
        RuntimeError: If kernel not available for current GPU
        ValueError: If tensor shapes or types are invalid
    
    Example:
        >>> A = torch.randn(128, 8192, dtype=torch.float8_e4m3fn, device='cuda')
        >>> B = torch.randn(1280, 8192, dtype=torch.float8_e4m3fn, device='cuda')
        >>> A_scale = torch.ones(128, dtype=torch.float32, device='cuda')
        >>> B_scale = torch.ones(1280, dtype=torch.float32, device='cuda')
        >>> out = fp8_gemm_decode(A, B, A_scale, B_scale)
        >>> print(out.shape)  # torch.Size([128, 1280])
    """
    # Validate availability
    gfx = get_gfx()
    if gfx not in ["gfx950", "gfx942"]:
        raise RuntimeError(
            f"fp8_gemm_decode requires MI355X (gfx950) or MI300X (gfx942), got {gfx}"
        )
    
    M, K = A.shape
    N = B.shape[0]
    
    # Validate shapes
    if B.shape[1] != K:
        raise ValueError(f"K dimension mismatch: A has K={K}, B has K={B.shape[1]}")
    
    # Validate alignment requirements
    if N % 128 != 0:
        raise ValueError(f"N must be divisible by 128, got N={N}")
    if K % 128 != 0:
        raise ValueError(f"K must be divisible by 128, got K={K}")
    if M < 16:
        raise ValueError(f"M must be at least 16, got M={M}")
    if K < 128:
        raise ValueError(f"K must be at least 128, got K={K}")
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)
    
    # Auto-select split_k
    if split_k <= 0:
        split_k = get_recommended_split_k(M, N, K)
    
    return fp8_gemm_decode_asm(A, B, A_scale, B_scale, out, split_k)


def get_llama70b_decode_shapes() -> list[Tuple[int, int, int]]:
    """
    Get the GEMM shapes for LLaMA 70B decode with TP=8.
    
    Returns list of (M, N, K) tuples for:
    - QKV projection: (batch, 1280, 8192)
    - O projection: (batch, 8192, 1024)
    - Gate/Up projection: (batch, 3584, 8192) 
    - Down projection: (batch, 8192, 28672)
    
    Where batch is typically 128 for decode.
    """
    return [
        (128, 1280, 8192),   # QKV projection
        (128, 8192, 1024),   # O projection
        (128, 3584, 8192),   # Gate/Up projection (each)
        (128, 8192, 28672),  # Down projection
    ]

