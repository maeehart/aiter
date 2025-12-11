# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# HipKittens-based FP8 GEMM for decode workloads
# Uses HipKittens library from Stanford's Hazy Research:
# https://github.com/HazyResearch/HipKittens

import os
import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops
from ..utility import dtypes
from ..jit.utils.chip_info import get_gfx


def gen_hipkittens_fp8_gemm_decode_fake_tensors(
    A: Tensor,          # [M, K] FP8
    B: Tensor,          # [N, K] FP8 (transposed weights)
    A_scale: Tensor,    # [M] or [1] float32
    B_scale: Tensor,    # [N] or [1] float32
    bias: Optional[Tensor] = None,  # [N] float32
    output_dtype: int = 0,  # 0=bf16, 1=fp16
) -> Tensor:
    M, K = A.shape
    N = B.shape[0]
    dtype = torch.bfloat16 if output_dtype == 0 else torch.float16
    return torch.empty(M, N, dtype=dtype, device=A.device)


@compile_ops(
    "module_hipkittens_fp8_gemm",
    fc_name="hipkittens_fp8_gemm_decode",
    gen_fake=gen_hipkittens_fp8_gemm_decode_fake_tensors,
)
def hipkittens_fp8_gemm_decode_kernel(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Optional[Tensor] = None,
    output_dtype: int = 0,
) -> Tensor: ...


def is_hipkittens_available() -> bool:
    """Check if HipKittens is available and supported on this GPU."""
    gfx = get_gfx()
    return gfx in ["gfx942", "gfx950"]


def hipkittens_fp8_gemm_decode(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Optional[Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """
    FP8 GEMM optimized for decode workloads using HipKittens.
    
    This kernel is optimized for small M (batch sizes ~1-256) typical in
    LLM decode phase, with large N and K (model dimensions).
    
    Args:
        A: Input activation tensor [M, K] in FP8 format
        B: Weight tensor [N, K] in FP8 format (row-major, will be transposed internally)
        A_scale: Per-token or scalar scale for A [M] or [1]
        B_scale: Per-channel or scalar scale for B [N] or [1]
        bias: Optional bias tensor [N]
        output_dtype: Output data type (torch.bfloat16 or torch.float16)
    
    Returns:
        Output tensor [M, N] in the specified output dtype
    
    Example:
        >>> A = torch.randn(128, 8192).to(torch.float8_e4m3fnuz).cuda()
        >>> B = torch.randn(1280, 8192).to(torch.float8_e4m3fnuz).cuda()
        >>> A_scale = torch.ones(128, device='cuda')
        >>> B_scale = torch.ones(1280, device='cuda')
        >>> out = hipkittens_fp8_gemm_decode(A, B, A_scale, B_scale)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA device"
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D tensors"
    assert A.shape[1] == B.shape[1], f"K dimension mismatch: {A.shape[1]} vs {B.shape[1]}"
    
    M, K = A.shape
    N = B.shape[0]
    
    # Convert output dtype to int for C++ interface
    dtype_int = 0 if output_dtype == torch.bfloat16 else 1
    
    # Ensure scales are float32
    if A_scale.dtype != torch.float32:
        A_scale = A_scale.float()
    if B_scale.dtype != torch.float32:
        B_scale = B_scale.float()
    
    # Broadcast scalar scales to per-token/per-channel
    if A_scale.numel() == 1:
        A_scale = A_scale.expand(M)
    if B_scale.numel() == 1:
        B_scale = B_scale.expand(N)
    
    return hipkittens_fp8_gemm_decode_kernel(A, B, A_scale, B_scale, bias, dtype_int)


def should_use_hipkittens_decode(M: int, N: int, K: int) -> bool:
    """
    Heuristic to decide if HipKittens decode kernel should be used.
    
    HipKittens decode kernel is optimized for:
    - Small M (1-256, typical decode batch sizes)
    - Large N and K (model dimensions like 8192, 28672, etc.)
    
    Args:
        M: Batch size / number of tokens
        N: Output dimension
        K: Input dimension
    
    Returns:
        True if HipKittens decode kernel should be used
    """
    if not is_hipkittens_available():
        return False
    
    # Optimized for decode: small M, large N and K
    is_decode_shape = M <= 256 and N >= 1024 and K >= 1024
    
    return is_decode_shape

