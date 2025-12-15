# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
HipKittens MoE Kernel Interface

This module provides a high-performance MoE implementation using HipKittens
with XCD-aware scheduling optimized for high batch sizes (prefill) on
AMD MI300X/MI325X GPUs.

Key optimizations:
1. XCD-aware workgroup scheduling to minimize L2 cache thrashing
2. 8-wave ping-pong pattern for compute/memory overlap
3. Expert-aware tiling to maximize weight reuse
4. Coalesced memory accesses using buffer loads

FP8 Blockscale Support:
- Supports FP8 (e4m3fnuz) weights with per-block dequantization scales
- Matches DeepSeek R1 production workload format (128x128 blocks)
- On-the-fly dequantization in the kernel
"""

import functools
from typing import Optional, Tuple

import torch

from aiter import dtypes
from aiter.fused_moe import moe_sorting, get_inter_dim
from aiter import ActivationType

# Import compile_ops decorator for JIT compilation
from aiter.jit.core import compile_ops

# Module name for the HipKittens MoE kernel
_HK_MOE_MODULE = "module_hipkittens_moe"

# FP8 blockscale configuration (matching DeepSeek R1)
FP8_SCALE_BLOCK_N = 128
FP8_SCALE_BLOCK_K = 128


# Block size configurations for different batch sizes
# Larger block sizes work better for high batch counts
HIPKITTENS_BLOCK_CONFIGS = {
    # (min_tokens, max_tokens): block_m
    (0, 512): 32,
    (512, 2048): 64,
    (2048, 8192): 64,      # High batch - use 64 for better L2 utilization
    (8192, float('inf')): 128,  # Very high batch
}


def get_hipkittens_block_m(num_tokens: int) -> int:
    """Get optimal block_m for HipKittens MoE based on token count."""
    for (min_t, max_t), block_m in HIPKITTENS_BLOCK_CONFIGS.items():
        if min_t <= num_tokens < max_t:
            return block_m
    return 64  # Default


def hipkittens_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    model_dim: int,
    moebuf_dtype: torch.dtype,
    block_size: int = 32,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    dispatch_policy: int = 0,
):
    """
    MoE sorting optimized for HipKittens kernels.
    
    This uses the standard AITER sorting but can be extended with
    XCD-aware reordering in the future.
    """
    return moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size,
        expert_mask,
        num_local_tokens,
        dispatch_policy,
    )


@compile_ops(_HK_MOE_MODULE, fc_name="hk_fused_moe_fwd")
def _hk_fused_moe_fwd_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    block_m: int,
) -> torch.Tensor:
    """Internal implementation that triggers JIT compilation."""
    pass


def hipkittens_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [expert, inter_dim*2, model_dim] for G1U1
    w2: torch.Tensor,  # [expert, model_dim, inter_dim]
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation: ActivationType = ActivationType.Silu,
    block_size_M: Optional[int] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
) -> torch.Tensor:
    """
    HipKittens-based fused MoE forward pass.
    
    This kernel is optimized for high batch sizes (prefill) where cache
    thrashing is a major performance issue. It uses XCD-aware scheduling
    to keep weight tiles in L2 cache across the chiplets.
    
    Args:
        hidden_states: Input tensor [num_tokens, model_dim]
        w1: Gate-Up projection [num_experts, inter_dim*2, model_dim]
        w2: Down projection [num_experts, model_dim, inter_dim]
        topk_weight: Routing weights [num_tokens, topk]
        topk_ids: Expert assignments [num_tokens, topk]
        expert_mask: Optional mask for expert parallelism
        activation: Activation function (default: SiLU for G1U1)
        block_size_M: Block size for M dimension (auto-selected if None)
        num_local_tokens: For dynamic batching
        moe_sorting_dispatch_policy: Sorting dispatch policy
        
    Returns:
        Output tensor [num_tokens, model_dim]
    """
    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    
    # Auto-select block size based on batch size
    if block_size_M is None:
        block_size_M = get_hipkittens_block_m(M)

    # Correctness requirement (current HipKittens kernels):
    # The kernels use a fixed 128-row tile internally and compute
    # `tile_id = row_start / block_m` to look up `sorted_expert_ids[tile_id]`.
    # If block_m != 128, a single kernel tile can span multiple sorting tiles
    # and the kernel will apply the wrong expert weights for part of the tile.
    # Until the kernels are updated to handle block_m != 128, force block_m=128.
    if block_size_M != 128:
        block_size_M = 128
    
    # Determine global expert count for EP
    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    
    dtype = hidden_states.dtype
    assert dtype in [dtypes.bf16], f"HipKittens MoE currently requires BFloat16, got {dtype}"
    
    # Verify G1U1 configuration
    isG1U1 = inter_dim != w1.shape[1]
    assert isG1U1, "HipKittens MoE currently only supports G1U1 (gate-up) configuration"
    assert activation == ActivationType.Silu, "HipKittens MoE currently only supports SiLU activation"
    
    # Perform MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = hipkittens_moe_sorting(
        topk_ids,
        topk_weight,
        global_E,
        model_dim,
        dtype,
        block_size_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    
    # Call HipKittens fused MoE via JIT-compiled wrapper
    output = _hk_fused_moe_fwd_impl(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        block_size_M,
    )
    
    return output


@compile_ops(_HK_MOE_MODULE, fc_name="hk_moe_stage1_fwd")
def hipkittens_moe_stage1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    block_m: int = 64,
) -> torch.Tensor:
    """
    HipKittens MoE Stage 1: Gate-Up projection.
    
    Computes output = hidden_states @ W1^T for G1U1 configuration.
    Uses XCD-aware scheduling for optimal L2 cache utilization.
    
    Args:
        hidden_states: Input [num_tokens, model_dim]
        w1: Gate-Up weights [num_experts, inter_dim*2, model_dim]
        sorted_ids: Sorted token IDs for expert routing
        sorted_expert_ids: Expert assignment per tile
        num_valid_ids: Valid token count per expert
        topk: Number of experts per token
        block_m: Block size for M dimension
        
    Returns:
        Stage 1 output [num_tokens, topk, inter_dim*2]
    """
    pass


@compile_ops(_HK_MOE_MODULE, fc_name="hk_moe_stage2_fwd")
def hipkittens_moe_stage2(
    intermediate: torch.Tensor,
    w2: torch.Tensor,
    output: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    topk: int,
    block_m: int = 64,
) -> None:
    """
    HipKittens MoE Stage 2: Down projection with weighted accumulation.
    
    Computes output += topk_weight * (intermediate @ W2^T)
    Uses XCD-aware scheduling matching Stage 1 pattern.
    
    Args:
        intermediate: Activated stage 1 output [num_tokens, topk, inter_dim]
        w2: Down projection [num_experts, model_dim, inter_dim]
        output: Pre-allocated output buffer [num_tokens, model_dim]
        sorted_ids: Sorted token IDs
        sorted_expert_ids: Expert assignment per tile
        num_valid_ids: Valid token count per expert
        topk_weights: Routing weights for accumulation
        topk: Number of experts per token
        block_m: Block size for M dimension
    """
    pass


def should_use_hipkittens_moe(
    num_tokens: int,
    model_dim: int,
    inter_dim: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
) -> bool:
    """
    Heuristic to determine if HipKittens MoE should be used.
    
    HipKittens MoE is optimized for high batch sizes where cache
    thrashing is the bottleneck. For small batches, the standard
    AITER kernels may be faster.
    
    Args:
        num_tokens: Number of tokens in batch
        model_dim: Hidden dimension
        inter_dim: Intermediate dimension
        num_experts: Number of experts
        topk: Experts per token
        dtype: Data type
        
    Returns:
        True if HipKittens MoE is recommended
    """
    # HipKittens is currently optimized for:
    # - BF16 data type
    # - High batch sizes (>1024 tokens)
    # - G1U1 configuration (gate-up projection)
    
    if dtype != torch.bfloat16:
        return False
    
    # For high batch sizes (prefill), HipKittens excels
    # due to XCD-aware scheduling
    if num_tokens >= 1024:
        return True
    
    return False


# ============================================================================
# FP8 Blockscale Interface
# ============================================================================

def compute_scale_dims(
    n_dim: int,
    k_dim: int,
    scale_block_n: int = FP8_SCALE_BLOCK_N,
    scale_block_k: int = FP8_SCALE_BLOCK_K,
) -> Tuple[int, int]:
    """Compute the scale tensor dimensions for blockscale quantization."""
    num_scale_n = (n_dim + scale_block_n - 1) // scale_block_n
    num_scale_k = (k_dim + scale_block_k - 1) // scale_block_k
    return num_scale_n, num_scale_k


@compile_ops(_HK_MOE_MODULE, fc_name="hk_fused_moe_fp8_fwd")
def _hk_fused_moe_fp8_fwd_impl(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    block_m: int,
) -> torch.Tensor:
    """Internal implementation that triggers JIT compilation."""
    pass


def hipkittens_fused_moe_fp8(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,  # [expert, inter_dim*2, model_dim] as FP8
    w2_fp8: torch.Tensor,  # [expert, model_dim, inter_dim] as FP8
    w1_scale: torch.Tensor,  # [expert, num_scale_n, num_scale_k]
    w2_scale: torch.Tensor,  # [expert, num_scale_n, num_scale_k]
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation: ActivationType = ActivationType.Silu,
    block_size_M: Optional[int] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
) -> torch.Tensor:
    """
    HipKittens-based fused MoE forward pass with FP8 weights.
    
    This kernel supports FP8 quantized weights with per-block dequantization
    scales, matching the DeepSeek R1 production workload format.
    
    Args:
        hidden_states: Input tensor [num_tokens, model_dim] bf16
        w1_fp8: Gate-Up projection [num_experts, inter_dim*2, model_dim] fp8
        w2_fp8: Down projection [num_experts, model_dim, inter_dim] fp8
        w1_scale: W1 scales [num_experts, ceil(N/128), ceil(K/128)] float
        w2_scale: W2 scales [num_experts, ceil(N/128), ceil(K/128)] float
        topk_weight: Routing weights [num_tokens, topk]
        topk_ids: Expert assignments [num_tokens, topk]
        expert_mask: Optional mask for expert parallelism
        activation: Activation function (default: SiLU for G1U1)
        block_size_M: Block size for M dimension (auto-selected if None)
        num_local_tokens: For dynamic batching
        moe_sorting_dispatch_policy: Sorting dispatch policy
        
    Returns:
        Output tensor [num_tokens, model_dim] bf16
    """
    M, topk = topk_ids.shape
    num_experts = w1_fp8.size(0)
    model_dim = hidden_states.size(1)
    inter_dim = w1_fp8.size(1) // 2  # G1U1
    
    # Auto-select block size based on batch size
    if block_size_M is None:
        block_size_M = get_hipkittens_block_m(M)
    
    # Force block_m=128 for correctness (see BF16 kernel comment)
    if block_size_M != 128:
        block_size_M = 128
    
    # Determine global expert count for EP
    global_E = num_experts
    if expert_mask is not None:
        global_E = expert_mask.numel()
    
    dtype = hidden_states.dtype
    assert dtype == dtypes.bf16, f"HipKittens MoE FP8 requires BFloat16 input, got {dtype}"
    
    # Verify G1U1 configuration
    assert activation == ActivationType.Silu, "HipKittens MoE FP8 only supports SiLU activation"
    
    # Perform MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = hipkittens_moe_sorting(
        topk_ids,
        topk_weight,
        global_E,
        model_dim,
        dtype,
        block_size_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    
    # Call HipKittens fused MoE FP8 via JIT-compiled wrapper
    output = _hk_fused_moe_fp8_fwd_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        block_size_M,
    )
    
    return output


@compile_ops(_HK_MOE_MODULE, fc_name="hk_moe_stage1_fp8_fwd")
def hipkittens_moe_stage1_fp8(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    block_m: int = 64,
) -> torch.Tensor:
    """
    HipKittens MoE Stage 1 with FP8 weights: Gate-Up projection.
    
    Computes output = hidden_states @ W1^T with FP8 dequantization.
    
    Args:
        hidden_states: Input [num_tokens, model_dim] bf16
        w1_fp8: Gate-Up weights [num_experts, inter_dim*2, model_dim] fp8
        w1_scale: Per-block scales [num_experts, num_scale_n, num_scale_k] float
        sorted_ids: Sorted token IDs for expert routing
        sorted_expert_ids: Expert assignment per tile
        num_valid_ids: Valid token count per expert
        topk: Number of experts per token
        block_m: Block size for M dimension
        
    Returns:
        Stage 1 output [num_tokens, topk, inter_dim*2]
    """
    pass


# ============================================================================
# FP8 with Fused Activation (Optimized)
# ============================================================================

@compile_ops(_HK_MOE_MODULE, fc_name="hk_fused_moe_fp8_fused_act_fwd")
def _hk_fused_moe_fp8_fused_act_fwd_impl(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    block_m: int,
) -> torch.Tensor:
    """Internal implementation that triggers JIT compilation."""
    pass


def hipkittens_fused_moe_fp8_fused_act(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation: ActivationType = ActivationType.Silu,
    block_size_M: Optional[int] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
) -> torch.Tensor:
    """
    HipKittens-based fused MoE forward pass with FP8 weights and FUSED activation.
    
    This version fuses the G1U1 activation (silu(gate) * up) into Stage 1,
    eliminating the separate activation kernel and reducing intermediate
    memory traffic by 50%.
    
    Benefits over hipkittens_fused_moe_fp8:
      - One less kernel launch
      - 50% reduction in intermediate memory traffic
      - Better for memory-bound workloads
    
    Trade-off:
      - 2x the weight loads in Stage 1 (loads both gate and up weights)
      - May be slower for very small batches where compute dominates
    
    Args:
        hidden_states: Input tensor [num_tokens, model_dim] bf16
        w1_fp8: Gate-Up projection [num_experts, inter_dim*2, model_dim] fp8
        w2_fp8: Down projection [num_experts, model_dim, inter_dim] fp8
        w1_scale: W1 scales [num_experts, ceil(N/128), ceil(K/128)] float
        w2_scale: W2 scales [num_experts, ceil(N/128), ceil(K/128)] float
        topk_weight: Routing weights [num_tokens, topk]
        topk_ids: Expert assignments [num_tokens, topk]
        expert_mask: Optional mask for expert parallelism
        activation: Activation function (default: SiLU for G1U1)
        block_size_M: Block size for M dimension (auto-selected if None)
        num_local_tokens: For dynamic batching
        moe_sorting_dispatch_policy: Sorting dispatch policy
        
    Returns:
        Output tensor [num_tokens, model_dim] bf16
    """
    M, topk = topk_ids.shape
    num_experts = w1_fp8.size(0)
    model_dim = hidden_states.size(1)
    inter_dim = w1_fp8.size(1) // 2  # G1U1
    
    # Auto-select block size based on batch size
    if block_size_M is None:
        block_size_M = get_hipkittens_block_m(M)
    
    # Force block_m=128 for correctness
    if block_size_M != 128:
        block_size_M = 128
    
    # Determine global expert count for EP
    global_E = num_experts
    if expert_mask is not None:
        global_E = expert_mask.numel()
    
    dtype = hidden_states.dtype
    assert dtype == dtypes.bf16, f"HipKittens MoE FP8 requires BFloat16 input, got {dtype}"
    
    # Verify G1U1 configuration
    assert activation == ActivationType.Silu, "HipKittens MoE FP8 only supports SiLU activation"
    
    # Perform MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = hipkittens_moe_sorting(
        topk_ids,
        topk_weight,
        global_E,
        model_dim,
        dtype,
        block_size_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    
    # Call HipKittens fused MoE FP8 with fused activation
    output = _hk_fused_moe_fp8_fused_act_fwd_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        block_size_M,
    )
    
    return output


# ============================================================================
# EXPERIMENTAL: FP8 with Fused Activation + Atomic-Free Stage 2
# ============================================================================

@compile_ops(_HK_MOE_MODULE, fc_name="hk_fused_moe_fp8_noatomic_fwd")
def _hk_fused_moe_fp8_noatomic_fwd_impl(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    block_m: int,
) -> torch.Tensor:
    """Internal implementation that triggers JIT compilation."""
    pass


def hipkittens_fused_moe_fp8_noatomic(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation: ActivationType = ActivationType.Silu,
    block_size_M: Optional[int] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
) -> torch.Tensor:
    """
    EXPERIMENTAL: HipKittens fused MoE with FP8 weights + fused activation + atomic-free Stage 2.
    
    This version eliminates atomic operations in Stage 2 by writing to a
    per-(token, topk_slot) buffer and then reducing. This can improve
    performance when atomic contention is high.
    
    Benefits over hipkittens_fused_moe_fp8_fused_act:
      - No atomic contention in Stage 2
      - Better parallelism for high-collision scenarios
    
    Trade-offs:
      - Extra memory: num_tokens * topk * model_dim * 4 bytes (~1.87 GB for 8k DeepSeek R1)
      - Extra reduction kernel launch
    
    Args:
        hidden_states: Input tensor [num_tokens, model_dim] bf16
        w1_fp8: Gate-Up projection [num_experts, inter_dim*2, model_dim] fp8
        w2_fp8: Down projection [num_experts, model_dim, inter_dim] fp8
        w1_scale: W1 scales [num_experts, ceil(N/128), ceil(K/128)] float
        w2_scale: W2 scales [num_experts, ceil(N/128), ceil(K/128)] float
        topk_weight: Routing weights [num_tokens, topk]
        topk_ids: Expert assignments [num_tokens, topk]
        expert_mask: Optional mask for expert parallelism
        activation: Activation function (default: SiLU for G1U1)
        block_size_M: Block size for M dimension (auto-selected if None)
        num_local_tokens: For dynamic batching
        moe_sorting_dispatch_policy: Sorting dispatch policy
        
    Returns:
        Output tensor [num_tokens, model_dim] bf16
    """
    M, topk = topk_ids.shape
    num_experts = w1_fp8.size(0)
    model_dim = hidden_states.size(1)
    inter_dim = w1_fp8.size(1) // 2  # G1U1
    
    if block_size_M is None:
        block_size_M = get_hipkittens_block_m(M)
    
    if block_size_M != 128:
        block_size_M = 128
    
    global_E = num_experts
    if expert_mask is not None:
        global_E = expert_mask.numel()
    
    dtype = hidden_states.dtype
    assert dtype == dtypes.bf16, f"HipKittens MoE FP8 requires BFloat16 input, got {dtype}"
    assert activation == ActivationType.Silu, "HipKittens MoE FP8 only supports SiLU activation"
    
    # Perform MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = hipkittens_moe_sorting(
        topk_ids,
        topk_weight,
        global_E,
        model_dim,
        dtype,
        block_size_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    
    # Call HipKittens fused MoE FP8 with fused activation + atomic-free Stage 2
    output = _hk_fused_moe_fp8_noatomic_fwd_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        block_size_M,
    )
    
    return output


# ============================================================================
# Fully Fused Kernel (Stage1 + Stage2 in one kernel)
# ============================================================================

@compile_ops(_HK_MOE_MODULE, fc_name="hk_fused_moe_fully_fused_fwd")
def _hk_fused_moe_fully_fused_fwd_impl(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    block_m: int,
) -> torch.Tensor:
    """Internal implementation wrapper for fully fused kernel."""
    pass


def hipkittens_fused_moe_fully_fused(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation: ActivationType = ActivationType.Silu,
    block_size_M: Optional[int] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
) -> torch.Tensor:
    """
    HipKittens FULLY FUSED MoE forward pass.
    
    Fuses Stage 1 (gate-up + SiLU) and Stage 2 (down projection) into a single kernel.
    Intermediate values stay in LDS, never going to global memory.
    
    Memory savings vs separate kernels (hipkittens_fused_moe_fp8_fused_act):
      - Eliminates intermediate write: sorted_M × inter_dim × 2B (~35MB at 8k batch)
      - Eliminates intermediate read: sorted_M × inter_dim × 2B (~35MB at 8k batch)
      - Total: ~66MB saved per forward pass
    
    This matches AITER's fused approach.
    
    Args:
        hidden_states: Input tensor [num_tokens, model_dim] bf16
        w1_fp8: Gate-Up projection [num_experts, inter_dim*2, model_dim] fp8
        w2_fp8: Down projection [num_experts, model_dim, inter_dim] fp8
        w1_scale: W1 scales [num_experts, ceil(N/128), ceil(K/128)] float
        w2_scale: W2 scales [num_experts, ceil(N/128), ceil(K/128)] float
        topk_weight: Routing weights [num_tokens, topk]
        topk_ids: Expert assignments [num_tokens, topk]
        expert_mask: Optional mask for expert parallelism
        activation: Activation function (default: SiLU for G1U1)
        block_size_M: Block size for M dimension (auto-selected if None)
        num_local_tokens: For dynamic batching
        moe_sorting_dispatch_policy: Sorting dispatch policy
        
    Returns:
        Output tensor [num_tokens, model_dim] bf16
    """
    M, topk = topk_ids.shape
    num_experts = w1_fp8.size(0)
    model_dim = hidden_states.size(1)
    inter_dim = w1_fp8.size(1) // 2  # G1U1
    
    # Use smaller block size for fused kernel (M_TILE=32)
    if block_size_M is None:
        block_size_M = 32  # Matches fused kernel's M_TILE
    
    # Force block_m=32 for fused kernel
    if block_size_M != 32:
        block_size_M = 32
    
    # Determine global expert count for EP
    global_E = num_experts
    if expert_mask is not None:
        global_E = expert_mask.numel()
    
    dtype = hidden_states.dtype
    assert dtype == dtypes.bf16, f"HipKittens MoE Fully Fused requires BFloat16 input, got {dtype}"
    assert activation == ActivationType.Silu, "HipKittens MoE Fully Fused only supports SiLU activation"
    
    # Perform MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = hipkittens_moe_sorting(
        topk_ids,
        topk_weight,
        global_E,
        model_dim,
        dtype,
        block_size_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    
    # Call HipKittens fully fused kernel
    output = _hk_fused_moe_fully_fused_fwd_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        block_size_M,
    )
    
    return output


# ============================================================================
# Streaming Fusion Kernel (intermediate in registers)
# ============================================================================

@compile_ops(_HK_MOE_MODULE, fc_name="hk_fused_moe_streaming_fwd")
def _hk_fused_moe_streaming_fwd_impl(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    block_m: int,
) -> torch.Tensor:
    """Internal implementation wrapper for streaming fusion kernel."""
    pass


def hipkittens_fused_moe_streaming(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation: ActivationType = ActivationType.Silu,
    block_size_M: Optional[int] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
) -> torch.Tensor:
    """
    HipKittens STREAMING FUSION MoE forward pass.
    
    Uses streaming approach where intermediate stays in registers, not LDS.
    Processes inter_dim in chunks, immediately multiplying by W2.
    
    This approach avoids the LDS buffer for intermediate values and
    reduces synchronization overhead compared to the LDS-based fusion.
    
    Args:
        hidden_states: Input tensor [num_tokens, model_dim] bf16
        w1_fp8: Gate-Up projection [num_experts, inter_dim*2, model_dim] fp8
        w2_fp8: Down projection [num_experts, model_dim, inter_dim] fp8
        w1_scale: W1 scales float
        w2_scale: W2 scales float
        topk_weight: Routing weights [num_tokens, topk]
        topk_ids: Expert assignments [num_tokens, topk]
        expert_mask: Optional mask for expert parallelism
        activation: Activation function (default: SiLU for G1U1)
        block_size_M: Block size for M dimension (auto-selected if None)
        num_local_tokens: For dynamic batching
        moe_sorting_dispatch_policy: Sorting dispatch policy
        
    Returns:
        Output tensor [num_tokens, model_dim] bf16
    """
    M, topk = topk_ids.shape
    num_experts = w1_fp8.size(0)
    model_dim = hidden_states.size(1)
    inter_dim = w1_fp8.size(1) // 2  # G1U1
    
    # Use smaller block size for streaming kernel (M_TILE=32)
    if block_size_M is None:
        block_size_M = 32  # Matches streaming kernel's M_TILE
    
    # Force block_m=32 for streaming kernel
    if block_size_M != 32:
        block_size_M = 32
    
    # Determine global expert count for EP
    global_E = num_experts
    if expert_mask is not None:
        global_E = expert_mask.numel()
    
    dtype = hidden_states.dtype
    assert dtype == dtypes.bf16, f"HipKittens MoE Streaming requires BFloat16 input, got {dtype}"
    assert activation == ActivationType.Silu, "HipKittens MoE Streaming only supports SiLU activation"
    
    # Perform MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = hipkittens_moe_sorting(
        topk_ids,
        topk_weight,
        global_E,
        model_dim,
        dtype,
        block_size_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    
    # Call HipKittens streaming fusion kernel
    output = _hk_fused_moe_streaming_fwd_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        block_size_M,
    )
    
    return output
