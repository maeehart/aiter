# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Debug script to find where scale divergence occurs.
"""

import torch
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting, fused_moe
from aiter import ActivationType, QuantType


def silu(x):
    return x * torch.sigmoid(x)


def python_sorted_moe(hidden, w1, w2, topk_weight, topk_ids, block_m=32):
    """Python reference using sorted indices."""
    num_tokens, model_dim = hidden.shape
    num_experts, inter_dim2, _ = w1.shape
    inter_dim = inter_dim2 // 2
    
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weight, num_experts, model_dim, hidden.dtype, block_m)
    
    sorted_M = sorted_ids.shape[0]
    
    # Stage 1
    intermediate = torch.zeros((sorted_M, inter_dim * 2), dtype=torch.float32, device="cuda")
    for row in range(sorted_M):
        packed_id = sorted_ids[row].item()
        token_id = packed_id & 0xFFFFFF
        tile_id = row // block_m
        expert_id = sorted_expert_ids[tile_id].item()
        
        if expert_id < 0 or expert_id >= num_experts or token_id >= num_tokens:
            continue
        
        hidden_row = hidden[token_id].float()
        w1_expert = w1[expert_id].float()
        intermediate[row] = hidden_row @ w1_expert.T
    
    # Activation
    gate = intermediate[:, :inter_dim]
    up = intermediate[:, inter_dim:]
    activated = silu(gate) * up
    
    # Stage 2
    output = torch.zeros((num_tokens, model_dim), dtype=torch.float32, device="cuda")
    for row in range(sorted_M):
        packed_id = sorted_ids[row].item()
        token_id = packed_id & 0xFFFFFF
        tile_id = row // block_m
        expert_id = sorted_expert_ids[tile_id].item()
        
        if expert_id < 0 or expert_id >= num_experts or token_id >= num_tokens:
            continue
        
        weight = sorted_weights[row].item()
        activated_row = activated[row]
        w2_expert = w2[expert_id].float()
        result = activated_row @ w2_expert.T
        output[token_id] += weight * result
    
    return output.to(torch.bfloat16)


def test_at_size(num_tokens, model_dim, inter_dim, num_experts, topk):
    """Test at specific size."""
    hidden = torch.randn((num_tokens, model_dim), dtype=torch.bfloat16, device="cuda") / 10
    w1 = torch.randn((num_experts, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda") / 10
    w2 = torch.randn((num_experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda") / 10
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    topk_w, topk_ids = fused_topk(hidden, scores, topk, True)
    
    # Python reference
    python_out = python_sorted_moe(hidden, w1, w2, topk_w, topk_ids, block_m=32)
    
    # AITER reference
    aiter_out = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                          activation=ActivationType.Silu,
                          quant_type=QuantType.No)
    
    diff = (python_out - aiter_out).abs()
    max_diff = diff.max().item()
    pct = (diff < 1.0).float().mean().item() * 100
    
    return max_diff, pct


def sweep_sizes():
    """Sweep through different sizes to find where divergence occurs."""
    print("="*70)
    print("Size Sweep: Finding where divergence occurs")
    print("="*70)
    
    num_experts = 8
    topk = 2
    
    # Test different model_dim values
    print("\n--- Varying model_dim (tokens=256, inter_dim=model_dim) ---")
    for model_dim in [128, 256, 512, 1024, 2048, 4096]:
        max_diff, pct = test_at_size(256, model_dim, model_dim, num_experts, topk)
        status = "✓" if pct > 99 else "✗"
        print(f"{status} model_dim={model_dim:4d}: max_diff={max_diff:.4f}, within_1.0={pct:.1f}%")
    
    # Test different token counts
    print("\n--- Varying tokens (model_dim=4096, inter_dim=4096) ---")
    for num_tokens in [16, 32, 64, 128, 256, 512, 1024]:
        max_diff, pct = test_at_size(num_tokens, 4096, 4096, num_experts, topk)
        status = "✓" if pct > 99 else "✗"
        print(f"{status} tokens={num_tokens:4d}: max_diff={max_diff:.4f}, within_1.0={pct:.1f}%")


def test_hipkittens_vs_python_ref():
    """Compare HipKittens against Python reference at different sizes."""
    from aiter.hipkittens_moe import hipkittens_fused_moe
    
    print("\n" + "="*70)
    print("HipKittens vs Python Reference")
    print("="*70)
    
    num_experts = 8
    topk = 2
    
    test_cases = [
        (64, 128),
        (64, 256),
        (64, 512),
        (64, 1024),
        (64, 2048),
        (64, 4096),
        (256, 4096),
        (512, 4096),
    ]
    
    for num_tokens, model_dim in test_cases:
        hidden = torch.randn((num_tokens, model_dim), dtype=torch.bfloat16, device="cuda") / 10
        w1 = torch.randn((num_experts, model_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda") / 10
        w2 = torch.randn((num_experts, model_dim, model_dim), dtype=torch.bfloat16, device="cuda") / 10
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
        topk_w, topk_ids = fused_topk(hidden, scores, topk, True)
        
        # Python reference
        python_out = python_sorted_moe(hidden, w1, w2, topk_w, topk_ids, block_m=32)
        
        # HipKittens
        hk_out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids,
                                       activation=ActivationType.Silu)
        
        diff = (python_out - hk_out).abs()
        max_diff = diff.max().item()
        pct = (diff < 1.0).float().mean().item() * 100
        
        status = "✓" if pct > 99 else "✗"
        print(f"{status} tokens={num_tokens:4d}, dim={model_dim:4d}: max_diff={max_diff:.4f}, within_1.0={pct:.1f}%")


def debug_specific_failure():
    """Debug a specific failing case."""
    from aiter.hipkittens_moe import (hipkittens_fused_moe, hipkittens_moe_stage1, 
                                       hipkittens_moe_sorting)
    
    print("\n" + "="*70)
    print("Debugging specific failure case")
    print("="*70)
    
    num_tokens = 64
    model_dim = 4096
    inter_dim = 4096
    num_experts = 8
    topk = 2
    block_m = 32
    
    hidden = torch.randn((num_tokens, model_dim), dtype=torch.bfloat16, device="cuda") / 10
    w1 = torch.randn((num_experts, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda") / 10
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    topk_w, topk_ids = fused_topk(hidden, scores, topk, True)
    
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_w, num_experts, model_dim, hidden.dtype, block_m)
    
    sorted_M = sorted_ids.shape[0]
    print(f"sorted_M: {sorted_M}")
    
    # HK Stage 1
    hk_intermediate = hipkittens_moe_stage1(
        hidden, w1, sorted_ids, sorted_expert_ids, num_valid_ids, topk, block_m)
    
    # Python Stage 1
    py_intermediate = torch.zeros((sorted_M, inter_dim * 2), dtype=torch.float32, device="cuda")
    for row in range(sorted_M):
        packed_id = sorted_ids[row].item()
        token_id = packed_id & 0xFFFFFF
        tile_id = row // block_m
        expert_id = sorted_expert_ids[tile_id].item()
        
        if expert_id < 0 or expert_id >= num_experts or token_id >= num_tokens:
            continue
        
        hidden_row = hidden[token_id].float()
        w1_expert = w1[expert_id].float()
        py_intermediate[row] = hidden_row @ w1_expert.T
    py_intermediate = py_intermediate.to(torch.bfloat16)
    
    diff = (hk_intermediate - py_intermediate).abs()
    print(f"\nStage 1 comparison:")
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Elements within 0.1: {(diff < 0.1).float().mean().item() * 100:.1f}%")
    
    # Check a specific row
    print(f"\n  Row 0:")
    print(f"    HK:  {hk_intermediate[0, :4]}")
    print(f"    Py:  {py_intermediate[0, :4]}")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    sweep_sizes()
    test_hipkittens_vs_python_ref()
    debug_specific_failure()

