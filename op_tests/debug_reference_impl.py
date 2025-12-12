# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Debug script to verify reference implementation matches AITER.
"""

import torch
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting, fused_moe
from aiter import ActivationType, QuantType


def silu(x):
    """Python SiLU implementation."""
    return x * torch.sigmoid(x)


def python_moe_full_ref(hidden, w1, w2, topk_weight, topk_ids):
    """
    Complete Python reference implementation of MoE.
    This should match AITER exactly.
    """
    num_tokens, model_dim = hidden.shape
    num_experts, inter_dim2, _ = w1.shape
    inter_dim = inter_dim2 // 2
    topk = topk_ids.shape[1]
    
    output = torch.zeros((num_tokens, model_dim), dtype=hidden.dtype, device=hidden.device)
    
    for token_idx in range(num_tokens):
        hidden_vec = hidden[token_idx].float()  # [model_dim]
        
        for k in range(topk):
            expert_id = topk_ids[token_idx, k].item()
            weight = topk_weight[token_idx, k].item()
            
            # Stage 1: hidden @ w1^T -> [inter_dim*2]
            # w1[expert] is [inter_dim*2, model_dim]
            w1_expert = w1[expert_id].float()  # [inter_dim*2, model_dim]
            stage1_out = hidden_vec @ w1_expert.T  # [model_dim] @ [model_dim, inter_dim*2] = [inter_dim*2]
            
            # Activation: SiLU(gate) * up
            gate = stage1_out[:inter_dim]
            up = stage1_out[inter_dim:]
            activated = silu(gate) * up  # [inter_dim]
            
            # Stage 2: activated @ w2^T -> [model_dim]
            # w2[expert] is [model_dim, inter_dim]
            w2_expert = w2[expert_id].float()  # [model_dim, inter_dim]
            stage2_out = activated @ w2_expert.T  # [inter_dim] @ [inter_dim, model_dim] = [model_dim]
            
            # Weight and accumulate
            output[token_idx] += (weight * stage2_out).to(hidden.dtype)
    
    return output


def test_python_vs_aiter():
    """Test Python reference against AITER."""
    print("="*60)
    print("Testing Python reference against AITER")
    print("="*60)
    
    cfg = {"model_dim": 256, "inter_dim": 256, "num_experts": 4, "topk": 2}
    num_tokens = 16
    
    hidden = torch.randn((num_tokens, cfg["model_dim"]), dtype=torch.bfloat16, device="cuda") / 10
    w1 = torch.randn((cfg["num_experts"], cfg["inter_dim"] * 2, cfg["model_dim"]), dtype=torch.bfloat16, device="cuda") / 10
    w2 = torch.randn((cfg["num_experts"], cfg["model_dim"], cfg["inter_dim"]), dtype=torch.bfloat16, device="cuda") / 10
    scores = torch.randn((num_tokens, cfg["num_experts"]), dtype=torch.float32, device="cuda")
    topk_w, topk_ids = fused_topk(hidden, scores, cfg["topk"], True)
    
    print(f"hidden shape: {hidden.shape}")
    print(f"w1 shape: {w1.shape}")
    print(f"w2 shape: {w2.shape}")
    print(f"topk_w shape: {topk_w.shape}")
    print(f"topk_ids shape: {topk_ids.shape}")
    
    # Python reference
    python_out = python_moe_full_ref(hidden, w1, w2, topk_w, topk_ids)
    
    # AITER reference
    aiter_out = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                          activation=ActivationType.Silu,
                          quant_type=QuantType.No)
    
    print(f"\nPython output sample: {python_out[0, :4]}")
    print(f"AITER output sample: {aiter_out[0, :4]}")
    
    diff = (python_out - aiter_out).abs()
    max_diff = diff.max().item()
    print(f"\nMax diff: {max_diff:.6f}")
    print(f"Elements within 0.1: {(diff < 0.1).float().mean().item() * 100:.1f}%")
    
    if max_diff < 0.1:
        print("✓ Python reference matches AITER!")
    else:
        print("✗ Python reference does NOT match AITER!")
        print("\nChecking individual operations...")
        check_stage1_layout(hidden, w1, topk_ids)


def check_stage1_layout(hidden, w1, topk_ids):
    """Check if Stage 1 layout is correct."""
    print("\n--- Stage 1 Layout Check ---")
    
    # For token 0, expert from topk
    expert_id = topk_ids[0, 0].item()
    
    hidden_vec = hidden[0].float()  # [model_dim]
    w1_expert = w1[expert_id].float()  # [inter_dim*2, model_dim]
    
    print(f"hidden_vec shape: {hidden_vec.shape}")
    print(f"w1_expert shape: {w1_expert.shape}")
    
    # Our computation: hidden @ w1^T
    # w1^T is [model_dim, inter_dim*2]
    result_ours = hidden_vec @ w1_expert.T
    print(f"hidden @ w1^T result shape: {result_ours.shape}")
    
    # Alternative: maybe it should be w1 @ hidden?
    # w1 @ hidden: [inter_dim*2, model_dim] @ [model_dim] = [inter_dim*2]
    result_alt = w1_expert @ hidden_vec
    print(f"w1 @ hidden result shape: {result_alt.shape}")
    
    # Check if they're the same (they should be for these shapes)
    diff = (result_ours - result_alt).abs().max()
    print(f"Diff between hidden@w1^T and w1@hidden: {diff:.6f}")


def test_simple_gemm():
    """Test if simple GEMM works correctly."""
    print("\n" + "="*60)
    print("Testing simple GEMM pattern")
    print("="*60)
    
    model_dim = 128
    inter_dim = 128
    
    # Single token, single expert case
    hidden = torch.randn((model_dim,), dtype=torch.float32, device="cuda")
    w1 = torch.randn((inter_dim * 2, model_dim), dtype=torch.float32, device="cuda")
    w2 = torch.randn((model_dim, inter_dim), dtype=torch.float32, device="cuda")
    
    # Stage 1: hidden @ w1^T = [model_dim] @ [model_dim, inter_dim*2] = [inter_dim*2]
    # But w1 is [inter_dim*2, model_dim], so w1^T is [model_dim, inter_dim*2]
    stage1_a = hidden @ w1.T  # This is [inter_dim*2]
    
    # Alternative interpretation: w1 @ hidden
    stage1_b = w1 @ hidden  # This is [inter_dim*2]
    
    print(f"hidden @ w1^T: {stage1_a[:4]}")
    print(f"w1 @ hidden:   {stage1_b[:4]}")
    print(f"Same: {torch.allclose(stage1_a, stage1_b)}")


def test_sorted_moe_reference():
    """
    Test a reference implementation that uses sorted indices like our kernels.
    """
    print("\n" + "="*60)
    print("Testing sorted MoE reference")
    print("="*60)
    
    cfg = {"model_dim": 128, "inter_dim": 128, "num_experts": 4, "topk": 2}
    num_tokens = 16
    
    hidden = torch.randn((num_tokens, cfg["model_dim"]), dtype=torch.bfloat16, device="cuda") / 10
    w1 = torch.randn((cfg["num_experts"], cfg["inter_dim"] * 2, cfg["model_dim"]), dtype=torch.bfloat16, device="cuda") / 10
    w2 = torch.randn((cfg["num_experts"], cfg["model_dim"], cfg["inter_dim"]), dtype=torch.bfloat16, device="cuda") / 10
    scores = torch.randn((num_tokens, cfg["num_experts"]), dtype=torch.float32, device="cuda")
    topk_w, topk_ids = fused_topk(hidden, scores, cfg["topk"], True)
    
    block_m = 32
    
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_w, cfg["num_experts"], cfg["model_dim"], hidden.dtype, block_m)
    
    sorted_M = sorted_ids.shape[0]
    inter_dim = cfg["inter_dim"]
    
    print(f"sorted_M: {sorted_M}")
    print(f"sorted_ids[:8]: {sorted_ids[:8]}")
    
    # Stage 1: sorted computation
    intermediate = torch.zeros((sorted_M, inter_dim * 2), dtype=torch.float32, device="cuda")
    
    for row in range(sorted_M):
        packed_id = sorted_ids[row].item()
        token_id = packed_id & 0xFFFFFF
        
        tile_id = row // block_m
        expert_id = sorted_expert_ids[tile_id].item()
        
        if expert_id < 0 or expert_id >= cfg["num_experts"]:
            continue
        if token_id < 0 or token_id >= num_tokens:
            continue
        
        hidden_row = hidden[token_id].float()
        w1_expert = w1[expert_id].float()
        intermediate[row] = hidden_row @ w1_expert.T
    
    intermediate = intermediate.to(torch.bfloat16)
    
    # Activation
    gate = intermediate[:, :inter_dim].float()
    up = intermediate[:, inter_dim:].float()
    activated = (silu(gate) * up).to(torch.bfloat16)
    
    # Stage 2: scatter back
    output = torch.zeros((num_tokens, cfg["model_dim"]), dtype=torch.float32, device="cuda")
    
    for row in range(sorted_M):
        packed_id = sorted_ids[row].item()
        token_id = packed_id & 0xFFFFFF
        
        tile_id = row // block_m
        expert_id = sorted_expert_ids[tile_id].item()
        
        if expert_id < 0 or expert_id >= cfg["num_experts"]:
            continue
        if token_id < 0 or token_id >= num_tokens:
            continue
        
        weight = sorted_weights[row].item()
        
        activated_row = activated[row].float()
        w2_expert = w2[expert_id].float()
        result = activated_row @ w2_expert.T
        output[token_id] += weight * result
    
    output = output.to(torch.bfloat16)
    
    # Compare with AITER
    aiter_out = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                          activation=ActivationType.Silu,
                          quant_type=QuantType.No)
    
    diff = (output - aiter_out).abs()
    print(f"\nSorted ref vs AITER:")
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Elements within 0.1: {(diff < 0.1).float().mean().item() * 100:.1f}%")
    
    print(f"\nSorted ref sample: {output[0, :4]}")
    print(f"AITER sample: {aiter_out[0, :4]}")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    test_simple_gemm()
    test_python_vs_aiter()
    test_sorted_moe_reference()

