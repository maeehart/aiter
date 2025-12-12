# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Direct comparison of AITER vs our implementation to find the exact difference.
"""

import torch
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting, fused_moe
from aiter import ActivationType, QuantType


def test_with_identity_weights():
    """Test with identity-like weights to trace the computation."""
    print("="*60)
    print("Test with identity-like weights")
    print("="*60)
    
    num_tokens = 4
    model_dim = 64
    inter_dim = 64
    num_experts = 2
    topk = 1
    
    # Create simple inputs
    hidden = torch.ones((num_tokens, model_dim), dtype=torch.bfloat16, device="cuda")
    
    # Create identity-like w1: each row i has 1 in column i
    # w1[expert, out_dim, in_dim] -> each output = sum of inputs
    w1 = torch.zeros((num_experts, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda")
    # Set gate and up to be all ones (will sum all inputs)
    w1[:, :, :] = 1.0 / model_dim  # Scale down so outputs are reasonable
    
    # w2 similar
    w2 = torch.zeros((num_experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda")
    w2[:, :, :] = 1.0 / inter_dim
    
    # Route all tokens to expert 0
    topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_w = torch.ones((num_tokens, topk), dtype=torch.float32, device="cuda")
    
    print(f"hidden shape: {hidden.shape}, sum: {hidden.sum().item()}")
    print(f"w1 shape: {w1.shape}, sum: {w1.sum().item()}")
    print(f"w2 shape: {w2.shape}, sum: {w2.sum().item()}")
    
    # AITER result
    aiter_out = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                          activation=ActivationType.Silu,
                          quant_type=QuantType.No)
    
    # HipKittens result
    from aiter.hipkittens_moe import hipkittens_fused_moe
    hk_out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids,
                                   activation=ActivationType.Silu)
    
    print(f"\nAITER output:")
    print(f"  shape: {aiter_out.shape}")
    print(f"  row 0: {aiter_out[0, :8]}")
    
    print(f"\nHipKittens output:")
    print(f"  shape: {hk_out.shape}")
    print(f"  row 0: {hk_out[0, :8]}")
    
    diff = (aiter_out - hk_out).abs()
    print(f"\nDiff: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")


def test_stage_by_stage():
    """
    Compare stage by stage to isolate where the difference comes from.
    """
    print("\n" + "="*60)
    print("Stage-by-stage comparison")
    print("="*60)
    
    num_tokens = 8
    model_dim = 4096
    inter_dim = 4096
    num_experts = 8
    topk = 2
    block_m = 32
    
    hidden = torch.randn((num_tokens, model_dim), dtype=torch.bfloat16, device="cuda") / 10
    w1 = torch.randn((num_experts, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda") / 10
    w2 = torch.randn((num_experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda") / 10
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    topk_w, topk_ids = fused_topk(hidden, scores, topk, True)
    
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_w, num_experts, model_dim, hidden.dtype, block_m)
    
    sorted_M = sorted_ids.shape[0]
    print(f"sorted_M: {sorted_M}")
    
    # Manual computation following AITER convention
    # Stage 1: For each sorted row, compute hidden[token] @ w1[expert].T
    manual_inter = torch.zeros((sorted_M, inter_dim * 2), dtype=torch.float32, device="cuda")
    
    for row in range(sorted_M):
        packed_id = sorted_ids[row].item()
        token_id = packed_id & 0xFFFFFF
        tile_id = row // block_m
        expert_id = sorted_expert_ids[tile_id].item()
        
        if expert_id < 0 or expert_id >= num_experts or token_id >= num_tokens:
            continue
        
        h = hidden[token_id].float()
        w = w1[expert_id].float()
        manual_inter[row] = h @ w.T
    
    manual_inter_bf16 = manual_inter.to(torch.bfloat16)
    
    # HK Stage 1
    from aiter.hipkittens_moe import hipkittens_moe_stage1
    hk_inter = hipkittens_moe_stage1(
        hidden, w1, sorted_ids, sorted_expert_ids, num_valid_ids, topk, block_m)
    
    diff_s1 = (manual_inter_bf16 - hk_inter).abs()
    print(f"\nStage 1 comparison:")
    print(f"  Max diff: {diff_s1.max().item():.6f}")
    print(f"  Manual row 0: {manual_inter_bf16[0, :4]}")
    print(f"  HK row 0: {hk_inter[0, :4]}")
    
    # Now check what AITER Stage 1 produces
    # The AITER intermediate is [token_num, topk, inter_dim] after G1U1 activation
    # Let's call ck_moe_stage1 directly
    
    # Actually, AITER's stage1 includes the activation
    # So intermediate in AITER = SiLU(gate) * up, which is [token_num, topk, inter_dim]
    
    # Let's trace through what values token 0 should have
    print("\n--- Detailed trace for row 0 ---")
    row = 0
    packed_id = sorted_ids[row].item()
    token_id = packed_id & 0xFFFFFF
    topk_slot = packed_id >> 24
    tile_id = row // block_m
    expert_id = sorted_expert_ids[tile_id].item()
    
    print(f"Row 0: token_id={token_id}, topk_slot={topk_slot}, expert_id={expert_id}")
    
    h = hidden[token_id].float()
    w = w1[expert_id].float()
    
    # Manual Stage 1
    stage1_out = h @ w.T  # [inter_dim*2]
    print(f"Stage 1 output (first 4): {stage1_out[:4]}")
    
    # G1U1 activation
    gate = stage1_out[:inter_dim]
    up = stage1_out[inter_dim:]
    def silu(x):
        return x * torch.sigmoid(x)
    activated = silu(gate) * up
    print(f"Activated (first 4): {activated[:4]}")
    
    # Stage 2
    w2_exp = w2[expert_id].float()
    stage2_out = activated @ w2_exp.T  # [model_dim]
    weight = sorted_weights[row].item()
    final = weight * stage2_out
    print(f"Stage 2 output (first 4): {final[:4]}")
    
    # Compare with AITER
    aiter_out = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                          activation=ActivationType.Silu,
                          quant_type=QuantType.No)
    print(f"\nAITER output for token {token_id} (first 4): {aiter_out[token_id, :4]}")


def test_output_layout():
    """
    Check if the output layout/indexing is the issue.
    """
    print("\n" + "="*60)
    print("Output layout comparison")
    print("="*60)
    
    num_tokens = 4
    model_dim = 128
    inter_dim = 128
    num_experts = 2
    topk = 2
    
    # Create inputs where each token-expert pair has distinct values
    hidden = torch.zeros((num_tokens, model_dim), dtype=torch.bfloat16, device="cuda")
    for i in range(num_tokens):
        hidden[i] = i + 1  # Token i has value i+1 in all dims
    
    # Simple weights
    w1 = torch.ones((num_experts, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda") / model_dim
    w2 = torch.ones((num_experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda") / inter_dim
    
    # Different weights for different experts
    scores = torch.zeros((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    for i in range(num_tokens):
        scores[i, i % num_experts] = 1.0  # Token i routes to expert i%num_experts
        scores[i, (i + 1) % num_experts] = 0.5
    
    topk_w, topk_ids = fused_topk(hidden, scores, topk, True)
    
    print(f"topk_ids:\n{topk_ids}")
    print(f"topk_w:\n{topk_w}")
    
    # AITER
    aiter_out = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                          activation=ActivationType.Silu,
                          quant_type=QuantType.No)
    
    # HipKittens
    from aiter.hipkittens_moe import hipkittens_fused_moe
    hk_out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids,
                                   activation=ActivationType.Silu)
    
    print(f"\nAITER output:")
    for i in range(num_tokens):
        print(f"  Token {i}: {aiter_out[i, :4]}")
    
    print(f"\nHipKittens output:")
    for i in range(num_tokens):
        print(f"  Token {i}: {hk_out[i, :4]}")
    
    diff = (aiter_out - hk_out).abs()
    print(f"\nMax diff: {diff.max().item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    test_with_identity_weights()
    test_output_layout()
    test_stage_by_stage()

