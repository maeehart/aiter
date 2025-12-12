# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Debug script for HipKittens MoE Stage 2
Tests Stage 2 in isolation to identify correctness issues.
"""

import torch
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting

CONFIGS = {
    "small": {"model_dim": 128, "inter_dim": 128, "num_experts": 4, "topk": 2},
    "medium": {"model_dim": 1024, "inter_dim": 1024, "num_experts": 8, "topk": 2},
    "large": {"model_dim": 4096, "inter_dim": 4096, "num_experts": 8, "topk": 2},
}


def create_inputs(num_tokens, cfg, dtype=torch.bfloat16, device="cuda"):
    """Create random inputs."""
    hidden = torch.randn((num_tokens, cfg["model_dim"]), dtype=dtype, device=device) / 10
    w2 = torch.randn((cfg["num_experts"], cfg["model_dim"], cfg["inter_dim"]), dtype=dtype, device=device) / 10
    scores = torch.randn((num_tokens, cfg["num_experts"]), dtype=torch.float32, device=device)
    topk_w, topk_ids = fused_topk(hidden, scores, cfg["topk"], True)
    return hidden, w2, topk_w, topk_ids


def reference_stage2(intermediate, w2, sorted_ids, sorted_expert_ids, sorted_weights, 
                     block_m, num_tokens, model_dim, num_experts):
    """
    Reference Stage 2 implementation.
    output[token_id] += weight * (intermediate[row] @ w2[expert]^T)
    """
    sorted_M = intermediate.shape[0]
    inter_dim = intermediate.shape[1]
    
    output = torch.zeros((num_tokens, model_dim), dtype=torch.float32, device=intermediate.device)
    
    for row in range(sorted_M):
        packed_id = sorted_ids[row].item()
        token_id = packed_id & 0xFFFFFF
        
        tile_id = row // block_m
        expert_id = sorted_expert_ids[tile_id].item()
        
        if expert_id < 0 or expert_id >= num_experts:
            continue
        if token_id < 0 or token_id >= num_tokens:
            continue
        
        weight = sorted_weights[row].item()
        
        # Matrix multiply: intermediate[row] @ w2[expert].T
        inter_row = intermediate[row].float()  # [inter_dim]
        w2_expert = w2[expert_id].float()  # [model_dim, inter_dim]
        result = inter_row @ w2_expert.T  # [model_dim]
        
        output[token_id] += weight * result
    
    return output.to(torch.bfloat16)


def test_stage2(num_tokens, cfg, label):
    """Test Stage 2 in isolation."""
    print(f"\n{'='*60}")
    print(f"Stage 2 Test - {label}: {num_tokens} tokens, model_dim={cfg['model_dim']}")
    print(f"{'='*60}")
    
    hidden, w2, topk_w, topk_ids = create_inputs(num_tokens, cfg)
    block_m = 32  # Fixed to match kernel
    
    # Do sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_w, cfg["num_experts"], cfg["model_dim"], hidden.dtype, block_m)
    
    sorted_M = sorted_ids.shape[0]
    inter_dim = cfg["inter_dim"]
    
    print(f"sorted_M: {sorted_M}")
    print(f"inter_dim: {inter_dim}")
    print(f"model_dim: {cfg['model_dim']}")
    
    # Create fake intermediate (normally this comes from Stage 1 + activation)
    intermediate = torch.randn((sorted_M, inter_dim), dtype=torch.bfloat16, device="cuda") / 10
    
    # Reference Stage 2
    ref_output = reference_stage2(
        intermediate, w2, sorted_ids, sorted_expert_ids, sorted_weights,
        block_m, num_tokens, cfg['model_dim'], cfg['num_experts'])
    
    # HipKittens Stage 2
    from aiter.hipkittens_moe import hipkittens_moe_stage2
    
    hk_output = torch.zeros((num_tokens, cfg['model_dim']), dtype=torch.bfloat16, device="cuda")
    hipkittens_moe_stage2(
        intermediate, w2, hk_output,
        sorted_ids, sorted_expert_ids, num_valid_ids, sorted_weights,
        cfg['topk'], block_m)
    
    # Compare
    diff = (ref_output - hk_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pct_close = (diff < 1.0).float().mean().item() * 100
    
    print(f"Max diff: {max_diff:.4f}")
    print(f"Mean diff: {mean_diff:.4f}")
    print(f"Elements within 1.0: {pct_close:.1f}%")
    
    print(f"Ref output sample [0, :4]: {ref_output[0, :4]}")
    print(f"HK output sample [0, :4]: {hk_output[0, :4]}")
    
    if pct_close < 99.0:
        print(f"✗ Stage 2 FAILED")
        
        # Find pattern of errors
        flat_diff = diff.flatten()
        err_mask = flat_diff > 1.0
        print(f"\nError pattern analysis:")
        print(f"  Total elements: {flat_diff.numel()}")
        print(f"  Error elements: {err_mask.sum().item()}")
        
        # Check if errors are in specific columns (model_dim positions)
        col_errors = diff.sum(dim=0)
        print(f"\nColumn error distribution (first 10):")
        for i in range(min(10, cfg['model_dim'])):
            print(f"  col {i}: {col_errors[i].item():.4f}")
        
        # Check specific row
        row_with_error = diff.sum(dim=1).argmax().item()
        print(f"\nRow with max error: {row_with_error}")
        print(f"  Ref: {ref_output[row_with_error, :4]}")
        print(f"  HK:  {hk_output[row_with_error, :4]}")
        
        return False
    else:
        print(f"✓ Stage 2 PASSED")
        return True


def test_stage2_detailed_small():
    """Detailed test with very small dimensions to trace exact behavior."""
    print("\n" + "="*60)
    print("Detailed Stage 2 test with minimal dimensions")
    print("="*60)
    
    num_tokens = 4
    cfg = {"model_dim": 64, "inter_dim": 64, "num_experts": 2, "topk": 1}
    
    hidden, w2, topk_w, topk_ids = create_inputs(num_tokens, cfg)
    block_m = 32
    
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_w, cfg["num_experts"], cfg["model_dim"], hidden.dtype, block_m)
    
    sorted_M = sorted_ids.shape[0]
    inter_dim = cfg["inter_dim"]
    
    print(f"sorted_M: {sorted_M}")
    print(f"sorted_ids: {sorted_ids[:8]}")
    print(f"sorted_expert_ids: {sorted_expert_ids}")
    print(f"sorted_weights: {sorted_weights[:8]}")
    
    # Create simple intermediate
    intermediate = torch.ones((sorted_M, inter_dim), dtype=torch.bfloat16, device="cuda") * 0.1
    
    # Reference
    ref_output = reference_stage2(
        intermediate, w2, sorted_ids, sorted_expert_ids, sorted_weights,
        block_m, num_tokens, cfg['model_dim'], cfg['num_experts'])
    
    # HipKittens
    from aiter.hipkittens_moe import hipkittens_moe_stage2
    
    hk_output = torch.zeros((num_tokens, cfg['model_dim']), dtype=torch.bfloat16, device="cuda")
    hipkittens_moe_stage2(
        intermediate, w2, hk_output,
        sorted_ids, sorted_expert_ids, num_valid_ids, sorted_weights,
        cfg['topk'], block_m)
    
    print(f"\nReference output:")
    for i in range(num_tokens):
        print(f"  token {i}: {ref_output[i, :8]}")
    
    print(f"\nHipKittens output:")
    for i in range(num_tokens):
        print(f"  token {i}: {hk_output[i, :8]}")
    
    diff = (ref_output - hk_output).abs()
    print(f"\nMax diff: {diff.max().item():.6f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Detailed small test
    test_stage2_detailed_small()
    
    # Test various sizes
    results = []
    for label, cfg in CONFIGS.items():
        for num_tokens in [32, 64, 256]:
            results.append(test_stage2(num_tokens, cfg, f"{label}-{num_tokens}"))
    
    print("\n" + "="*60)
    print(f"Summary: {sum(results)}/{len(results)} passed")
    print("="*60)

