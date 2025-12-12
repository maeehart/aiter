# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Debug script for full HipKittens MoE pipeline
Tests each stage and the full pipeline at various sizes.
"""

import torch
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting, fused_moe
from aiter import ActivationType, QuantType

# Config for debugging
SMALL_CONFIG = {
    "model_dim": 128,
    "inter_dim": 128,
    "num_experts": 4,
    "topk": 2,
}

MEDIUM_CONFIG = {
    "model_dim": 1024,
    "inter_dim": 1024,
    "num_experts": 8,
    "topk": 2,
}

LARGE_CONFIG = {
    "model_dim": 4096,
    "inter_dim": 4096,
    "num_experts": 8,
    "topk": 2,
}


def create_inputs(num_tokens, cfg, dtype=torch.bfloat16, device="cuda"):
    """Create random inputs."""
    hidden = torch.randn((num_tokens, cfg["model_dim"]), dtype=dtype, device=device) / 10
    w1 = torch.randn((cfg["num_experts"], cfg["inter_dim"] * 2, cfg["model_dim"]), dtype=dtype, device=device) / 10
    w2 = torch.randn((cfg["num_experts"], cfg["model_dim"], cfg["inter_dim"]), dtype=dtype, device=device) / 10
    scores = torch.randn((num_tokens, cfg["num_experts"]), dtype=torch.float32, device=device)
    topk_w, topk_ids = fused_topk(hidden, scores, cfg["topk"], True)
    return hidden, w1, w2, topk_w, topk_ids


def test_full_pipeline(num_tokens, cfg, label):
    """Test full MoE pipeline."""
    print(f"\n{'='*60}")
    print(f"{label}: {num_tokens} tokens, model_dim={cfg['model_dim']}")
    print(f"{'='*60}")
    
    hidden, w1, w2, topk_w, topk_ids = create_inputs(num_tokens, cfg)
    
    # Reference: AITER fused_moe
    ref_output = fused_moe(hidden, w1, w2, topk_w, topk_ids, 
                           activation=ActivationType.Silu, 
                           quant_type=QuantType.No)
    
    # HipKittens
    from aiter.hipkittens_moe import hipkittens_fused_moe
    hk_output = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, 
                                      activation=ActivationType.Silu)
    
    # Compare
    diff = (ref_output - hk_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pct_close = (diff < 1.0).float().mean().item() * 100
    
    print(f"Max diff: {max_diff:.4f}")
    print(f"Mean diff: {mean_diff:.4f}")
    print(f"Elements within 1.0: {pct_close:.1f}%")
    
    print(f"Ref output sample: {ref_output[0, :4]}")
    print(f"HK output sample: {hk_output[0, :4]}")
    
    if pct_close < 99.0:
        print(f"✗ FAILED - {pct_close:.1f}% elements correct")
        
        # Find worst element
        flat_diff = diff.flatten()
        worst_idx = flat_diff.argmax().item()
        row = worst_idx // cfg['model_dim']
        col = worst_idx % cfg['model_dim']
        print(f"\nWorst diff at [{row}, {col}]:")
        print(f"  Ref: {ref_output[row, col].item():.4f}")
        print(f"  HK:  {hk_output[row, col].item():.4f}")
        return False
    else:
        print(f"✓ PASSED")
        return True


def test_stage1_various_sizes():
    """Test Stage 1 at various sizes."""
    from aiter.hipkittens_moe import hipkittens_moe_stage1
    
    print("\n" + "="*60)
    print("Testing Stage 1 at various sizes")
    print("="*60)
    
    tests = [
        (16, SMALL_CONFIG, "Small"),
        (128, SMALL_CONFIG, "Small-128"),
        (256, MEDIUM_CONFIG, "Medium-256"),
        (512, MEDIUM_CONFIG, "Medium-512"),
        (1024, LARGE_CONFIG, "Large-1024"),
    ]
    
    for num_tokens, cfg, label in tests:
        print(f"\n--- {label}: {num_tokens} tokens ---")
        
        hidden, w1, w2, topk_w, topk_ids = create_inputs(num_tokens, cfg)
        block_m = 32  # Fixed to match kernel BLOCK_M
        
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
            topk_ids, topk_w, cfg["num_experts"], cfg["model_dim"], hidden.dtype, block_m)
        
        # Reference Stage 1 (manual computation)
        sorted_M = sorted_ids.shape[0]
        inter_dim = w1.shape[1] // 2
        ref_output = torch.zeros((sorted_M, inter_dim * 2), dtype=hidden.dtype, device=hidden.device)
        
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
            result = hidden_row @ w1_expert.T
            ref_output[row] = result.to(hidden.dtype)
        
        # HipKittens Stage 1
        hk_output = hipkittens_moe_stage1(
            hidden, w1, sorted_ids, sorted_expert_ids, num_valid_ids, cfg["topk"], block_m)
        
        # Compare
        diff = (ref_output - hk_output).abs()
        max_diff = diff.max().item()
        pct_close = (diff < 1.0).float().mean().item() * 100
        
        print(f"Max diff: {max_diff:.4f}, Elements within 1.0: {pct_close:.1f}%")
        
        if pct_close < 99.0:
            print(f"✗ Stage 1 FAILED")
        else:
            print(f"✓ Stage 1 PASSED")


def test_scaling():
    """Test to see if there's a systematic scaling issue."""
    print("\n" + "="*60)
    print("Testing for scaling issues")
    print("="*60)
    
    cfg = MEDIUM_CONFIG
    num_tokens = 256
    
    hidden, w1, w2, topk_w, topk_ids = create_inputs(num_tokens, cfg)
    
    # Reference
    ref_output = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                           activation=ActivationType.Silu,
                           quant_type=QuantType.No)
    
    # HipKittens
    from aiter.hipkittens_moe import hipkittens_fused_moe
    hk_output = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids,
                                      activation=ActivationType.Silu)
    
    # Check ratio
    ref_flat = ref_output.flatten().float()
    hk_flat = hk_output.flatten().float()
    
    # Filter out near-zero values
    mask = ref_flat.abs() > 0.01
    if mask.sum() > 0:
        ref_nonzero = ref_flat[mask]
        hk_nonzero = hk_flat[mask]
        ratios = hk_nonzero / ref_nonzero
        
        print(f"Ratio stats (HK/Ref) for non-zero elements:")
        print(f"  Mean: {ratios.mean().item():.4f}")
        print(f"  Std: {ratios.std().item():.4f}")
        print(f"  Min: {ratios.min().item():.4f}")
        print(f"  Max: {ratios.max().item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test Stage 1 at various sizes first
    test_stage1_various_sizes()
    
    # Test scaling issues
    test_scaling()
    
    # Test full pipeline at various sizes
    results = []
    results.append(test_full_pipeline(32, SMALL_CONFIG, "Small-32"))
    results.append(test_full_pipeline(64, SMALL_CONFIG, "Small-64"))
    results.append(test_full_pipeline(128, MEDIUM_CONFIG, "Medium-128"))
    results.append(test_full_pipeline(256, MEDIUM_CONFIG, "Medium-256"))
    results.append(test_full_pipeline(512, LARGE_CONFIG, "Large-512"))
    results.append(test_full_pipeline(1024, LARGE_CONFIG, "Large-1024"))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Passed: {sum(results)}/{len(results)}")

