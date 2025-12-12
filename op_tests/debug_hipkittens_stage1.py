# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Debug script for HipKittens MoE Stage 1
Tests stage 1 in isolation to identify correctness issues.
"""

import torch
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting

# Simple configuration for debugging
DEBUG_CONFIG = {
    "model_dim": 128,      # Small for debugging
    "inter_dim": 128,      # Small
    "num_experts": 4,
    "topk": 2,
    "num_tokens": 16,      # Small batch
}


def create_simple_inputs(cfg, dtype=torch.bfloat16, device="cuda"):
    """Create simple deterministic inputs for debugging."""
    hidden = torch.randn((cfg["num_tokens"], cfg["model_dim"]), dtype=dtype, device=device)
    # Use small identity-like weights for easier debugging
    w1 = torch.randn((cfg["num_experts"], cfg["inter_dim"] * 2, cfg["model_dim"]), dtype=dtype, device=device) / 10
    scores = torch.randn((cfg["num_tokens"], cfg["num_experts"]), dtype=torch.float32, device=device)
    topk_w, topk_ids = fused_topk(hidden, scores, cfg["topk"], True)
    return hidden, w1, topk_w, topk_ids


def reference_stage1(hidden_states, w1, sorted_ids, sorted_expert_ids, block_m, num_tokens):
    """
    Reference implementation for Stage 1 in Python.
    Stage 1: output[sorted_row] = hidden_states[token_id] @ w1[expert]^T
    """
    sorted_M = sorted_ids.shape[0]
    inter_dim = w1.shape[1] // 2  # G1U1
    model_dim = w1.shape[2]
    
    output = torch.zeros((sorted_M, inter_dim * 2), dtype=hidden_states.dtype, device=hidden_states.device)
    
    for row in range(sorted_M):
        packed_id = sorted_ids[row].item()
        token_id = packed_id & 0xFFFFFF
        
        # Get expert from tile
        tile_id = row // block_m
        expert_id = sorted_expert_ids[tile_id].item()
        
        if expert_id < 0 or expert_id >= w1.shape[0]:
            continue  # Invalid expert
        
        if token_id < 0 or token_id >= num_tokens:
            continue  # Invalid token
            
        # Matrix multiply: hidden[token_id] @ w1[expert].T
        hidden_row = hidden_states[token_id].float()  # [model_dim]
        w1_expert = w1[expert_id].float()  # [inter_dim*2, model_dim]
        result = hidden_row @ w1_expert.T  # [inter_dim*2]
        output[row] = result.to(hidden_states.dtype)
    
    return output


def test_stage1_reference():
    """Compare HipKittens Stage 1 with reference implementation."""
    cfg = DEBUG_CONFIG
    hidden, w1, topk_w, topk_ids = create_simple_inputs(cfg)
    
    block_m = 32  # Match kernel BLOCK_M
    
    # Do sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_w,
        cfg["num_experts"],
        cfg["model_dim"],
        hidden.dtype,
        block_m,
    )
    
    print(f"sorted_ids shape: {sorted_ids.shape}")
    print(f"sorted_expert_ids shape: {sorted_expert_ids.shape}")
    print(f"block_m: {block_m}")
    print(f"sorted_M: {sorted_ids.shape[0]}")
    
    # Print some sorted_ids and decoded token_ids
    print("\nFirst 10 sorted_ids and decoded token_ids:")
    for i in range(min(10, sorted_ids.shape[0])):
        packed = sorted_ids[i].item()
        token_id = packed & 0xFFFFFF
        topk_slot = packed >> 24
        print(f"  row {i}: packed={packed}, token_id={token_id}, topk_slot={topk_slot}")
    
    # Print expert assignments
    print(f"\nsorted_expert_ids (one per {block_m} rows):")
    for i in range(min(10, sorted_expert_ids.shape[0])):
        print(f"  tile {i}: expert={sorted_expert_ids[i].item()}")
    
    # Reference stage 1
    ref_output = reference_stage1(hidden, w1, sorted_ids, sorted_expert_ids, block_m, cfg["num_tokens"])
    
    print(f"\nReference output shape: {ref_output.shape}")
    print(f"Reference output sample: {ref_output[0, :8]}")
    
    # Now call the HipKittens stage 1
    try:
        from aiter.hipkittens_moe import hipkittens_moe_stage1
        
        hk_output = hipkittens_moe_stage1(
            hidden, w1,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            cfg["topk"], block_m
        )
        
        print(f"\nHipKittens output shape: {hk_output.shape}")
        print(f"HipKittens output sample: {hk_output[0, :8]}")
        
        # Compare
        try:
            checkAllclose(ref_output, hk_output, rtol=0.05, atol=0.5, msg="Stage 1 comparison")
            print("\n✓ Stage 1 PASSED!")
        except AssertionError as e:
            print(f"\n✗ Stage 1 FAILED")
            
            # Find first mismatch
            diff = (ref_output - hk_output).abs()
            max_diff_idx = diff.argmax()
            row = max_diff_idx // ref_output.shape[1]
            col = max_diff_idx % ref_output.shape[1]
            print(f"\nMax diff at [{row}, {col}]:")
            print(f"  Reference: {ref_output[row, col]}")
            print(f"  HipKittens: {hk_output[row, col]}")
            print(f"  Diff: {diff[row, col]}")
            
            # Check how many elements match
            close_mask = (diff < 1.0)
            pct_close = close_mask.float().mean().item() * 100
            print(f"\nElements within tolerance: {pct_close:.1f}%")
            
    except ImportError as e:
        print(f"Could not import hipkittens_moe: {e}")
        print("Testing reference implementation only.")


def test_simple_matmul():
    """Test that the basic matmul operation works."""
    print("\n" + "="*60)
    print("Testing simple matrix multiply pattern")
    print("="*60)
    
    cfg = DEBUG_CONFIG
    hidden, w1, topk_w, topk_ids = create_simple_inputs(cfg)
    
    # Manual computation for token 0, expert 0
    expert_id = 0
    token_id = 0
    
    hidden_vec = hidden[token_id].float()  # [model_dim]
    w1_mat = w1[expert_id].float()  # [inter_dim*2, model_dim]
    
    result = hidden_vec @ w1_mat.T  # [inter_dim*2]
    
    print(f"hidden[{token_id}] shape: {hidden_vec.shape}, first 4: {hidden_vec[:4]}")
    print(f"w1[{expert_id}] shape: {w1_mat.shape}")
    print(f"result shape: {result.shape}, first 4: {result[:4]}")
    

if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("="*60)
    print("HipKittens MoE Stage 1 Debug Test")
    print("="*60)
    
    test_simple_matmul()
    
    print("\n" + "="*60)
    print("Testing Stage 1 vs Reference")
    print("="*60)
    
    test_stage1_reference()

