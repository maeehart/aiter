# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Debug script for HipKittens MoE Activation (G1U1 SiLU)
Tests the activation step in isolation.
"""

import torch
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting, fused_moe
from aiter import ActivationType, QuantType


def silu(x):
    """Python SiLU implementation."""
    return x * torch.sigmoid(x)


def reference_g1u1_activation(intermediate):
    """
    Reference G1U1 activation.
    intermediate: [M, inter_dim*2]
    Returns: [M, inter_dim] - activated first half
    """
    inter_dim = intermediate.shape[1] // 2
    gate = intermediate[:, :inter_dim].float()
    up = intermediate[:, inter_dim:].float()
    result = silu(gate) * up
    return result.to(intermediate.dtype)


def test_activation_sizes():
    """Test activation at various sizes."""
    print("="*60)
    print("Testing G1U1 Activation at various sizes")
    print("="*60)
    
    test_cases = [
        (100, 64),      # Small
        (100, 256),     # Medium
        (100, 1024),    # Large  
        (100, 4096),    # Very large (this is where full pipeline fails)
        (1000, 4096),   # High M
    ]
    
    for sorted_M, inter_dim in test_cases:
        print(f"\n--- M={sorted_M}, inter_dim={inter_dim} ---")
        
        # Create random intermediate tensor
        intermediate = torch.randn((sorted_M, inter_dim * 2), dtype=torch.bfloat16, device="cuda") / 10
        
        # Reference
        ref_activated = reference_g1u1_activation(intermediate)
        
        # Make a copy for the kernel to modify
        inter_copy = intermediate.clone()
        
        # The activation is done inline and then sliced
        # Let's simulate what the C++ code does
        # Call the activation kernel by importing the module
        
        # We can test by just computing what the sliced output should be
        # The activation writes to the first half, then we slice
        expected_sliced = ref_activated
        
        # Simulate the slice + contiguous
        inter_first_half = inter_copy[:, :inter_dim].contiguous()
        
        print(f"inter_copy stride: {inter_copy.stride()}")
        print(f"inter_first_half stride: {inter_first_half.stride()}")
        print(f"inter_first_half is contiguous: {inter_first_half.is_contiguous()}")
        
        # The issue: we need to check if slicing BEFORE activation gives wrong data
        # Let's trace through what happens in hk_moe_torch.cu:
        # 1. Stage 1 fills intermediate with gate-up projection
        # 2. Activation kernel modifies first half in-place
        # 3. We slice and make contiguous
        
        # The activation kernel should be modifying intermediate in-place
        # Let's check if our Python reference matches what an in-place op would do
        
        # Simulate in-place activation
        gate = inter_copy[:, :inter_dim].float()
        up = inter_copy[:, inter_dim:].float()
        activated = silu(gate) * up
        inter_copy[:, :inter_dim] = activated.to(torch.bfloat16)
        
        # Now slice
        sliced = inter_copy[:, :inter_dim].contiguous()
        
        # Compare
        diff = (ref_activated - sliced).abs()
        max_diff = diff.max().item()
        print(f"Max diff between ref and simulated: {max_diff:.6f}")
        
        if max_diff > 0.01:
            print("✗ MISMATCH")
        else:
            print("✓ OK")


def test_full_pipeline_with_activation_check():
    """Test full pipeline and check intermediate values."""
    print("\n" + "="*60)
    print("Testing full pipeline with activation debugging")
    print("="*60)
    
    cfg = {"model_dim": 4096, "inter_dim": 4096, "num_experts": 8, "topk": 2}
    num_tokens = 512
    
    hidden = torch.randn((num_tokens, cfg["model_dim"]), dtype=torch.bfloat16, device="cuda") / 10
    w1 = torch.randn((cfg["num_experts"], cfg["inter_dim"] * 2, cfg["model_dim"]), dtype=torch.bfloat16, device="cuda") / 10
    w2 = torch.randn((cfg["num_experts"], cfg["model_dim"], cfg["inter_dim"]), dtype=torch.bfloat16, device="cuda") / 10
    scores = torch.randn((num_tokens, cfg["num_experts"]), dtype=torch.float32, device="cuda")
    topk_w, topk_ids = fused_topk(hidden, scores, cfg["topk"], True)
    
    block_m = 32
    
    # Do sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_w, cfg["num_experts"], cfg["model_dim"], hidden.dtype, block_m)
    
    sorted_M = sorted_ids.shape[0]
    inter_dim = cfg["inter_dim"]
    
    print(f"sorted_M: {sorted_M}")
    
    # Manual Stage 1
    from aiter.hipkittens_moe import hipkittens_moe_stage1
    
    intermediate_hk = hipkittens_moe_stage1(
        hidden, w1, sorted_ids, sorted_expert_ids, num_valid_ids, cfg["topk"], block_m)
    
    print(f"Stage 1 output shape: {intermediate_hk.shape}")
    print(f"Stage 1 output sample: {intermediate_hk[0, :4]}")
    
    # Reference activation
    ref_activated = reference_g1u1_activation(intermediate_hk)
    
    print(f"Ref activated shape: {ref_activated.shape}")
    print(f"Ref activated sample: {ref_activated[0, :4]}")
    
    # Now manually do what hk_moe_torch.cu does for activation
    # The kernel is apply_g1u1_activation_kernel
    
    # For comparison, let's also check the AITER reference
    ref_output = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                           activation=ActivationType.Silu,
                           quant_type=QuantType.No)
    
    print(f"\nAITER reference output sample: {ref_output[0, :4]}")
    
    # HipKittens full pipeline
    from aiter.hipkittens_moe import hipkittens_fused_moe
    hk_output = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids,
                                      activation=ActivationType.Silu)
    
    print(f"HipKittens output sample: {hk_output[0, :4]}")
    
    diff = (ref_output - hk_output).abs()
    print(f"\nMax diff: {diff.max().item():.4f}")
    print(f"Elements within 1.0: {(diff < 1.0).float().mean().item() * 100:.1f}%")


def test_manual_pipeline():
    """
    Manually reconstruct the full pipeline to identify exactly where it breaks.
    """
    print("\n" + "="*60)
    print("Manual Pipeline Reconstruction")
    print("="*60)
    
    cfg = {"model_dim": 4096, "inter_dim": 4096, "num_experts": 8, "topk": 2}
    num_tokens = 256
    
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
    
    # Stage 1 using HK
    from aiter.hipkittens_moe import hipkittens_moe_stage1, hipkittens_moe_stage2
    
    intermediate = hipkittens_moe_stage1(
        hidden, w1, sorted_ids, sorted_expert_ids, num_valid_ids, cfg["topk"], block_m)
    
    print(f"Step 1: HK Stage 1 - OK (shape {intermediate.shape})")
    
    # Activation (Python reference)
    activated = reference_g1u1_activation(intermediate)
    print(f"Step 2: Python Activation - OK (shape {activated.shape})")
    
    # Stage 2 using HK with Python-activated intermediate
    output_manual = torch.zeros((num_tokens, cfg["model_dim"]), dtype=torch.bfloat16, device="cuda")
    hipkittens_moe_stage2(
        activated, w2, output_manual,
        sorted_ids, sorted_expert_ids, num_valid_ids, sorted_weights,
        cfg["topk"], block_m)
    
    print(f"Step 3: HK Stage 2 - OK (shape {output_manual.shape})")
    
    # Compare with AITER reference
    ref_output = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                           activation=ActivationType.Silu,
                           quant_type=QuantType.No)
    
    diff = (ref_output - output_manual).abs()
    print(f"\nManual pipeline vs AITER:")
    print(f"  Max diff: {diff.max().item():.4f}")
    print(f"  Elements within 1.0: {(diff < 1.0).float().mean().item() * 100:.1f}%")
    
    # Now compare with full HK pipeline
    from aiter.hipkittens_moe import hipkittens_fused_moe
    hk_output = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids,
                                      activation=ActivationType.Silu)
    
    diff_hk = (ref_output - hk_output).abs()
    print(f"\nFull HK pipeline vs AITER:")
    print(f"  Max diff: {diff_hk.max().item():.4f}")
    print(f"  Elements within 1.0: {(diff_hk < 1.0).float().mean().item() * 100:.1f}%")
    
    # Compare manual vs full HK
    diff_manual_hk = (output_manual - hk_output).abs()
    print(f"\nManual pipeline vs Full HK pipeline:")
    print(f"  Max diff: {diff_manual_hk.max().item():.4f}")
    print(f"  Elements within 1.0: {(diff_manual_hk < 1.0).float().mean().item() * 100:.1f}%")
    
    if (diff < 1.0).float().mean().item() > 0.99:
        print("\n✓ Manual pipeline (HK Stage 1 + Python Act + HK Stage 2) works!")
        print("  The issue is in the C++ activation kernel.")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    test_activation_sizes()
    test_full_pipeline_with_activation_check()
    test_manual_pipeline()

