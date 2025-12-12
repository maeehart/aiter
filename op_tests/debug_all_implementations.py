# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Compare all MoE implementations:
1. torch_moe - Pure PyTorch reference (ground truth)
2. fused_moe - AITER implementation (ASM/CK)
3. hipkittens_fused_moe - HipKittens implementation

This helps identify where differences originate.
"""

import torch
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting, fused_moe, torch_moe
from aiter import ActivationType, QuantType

try:
    from aiter.hipkittens_moe import hipkittens_fused_moe
    HK_AVAILABLE = True
except ImportError:
    HK_AVAILABLE = False
    print("HipKittens MoE not available")


def create_inputs(num_tokens, model_dim, inter_dim, num_experts, topk, 
                  dtype=torch.bfloat16, device="cuda"):
    """Create random inputs."""
    hidden = torch.randn((num_tokens, model_dim), dtype=dtype, device=device) / 10
    w1 = torch.randn((num_experts, inter_dim * 2, model_dim), dtype=dtype, device=device) / 10
    w2 = torch.randn((num_experts, model_dim, inter_dim), dtype=dtype, device=device) / 10
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device)
    topk_w, topk_ids = fused_topk(hidden, scores, topk, True)
    return hidden, w1, w2, topk_w, topk_ids


def compare_two(name1, out1, name2, out2, tol=1.0):
    """Compare two outputs and print statistics."""
    diff = (out1 - out2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    pct = (diff < tol).float().mean().item() * 100
    
    status = "✓" if pct > 99.0 else "✗"
    print(f"  {status} {name1} vs {name2}: max={max_diff:.4f}, mean={mean_diff:.4f}, {pct:.1f}% within {tol}")
    return pct > 99.0


def test_all_implementations(num_tokens, model_dim, inter_dim, num_experts, topk):
    """Test all implementations and compare."""
    print(f"\n{'='*70}")
    print(f"Testing: tokens={num_tokens}, model_dim={model_dim}, inter_dim={inter_dim}")
    print(f"         experts={num_experts}, topk={topk}")
    print(f"{'='*70}")
    
    hidden, w1, w2, topk_w, topk_ids = create_inputs(
        num_tokens, model_dim, inter_dim, num_experts, topk)
    
    # 1. PyTorch reference (ground truth)
    torch_out = torch_moe(hidden, w1, w2, topk_w, topk_ids, 
                          activation=ActivationType.Silu)
    print(f"PyTorch ref: {torch_out[0, :4]}")
    
    # 2. AITER fused_moe
    aiter_out = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                          activation=ActivationType.Silu,
                          quant_type=QuantType.No)
    print(f"AITER:       {aiter_out[0, :4]}")
    
    # 3. HipKittens (if available)
    if HK_AVAILABLE:
        hk_out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids,
                                       activation=ActivationType.Silu)
        print(f"HipKittens:  {hk_out[0, :4]}")
    
    print("\nComparisons:")
    results = {}
    
    # Compare PyTorch vs AITER
    results['torch_vs_aiter'] = compare_two("PyTorch", torch_out, "AITER", aiter_out)
    
    # Compare HipKittens vs others
    if HK_AVAILABLE:
        results['torch_vs_hk'] = compare_two("PyTorch", torch_out, "HipKittens", hk_out)
        results['aiter_vs_hk'] = compare_two("AITER", aiter_out, "HipKittens", hk_out)
    
    return results


def run_size_sweep():
    """Run tests at various sizes."""
    print("\n" + "="*70)
    print("SIZE SWEEP: Finding where implementations diverge")
    print("="*70)
    
    test_cases = [
        # (tokens, model_dim, inter_dim, experts, topk)
        (16, 64, 64, 4, 2),
        (16, 128, 128, 4, 2),
        (16, 256, 256, 4, 2),
        (32, 512, 512, 8, 2),
        (64, 1024, 1024, 8, 2),
        (64, 2048, 2048, 8, 2),
        (64, 4096, 4096, 8, 2),
        (256, 4096, 4096, 8, 2),
    ]
    
    results = []
    for tokens, model_dim, inter_dim, experts, topk in test_cases:
        result = test_all_implementations(tokens, model_dim, inter_dim, experts, topk)
        results.append({
            'config': (tokens, model_dim, inter_dim, experts, topk),
            'results': result
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<40} | {'Torch vs AITER':<15} | {'Torch vs HK':<15}")
    print("-"*70)
    
    for r in results:
        cfg = f"t={r['config'][0]}, d={r['config'][1]}"
        torch_aiter = "PASS" if r['results'].get('torch_vs_aiter', False) else "FAIL"
        torch_hk = "PASS" if r['results'].get('torch_vs_hk', False) else "FAIL"
        print(f"{cfg:<40} | {torch_aiter:<15} | {torch_hk:<15}")


def detailed_comparison():
    """Detailed comparison at a specific size."""
    print("\n" + "="*70)
    print("DETAILED COMPARISON at model_dim=4096")
    print("="*70)
    
    num_tokens = 32
    model_dim = 4096
    inter_dim = 4096
    num_experts = 8
    topk = 2
    
    hidden, w1, w2, topk_w, topk_ids = create_inputs(
        num_tokens, model_dim, inter_dim, num_experts, topk)
    
    # PyTorch reference
    torch_out = torch_moe(hidden, w1, w2, topk_w, topk_ids,
                          activation=ActivationType.Silu)
    
    # AITER
    aiter_out = fused_moe(hidden, w1, w2, topk_w, topk_ids,
                          activation=ActivationType.Silu,
                          quant_type=QuantType.No)
    
    # HipKittens
    if HK_AVAILABLE:
        hk_out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids,
                                       activation=ActivationType.Silu)
    
    # Detailed analysis
    print("\n--- PyTorch vs AITER ---")
    diff_torch_aiter = (torch_out - aiter_out).abs()
    print(f"Max diff: {diff_torch_aiter.max().item():.4f}")
    print(f"Mean diff: {diff_torch_aiter.mean().item():.4f}")
    print(f"Std diff: {diff_torch_aiter.std().item():.4f}")
    
    # Check ratio
    torch_flat = torch_out.flatten().float()
    aiter_flat = aiter_out.flatten().float()
    mask = torch_flat.abs() > 0.1
    if mask.sum() > 0:
        ratios = aiter_flat[mask] / torch_flat[mask]
        print(f"Ratio (AITER/Torch) for |torch|>0.1:")
        print(f"  Mean: {ratios.mean().item():.4f}")
        print(f"  Std: {ratios.std().item():.4f}")
    
    if HK_AVAILABLE:
        print("\n--- PyTorch vs HipKittens ---")
        diff_torch_hk = (torch_out - hk_out).abs()
        print(f"Max diff: {diff_torch_hk.max().item():.4f}")
        print(f"Mean diff: {diff_torch_hk.mean().item():.4f}")
        
        print("\n--- AITER vs HipKittens ---")
        diff_aiter_hk = (aiter_out - hk_out).abs()
        print(f"Max diff: {diff_aiter_hk.max().item():.4f}")
        print(f"Mean diff: {diff_aiter_hk.mean().item():.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    run_size_sweep()
    detailed_comparison()

