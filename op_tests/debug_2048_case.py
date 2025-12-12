# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Debug the edge case at model_dim=2048 where HipKittens shows larger errors.
"""

import torch
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe
from aiter import ActivationType, QuantType
from aiter.hipkittens_moe import hipkittens_fused_moe


def analyze_2048_case():
    """Analyze why there are outliers at model_dim=2048."""
    torch.manual_seed(42)
    
    num_tokens = 64
    model_dim = 2048
    inter_dim = 2048
    num_experts = 8
    topk = 2
    
    print("="*70)
    print(f"Analyzing model_dim={model_dim}, tokens={num_tokens}")
    print("="*70)
    
    hidden = torch.randn((num_tokens, model_dim), dtype=torch.bfloat16, device="cuda") / 10
    w1 = torch.randn((num_experts, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda") / 10
    w2 = torch.randn((num_experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda") / 10
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    topk_w, topk_ids = fused_topk(hidden, scores, topk, True)
    
    # PyTorch reference
    torch_out = torch_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
    
    # HipKittens
    hk_out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
    
    # Find outliers
    diff = (torch_out - hk_out).abs()
    max_diff = diff.max().item()
    
    # Find rows with largest errors
    row_errors = diff.max(dim=1).values
    worst_rows = torch.argsort(row_errors, descending=True)[:10]
    
    print(f"\nOverall max diff: {max_diff:.4f}")
    print(f"Mean diff: {diff.mean().item():.4f}")
    print(f"Elements within 1.0: {(diff < 1.0).float().mean().item() * 100:.1f}%")
    
    print("\n10 rows with largest errors:")
    for row in worst_rows:
        row_max = row_errors[row].item()
        print(f"  Row {row.item():3d}: max_err={row_max:.4f}")
    
    # Analyze the worst row
    worst_row = worst_rows[0].item()
    print(f"\nWorst row ({worst_row}) details:")
    print(f"  PyTorch: {torch_out[worst_row, :4]}")
    print(f"  HK:      {hk_out[worst_row, :4]}")
    print(f"  topk_ids for this token: {topk_ids[worst_row]}")
    print(f"  topk_weights for this token: {topk_w[worst_row]}")


def test_varying_tokens_at_2048():
    """Test if the issue is related to token count."""
    torch.manual_seed(42)
    
    model_dim = 2048
    inter_dim = 2048
    num_experts = 8
    topk = 2
    
    print("\n" + "="*70)
    print("Testing different token counts at model_dim=2048")
    print("="*70)
    
    for num_tokens in [16, 32, 64, 128, 256]:
        hidden = torch.randn((num_tokens, model_dim), dtype=torch.bfloat16, device="cuda") / 10
        w1 = torch.randn((num_experts, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda") / 10
        w2 = torch.randn((num_experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda") / 10
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
        topk_w, topk_ids = fused_topk(hidden, scores, topk, True)
        
        torch_out = torch_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
        hk_out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
        
        diff = (torch_out - hk_out).abs()
        max_diff = diff.max().item()
        pct = (diff < 1.0).float().mean().item() * 100
        
        status = "✓" if pct > 99.0 else "✗"
        print(f"  {status} tokens={num_tokens:4d}: max={max_diff:.4f}, {pct:.1f}% within 1.0")


def test_at_4096_detailed():
    """Test at 4096 with different token counts."""
    torch.manual_seed(42)
    
    model_dim = 4096
    inter_dim = 4096
    num_experts = 8
    topk = 2
    
    print("\n" + "="*70)
    print("Testing different token counts at model_dim=4096")
    print("="*70)
    
    for num_tokens in [16, 32, 64, 128, 256, 512]:
        hidden = torch.randn((num_tokens, model_dim), dtype=torch.bfloat16, device="cuda") / 10
        w1 = torch.randn((num_experts, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda") / 10
        w2 = torch.randn((num_experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda") / 10
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
        topk_w, topk_ids = fused_topk(hidden, scores, topk, True)
        
        torch_out = torch_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
        hk_out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
        
        diff = (torch_out - hk_out).abs()
        max_diff = diff.max().item()
        pct = (diff < 1.0).float().mean().item() * 100
        
        status = "✓" if pct > 99.0 else "✗"
        print(f"  {status} tokens={num_tokens:4d}: max={max_diff:.4f}, {pct:.1f}% within 1.0")


if __name__ == "__main__":
    analyze_2048_case()
    test_varying_tokens_at_2048()
    test_at_4096_detailed()

