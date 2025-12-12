#!/usr/bin/env python3
"""
Profile HipKittens MoE kernels using rocprof.

Usage:
    # Basic profiling with counters
    rocprofv3 --stats python profile_hipkittens_moe.py
    
    # Advanced Thread Tracing (for rocprof-compute-viewer)
    rocprofv3 --att --att-kernel "hk_moe" python profile_hipkittens_moe.py
    
    # With activity summary  
    rocprofv3 --att-activity 10 --att-kernel "hk_moe" python profile_hipkittens_moe.py
    
    # Hardware counters for MMA utilization
    rocprofv3 --pmc "SQ_VALU_MFMA_BUSY_CYCLES,SQ_INSTS_MFMA,SQ_INSTS_VALU,SQ_BUSY_CU_CYCLES" python profile_hipkittens_moe.py
"""

import torch
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiter.fused_moe import fused_topk
from aiter.hipkittens_moe import hipkittens_fused_moe
from aiter import ActivationType

def profile_hipkittens():
    """Run HipKittens MoE kernel for profiling."""
    
    # Use a reasonable batch size
    batch_size = 4096
    model_dim = 4096
    inter_dim = 4096
    num_experts = 8
    topk = 2
    
    print(f"Profiling HipKittens MoE:")
    print(f"  batch_size={batch_size}")
    print(f"  model_dim={model_dim}")
    print(f"  inter_dim={inter_dim}")
    print(f"  num_experts={num_experts}")
    print(f"  topk={topk}")
    
    # Create test tensors
    torch.manual_seed(42)
    hidden = torch.randn(batch_size, model_dim, dtype=torch.bfloat16, device='cuda')
    w1 = torch.randn(num_experts, inter_dim * 2, model_dim, dtype=torch.bfloat16, device='cuda')
    w2 = torch.randn(num_experts, model_dim, inter_dim, dtype=torch.bfloat16, device='cuda')
    scores = torch.randn(batch_size, num_experts, dtype=torch.float32, device='cuda')
    
    # Get topk routing
    topk_w, topk_ids = fused_topk(hidden, scores, topk, True)
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
        torch.cuda.synchronize()
    
    # Profile runs
    num_runs = 10
    print(f"Running {num_runs} iterations for profiling...")
    
    torch.cuda.synchronize()
    
    for i in range(num_runs):
        out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
    
    torch.cuda.synchronize()
    
    print("Profiling complete!")
    print(f"Output shape: {out.shape}")
    
    # Verify correctness
    print("Verifying correctness...")
    expected = torch.zeros_like(hidden)
    for b in range(batch_size):
        for k in range(topk):
            expert_id = topk_ids[b, k].item()
            weight = topk_w[b, k].item()
            x = hidden[b:b+1]
            gate_up = x @ w1[expert_id].T
            gate = gate_up[:, :inter_dim]
            up = gate_up[:, inter_dim:]
            activated = torch.nn.functional.silu(gate) * up
            down = activated @ w2[expert_id].T
            expected[b] += weight * down.squeeze(0)
    
    diff = (out.float() - expected.float()).abs().max().item()
    print(f"Max abs diff vs reference: {diff}")

if __name__ == "__main__":
    profile_hipkittens()

