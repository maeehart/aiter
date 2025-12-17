#!/usr/bin/env python3
"""Profile MoE kernel with rocprof for detailed performance analysis."""

import os
import torch

os.environ['VLLM_ROCM_USE_AITER'] = '1'

from aiter.fused_moe import fused_moe, QuantType, ActivationType

FP8_DTYPE = torch.float8_e4m3fnuz

def run_kernel(batch_size: int = 8192, num_iters: int = 10):
    """Run the kernel for profiling."""
    num_experts = 256
    hidden_size = 7168
    intermediate_size = 256
    topk = 8
    device = "cuda"
    
    shard_intermediate_size = intermediate_size * 2
    
    # Create inputs
    hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
    
    # FP8 weights
    w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, 
                    device=device, dtype=torch.bfloat16).to(FP8_DTYPE)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, 
                    device=device, dtype=torch.bfloat16).to(FP8_DTYPE)
    
    # Scales
    scale_shape_w1 = (num_experts, shard_intermediate_size // 128, hidden_size // 128)
    scale_shape_w2 = (num_experts, hidden_size // 128, intermediate_size // 128)
    w1_scale = torch.ones(scale_shape_w1, device=device, dtype=torch.float32) * 0.1
    w2_scale = torch.ones(scale_shape_w2, device=device, dtype=torch.float32) * 0.1
    
    # Random routing
    topk_weights = torch.rand(batch_size, topk, device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(0, num_experts, (batch_size, topk), device=device, dtype=torch.int32)
    
    # Warmup
    for _ in range(5):
        _ = fused_moe(
            hidden_states, w1, w2,
            topk_weights, topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x128,
            w1_scale=w1_scale, w2_scale=w2_scale,
        )
        torch.cuda.synchronize()
    
    print(f"Running {num_iters} iterations with batch_size={batch_size}")
    
    # Profile iterations
    for i in range(num_iters):
        _ = fused_moe(
            hidden_states, w1, w2,
            topk_weights, topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x128,
            w1_scale=w1_scale, w2_scale=w2_scale,
        )
        torch.cuda.synchronize()
    
    print("Done")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--num-iters', type=int, default=10)
    args = parser.parse_args()
    run_kernel(args.batch_size, args.num_iters)
