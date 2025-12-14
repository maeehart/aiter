#!/usr/bin/env python3
"""Profile HipKittens MoE kernels using rocprofv3.

This script is intended to be run under rocprofv3 for traces/PMCs.
We intentionally do NOT compute a full PyTorch reference here (too slow and pollutes profiles).

Example:
  rocprofv3 --pmc "SQ_INSTS_MFMA,SQ_INSTS_VALU,SQ_BUSY_CU_CYCLES" -- \
    python op_tests/profile_hipkittens_moe.py --batch-size 8192 --warmup 3 --iters 10
"""

import argparse
import os
import sys

import torch

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiter.fused_moe import fused_topk
from aiter.hipkittens_moe import hipkittens_fused_moe
from aiter import ActivationType


def main(batch_size: int, warmup: int, iters: int, seed: int) -> None:
    model_dim = 4096
    inter_dim = 4096
    num_experts = 8
    topk = 2

    print("Profiling HipKittens MoE:")
    print(f"  batch_size={batch_size}")
    print(f"  model_dim={model_dim}")
    print(f"  inter_dim={inter_dim}")
    print(f"  num_experts={num_experts}")
    print(f"  topk={topk}")

    torch.manual_seed(seed)
    hidden = torch.randn((batch_size, model_dim), dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn((num_experts, inter_dim * 2, model_dim), dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn((num_experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((batch_size, num_experts), dtype=torch.float32, device="cuda")

    topk_w, topk_ids = fused_topk(hidden, scores, topk, True)

    print("Warming up...")
    for _ in range(warmup):
        _ = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
        torch.cuda.synchronize()

    print(f"Running {iters} iterations for profiling...")
    torch.cuda.synchronize()

    for _ in range(iters):
        out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)

    torch.cuda.synchronize()
    print("Profiling complete!")
    print(f"Output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main(batch_size=args.batch_size, warmup=args.warmup, iters=args.iters, seed=args.seed)
