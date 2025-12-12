# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
HipKittens MoE Benchmark Script

Benchmarks HipKittens MoE kernel against baseline AITER for high batch sizes.
Based on CSV data showing performance issues at 4000-8600 tokens in prefill.

Usage:
    python test_hipkittens_moe.py
    python test_hipkittens_moe.py --batch-sizes 4000 6000 8000
"""

import argparse
import torch
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest
from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter import ActivationType, QuantType

try:
    from aiter.hipkittens_moe import hipkittens_fused_moe
    HIPKITTENS_AVAILABLE = True
except ImportError as e:
    print(f"HipKittens MoE not available: {e}")
    HIPKITTENS_AVAILABLE = False

# High batch sizes from CSV (tokens causing cache thrashing)
HIGH_BATCH_SIZES = [480, 1547, 2907, 4132, 4644, 5913, 7339, 8613]

# MoE config from CSV: model_dim=7168, experts=256, topk=8
# Note: Using smaller config for testing; scale up when GPU memory is sufficient
MOE_CONFIG_FULL = {"model_dim": 7168, "inter_dim": 7168, "num_experts": 256, "topk": 8}
MOE_CONFIG_SMALL = {"model_dim": 4096, "inter_dim": 4096, "num_experts": 8, "topk": 2}
MOE_CONFIG = MOE_CONFIG_SMALL  # Use smaller config for testing


def create_inputs(batch_size, cfg, dtype=torch.bfloat16, device="cuda"):
    hidden = torch.randn((batch_size, cfg["model_dim"]), dtype=dtype, device=device) / 10
    w1 = torch.randn((cfg["num_experts"], cfg["inter_dim"] * 2, cfg["model_dim"]), dtype=dtype, device=device) / 10
    w2 = torch.randn((cfg["num_experts"], cfg["model_dim"], cfg["inter_dim"]), dtype=dtype, device=device) / 10
    scores = torch.randn((batch_size, cfg["num_experts"]), dtype=torch.float32, device=device)
    topk_w, topk_ids = fused_topk(hidden, scores, cfg["topk"], True)
    return hidden, w1, w2, topk_w, topk_ids


@perftest()
def bench_aiter(hidden, w1, w2, topk_w, topk_ids):
    return fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu, quant_type=QuantType.No)


@perftest()
def bench_hipkittens(hidden, w1, w2, topk_w, topk_ids):
    return hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)


def calc_flops(bs, cfg):
    return bs * cfg["topk"] * 2 * cfg["model_dim"] * cfg["inter_dim"] * 3


def run_benchmark(batch_sizes):
    print(f"\n{'='*70}")
    print(f"HipKittens MoE Benchmark - High Batch Sizes")
    print(f"Config: model_dim={MOE_CONFIG['model_dim']}, inter_dim={MOE_CONFIG['inter_dim']}")
    print(f"        experts={MOE_CONFIG['num_experts']}, topk={MOE_CONFIG['topk']}")
    print(f"{'='*70}\n")
    
    results = []
    for bs in batch_sizes:
        print(f"\n--- Batch: {bs} tokens ---")
        hidden, w1, w2, topk_w, topk_ids = create_inputs(bs, MOE_CONFIG)
        flops = calc_flops(bs, MOE_CONFIG)
        
        try:
            aiter_out, aiter_us = bench_aiter(hidden, w1, w2, topk_w, topk_ids)
            aiter_tf = flops / (aiter_us * 1e-6) / 1e12
            print(f"  AITER:      {aiter_us:8.2f} us, {aiter_tf:6.2f} TFLOPs")
        except Exception as e:
            print(f"  AITER:      Failed - {e}")
            aiter_out, aiter_us, aiter_tf = None, float('inf'), 0
        
        if HIPKITTENS_AVAILABLE:
            try:
                hk_out, hk_us = bench_hipkittens(hidden, w1, w2, topk_w, topk_ids)
                hk_tf = flops / (hk_us * 1e-6) / 1e12
                speedup = aiter_us / hk_us if hk_us > 0 else 0
                print(f"  HipKittens: {hk_us:8.2f} us, {hk_tf:6.2f} TFLOPs, {speedup:.2f}x speedup")
                if aiter_out is not None:
                    try:
                        checkAllclose(aiter_out, hk_out, rtol=0.01, atol=1.0, msg=f"batch={bs}")
                        print(f"  Correctness: PASS")
                    except AssertionError as e:
                        print(f"  Correctness: FAIL")
            except Exception as e:
                print(f"  HipKittens: Failed - {e}")
                hk_us, hk_tf, speedup = float('inf'), 0, 0
        else:
            hk_us, hk_tf, speedup = float('inf'), 0, 0
        
        results.append({"batch": bs, "aiter_us": aiter_us, "hk_us": hk_us, "speedup": speedup})
    
    print(f"\n{'='*70}")
    print(f"{'Batch':>8} | {'AITER (us)':>12} | {'HK (us)':>12} | {'Speedup':>8}")
    print("-" * 50)
    for r in results:
        sp = f"{r['speedup']:.2f}x" if r['speedup'] > 0 else "N/A"
        print(f"{r['batch']:>8} | {r['aiter_us']:>12.2f} | {r['hk_us']:>12.2f} | {sp:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HipKittens MoE Benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=HIGH_BATCH_SIZES)
    parser.add_argument("--quick", action="store_true", help="Quick run")
    args = parser.parse_args()
    
    if args.quick:
        run_benchmark([1024, 4096, 8192])
    else:
        run_benchmark(args.batch_sizes)

