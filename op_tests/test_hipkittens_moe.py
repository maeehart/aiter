# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
HipKittens MoE Benchmark Script

Benchmarks HipKittens MoE kernel against baseline AITER for high batch sizes.
Based on CSV data showing performance issues at 4000-8600 tokens in prefill.

Supports both BF16 weights and FP8 blockscale weights (DeepSeek R1 format).

Usage:
    # Quick test with 3 batch sizes (1024, 4096, 8192)
    python test_hipkittens_moe.py --quick
    
    # Full benchmark with all high batch sizes from CSV analysis
    python test_hipkittens_moe.py
    
    # Custom batch sizes
    python test_hipkittens_moe.py --batch-sizes 4000 6000 8000
    
    # FP8 blockscale benchmark (DeepSeek R1 config)
    python test_hipkittens_moe.py --fp8 --deepseek-r1

Output:
    - Timing comparison (AITER vs HipKittens)
    - TFLOPs achieved by each implementation
    - Correctness check against PyTorch reference
"""

import argparse
import torch
import math
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest
from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter import ActivationType, QuantType
from aiter import pertoken_quant
from einops import rearrange

try:
    from aiter.hipkittens_moe import (
        hipkittens_fused_moe,
        hipkittens_fused_moe_fp8,
        hipkittens_fused_moe_fp8_fused_act,
        hipkittens_fused_moe_fp8_noatomic,
        hipkittens_fused_moe_fully_fused,
    )
    HIPKITTENS_AVAILABLE = True
    HIPKITTENS_FP8_AVAILABLE = True
except ImportError as e:
    print(f"HipKittens MoE not available: {e}")
    HIPKITTENS_AVAILABLE = False
    HIPKITTENS_FP8_AVAILABLE = False

# Use PyTorch reference as ground truth (AITER ASM kernels have numerical differences at large sizes)
USE_PYTORCH_REF = True

# High batch sizes from CSV (tokens causing cache thrashing)
HIGH_BATCH_SIZES = [480, 1547, 2907, 4132, 4644, 5913, 7339, 8613]

# AITER production kernel timings from CSV (nanoseconds)
# Kernel: aiter::fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_ps_32x256
AITER_CSV_TIMINGS_NS = {
    8613: 2469368.5,
    7339: 1932024.0,
    5913: 1677291.5,
    4644: 1547635.0,
    4132: 1291363.0,
    2907: 984384.0,
    2072: 811535.7,
    1547: 599607.8,
    1192: 522264.0,
    480:  353069.0,
}

# MoE config from CSV: model_dim=7168, experts=256, topk=8
# Note: Using smaller config for testing; scale up when GPU memory is sufficient
MOE_CONFIG_FULL = {"model_dim": 7168, "inter_dim": 7168, "num_experts": 256, "topk": 8}
MOE_CONFIG_SMALL = {"model_dim": 4096, "inter_dim": 4096, "num_experts": 8, "topk": 2}
MOE_CONFIG = MOE_CONFIG_SMALL  # Use smaller config for testing

# DeepSeek R1 MoE config from CSV roofline data
# model_dim=7168, inter_dim=256, experts=256, topk=8
DEEPSEEK_R1_CONFIG = {"model_dim": 7168, "inter_dim": 256, "num_experts": 256, "topk": 8}

# FP8 blockscale configuration (128x128 blocks)
FP8_SCALE_BLOCK_N = 128
FP8_SCALE_BLOCK_K = 128


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
def bench_aiter_fp8(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids):
    """Benchmark AITER production FP8 blockscale kernel."""
    return fused_moe(hidden, w1_fp8, w2_fp8, topk_w, topk_ids, 
                     activation=ActivationType.Silu, 
                     quant_type=QuantType.per_128x128,
                     w1_scale=w1_scale, w2_scale=w2_scale)


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
    if USE_PYTORCH_REF:
        print(f"        Correctness: vs PyTorch reference (ground truth)")
    else:
        print(f"        Correctness: vs AITER")
    print(f"{'='*70}\n")
    
    results = []
    for bs in batch_sizes:
        print(f"\n--- Batch: {bs} tokens ---")
        hidden, w1, w2, topk_w, topk_ids = create_inputs(bs, MOE_CONFIG)
        flops = calc_flops(bs, MOE_CONFIG)
        
        # Get reference output for correctness check
        if USE_PYTORCH_REF:
            ref_out = torch_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
            ref_name = "PyTorch"
        else:
            ref_out = None
            ref_name = "AITER"
        
        try:
            aiter_out, aiter_us = bench_aiter(hidden, w1, w2, topk_w, topk_ids)
            aiter_tf = flops / (aiter_us * 1e-6) / 1e12
            print(f"  AITER:      {aiter_us:8.2f} us, {aiter_tf:6.2f} TFLOPs")
            if ref_out is None:
                ref_out = aiter_out
        except Exception as e:
            print(f"  AITER:      Failed - {e}")
            aiter_out, aiter_us, aiter_tf = None, float('inf'), 0
        
        if HIPKITTENS_AVAILABLE:
            try:
                hk_out, hk_us = bench_hipkittens(hidden, w1, w2, topk_w, topk_ids)
                hk_tf = flops / (hk_us * 1e-6) / 1e12
                speedup = aiter_us / hk_us if hk_us > 0 else 0
                print(f"  HipKittens: {hk_us:8.2f} us, {hk_tf:6.2f} TFLOPs, {speedup:.2f}x speedup")
                if ref_out is not None:
                    # checkAllclose returns mismatch ratio (0.0 means perfect match).
                    mismatch = checkAllclose(
                        ref_out,
                        hk_out,
                        rtol=0.01,
                        atol=1.0,
                        msg=f"batch={bs}",
                        printLog=False,
                    )
                    verdict = "PASS" if mismatch == 0 else f"FAIL (mismatch={mismatch:.1%})"
                    print(f"  Correctness vs {ref_name}: {verdict}")
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


def quantize_weights_blockscale(w, scale_blk_n=128, scale_blk_k=128):
    """
    Quantize weights to FP8 with blockscale.
    
    Args:
        w: Weight tensor [E, N, K] in bf16/fp32
        scale_blk_n: Block size in N dimension
        scale_blk_k: Block size in K dimension
        
    Returns:
        w_fp8: Quantized weights [E, N, K] as FP8
        w_scale: Scales [E, ceil(N/blk_n), ceil(K/blk_k)] as float32
    """
    E, N, K = w.shape
    quant_dtype = dtypes.fp8
    
    # Reshape for block quantization
    num_blk_n = math.ceil(N / scale_blk_n)
    num_blk_k = math.ceil(K / scale_blk_k)
    
    # Pad if needed
    N_padded = num_blk_n * scale_blk_n
    K_padded = num_blk_k * scale_blk_k
    
    if N_padded != N or K_padded != K:
        w_padded = torch.zeros((E, N_padded, K_padded), dtype=w.dtype, device=w.device)
        w_padded[:, :N, :K] = w
        w = w_padded
    
    # Rearrange for block quantization
    tmp = rearrange(
        w.view(E, num_blk_n, scale_blk_n, num_blk_k, scale_blk_k),
        "e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)",
    ).contiguous()
    
    # Quantize per block
    w_q, w_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
    
    # Reshape back
    w_q = rearrange(
        w_q.view(E, num_blk_n, num_blk_k, scale_blk_n, scale_blk_k),
        "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
    ).contiguous()
    
    # Take original size
    w_q = w_q[:, :N, :K].contiguous()
    w_scale = w_scale.view(E, num_blk_n, num_blk_k)
    
    return w_q, w_scale


def create_fp8_inputs(batch_size, cfg, dtype=torch.bfloat16, device="cuda"):
    """Create inputs with FP8 quantized weights for DeepSeek R1 format."""
    hidden = torch.randn((batch_size, cfg["model_dim"]), dtype=dtype, device=device) / 10
    
    # Create bf16 weights first
    w1_bf16 = torch.randn((cfg["num_experts"], cfg["inter_dim"] * 2, cfg["model_dim"]), dtype=dtype, device=device) / 10
    w2_bf16 = torch.randn((cfg["num_experts"], cfg["model_dim"], cfg["inter_dim"]), dtype=dtype, device=device) / 10
    
    # Quantize to FP8 with blockscale
    w1_fp8, w1_scale = quantize_weights_blockscale(w1_bf16, FP8_SCALE_BLOCK_N, FP8_SCALE_BLOCK_K)
    w2_fp8, w2_scale = quantize_weights_blockscale(w2_bf16, FP8_SCALE_BLOCK_N, FP8_SCALE_BLOCK_K)
    
    scores = torch.randn((batch_size, cfg["num_experts"]), dtype=torch.float32, device=device)
    topk_w, topk_ids = fused_topk(hidden, scores, cfg["topk"], True)
    
    return hidden, w1_bf16, w2_bf16, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids


@perftest()
def bench_hipkittens_fp8(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids):
    return hipkittens_fused_moe_fp8(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)


@perftest()
def bench_hipkittens_fp8_fused_act(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids):
    return hipkittens_fused_moe_fp8_fused_act(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)


@perftest()
def bench_hipkittens_fp8_noatomic(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids):
    return hipkittens_fused_moe_fp8_noatomic(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)


@perftest()
def bench_hipkittens_fp8_fully_fused(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids):
    return hipkittens_fused_moe_fully_fused(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)


def run_fp8_benchmark(batch_sizes, cfg, compare_fused_act=True, compare_noatomic=False, compare_fully_fused=False):
    """Run FP8 blockscale benchmark.
    
    Args:
        batch_sizes: List of batch sizes to test
        cfg: MoE configuration dict
        compare_fused_act: If True, also benchmark the fused activation variant
        compare_noatomic: If True, also benchmark the atomic-free variant
    """
    print(f"\n{'='*70}")
    print(f"HipKittens MoE FP8 Blockscale Benchmark")
    print(f"Config: model_dim={cfg['model_dim']}, inter_dim={cfg['inter_dim']}")
    print(f"        experts={cfg['num_experts']}, topk={cfg['topk']}")
    print(f"        FP8 blockscale: {FP8_SCALE_BLOCK_N}x{FP8_SCALE_BLOCK_K}")
    if compare_fused_act:
        print(f"        Comparing: FP8 vs FP8+FusedAct")
    print(f"{'='*70}\n")
    
    results = []
    for bs in batch_sizes:
        print(f"\n--- Batch: {bs} tokens ---")
        
        try:
            hidden, w1_bf16, w2_bf16, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids = create_fp8_inputs(bs, cfg)
        except Exception as e:
            print(f"  Setup failed: {e}")
            continue
        
        flops = calc_flops(bs, cfg)
        
        # Reference: PyTorch with dequantized weights
        try:
            ref_out = torch_moe(hidden, w1_bf16, w2_bf16, topk_w, topk_ids, activation=ActivationType.Silu)
        except Exception as e:
            print(f"  PyTorch reference failed: {e}")
            ref_out = None
        
        hk_fp8_us, hk_fp8_tf = float('inf'), 0
        hk_fused_us, hk_fused_tf = float('inf'), 0
        aiter_fp8_us, aiter_fp8_tf = float('inf'), 0
        
        # AITER production FP8 (baseline)
        try:
            aiter_fp8_out, aiter_fp8_us = bench_aiter_fp8(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)
            aiter_fp8_tf = flops / (aiter_fp8_us * 1e-6) / 1e12
            print(f"  AITER FP8:      {aiter_fp8_us:8.2f} us, {aiter_fp8_tf:6.2f} TFLOPs")
            
            if ref_out is not None:
                mismatch = checkAllclose(
                    ref_out, aiter_fp8_out, rtol=0.05, atol=0.1, msg=f"batch={bs}", printLog=False
                )
                verdict = "PASS" if mismatch == 0 else f"WARN (mismatch={mismatch:.1%})"
                print(f"    Correctness: {verdict}")
        except Exception as e:
            print(f"  AITER FP8:      Failed - {e}")
        
        # HipKittens FP8 (non-fused)
        if HIPKITTENS_FP8_AVAILABLE:
            try:
                hk_fp8_out, hk_fp8_us = bench_hipkittens_fp8(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)
                hk_fp8_tf = flops / (hk_fp8_us * 1e-6) / 1e12
                print(f"  HK FP8:         {hk_fp8_us:8.2f} us, {hk_fp8_tf:6.2f} TFLOPs")
                
                if ref_out is not None:
                    mismatch = checkAllclose(
                        ref_out, hk_fp8_out, rtol=0.05, atol=0.1, msg=f"batch={bs}", printLog=False
                    )
                    verdict = "PASS" if mismatch == 0 else f"WARN (mismatch={mismatch:.1%})"
                    print(f"    Correctness: {verdict}")
            except Exception as e:
                print(f"  HK FP8:         Failed - {e}")
        
        # HipKittens FP8 + Fused Activation
        if compare_fused_act and HIPKITTENS_FP8_AVAILABLE:
            try:
                hk_fused_out, hk_fused_us = bench_hipkittens_fp8_fused_act(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)
                hk_fused_tf = flops / (hk_fused_us * 1e-6) / 1e12
                speedup = hk_fp8_us / hk_fused_us if hk_fused_us > 0 else 0
                print(f"  HK FP8+Fused:   {hk_fused_us:8.2f} us, {hk_fused_tf:6.2f} TFLOPs ({speedup:.2f}x vs non-fused)")
                
                if ref_out is not None:
                    mismatch = checkAllclose(
                        ref_out, hk_fused_out, rtol=0.05, atol=0.1, msg=f"batch={bs}", printLog=False
                    )
                    verdict = "PASS" if mismatch == 0 else f"WARN (mismatch={mismatch:.1%})"
                    print(f"    Correctness: {verdict}")
            except Exception as e:
                print(f"  HK FP8+Fused:   Failed - {e}")
        
        # HipKittens FP8 + Fused Activation + No Atomics (EXPERIMENTAL)
        hk_noatomic_us, hk_noatomic_tf = float('inf'), 0
        if compare_noatomic and HIPKITTENS_FP8_AVAILABLE:
            try:
                hk_noatomic_out, hk_noatomic_us = bench_hipkittens_fp8_noatomic(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)
                hk_noatomic_tf = flops / (hk_noatomic_us * 1e-6) / 1e12
                speedup_vs_fused = hk_fused_us / hk_noatomic_us if hk_noatomic_us > 0 else 0
                print(f"  HK NoAtomic:    {hk_noatomic_us:8.2f} us, {hk_noatomic_tf:6.2f} TFLOPs ({speedup_vs_fused:.2f}x vs fused)")
                
                if ref_out is not None:
                    mismatch = checkAllclose(
                        ref_out, hk_noatomic_out, rtol=0.05, atol=0.1, msg=f"batch={bs}", printLog=False
                    )
                    verdict = "PASS" if mismatch == 0 else f"WARN (mismatch={mismatch:.1%})"
                    print(f"    Correctness: {verdict}")
            except Exception as e:
                print(f"  HK NoAtomic:    Failed - {e}")
        
        # HipKittens FP8 Fully Fused (Stage1+Stage2 in single kernel)
        hk_fullyfused_us, hk_fullyfused_tf = float('inf'), 0
        if compare_fully_fused and HIPKITTENS_FP8_AVAILABLE:
            try:
                hk_fullyfused_out, hk_fullyfused_us = bench_hipkittens_fp8_fully_fused(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)
                hk_fullyfused_tf = flops / (hk_fullyfused_us * 1e-6) / 1e12
                speedup_vs_fused = hk_fused_us / hk_fullyfused_us if hk_fullyfused_us > 0 else 0
                print(f"  HK FullyFused:  {hk_fullyfused_us:8.2f} us, {hk_fullyfused_tf:6.2f} TFLOPs ({speedup_vs_fused:.2f}x vs fused)")
                
                if ref_out is not None:
                    mismatch = checkAllclose(
                        ref_out, hk_fullyfused_out, rtol=0.05, atol=0.1, msg=f"batch={bs}", printLog=False
                    )
                    verdict = "PASS" if mismatch == 0 else f"WARN (mismatch={mismatch:.1%})"
                    print(f"    Correctness: {verdict}")
                    if mismatch > 0:
                        # Debug: show more info about the mismatch
                        nan_count = torch.isnan(hk_fullyfused_out).sum().item()
                        inf_count = torch.isinf(hk_fullyfused_out).sum().item()
                        if nan_count > 0:
                            print(f"    DEBUG: NaN count={nan_count}, Inf count={inf_count}")
                            print(f"    DEBUG: Output mean={hk_fullyfused_out.nanmean().item():.6f}, std={hk_fullyfused_out.std().item():.6f}")
            except Exception as e:
                print(f"  HK FullyFused:  Failed - {e}")
                import traceback
                traceback.print_exc()
        
        results.append({
            "batch": bs, 
            "aiter_fp8_us": aiter_fp8_us, "aiter_fp8_tf": aiter_fp8_tf,
            "hk_fp8_us": hk_fp8_us, "hk_fp8_tf": hk_fp8_tf,
            "hk_fused_us": hk_fused_us, "hk_fused_tf": hk_fused_tf,
            "hk_noatomic_us": hk_noatomic_us, "hk_noatomic_tf": hk_noatomic_tf,
            "hk_fullyfused_us": hk_fullyfused_us, "hk_fullyfused_tf": hk_fullyfused_tf,
        })
    
    print(f"\n{'='*80}")
    print("SUMMARY: HipKittens vs AITER Production (LIVE comparison)")
    print("-" * 80)
    print(f"{'Batch':>8} | {'AITER (us)':>10} | {'HK Fused (us)':>12} | {'Gap':>8} | {'HK TFLOPs':>10}")
    print("-" * 80)
    for r in results:
        aiter_us = r.get('aiter_fp8_us', float('inf'))
        hk_us = r.get('hk_fused_us', float('inf'))
        if aiter_us < float('inf') and hk_us < float('inf'):
            gap = hk_us / aiter_us
            print(f"{r['batch']:>8} | {aiter_us:>10.2f} | {hk_us:>12.2f} | {gap:>7.2f}x | {r['hk_fused_tf']:>10.2f}")
        elif hk_us < float('inf'):
            print(f"{r['batch']:>8} | {'N/A':>10} | {hk_us:>12.2f} | {'N/A':>8} | {r['hk_fused_tf']:>10.2f}")
    
    print(f"\n{'='*80}")
    print("Note: Gap > 1.0x means HipKittens is slower than AITER production.")
    print("Target: Reduce gap to ~1.0x (match AITER) or <1.0x (beat AITER).")


def profile_stages(batch_sizes, cfg, num_iters=10):
    """Profile individual Stage1 and Stage2 timing using CUDA events."""
    from aiter.fused_moe import moe_sorting
    from aiter.hipkittens_moe import hipkittens_moe_sorting
    
    # Import C++ module for direct stage calls
    try:
        from aiter.hipkittens_moe import _HK_MOE_MODULE
        from aiter.jit.core import compile_ops
    except ImportError:
        print("Cannot import HipKittens module for stage profiling")
        return
    
    print(f"\n{'='*70}")
    print(f"HipKittens MoE Stage Profiling (FP8 Fused Activation)")
    print(f"Config: model_dim={cfg['model_dim']}, inter_dim={cfg['inter_dim']}")
    print(f"        experts={cfg['num_experts']}, topk={cfg['topk']}")
    print(f"{'='*70}\n")
    
    for bs in batch_sizes:
        print(f"\n--- Batch: {bs} tokens ---")
        
        try:
            hidden, w1_bf16, w2_bf16, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids = create_fp8_inputs(bs, cfg)
        except Exception as e:
            print(f"  Setup failed: {e}")
            continue
        
        # Run full fused kernel and profile with CUDA events
        # Warmup
        for _ in range(3):
            _ = hipkittens_fused_moe_fp8_fused_act(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)
            torch.cuda.synchronize()
        
        # Profile full kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(num_iters):
            _ = hipkittens_fused_moe_fp8_fused_act(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)
        end_event.record()
        torch.cuda.synchronize()
        
        total_us = start_event.elapsed_time(end_event) * 1000 / num_iters  # ms to us
        flops = calc_flops(bs, cfg)
        tflops = flops / (total_us * 1e-6) / 1e12
        
        print(f"  Total FP8+Fused: {total_us:.2f} us, {tflops:.2f} TFLOPs")
        
        # Profile with markers using rocprof (if available)
        # For now, we output the total timing - detailed stage breakdown requires rocprof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HipKittens MoE Benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=HIGH_BATCH_SIZES)
    parser.add_argument("--quick", action="store_true", help="Quick run")
    parser.add_argument("--fp8", action="store_true", help="Run FP8 blockscale benchmark")
    parser.add_argument("--deepseek-r1", action="store_true", help="Use DeepSeek R1 config (model_dim=7168, inter_dim=256, experts=256, topk=8)")
    parser.add_argument("--profile-stages", action="store_true", help="Profile Stage1 vs Stage2 timing")
    parser.add_argument("--noatomic", action="store_true", help="Also benchmark atomic-free Stage 2 variant")
    parser.add_argument("--fully-fused", action="store_true", help="Also benchmark fully fused (Stage1+Stage2) kernel")
    args = parser.parse_args()
    
    if args.profile_stages:
        cfg = DEEPSEEK_R1_CONFIG if args.deepseek_r1 else MOE_CONFIG
        profile_stages(args.batch_sizes if not args.quick else [2048, 4096, 8192], cfg)
    elif args.fp8:
        cfg = DEEPSEEK_R1_CONFIG if args.deepseek_r1 else MOE_CONFIG
        if args.quick:
            run_fp8_benchmark([512, 1024, 2048], cfg, compare_noatomic=args.noatomic, compare_fully_fused=args.fully_fused)
        else:
            run_fp8_benchmark(args.batch_sizes, cfg, compare_noatomic=args.noatomic, compare_fully_fused=args.fully_fused)
    else:
        if args.quick:
            run_benchmark([1024, 4096, 8192])
        else:
            run_benchmark(args.batch_sizes)

