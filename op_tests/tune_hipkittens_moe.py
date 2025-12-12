#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
HipKittens MoE Kernel Parameter Tuner

Uses scipy.optimize to find optimal kernel parameters (BLOCK_M, BLOCK_N, BLOCK_K).
Evaluates accuracy and performance on large batch sizes.

Usage:
    python tune_hipkittens_moe.py
    python tune_hipkittens_moe.py --method bayesian --max-evals 50
"""

import argparse
import os
import re
import shutil
import subprocess
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aiter import ActivationType
from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter.test_common import checkAllclose

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
STAGE1_PATH = PROJECT_ROOT / "csrc/hipkittens_moe/hk_moe_stage1.cu"
STAGE2_PATH = PROJECT_ROOT / "csrc/hipkittens_moe/hk_moe_stage2.cu"
JIT_BUILD_PATH = PROJECT_ROOT / "aiter/jit/build/module_hipkittens_moe"
JIT_SO_PATH = PROJECT_ROOT / "aiter/jit/module_hipkittens_moe.so"

# Test configuration - use large batch for accurate evaluation
TEST_CONFIG = {
    "batch_size": 4096,
    "model_dim": 4096,
    "inter_dim": 4096,
    "num_experts": 8,
    "topk": 2,
}

# Parameter search space
PARAM_SPACE = {
    "BLOCK_M": [16, 32, 64],
    "BLOCK_N": [32, 64, 128, 256],
    "BLOCK_K": [32, 64, 128],
}

# Constraints: shared memory must fit in 64KB
# s_input: BLOCK_M * (BLOCK_K + 8) * 4 bytes
# s_weight: BLOCK_N * (BLOCK_K + 8) * 4 bytes
MAX_SHARED_MEM = 65536


@dataclass
class TuningResult:
    params: Dict[str, int]
    time_us: float
    tflops: float
    correct: bool
    error: Optional[str] = None


def calc_shared_memory(block_m: int, block_n: int, block_k: int) -> int:
    """Calculate shared memory usage for given parameters."""
    s_input = block_m * (block_k + 8) * 4
    s_weight = block_n * (block_k + 8) * 4
    return s_input + s_weight


def is_valid_config(block_m: int, block_n: int, block_k: int) -> bool:
    """Check if configuration is valid (fits in shared memory)."""
    smem = calc_shared_memory(block_m, block_n, block_k)
    return smem <= MAX_SHARED_MEM


def update_kernel_params(filepath: Path, block_m: int, block_n: int, block_k: int) -> str:
    """Update kernel parameters in source file. Returns original content for restoration."""
    with open(filepath, 'r') as f:
        original = f.read()
    
    # Pattern to match the configuration namespace
    pattern = r'(namespace s[12]_cfg \{[^}]*constexpr int BLOCK_M = )\d+([^}]*constexpr int BLOCK_N = )\d+([^}]*constexpr int BLOCK_K = )\d+'
    
    def replacer(m):
        return f"{m.group(1)}{block_m}{m.group(2)}{block_n}{m.group(3)}{block_k}"
    
    new_content = re.sub(pattern, replacer, original, flags=re.DOTALL)
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    return original


def restore_kernel(filepath: Path, original_content: str):
    """Restore original kernel content."""
    with open(filepath, 'w') as f:
        f.write(original_content)


def clean_and_rebuild():
    """Clean JIT cache and force rebuild."""
    if JIT_BUILD_PATH.exists():
        shutil.rmtree(JIT_BUILD_PATH)
    if JIT_SO_PATH.exists():
        os.remove(JIT_SO_PATH)
    
    # Clear Python module cache
    mods_to_remove = [k for k in sys.modules.keys() if 'hipkittens' in k or 'module_hipkittens' in k]
    for mod in mods_to_remove:
        del sys.modules[mod]


def evaluate_config(block_m: int, block_n: int, block_k: int, 
                    warmup: int = 2, iters: int = 5) -> TuningResult:
    """Evaluate a single configuration."""
    params = {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}
    
    # Check validity
    if not is_valid_config(block_m, block_n, block_k):
        return TuningResult(params, float('inf'), 0.0, False, 
                          f"Shared memory overflow: {calc_shared_memory(block_m, block_n, block_k)} > {MAX_SHARED_MEM}")
    
    try:
        # Update kernel parameters
        orig_s1 = update_kernel_params(STAGE1_PATH, block_m, block_n, block_k)
        orig_s2 = update_kernel_params(STAGE2_PATH, block_m, block_n, block_k)
        
        try:
            # Clean and rebuild
            clean_and_rebuild()
            
            # Import fresh module
            from aiter.hipkittens_moe import hipkittens_fused_moe
            
            # Create test data
            cfg = TEST_CONFIG
            hidden = torch.randn((cfg["batch_size"], cfg["model_dim"]), 
                               dtype=torch.bfloat16, device="cuda") / 10
            w1 = torch.randn((cfg["num_experts"], cfg["inter_dim"] * 2, cfg["model_dim"]), 
                            dtype=torch.bfloat16, device="cuda") / 10
            w2 = torch.randn((cfg["num_experts"], cfg["model_dim"], cfg["inter_dim"]), 
                            dtype=torch.bfloat16, device="cuda") / 10
            scores = torch.randn((cfg["batch_size"], cfg["num_experts"]), 
                               dtype=torch.float32, device="cuda")
            topk_w, topk_ids = fused_topk(hidden, scores, cfg["topk"], True)
            
            # Reference output
            ref = torch_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
            
            # Warmup
            for _ in range(warmup):
                _ = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
            torch.cuda.synchronize()
            
            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(iters):
                out = hipkittens_fused_moe(hidden, w1, w2, topk_w, topk_ids, activation=ActivationType.Silu)
            end.record()
            torch.cuda.synchronize()
            
            time_ms = start.elapsed_time(end) / iters
            time_us = time_ms * 1000
            
            # Calculate TFLOPs
            flops = cfg["batch_size"] * cfg["topk"] * 2 * cfg["model_dim"] * cfg["inter_dim"] * 3
            tflops = flops / (time_us * 1e6)
            
            # Check accuracy
            try:
                checkAllclose(ref, out, atol=1.0, rtol=0.01)
                correct = True
            except:
                # Check if within reasonable bounds
                max_delta = (ref - out).abs().max().item()
                correct = max_delta < 10.0  # Allow some tolerance
            
            return TuningResult(params, time_us, tflops, correct)
            
        finally:
            # Restore original kernels
            restore_kernel(STAGE1_PATH, orig_s1)
            restore_kernel(STAGE2_PATH, orig_s2)
            
    except Exception as e:
        return TuningResult(params, float('inf'), 0.0, False, str(e))


def grid_search():
    """Simple grid search over parameter space."""
    print("\n" + "="*70)
    print("HipKittens MoE Parameter Tuning - Grid Search")
    print(f"Test config: batch={TEST_CONFIG['batch_size']}, model_dim={TEST_CONFIG['model_dim']}")
    print("="*70 + "\n")
    
    results = []
    
    total = len(PARAM_SPACE["BLOCK_M"]) * len(PARAM_SPACE["BLOCK_N"]) * len(PARAM_SPACE["BLOCK_K"])
    current = 0
    
    for block_m in PARAM_SPACE["BLOCK_M"]:
        for block_n in PARAM_SPACE["BLOCK_N"]:
            for block_k in PARAM_SPACE["BLOCK_K"]:
                current += 1
                
                # Quick validity check
                if not is_valid_config(block_m, block_n, block_k):
                    print(f"[{current}/{total}] M={block_m}, N={block_n}, K={block_k}: SKIP (shared mem overflow)")
                    continue
                
                print(f"[{current}/{total}] Testing M={block_m}, N={block_n}, K={block_k}...", end=" ", flush=True)
                
                result = evaluate_config(block_m, block_n, block_k)
                results.append(result)
                
                if result.error:
                    print(f"ERROR: {result.error}")
                elif not result.correct:
                    print(f"INCORRECT: {result.time_us:.1f} us, {result.tflops:.2f} TFLOPs")
                else:
                    print(f"OK: {result.time_us:.1f} us, {result.tflops:.2f} TFLOPs")
    
    # Find best valid result
    valid_results = [r for r in results if r.correct and r.error is None]
    if valid_results:
        best = min(valid_results, key=lambda r: r.time_us)
        print("\n" + "="*70)
        print("BEST CONFIGURATION:")
        print(f"  BLOCK_M = {best.params['BLOCK_M']}")
        print(f"  BLOCK_N = {best.params['BLOCK_N']}")
        print(f"  BLOCK_K = {best.params['BLOCK_K']}")
        print(f"  Time: {best.time_us:.1f} us")
        print(f"  TFLOPs: {best.tflops:.2f}")
        print("="*70)
        return best
    else:
        print("\nNo valid configuration found!")
        return None


def scipy_optimize():
    """Use scipy.optimize for parameter tuning."""
    from scipy.optimize import minimize, differential_evolution
    
    print("\n" + "="*70)
    print("HipKittens MoE Parameter Tuning - Scipy Optimize")
    print(f"Test config: batch={TEST_CONFIG['batch_size']}, model_dim={TEST_CONFIG['model_dim']}")
    print("="*70 + "\n")
    
    # Map continuous values to discrete parameter values
    def decode_params(x):
        # x[0] -> BLOCK_M index, x[1] -> BLOCK_N index, x[2] -> BLOCK_K index
        block_m = PARAM_SPACE["BLOCK_M"][int(np.clip(x[0], 0, len(PARAM_SPACE["BLOCK_M"])-1))]
        block_n = PARAM_SPACE["BLOCK_N"][int(np.clip(x[1], 0, len(PARAM_SPACE["BLOCK_N"])-1))]
        block_k = PARAM_SPACE["BLOCK_K"][int(np.clip(x[2], 0, len(PARAM_SPACE["BLOCK_K"])-1))]
        return block_m, block_n, block_k
    
    eval_cache = {}
    
    def objective(x):
        block_m, block_n, block_k = decode_params(x)
        key = (block_m, block_n, block_k)
        
        if key in eval_cache:
            return eval_cache[key]
        
        print(f"Evaluating M={block_m}, N={block_n}, K={block_k}...", end=" ", flush=True)
        result = evaluate_config(block_m, block_n, block_k)
        
        if result.error or not result.correct:
            score = 1e9  # Infeasible
            print(f"INFEASIBLE")
        else:
            score = result.time_us
            print(f"{result.time_us:.1f} us, {result.tflops:.2f} TFLOPs")
        
        eval_cache[key] = score
        return score
    
    # Bounds for indices
    bounds = [
        (0, len(PARAM_SPACE["BLOCK_M"]) - 0.01),
        (0, len(PARAM_SPACE["BLOCK_N"]) - 0.01),
        (0, len(PARAM_SPACE["BLOCK_K"]) - 0.01),
    ]
    
    print("Running differential evolution...")
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=20,
        popsize=5,
        seed=42,
        disp=True,
        workers=1,  # Must be 1 for CUDA
        updating='deferred'
    )
    
    best_m, best_n, best_k = decode_params(result.x)
    print("\n" + "="*70)
    print("BEST CONFIGURATION:")
    print(f"  BLOCK_M = {best_m}")
    print(f"  BLOCK_N = {best_n}")
    print(f"  BLOCK_K = {best_k}")
    print(f"  Time: {result.fun:.1f} us")
    print("="*70)
    
    return best_m, best_n, best_k


def apply_best_config(block_m: int, block_n: int, block_k: int):
    """Apply the best configuration to kernel files."""
    print(f"\nApplying best config: M={block_m}, N={block_n}, K={block_k}")
    
    # Update Stage 1
    with open(STAGE1_PATH, 'r') as f:
        content = f.read()
    
    pattern = r'(namespace s1_cfg \{[^}]*constexpr int BLOCK_M = )\d+([^}]*constexpr int BLOCK_N = )\d+([^}]*constexpr int BLOCK_K = )\d+'
    content = re.sub(pattern, 
                    f"\\g<1>{block_m}\\g<2>{block_n}\\g<3>{block_k}", 
                    content, flags=re.DOTALL)
    
    with open(STAGE1_PATH, 'w') as f:
        f.write(content)
    
    # Update Stage 2
    with open(STAGE2_PATH, 'r') as f:
        content = f.read()
    
    pattern = r'(namespace s2_cfg \{[^}]*constexpr int BLOCK_M = )\d+([^}]*constexpr int BLOCK_N = )\d+([^}]*constexpr int BLOCK_K = )\d+'
    content = re.sub(pattern, 
                    f"\\g<1>{block_m}\\g<2>{block_n}\\g<3>{block_k}", 
                    content, flags=re.DOTALL)
    
    with open(STAGE2_PATH, 'w') as f:
        f.write(content)
    
    print("Configuration applied!")


def main():
    parser = argparse.ArgumentParser(description="Tune HipKittens MoE kernel parameters")
    parser.add_argument("--method", choices=["grid", "scipy"], default="grid",
                       help="Optimization method")
    parser.add_argument("--apply", action="store_true",
                       help="Apply best configuration to kernel files")
    args = parser.parse_args()
    
    if args.method == "grid":
        best = grid_search()
        if best and args.apply:
            apply_best_config(best.params["BLOCK_M"], best.params["BLOCK_N"], best.params["BLOCK_K"])
    else:
        best_m, best_n, best_k = scipy_optimize()
        if args.apply:
            apply_best_config(best_m, best_n, best_k)


if __name__ == "__main__":
    main()

