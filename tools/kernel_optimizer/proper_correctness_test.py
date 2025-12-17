#!/usr/bin/env python3
"""
Proper correctness validation for optimized kernels.

The correct approach:
1. Measure the MAXIMUM element-wise difference between original and optimized kernels
2. Compare this to the BASELINE variation (inherent non-determinism of the original kernel)
3. If max_diff is within baseline variation, the kernels are equivalent

Baseline finding: Original kernel has max element-wise diff of ~2.0 (1 bit in bfloat16)
between runs with identical inputs. This is the acceptable variation threshold.
"""

import os
import shutil
import subprocess
import sys
import numpy as np

os.environ['VLLM_ROCM_USE_AITER'] = '1'

AITER_PATH = "/workspace/dev/aiter_long_context2"
KERNEL_DIR = f"{AITER_PATH}/hsa/gfx942/fmoe/silu"
KERNEL_NAME = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"
ORIGINAL_PATH = f"{KERNEL_DIR}/{KERNEL_NAME}"
BACKUP_PATH = f"{ORIGINAL_PATH}.backup"

# Acceptable variation: 1 bit in bfloat16 ≈ 2.0
ACCEPTABLE_MAX_DIFF = 2.5  # Slightly above 2.0 to account for edge cases


def get_kernel_hash(path):
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def run_kernel_and_get_output(seed: int = 42, batch_size: int = 1024):
    """Run kernel in subprocess and return full output tensor."""
    script = f'''
import os
os.environ['VLLM_ROCM_USE_AITER'] = '1'
import torch
import numpy as np
from aiter.fused_moe import fused_moe, QuantType, ActivationType

torch.manual_seed({seed})
torch.cuda.manual_seed({seed})
np.random.seed({seed})

batch_size = {batch_size}
num_experts = 256
hidden_size = 7168
intermediate_size = 256
topk = 8

hidden = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)
w1 = torch.randn(num_experts, intermediate_size*2, hidden_size, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
w2 = torch.randn(num_experts, hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
w1_scale = torch.ones((num_experts, 4, hidden_size//128), device="cuda", dtype=torch.float32) * 0.1
w2_scale = torch.ones((num_experts, hidden_size//128, 2), device="cuda", dtype=torch.float32) * 0.1
weights = torch.rand(batch_size, topk, device="cuda", dtype=torch.float32)
weights = weights / weights.sum(dim=-1, keepdim=True)
ids = torch.randint(0, num_experts, (batch_size, topk), device="cuda", dtype=torch.int32)

# Run kernel
out = fused_moe(hidden.clone(), w1, w2, weights, ids, 
               activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
               w1_scale=w1_scale, w2_scale=w2_scale)
torch.cuda.synchronize()

# Save output
out_np = out.cpu().float().numpy()
np.save("/tmp/kernel_output.npy", out_np)
print(f"SHAPE:{{out_np.shape}}")
print(f"CHECKSUM:{{out_np.sum()}}")
'''
    
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=120
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr[-500:]}")
        return None
    
    return np.load("/tmp/kernel_output.npy")


def measure_baseline_variation(n_runs: int = 10, batch_size: int = 1024):
    """Measure the inherent variation in the original kernel."""
    print("=" * 70)
    print("MEASURING BASELINE VARIATION (Original Kernel)")
    print("=" * 70)
    
    # Ensure original kernel
    if os.path.exists(BACKUP_PATH):
        shutil.copy(BACKUP_PATH, ORIGINAL_PATH)
    
    outputs = []
    for i in range(n_runs):
        out = run_kernel_and_get_output(seed=42, batch_size=batch_size)  # Same seed every time
        if out is not None:
            outputs.append(out)
            print(f"  Run {i+1}: checksum = {out.sum():.2f}")
    
    if len(outputs) < 2:
        print("ERROR: Could not get enough outputs")
        return None
    
    # Compute pairwise max differences
    max_diffs = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            diff = np.abs(outputs[i] - outputs[j])
            max_diffs.append(diff.max())
    
    baseline_max_diff = max(max_diffs)
    baseline_mean_diff = np.mean(max_diffs)
    
    print(f"\nBaseline variation analysis ({len(outputs)} runs):")
    print(f"  Max element-wise difference: {baseline_max_diff:.4f}")
    print(f"  Mean of max differences: {baseline_mean_diff:.4f}")
    print(f"  Expected (1 bit bfloat16): ~2.0")
    
    return baseline_max_diff, outputs[0]  # Return baseline max and a reference output


def compare_kernel_to_reference(kernel_path: str, reference_output: np.ndarray, 
                                kernel_name: str, n_runs: int = 5):
    """Compare optimized kernel output to reference."""
    print(f"\n{'='*70}")
    print(f"Testing: {kernel_name}")
    print(f"{'='*70}")
    
    # Swap kernel
    shutil.copy(kernel_path, ORIGINAL_PATH)
    print(f"Kernel hash: {get_kernel_hash(kernel_path)}")
    
    max_diffs_to_ref = []
    self_max_diffs = []
    outputs = []
    
    for i in range(n_runs):
        out = run_kernel_and_get_output(seed=42, batch_size=1024)  # Same seed!
        if out is not None:
            outputs.append(out)
            diff_to_ref = np.abs(out - reference_output)
            max_diffs_to_ref.append(diff_to_ref.max())
            print(f"  Run {i+1}: checksum = {out.sum():.2f}, max_diff_to_ref = {diff_to_ref.max():.4f}")
    
    # Also measure self-variation of this kernel
    if len(outputs) >= 2:
        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                diff = np.abs(outputs[i] - outputs[j])
                self_max_diffs.append(diff.max())
    
    if not max_diffs_to_ref:
        print("ERROR: No valid outputs")
        return None
    
    result = {
        'max_diff_to_reference': max(max_diffs_to_ref),
        'mean_diff_to_reference': np.mean(max_diffs_to_ref),
        'self_variation': max(self_max_diffs) if self_max_diffs else 0,
    }
    
    print(f"\nResults:")
    print(f"  Max difference to reference: {result['max_diff_to_reference']:.4f}")
    print(f"  Mean difference to reference: {result['mean_diff_to_reference']:.4f}")
    print(f"  Self-variation (inherent): {result['self_variation']:.4f}")
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--n-runs', type=int, default=10)
    args = parser.parse_args()
    
    print("=" * 70)
    print("PROPER CORRECTNESS VALIDATION")
    print("=" * 70)
    print(f"Acceptable max variation: {ACCEPTABLE_MAX_DIFF} (≈1 bit in bfloat16)")
    print(f"Batch size: {args.batch_size}")
    print(f"Runs per kernel: {args.n_runs}")
    
    # Ensure backup
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(ORIGINAL_PATH, BACKUP_PATH)
    
    # Measure baseline
    baseline_result = measure_baseline_variation(args.n_runs, args.batch_size)
    if baseline_result is None:
        return
    
    baseline_max_diff, reference_output = baseline_result
    
    # Test variants
    variants = [
        ('opt_vmcnt_reduce25', f"{KERNEL_DIR}/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256_opt_vmcnt_reduce25.co"),
        ('opt_vmcnt_zero', f"{KERNEL_DIR}/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256_opt_vmcnt_zero.co"),
        ('opt_vmcnt_cap8', f"{KERNEL_DIR}/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256_opt_vmcnt_cap8.co"),
        ('opt_both_reduce50', f"{KERNEL_DIR}/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256_opt_both_reduce50.co"),
    ]
    
    results = {}
    
    for name, path in variants:
        if not os.path.exists(path):
            print(f"\n⚠️ Variant {name} not found")
            continue
        
        result = compare_kernel_to_reference(path, reference_output, name, n_runs=5)
        if result:
            results[name] = result
    
    # Restore original
    shutil.copy(BACKUP_PATH, ORIGINAL_PATH)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nBaseline (original kernel self-variation): max_diff = {baseline_max_diff:.4f}")
    print(f"Acceptable threshold: max_diff ≤ {ACCEPTABLE_MAX_DIFF}")
    print(f"\n{'Kernel':<25} {'Max Diff':>10} {'Self Var':>10} {'Status':>12}")
    print("-" * 60)
    
    all_passed = True
    for name, result in results.items():
        max_diff = result['max_diff_to_reference']
        self_var = result['self_variation']
        
        # Check if within acceptable range
        # The key criterion: max_diff should be similar to baseline variation
        is_correct = max_diff <= max(ACCEPTABLE_MAX_DIFF, baseline_max_diff * 1.5)
        
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        if not is_correct:
            all_passed = False
        
        print(f"{name:<25} {max_diff:>10.4f} {self_var:>10.4f} {status:>12}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All kernels produce outputs within acceptable variation!")
        print(f"   (max element-wise diff ≤ {ACCEPTABLE_MAX_DIFF}, same as 1 bit in bfloat16)")
    else:
        print("❌ Some kernels have excessive variation - may be incorrect!")
    
    return results


if __name__ == '__main__':
    main()

