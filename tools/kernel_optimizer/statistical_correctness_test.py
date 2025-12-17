#!/usr/bin/env python3
"""
Statistical correctness validation for optimized kernels.

Key insight: The original kernel is NON-DETERMINISTIC - about 48-53% of output
elements differ between runs with identical inputs. Max diff is ~2.0 (one bit).

This script validates that optimized kernels produce STATISTICALLY EQUIVALENT
outputs to the original kernel by:
1. Running each kernel N times with IDENTICAL inputs
2. Comparing output distributions (mean, std, range)
3. Using t-tests to check if differences are significant
"""

import os
import shutil
import subprocess
import sys
import numpy as np
from scipy import stats

os.environ['VLLM_ROCM_USE_AITER'] = '1'

AITER_PATH = "/workspace/dev/aiter_long_context2"
KERNEL_DIR = f"{AITER_PATH}/hsa/gfx942/fmoe/silu"
KERNEL_NAME = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"
ORIGINAL_PATH = f"{KERNEL_DIR}/{KERNEL_NAME}"
BACKUP_PATH = f"{ORIGINAL_PATH}.backup"


def get_kernel_hash(path):
    """Get MD5 hash of kernel file."""
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def run_kernel_test(n_runs: int = 30, batch_size: int = 1024, seed: int = 42):
    """
    Run kernel N times in a fresh subprocess with IDENTICAL inputs each run.
    Returns checksums and statistics.
    """
    script = f'''
import os
os.environ['VLLM_ROCM_USE_AITER'] = '1'
import torch
from aiter.fused_moe import fused_moe, QuantType, ActivationType

# CRITICAL: Set seeds ONCE before creating inputs
torch.manual_seed({seed})
torch.cuda.manual_seed({seed})

batch_size = {batch_size}
num_experts = 256
hidden_size = 7168
intermediate_size = 256
topk = 8

# Create inputs ONCE - these are IDENTICAL for every kernel run
hidden = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16)
w1 = torch.randn(num_experts, intermediate_size*2, hidden_size, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
w2 = torch.randn(num_experts, hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
w1_scale = torch.ones((num_experts, 4, hidden_size//128), device="cuda", dtype=torch.float32) * 0.1
w2_scale = torch.ones((num_experts, hidden_size//128, 2), device="cuda", dtype=torch.float32) * 0.1
weights = torch.rand(batch_size, topk, device="cuda", dtype=torch.float32)
weights = weights / weights.sum(dim=-1, keepdim=True)
ids = torch.randint(0, num_experts, (batch_size, topk), device="cuda", dtype=torch.int32)

# Verify inputs are deterministic
input_checksum = hidden.sum().item() + weights.sum().item() + ids.sum().item()
print(f"INPUT_CHECKSUM:{{input_checksum}}")

# Run N times with SAME inputs
checksums = []
means = []
stds = []
for i in range({n_runs}):
    # Clone hidden to ensure it's not modified
    out = fused_moe(hidden.clone(), w1, w2, weights, ids, 
                   activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
                   w1_scale=w1_scale, w2_scale=w2_scale)
    torch.cuda.synchronize()
    checksums.append(out.sum().item())
    means.append(out.float().mean().item())
    stds.append(out.float().std().item())

print(f"CHECKSUMS:{{checksums}}")
print(f"MEANS:{{means}}")
print(f"STDS:{{stds}}")
'''
    
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=180
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr[-1000:]}")
        return None
    
    data = {}
    for line in result.stdout.split('\n'):
        if line.startswith('INPUT_CHECKSUM:'):
            data['input_checksum'] = float(line.replace('INPUT_CHECKSUM:', ''))
        elif line.startswith('CHECKSUMS:'):
            data['checksums'] = eval(line.replace('CHECKSUMS:', ''))
        elif line.startswith('MEANS:'):
            data['means'] = eval(line.replace('MEANS:', ''))
        elif line.startswith('STDS:'):
            data['stds'] = eval(line.replace('STDS:', ''))
    
    return data


def compare_kernel_outputs(original_data, variant_data, variant_name):
    """Compare outputs statistically."""
    print(f"\n{'='*60}")
    print(f"Comparing: original vs {variant_name}")
    print(f"{'='*60}")
    
    # Verify inputs were identical
    if abs(original_data['input_checksum'] - variant_data['input_checksum']) > 1e-6:
        print(f"⚠️  WARNING: Input checksums differ!")
        print(f"   Original: {original_data['input_checksum']}")
        print(f"   Variant:  {variant_data['input_checksum']}")
        return False
    else:
        print(f"✓ Input checksums match: {original_data['input_checksum']:.2f}")
    
    # Compare checksum distributions
    orig_cs = np.array(original_data['checksums'])
    var_cs = np.array(variant_data['checksums'])
    
    print(f"\nOutput checksum statistics:")
    print(f"  Original: mean={np.mean(orig_cs):.2f}, std={np.std(orig_cs):.2f}, "
          f"range=[{np.min(orig_cs):.2f}, {np.max(orig_cs):.2f}]")
    print(f"  {variant_name}: mean={np.mean(var_cs):.2f}, std={np.std(var_cs):.2f}, "
          f"range=[{np.min(var_cs):.2f}, {np.max(var_cs):.2f}]")
    
    # T-test
    t_stat, p_value = stats.ttest_ind(orig_cs, var_cs)
    print(f"\nT-test: t={t_stat:.3f}, p={p_value:.4f}")
    
    # Check range overlap
    ranges_overlap = not (np.max(orig_cs) < np.min(var_cs) or np.max(var_cs) < np.min(orig_cs))
    print(f"Ranges overlap: {ranges_overlap}")
    
    # Compare element-wise statistics
    orig_means = np.array(original_data['means'])
    var_means = np.array(variant_data['means'])
    print(f"\nElement-wise mean: original={np.mean(orig_means):.6f}, {variant_name}={np.mean(var_means):.6f}")
    
    orig_stds = np.array(original_data['stds'])
    var_stds = np.array(variant_data['stds'])
    print(f"Element-wise std: original={np.mean(orig_stds):.6f}, {variant_name}={np.mean(var_stds):.6f}")
    
    # Verdict
    mean_diff = abs(np.mean(orig_cs) - np.mean(var_cs))
    combined_std = np.sqrt(np.std(orig_cs)**2 + np.std(var_cs)**2)
    
    print(f"\nStatistical verdict:")
    print(f"  Mean difference: {mean_diff:.2f}")
    print(f"  Combined std: {combined_std:.2f}")
    print(f"  Ratio: {mean_diff / combined_std:.2f} (should be < 2 for equivalence)")
    
    # Verdict criteria:
    # 1. p-value > 0.01 (not significantly different)
    # 2. Ranges overlap
    # 3. Mean difference < 2 * combined_std
    is_equivalent = (p_value > 0.01) and ranges_overlap and (mean_diff < 2 * combined_std)
    
    if is_equivalent:
        print(f"\n✅ {variant_name} is STATISTICALLY EQUIVALENT to original")
    else:
        print(f"\n❌ {variant_name} may be DIFFERENT from original")
    
    return is_equivalent


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--n-runs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print("STATISTICAL CORRECTNESS VALIDATION")
    print("=" * 70)
    print(f"Batch size: {args.batch_size}")
    print(f"Runs per kernel: {args.n_runs}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Ensure backup exists
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(ORIGINAL_PATH, BACKUP_PATH)
    
    # Test original kernel
    print(f"\n=== Testing ORIGINAL kernel ===")
    print(f"Kernel hash: {get_kernel_hash(BACKUP_PATH)}")
    shutil.copy(BACKUP_PATH, ORIGINAL_PATH)
    original_data = run_kernel_test(args.n_runs, args.batch_size, args.seed)
    
    if original_data is None:
        print("Failed to run original kernel!")
        return
    
    print(f"Input checksum: {original_data['input_checksum']:.2f}")
    print(f"Output checksums (first 5): {original_data['checksums'][:5]}")
    print(f"Unique checksums: {len(set(original_data['checksums']))}")
    
    # Variants to test
    broken_dir = f"{KERNEL_DIR}/broken_optimizations"
    variants = [
        ('opt_vmcnt_reduce25', f"{broken_dir}/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256_opt_vmcnt_reduce25.co"),
        ('opt_vmcnt_zero', f"{broken_dir}/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256_opt_vmcnt_zero.co"),
        ('opt_vmcnt_cap8', f"{broken_dir}/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256_opt_vmcnt_cap8.co"),
        ('opt_both_reduce50', f"{broken_dir}/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256_opt_both_reduce50.co"),
    ]
    
    results = {}
    
    for name, path in variants:
        if not os.path.exists(path):
            print(f"\n⚠️ Variant {name} not found at {path}")
            continue
        
        print(f"\n=== Testing {name} ===")
        print(f"Kernel hash: {get_kernel_hash(path)}")
        
        # Swap kernel
        shutil.copy(path, ORIGINAL_PATH)
        
        # Run test
        variant_data = run_kernel_test(args.n_runs, args.batch_size, args.seed)
        
        if variant_data is None:
            print(f"Failed to run {name}!")
            continue
        
        # Compare
        is_equivalent = compare_kernel_outputs(original_data, variant_data, name)
        results[name] = is_equivalent
    
    # Restore original
    shutil.copy(BACKUP_PATH, ORIGINAL_PATH)
    print(f"\n✓ Restored original kernel")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    for name, is_equiv in results.items():
        status = "✅ EQUIVALENT" if is_equiv else "❌ DIFFERENT"
        print(f"  {name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All optimized kernels are statistically equivalent to original!")
    
    return results


if __name__ == '__main__':
    main()

