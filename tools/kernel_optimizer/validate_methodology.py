#!/usr/bin/env python3
"""
Validate our testing methodology:
1. Verify that copying the kernel without changes produces identical results
2. Verify that all kernels receive exactly the same inputs
3. Check for any issues with kernel loading/unloading
"""

import os
import shutil
import torch
import numpy as np

os.environ['VLLM_ROCM_USE_AITER'] = '1'

AITER_PATH = "/workspace/dev/aiter_long_context2"
KERNEL_DIR = f"{AITER_PATH}/hsa/gfx942/fmoe/silu"
KERNEL_NAME = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"
ORIGINAL_PATH = f"{KERNEL_DIR}/{KERNEL_NAME}"
BACKUP_PATH = f"{ORIGINAL_PATH}.backup"

FP8_DTYPE = torch.float8_e4m3fnuz


def create_deterministic_inputs(batch_size: int, seed: int = 12345):
    """Create deterministic inputs that can be exactly reproduced."""
    # Set all random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    num_experts = 256
    hidden_size = 7168
    intermediate_size = 256
    topk = 8
    device = "cuda"
    
    shard_intermediate_size = intermediate_size * 2
    
    # Create inputs - use specific values for reproducibility
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
    
    # Deterministic routing
    topk_weights = torch.rand(batch_size, topk, device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.randint(0, num_experts, (batch_size, topk), device=device, dtype=torch.int32)
    
    return {
        'hidden_states': hidden_states.clone(),
        'w1': w1.clone(),
        'w2': w2.clone(),
        'w1_scale': w1_scale.clone(),
        'w2_scale': w2_scale.clone(),
        'topk_weights': topk_weights.clone(),
        'topk_ids': topk_ids.clone(),
    }


def run_kernel(inputs: dict) -> torch.Tensor:
    """Run the kernel with given inputs."""
    from aiter.fused_moe import fused_moe, QuantType, ActivationType
    
    output = fused_moe(
        inputs['hidden_states'].clone(),  # Clone to ensure isolation
        inputs['w1'],
        inputs['w2'],
        inputs['topk_weights'],
        inputs['topk_ids'],
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,
        w1_scale=inputs['w1_scale'],
        w2_scale=inputs['w2_scale'],
    )
    torch.cuda.synchronize()
    return output


def compare_outputs(out1: torch.Tensor, out2: torch.Tensor, name1: str, name2: str):
    """Compare two outputs and print detailed comparison."""
    abs_diff = torch.abs(out1.float() - out2.float())
    rel_diff = abs_diff / (torch.abs(out1.float()) + 1e-8)
    
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    mean_abs = abs_diff.mean().item()
    
    # Check exact match
    exact_match = torch.equal(out1, out2)
    
    # Check within tolerance
    atol, rtol = 1e-3, 1e-3
    within_atol = abs_diff <= atol
    within_rtol = rel_diff <= rtol
    within_tolerance = within_atol | within_rtol
    pct_match = within_tolerance.float().mean().item() * 100
    
    print(f"\nComparing {name1} vs {name2}:")
    print(f"  Exact match: {exact_match}")
    print(f"  Max absolute error: {max_abs:.6e}")
    print(f"  Max relative error: {max_rel:.6e}")
    print(f"  Mean absolute error: {mean_abs:.6e}")
    print(f"  Elements within tolerance: {pct_match:.4f}%")
    
    if exact_match:
        return "EXACT_MATCH"
    elif pct_match > 99.99:
        return "NEAR_MATCH"
    elif pct_match > 99.0:
        return "ACCEPTABLE"
    else:
        return "MISMATCH"


def test_1_input_reproducibility():
    """Test 1: Verify inputs are reproducible."""
    print("\n" + "=" * 70)
    print("TEST 1: Input Reproducibility")
    print("=" * 70)
    
    batch_size = 2048
    
    # Create inputs twice with same seed
    inputs1 = create_deterministic_inputs(batch_size, seed=12345)
    inputs2 = create_deterministic_inputs(batch_size, seed=12345)
    
    all_match = True
    for key in inputs1:
        match = torch.equal(inputs1[key], inputs2[key])
        print(f"  {key}: {'✓ MATCH' if match else '✗ DIFFER'}")
        if not match:
            all_match = False
    
    return all_match


def test_2_same_kernel_multiple_runs():
    """Test 2: Same kernel, same inputs, multiple runs should give same output."""
    print("\n" + "=" * 70)
    print("TEST 2: Same Kernel Multiple Runs (Same Inputs)")
    print("=" * 70)
    
    batch_size = 2048
    inputs = create_deterministic_inputs(batch_size, seed=12345)
    
    print("Running kernel 5 times with identical inputs...")
    outputs = []
    for i in range(5):
        # Clone inputs for each run
        run_inputs = {k: v.clone() for k, v in inputs.items()}
        out = run_kernel(run_inputs)
        outputs.append(out.clone())
        print(f"  Run {i+1}: output checksum = {out.sum().item():.6f}")
    
    # Compare all outputs
    all_same = True
    for i in range(1, 5):
        result = compare_outputs(outputs[0], outputs[i], "Run 1", f"Run {i+1}")
        if result != "EXACT_MATCH":
            all_same = False
    
    return all_same


def test_3_kernel_copy_no_modification():
    """Test 3: Copy kernel file without modification, should produce same results."""
    print("\n" + "=" * 70)
    print("TEST 3: Kernel Copy Without Modification")
    print("=" * 70)
    
    batch_size = 2048
    
    # Ensure backup exists
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(ORIGINAL_PATH, BACKUP_PATH)
        print(f"Created backup: {BACKUP_PATH}")
    
    # Create inputs ONCE
    inputs = create_deterministic_inputs(batch_size, seed=54321)
    
    # Run with original kernel
    print("\nRunning with original kernel...")
    run_inputs1 = {k: v.clone() for k, v in inputs.items()}
    output_original = run_kernel(run_inputs1)
    print(f"  Output checksum: {output_original.sum().item():.6f}")
    
    # Copy kernel to itself (simulates our swap mechanism)
    print("\nCopying kernel file to itself (no actual change)...")
    kernel_data = open(ORIGINAL_PATH, 'rb').read()
    open(ORIGINAL_PATH, 'wb').write(kernel_data)
    
    # Need to restart Python/reload module to pick up new kernel?
    # Actually, AITER caches the loaded module, so let's test this
    
    print("\nRunning with 'copied' kernel (same file)...")
    run_inputs2 = {k: v.clone() for k, v in inputs.items()}
    output_copied = run_kernel(run_inputs2)
    print(f"  Output checksum: {output_copied.sum().item():.6f}")
    
    result = compare_outputs(output_original, output_copied, "Original", "Copied")
    
    # Restore from backup
    shutil.copy(BACKUP_PATH, ORIGINAL_PATH)
    
    return result == "EXACT_MATCH"


def test_4_verify_inputs_passed_correctly():
    """Test 4: Verify that the inputs we pass are actually used."""
    print("\n" + "=" * 70)
    print("TEST 4: Verify Inputs Are Actually Used")
    print("=" * 70)
    
    batch_size = 1024
    
    # Create two different inputs
    inputs1 = create_deterministic_inputs(batch_size, seed=11111)
    inputs2 = create_deterministic_inputs(batch_size, seed=22222)
    
    print("Running kernel with two different inputs...")
    output1 = run_kernel(inputs1)
    output2 = run_kernel(inputs2)
    
    print(f"  Output 1 checksum: {output1.sum().item():.6f}")
    print(f"  Output 2 checksum: {output2.sum().item():.6f}")
    
    # They should be DIFFERENT
    result = compare_outputs(output1, output2, "Inputs1", "Inputs2")
    
    different = result != "EXACT_MATCH"
    print(f"\nOutputs are different (as expected): {different}")
    
    return different


def test_5_file_swap_simulation():
    """Test 5: Simulate our file swap methodology with a copy of the original."""
    print("\n" + "=" * 70)
    print("TEST 5: File Swap Simulation (Our Methodology)")
    print("=" * 70)
    
    batch_size = 2048
    
    # Create a true copy of the kernel
    temp_copy_path = f"{KERNEL_DIR}/temp_original_copy.co"
    shutil.copy(ORIGINAL_PATH, temp_copy_path)
    print(f"Created temp copy: {temp_copy_path}")
    
    # Ensure backup
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(ORIGINAL_PATH, BACKUP_PATH)
    
    # Create inputs ONCE before any swapping
    inputs = create_deterministic_inputs(batch_size, seed=99999)
    
    # Run 1: With original kernel
    print("\nRun 1: Original kernel...")
    run_inputs1 = {k: v.clone() for k, v in inputs.items()}
    output1 = run_kernel(run_inputs1)
    checksum1 = output1.sum().item()
    print(f"  Checksum: {checksum1:.6f}")
    
    # Swap in the copy (which is identical to original)
    print("\nSwapping in the copy (identical to original)...")
    shutil.copy(temp_copy_path, ORIGINAL_PATH)
    
    # Run 2: With swapped kernel (should be identical)
    print("Run 2: After swap (should be identical)...")
    run_inputs2 = {k: v.clone() for k, v in inputs.items()}
    output2 = run_kernel(run_inputs2)
    checksum2 = output2.sum().item()
    print(f"  Checksum: {checksum2:.6f}")
    
    result = compare_outputs(output1, output2, "Before swap", "After swap")
    
    # Restore and cleanup
    shutil.copy(BACKUP_PATH, ORIGINAL_PATH)
    os.remove(temp_copy_path)
    print("\nCleaned up temp files")
    
    return result == "EXACT_MATCH" or result == "NEAR_MATCH"


def main():
    print("=" * 70)
    print("METHODOLOGY VALIDATION")
    print("=" * 70)
    print("Checking if our testing approach is sound...")
    
    results = {}
    
    results['test1_inputs_reproducible'] = test_1_input_reproducibility()
    results['test2_same_kernel_same_output'] = test_2_same_kernel_multiple_runs()
    results['test3_kernel_copy_same_output'] = test_3_kernel_copy_no_modification()
    results['test4_different_inputs_different_output'] = test_4_verify_inputs_passed_correctly()
    results['test5_file_swap_works'] = test_5_file_swap_simulation()
    
    # Summary
    print("\n" + "=" * 70)
    print("METHODOLOGY VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All methodology checks passed!")
        print("   Our testing approach is sound - any differences are from kernel changes.")
    else:
        print("\n❌ Some methodology checks failed!")
        print("   Need to investigate the testing approach before trusting results.")
    
    return all_passed


if __name__ == '__main__':
    main()

