#!/usr/bin/env python3
"""
Binary Patch Optimizer for AMDGPU Kernels

This script performs black-box optimization on GPU kernel binaries by:
1. Identifying patchable locations (s_waitcnt instructions)
2. Trying different waitcnt values
3. Benchmarking each variant
4. Keeping the best performing patches

Key insight for cache thrashing:
- s_waitcnt controls how many outstanding memory ops are allowed
- Lower vmcnt = less parallelism but better cache locality
- Higher vmcnt = more parallelism but more cache pressure

For large batches (8k+), reducing vmcnt may help cache locality.
"""

import os
import sys
import shutil
import struct
import subprocess
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Paths
AITER_PATH = "/workspace/dev/aiter_long_context2"
KERNEL_DIR = f"{AITER_PATH}/hsa/gfx942/fmoe/silu"
KERNEL_NAME = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256.co"
ORIGINAL_KERNEL = f"{KERNEL_DIR}/{KERNEL_NAME}"
BACKUP_KERNEL = f"{KERNEL_DIR}/{KERNEL_NAME}.backup"
WORK_DIR = "/workspace/dev/vllm/benchmarks/kernels/patch_work"

@dataclass
class WaitcntLocation:
    """Represents a s_waitcnt instruction location."""
    offset: int  # File offset
    original_bytes: bytes  # Original 4 bytes
    vmcnt: int  # Original vmcnt value
    lgkmcnt: int  # Original lgkmcnt value
    expcnt: int  # Original expcnt value
    
    def describe(self) -> str:
        return f"offset=0x{self.offset:04X}, vmcnt={self.vmcnt}, lgkmcnt={self.lgkmcnt}, expcnt={self.expcnt}"

def decode_waitcnt(imm16: int) -> Tuple[int, int, int]:
    """Decode s_waitcnt immediate to (vmcnt, lgkmcnt, expcnt).
    
    GFX9+ encoding:
    - vmcnt[3:0] = bits [3:0]
    - vmcnt[5:4] = bits [15:14] 
    - lgkmcnt = bits [11:8]
    - expcnt = bits [6:4]
    """
    vmcnt_lo = imm16 & 0xF
    vmcnt_hi = (imm16 >> 14) & 0x3
    vmcnt = vmcnt_lo | (vmcnt_hi << 4)
    lgkmcnt = (imm16 >> 8) & 0xF
    expcnt = (imm16 >> 4) & 0x7
    return vmcnt, lgkmcnt, expcnt

def encode_waitcnt(vmcnt: int, lgkmcnt: int, expcnt: int) -> int:
    """Encode (vmcnt, lgkmcnt, expcnt) to s_waitcnt immediate."""
    vmcnt_lo = vmcnt & 0xF
    vmcnt_hi = (vmcnt >> 4) & 0x3
    imm16 = vmcnt_lo | (expcnt << 4) | (lgkmcnt << 8) | (vmcnt_hi << 14)
    return imm16

def find_waitcnt_locations(kernel_data: bytes) -> List[WaitcntLocation]:
    """Find all s_waitcnt instructions in the kernel binary."""
    locations = []
    
    # s_waitcnt opcode is 0xBF8C (little endian: 8C BF)
    # Full instruction is 4 bytes: imm16_lo, imm16_hi, 0x8C, 0xBF
    i = 0
    while i < len(kernel_data) - 4:
        # Look for s_waitcnt pattern (0xBF8C in bytes 2-3)
        if kernel_data[i+2:i+4] == b'\x8c\xbf':
            imm16 = struct.unpack('<H', kernel_data[i:i+2])[0]
            vmcnt, lgkmcnt, expcnt = decode_waitcnt(imm16)
            
            loc = WaitcntLocation(
                offset=i,
                original_bytes=kernel_data[i:i+4],
                vmcnt=vmcnt,
                lgkmcnt=lgkmcnt,
                expcnt=expcnt,
            )
            locations.append(loc)
            i += 4
        else:
            i += 1
    
    return locations

def apply_patch(kernel_data: bytes, offset: int, vmcnt: int, lgkmcnt: int, expcnt: int) -> bytes:
    """Apply a single waitcnt patch."""
    imm16 = encode_waitcnt(vmcnt, lgkmcnt, expcnt)
    patched = bytearray(kernel_data)
    patched[offset:offset+2] = struct.pack('<H', imm16)
    return bytes(patched)

def run_benchmark(batch_size: int = 8192, num_iters: int = 20) -> Optional[float]:
    """Run benchmark and return average time in microseconds."""
    try:
        result = subprocess.run(
            [
                "python", "-c", f"""
import os
os.environ['VLLM_ROCM_USE_AITER'] = '1'

import torch
import time

# Direct AITER import for benchmark
from aiter.fused_moe import fused_moe as aiter_fused_moe
from aiter.fused_moe import QuantType, ActivationType

# Setup
batch_size = {batch_size}
num_experts = 256
hidden_size = 7168
intermediate_size = 256  # After TP8
topk = 8
device = "cuda"
FP8_DTYPE = torch.float8_e4m3fnuz

# Create inputs
hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

# FP8 weights (note: AITER expects [E, N, K] layout)
shard_intermediate_size = intermediate_size * 2  # gate + up
w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, device=device, dtype=torch.bfloat16).to(FP8_DTYPE)
w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16).to(FP8_DTYPE)

# Scales for blockscale quantization [E, N//128, K//128]
scale_shape_w1 = (num_experts, shard_intermediate_size // 128, hidden_size // 128)
scale_shape_w2 = (num_experts, hidden_size // 128, intermediate_size // 128)
w1_scale = torch.ones(scale_shape_w1, device=device, dtype=torch.float32) * 0.1
w2_scale = torch.ones(scale_shape_w2, device=device, dtype=torch.float32) * 0.1

# Generate random topk routing
topk_weights = torch.rand(batch_size, topk, device=device, dtype=torch.float32)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # normalize
topk_ids = torch.randint(0, num_experts, (batch_size, topk), device=device, dtype=torch.int32)

# Warmup
for _ in range(5):
    _ = aiter_fused_moe(
        hidden_states, w1, w2,
        topk_weights, topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,  # FP8 blockscale
        w1_scale=w1_scale, w2_scale=w2_scale,
    )
    torch.cuda.synchronize()

# Benchmark
start_events = [torch.cuda.Event(enable_timing=True) for _ in range({num_iters})]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range({num_iters})]

for i in range({num_iters}):
    start_events[i].record()
    _ = aiter_fused_moe(
        hidden_states, w1, w2,
        topk_weights, topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,  # FP8 blockscale
        w1_scale=w1_scale, w2_scale=w2_scale,
    )
    end_events[i].record()

torch.cuda.synchronize()
times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]  # us
avg_time = sum(times) / len(times)
print(f"BENCHMARK_RESULT:{{avg_time:.2f}}")
"""
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode != 0:
            print(f"Benchmark failed: {result.stderr[:500]}")
            return None
        
        # Parse result
        for line in result.stdout.split('\n'):
            if line.startswith('BENCHMARK_RESULT:'):
                return float(line.split(':')[1])
        
        print(f"No result found in output: {result.stdout[:500]}")
        return None
        
    except subprocess.TimeoutExpired:
        print("Benchmark timed out")
        return None
    except Exception as e:
        print(f"Benchmark error: {e}")
        return None

def save_kernel(kernel_data: bytes, path: str):
    """Save kernel binary."""
    with open(path, 'wb') as f:
        f.write(kernel_data)
    # Make executable
    os.chmod(path, 0o755)

def load_kernel(path: str) -> bytes:
    """Load kernel binary."""
    with open(path, 'rb') as f:
        return f.read()

def optimize_waitcnt(
    locations: List[WaitcntLocation],
    kernel_data: bytes,
    batch_size: int = 8192,
    num_iterations: int = 50,
    population_size: int = 10,
) -> Tuple[bytes, Dict]:
    """
    Genetic algorithm-style optimization of waitcnt values.
    
    Strategy for cache thrashing:
    1. Try reducing vmcnt values (less outstanding memory ops)
    2. Try adding more strict synchronization
    """
    print(f"\n=== Starting Optimization ===")
    print(f"Found {len(locations)} waitcnt locations")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {num_iterations}")
    
    # Filter to only locations with vmcnt > 0 (these affect memory pipeline)
    patchable = [loc for loc in locations if loc.vmcnt > 0]
    print(f"Patchable locations (vmcnt > 0): {len(patchable)}")
    
    if not patchable:
        print("No patchable locations found!")
        return kernel_data, {}
    
    # Backup original
    if not os.path.exists(BACKUP_KERNEL):
        shutil.copy(ORIGINAL_KERNEL, BACKUP_KERNEL)
        print(f"Backed up original to {BACKUP_KERNEL}")
    
    # Get baseline
    print("\n--- Baseline Performance ---")
    save_kernel(kernel_data, ORIGINAL_KERNEL)
    baseline_time = run_benchmark(batch_size)
    if baseline_time is None:
        print("Failed to get baseline!")
        return kernel_data, {}
    print(f"Baseline: {baseline_time:.2f} us")
    
    best_kernel = kernel_data
    best_time = baseline_time
    best_patches = {}
    
    # Optimization strategies to try
    strategies = [
        # Strategy 1: Reduce all vmcnt by half
        ("reduce_vmcnt_half", lambda v, l, e: (max(0, v // 2), l, e)),
        # Strategy 2: Set all vmcnt to 0 (maximum synchronization)
        ("vmcnt_zero", lambda v, l, e: (0, l, e)),
        # Strategy 3: Reduce vmcnt by 25%
        ("reduce_vmcnt_25", lambda v, l, e: (max(0, v - v // 4), l, e)),
        # Strategy 4: Cap vmcnt at 8
        ("cap_vmcnt_8", lambda v, l, e: (min(v, 8), l, e)),
        # Strategy 5: Cap vmcnt at 4
        ("cap_vmcnt_4", lambda v, l, e: (min(v, 4), l, e)),
        # Strategy 6: Set lgkmcnt to 0 everywhere
        ("lgkmcnt_zero", lambda v, l, e: (v, 0, e)),
        # Strategy 7: Full sync everywhere
        ("full_sync", lambda v, l, e: (0, 0, 0)),
        # Strategy 8: Only modify high vmcnt (>12) locations
        ("reduce_high_vmcnt", lambda v, l, e: (min(v, 8), l, e) if v > 12 else (v, l, e)),
    ]
    
    results = []
    
    for strategy_name, transform in strategies:
        print(f"\n--- Testing: {strategy_name} ---")
        
        # Apply patches
        patched = kernel_data
        patches = {}
        for loc in patchable:
            new_vmcnt, new_lgkmcnt, new_expcnt = transform(loc.vmcnt, loc.lgkmcnt, loc.expcnt)
            if (new_vmcnt, new_lgkmcnt, new_expcnt) != (loc.vmcnt, loc.lgkmcnt, loc.expcnt):
                patched = apply_patch(patched, loc.offset, new_vmcnt, new_lgkmcnt, new_expcnt)
                patches[loc.offset] = {
                    'old': (loc.vmcnt, loc.lgkmcnt, loc.expcnt),
                    'new': (new_vmcnt, new_lgkmcnt, new_expcnt),
                }
        
        if not patches:
            print("  No changes for this strategy")
            continue
        
        print(f"  Applied {len(patches)} patches")
        
        # Test patched kernel
        save_kernel(patched, ORIGINAL_KERNEL)
        time_result = run_benchmark(batch_size)
        
        if time_result is None:
            print(f"  CRASHED - reverting")
            save_kernel(kernel_data, ORIGINAL_KERNEL)
            continue
        
        speedup = baseline_time / time_result
        results.append((strategy_name, time_result, speedup, patches))
        print(f"  Time: {time_result:.2f} us (speedup: {speedup:.3f}x)")
        
        if time_result < best_time:
            best_time = time_result
            best_kernel = patched
            best_patches = patches
            print(f"  NEW BEST!")
    
    # Restore original for now
    save_kernel(kernel_data, ORIGINAL_KERNEL)
    
    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Baseline: {baseline_time:.2f} us")
    print(f"Best:     {best_time:.2f} us ({baseline_time/best_time:.3f}x speedup)")
    print("\nAll results:")
    for name, time_val, speedup, _ in sorted(results, key=lambda x: x[1]):
        status = "BEST" if time_val == best_time and time_val < baseline_time else ""
        print(f"  {name:25s}: {time_val:8.2f} us ({speedup:.3f}x) {status}")
    
    return best_kernel, best_patches

def optimize_targeted(
    locations: List[WaitcntLocation],
    kernel_data: bytes,
    batch_size: int = 8192,
) -> Tuple[bytes, Dict]:
    """More targeted optimization strategies."""
    print(f"\n=== Targeted Optimization ===")
    
    # Only target locations with specific vmcnt values
    high_vmcnt_locs = [loc for loc in locations if loc.vmcnt >= 16]
    medium_vmcnt_locs = [loc for loc in locations if 8 <= loc.vmcnt < 16]
    
    print(f"High vmcnt (>=16) locations: {len(high_vmcnt_locs)}")
    print(f"Medium vmcnt (8-15) locations: {len(medium_vmcnt_locs)}")
    
    # Backup original
    if not os.path.exists(BACKUP_KERNEL):
        shutil.copy(ORIGINAL_KERNEL, BACKUP_KERNEL)
    
    # Get baseline
    print("\n--- Baseline ---")
    save_kernel(kernel_data, ORIGINAL_KERNEL)
    baseline_time = run_benchmark(batch_size)
    if baseline_time is None:
        print("Failed to get baseline!")
        return kernel_data, {}
    print(f"Baseline: {baseline_time:.2f} us")
    
    strategies = [
        # Target only high vmcnt locations
        ("high_vmcnt_to_0", high_vmcnt_locs, lambda v, l, e: (0, l, e)),
        ("high_vmcnt_to_4", high_vmcnt_locs, lambda v, l, e: (4, l, e)),
        ("high_vmcnt_to_8", high_vmcnt_locs, lambda v, l, e: (8, l, e)),
        # Target only medium vmcnt  
        ("medium_vmcnt_to_0", medium_vmcnt_locs, lambda v, l, e: (0, l, e)),
        ("medium_vmcnt_to_4", medium_vmcnt_locs, lambda v, l, e: (4, l, e)),
        # Combined but different values
        ("high_to_4_med_to_0", high_vmcnt_locs + medium_vmcnt_locs, 
         lambda v, l, e: (0, l, e) if v < 16 else (4, l, e)),
    ]
    
    best_time = baseline_time
    best_kernel = kernel_data
    results = []
    
    for name, locs, transform in strategies:
        print(f"\n--- Testing: {name} ---")
        
        if not locs:
            print("  No locations to patch")
            continue
            
        patched = kernel_data
        for loc in locs:
            new_vmcnt, new_lgkmcnt, new_expcnt = transform(loc.vmcnt, loc.lgkmcnt, loc.expcnt)
            patched = apply_patch(patched, loc.offset, new_vmcnt, new_lgkmcnt, new_expcnt)
        
        save_kernel(patched, ORIGINAL_KERNEL)
        time_result = run_benchmark(batch_size)
        
        if time_result is None:
            print(f"  CRASHED")
            save_kernel(kernel_data, ORIGINAL_KERNEL)
            continue
        
        speedup = baseline_time / time_result
        results.append((name, time_result, speedup))
        print(f"  Time: {time_result:.2f} us (speedup: {speedup:.3f}x)")
        
        if time_result < best_time:
            best_time = time_result
            best_kernel = patched
            print(f"  NEW BEST!")
    
    # Restore original
    save_kernel(kernel_data, ORIGINAL_KERNEL)
    
    print("\n" + "=" * 50)
    print("TARGETED OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Baseline: {baseline_time:.2f} us")
    print(f"Best:     {best_time:.2f} us ({baseline_time/best_time:.3f}x)")
    for name, time_val, speedup in sorted(results, key=lambda x: x[1]):
        status = "BEST" if time_val == best_time else ""
        print(f"  {name:25s}: {time_val:8.2f} us ({speedup:.3f}x) {status}")
    
    return best_kernel, {}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Binary patch optimizer for AMDGPU kernels")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size for benchmarking")
    parser.add_argument("--iterations", type=int, default=50, help="Optimization iterations")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't optimize")
    parser.add_argument("--apply-best", action="store_true", help="Apply best patches permanently")
    parser.add_argument("--restore", action="store_true", help="Restore original kernel")
    parser.add_argument("--targeted", action="store_true", help="Run targeted optimization")
    args = parser.parse_args()
    
    os.makedirs(WORK_DIR, exist_ok=True)
    
    if args.restore:
        if os.path.exists(BACKUP_KERNEL):
            shutil.copy(BACKUP_KERNEL, ORIGINAL_KERNEL)
            print(f"Restored original kernel from backup")
        else:
            print("No backup found!")
        return
    
    # Load kernel
    print(f"Loading kernel: {ORIGINAL_KERNEL}")
    kernel_data = load_kernel(ORIGINAL_KERNEL)
    print(f"Kernel size: {len(kernel_data)} bytes")
    
    # Find waitcnt locations
    locations = find_waitcnt_locations(kernel_data)
    print(f"\nFound {len(locations)} s_waitcnt instructions:")
    
    for i, loc in enumerate(locations[:20]):  # Show first 20
        print(f"  [{i:2d}] {loc.describe()}")
    if len(locations) > 20:
        print(f"  ... and {len(locations) - 20} more")
    
    if args.analyze_only:
        return
    
    if args.targeted:
        # Run targeted optimization
        best_kernel, best_patches = optimize_targeted(
            locations, kernel_data, 
            batch_size=args.batch_size,
        )
    else:
        # Run standard optimization
        best_kernel, best_patches = optimize_waitcnt(
        locations, kernel_data, 
        batch_size=args.batch_size,
        num_iterations=args.iterations,
    )
    
    if args.apply_best and best_patches:
        print("\nApplying best patches permanently...")
        save_kernel(best_kernel, ORIGINAL_KERNEL)
        print("Done! Run with --restore to revert.")

if __name__ == '__main__':
    main()
