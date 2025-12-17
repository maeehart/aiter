#!/usr/bin/env python3
"""
Cache policy optimizer for MoE kernels.

Modifies cache hints (glc, slc) in memory instructions to optimize
for different workload sizes.

glc (bit 14) = 1: Globally coherent, bypass L1 cache
slc (bit 17) = 1: System level coherent, bypass L2 cache
"""

import os
import shutil
import struct
import subprocess
import sys
import numpy as np
from typing import List, Dict, Tuple

AITER_PATH = "/workspace/dev/aiter_long_context2"
KERNEL_DIR = f"{AITER_PATH}/hsa/gfx942/fmoe/silu"
KERNEL_NAME = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"
ORIGINAL_PATH = f"{KERNEL_DIR}/{KERNEL_NAME}"
BACKUP_PATH = f"{ORIGINAL_PATH}.backup"


def find_memory_instructions(data: bytes) -> Dict[str, List[Dict]]:
    """Find all memory instructions that can have cache hints modified."""
    results = {
        'global_load': [],
        'buffer_load': [],
        'buffer_store': [],
    }
    
    i = 0
    while i < len(data) - 8:
        # Check 4-byte instruction at position i
        instr = struct.unpack('<I', data[i:i+4])[0]
        
        # global_load_dword: opcode DC5xxxxx
        if (instr >> 24) == 0xDC and ((instr >> 16) & 0xF0) == 0x50:
            glc = (instr >> 14) & 1
            slc = (instr >> 17) & 1
            results['global_load'].append({
                'offset': i,
                'instr': instr,
                'glc': glc,
                'slc': slc,
            })
            i += 8  # global_load is 8 bytes
            continue
        
        # buffer_load: opcode E05xxxxx
        if (instr >> 24) == 0xE0 and ((instr >> 16) & 0xF0) == 0x50:
            glc = (instr >> 14) & 1
            slc = (instr >> 17) & 1
            results['buffer_load'].append({
                'offset': i,
                'instr': instr,
                'glc': glc,
                'slc': slc,
            })
            i += 8
            continue
        
        # buffer_store: opcode E07xxxxx  
        if (instr >> 24) == 0xE0 and ((instr >> 16) & 0xF0) == 0x70:
            glc = (instr >> 14) & 1
            slc = (instr >> 17) & 1
            results['buffer_store'].append({
                'offset': i,
                'instr': instr,
                'glc': glc,
                'slc': slc,
            })
            i += 8
            continue
        
        i += 4
    
    return results


def modify_cache_bits(instr: int, glc: int, slc: int) -> int:
    """Modify cache bits in instruction."""
    # Clear bits 14 and 17
    modified = instr & ~((1 << 14) | (1 << 17))
    # Set new values
    modified |= (glc << 14) | (slc << 17)
    return modified


def apply_cache_strategy(data: bytes, instructions: Dict, strategy: str) -> bytes:
    """Apply a cache strategy to the kernel."""
    patched = bytearray(data)
    count = 0
    
    if strategy == 'glc_global':
        # Set glc=1 for global loads only
        for instr in instructions['global_load']:
            new_instr = modify_cache_bits(instr['instr'], glc=1, slc=0)
            patched[instr['offset']:instr['offset']+4] = struct.pack('<I', new_instr)
            count += 1
    
    elif strategy == 'slc_global':
        # Set slc=1 for global loads (bypass L2)
        for instr in instructions['global_load']:
            new_instr = modify_cache_bits(instr['instr'], glc=0, slc=1)
            patched[instr['offset']:instr['offset']+4] = struct.pack('<I', new_instr)
            count += 1
    
    elif strategy == 'glc_buffer':
        # Set glc=1 for buffer loads
        for instr in instructions['buffer_load']:
            new_instr = modify_cache_bits(instr['instr'], glc=1, slc=0)
            patched[instr['offset']:instr['offset']+4] = struct.pack('<I', new_instr)
            count += 1
    
    elif strategy == 'slc_buffer':
        # Set slc=1 for buffer loads (bypass L2)
        for instr in instructions['buffer_load']:
            new_instr = modify_cache_bits(instr['instr'], glc=0, slc=1)
            patched[instr['offset']:instr['offset']+4] = struct.pack('<I', new_instr)
            count += 1
    
    elif strategy == 'slc_all':
        # Set slc=1 for all memory ops (bypass L2 completely)
        for cat in ['global_load', 'buffer_load']:
            for instr in instructions[cat]:
                new_instr = modify_cache_bits(instr['instr'], glc=0, slc=1)
                patched[instr['offset']:instr['offset']+4] = struct.pack('<I', new_instr)
                count += 1
    
    elif strategy == 'glc_slc_all':
        # Set both glc=1 and slc=1 (no caching)
        for cat in ['global_load', 'buffer_load']:
            for instr in instructions[cat]:
                new_instr = modify_cache_bits(instr['instr'], glc=1, slc=1)
                patched[instr['offset']:instr['offset']+4] = struct.pack('<I', new_instr)
                count += 1
    
    return bytes(patched), count


def run_benchmark(batch_size: int = 8192, n_runs: int = 20) -> Tuple[float, float, List[float]]:
    """Run benchmark in subprocess and return (mean, std, all_values)."""
    script = f'''
import os
os.environ['VLLM_ROCM_USE_AITER'] = '1'
import torch
from aiter.fused_moe import fused_moe, QuantType, ActivationType

torch.manual_seed(42)
torch.cuda.manual_seed(42)

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

# Warmup
for _ in range(10):
    out = fused_moe(hidden.clone(), w1, w2, weights, ids, 
                   activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
                   w1_scale=w1_scale, w2_scale=w2_scale)
    torch.cuda.synchronize()

# Benchmark
times = []
for _ in range({n_runs}):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fused_moe(hidden.clone(), w1, w2, weights, ids, 
                   activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
                   w1_scale=w1_scale, w2_scale=w2_scale)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # us

print(f"TIMES:{{times}}")
print(f"CHECKSUM:{{out.sum().item()}}")
'''
    
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True, text=True, timeout=120
    )
    
    if result.returncode != 0:
        print(f"Error stdout: {result.stdout[-500:]}")
        print(f"Error stderr: {result.stderr[-500:]}")
        return None, None, None
    
    times = None
    checksum = None
    for line in result.stdout.split('\n'):
        if line.startswith('TIMES:'):
            # Parse list from string representation
            times = eval(line.replace('TIMES:', ''))
        if line.startswith('CHECKSUM:'):
            checksum = float(line.replace('CHECKSUM:', ''))
    
    if times:
        return np.mean(times), np.std(times), times
    return None, None, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--n-runs', type=int, default=30)
    args = parser.parse_args()
    
    print("=" * 70)
    print("CACHE POLICY OPTIMIZER")
    print("=" * 70)
    print(f"Batch size: {args.batch_size}")
    print(f"Runs per strategy: {args.n_runs}")
    
    # Ensure backup
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(ORIGINAL_PATH, BACKUP_PATH)
    
    # Load and analyze kernel
    with open(BACKUP_PATH, 'rb') as f:
        original_data = f.read()
    
    instructions = find_memory_instructions(original_data)
    
    print(f"\nMemory instructions found:")
    print(f"  global_load: {len(instructions['global_load'])}")
    print(f"  buffer_load: {len(instructions['buffer_load'])}")
    print(f"  buffer_store: {len(instructions['buffer_store'])}")
    
    # Show current cache settings
    print(f"\nCurrent cache settings:")
    for cat in ['global_load', 'buffer_load']:
        for instr in instructions[cat][:3]:  # Show first 3
            print(f"  {cat} @ 0x{instr['offset']:X}: glc={instr['glc']}, slc={instr['slc']}")
    
    # Strategies to test
    strategies = [
        ('baseline', None),
        ('glc_buffer', 'Buffer loads bypass L1'),
        ('slc_buffer', 'Buffer loads bypass L2'),
        ('slc_all', 'All loads bypass L2'),
        # ('glc_slc_all', 'All loads bypass both caches'),  # Usually too aggressive
    ]
    
    results = {}
    
    for name, desc in strategies:
        print(f"\n=== Testing: {name} ===")
        if desc:
            print(f"    ({desc})")
        
        if name == 'baseline':
            shutil.copy(BACKUP_PATH, ORIGINAL_PATH)
            patched_count = 0
        else:
            patched, patched_count = apply_cache_strategy(original_data, instructions, name)
            with open(ORIGINAL_PATH, 'wb') as f:
                f.write(patched)
            os.chmod(ORIGINAL_PATH, 0o755)
        
        print(f"    Modified {patched_count} instructions")
        
        mean, std, times = run_benchmark(args.batch_size, args.n_runs)
        
        if mean is not None:
            results[name] = {'mean': mean, 'std': std, 'times': times}
            print(f"    Time: {mean:.2f} ± {std:.2f} μs")
        else:
            print(f"    FAILED")
    
    # Restore original
    shutil.copy(BACKUP_PATH, ORIGINAL_PATH)
    
    # Summary
    if 'baseline' in results:
        baseline_mean = results['baseline']['mean']
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Strategy':<20} {'Time (μs)':<15} {'Speedup':<10}")
        print("-" * 45)
        
        for name in results:
            mean = results[name]['mean']
            speedup = baseline_mean / mean
            marker = " ★" if speedup > 1.01 else ""
            print(f"{name:<20} {mean:>10.2f} ± {results[name]['std']:<5.1f} {speedup:>7.3f}x{marker}")


if __name__ == '__main__':
    main()

