#!/usr/bin/env python3
"""
Advanced binary optimizer with more sophisticated strategies for MoE kernels.

Optimization targets:
1. vmcnt (vector memory count) - affects global memory pipelining
2. lgkmcnt (LDS/GDS/scalar memory count) - affects LDS pipelining
3. s_nop removal - remove unnecessary stalls
4. Combined strategies
"""

import os
import shutil
import struct
import subprocess
import numpy as np
from typing import Optional, Tuple, Dict, List

AITER_PATH = "/workspace/dev/aiter_long_context2"
KERNEL_DIR = f"{AITER_PATH}/hsa/gfx942/fmoe/silu"
KERNEL_NAME = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"
ORIGINAL_KERNEL = f"{KERNEL_DIR}/{KERNEL_NAME}"
BACKUP_KERNEL = f"{KERNEL_DIR}/{KERNEL_NAME}.backup"


def decode_waitcnt(imm16: int) -> Tuple[int, int, int]:
    """Decode s_waitcnt to (vmcnt, lgkmcnt, expcnt)."""
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
    return vmcnt_lo | (expcnt << 4) | (lgkmcnt << 8) | (vmcnt_hi << 14)


def find_instructions(kernel_data: bytes) -> Dict:
    """Find all patchable instructions."""
    results = {
        'waitcnt': [],
        's_nop': [],
    }
    
    i = 0
    while i < len(kernel_data) - 4:
        # s_waitcnt: opcode 0xBF8C
        if kernel_data[i+2:i+4] == b'\x8c\xbf':
            imm16 = struct.unpack('<H', kernel_data[i:i+2])[0]
            vmcnt, lgkmcnt, expcnt = decode_waitcnt(imm16)
            results['waitcnt'].append({
                'offset': i,
                'vmcnt': vmcnt,
                'lgkmcnt': lgkmcnt,
                'expcnt': expcnt,
            })
            i += 4
        # s_nop: opcode 0xBF80 with 00 00 immediate
        elif kernel_data[i:i+4] == b'\x00\x00\x80\xbf':
            results['s_nop'].append({'offset': i})
            i += 4
        else:
            i += 1
    
    return results


def apply_waitcnt_patch(data: bytes, offset: int, vmcnt: int, lgkmcnt: int, expcnt: int) -> bytes:
    """Apply a single waitcnt patch."""
    imm16 = encode_waitcnt(vmcnt, lgkmcnt, expcnt)
    patched = bytearray(data)
    patched[offset:offset+2] = struct.pack('<H', imm16)
    return bytes(patched)


def replace_with_nop(data: bytes, offset: int) -> bytes:
    """Replace an instruction with s_nop 0."""
    patched = bytearray(data)
    patched[offset:offset+4] = b'\x00\x00\x80\xbf'  # s_nop 0
    return bytes(patched)


def run_benchmark(batch_size: int = 16000, num_iters: int = 20) -> Optional[float]:
    """Run benchmark and return average time in microseconds."""
    try:
        result = subprocess.run(
            ["python", "-c", f"""
import os
os.environ['VLLM_ROCM_USE_AITER'] = '1'
import torch
from aiter.fused_moe import fused_moe, QuantType, ActivationType

FP8_DTYPE = torch.float8_e4m3fnuz
batch_size = {batch_size}
num_experts = 256
hidden_size = 7168
intermediate_size = 256
topk = 8
device = "cuda"

shard_intermediate_size = intermediate_size * 2
hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, device=device, dtype=torch.bfloat16).to(FP8_DTYPE)
w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16).to(FP8_DTYPE)
scale_shape_w1 = (num_experts, shard_intermediate_size // 128, hidden_size // 128)
scale_shape_w2 = (num_experts, hidden_size // 128, intermediate_size // 128)
w1_scale = torch.ones(scale_shape_w1, device=device, dtype=torch.float32) * 0.1
w2_scale = torch.ones(scale_shape_w2, device=device, dtype=torch.float32) * 0.1
topk_weights = torch.rand(batch_size, topk, device=device, dtype=torch.float32)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
topk_ids = torch.randint(0, num_experts, (batch_size, topk), device=device, dtype=torch.int32)

# Warmup
for _ in range(5):
    _ = fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
        w1_scale=w1_scale, w2_scale=w2_scale)
    torch.cuda.synchronize()

# Benchmark
start_events = [torch.cuda.Event(enable_timing=True) for _ in range({num_iters})]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range({num_iters})]
for i in range({num_iters}):
    start_events[i].record()
    _ = fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
        w1_scale=w1_scale, w2_scale=w2_scale)
    end_events[i].record()

torch.cuda.synchronize()
times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
print(f"RESULT:{{sum(times)/len(times):.2f}}")
"""],
            capture_output=True, text=True, timeout=120,
        )
        
        if result.returncode != 0:
            return None
        
        for line in result.stdout.split('\n'):
            if line.startswith('RESULT:'):
                return float(line.split(':')[1])
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def save_kernel(data: bytes, path: str):
    """Save kernel binary."""
    with open(path, 'wb') as f:
        f.write(data)
    os.chmod(path, 0o755)


def load_kernel(path: str) -> bytes:
    """Load kernel binary."""
    with open(path, 'rb') as f:
        return f.read()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Advanced MoE kernel optimizer")
    parser.add_argument("--batch-size", type=int, default=16000)
    parser.add_argument("--num-iters", type=int, default=20)
    args = parser.parse_args()
    
    # Backup and load kernel
    if not os.path.exists(BACKUP_KERNEL):
        shutil.copy(ORIGINAL_KERNEL, BACKUP_KERNEL)
    kernel_data = load_kernel(BACKUP_KERNEL)
    
    # Find patchable instructions
    instructions = find_instructions(kernel_data)
    print(f"Found {len(instructions['waitcnt'])} s_waitcnt instructions")
    print(f"Found {len(instructions['s_nop'])} s_nop instructions")
    
    # Analyze waitcnt distribution
    lgkmcnt_dist = {}
    vmcnt_dist = {}
    for w in instructions['waitcnt']:
        lgkmcnt_dist[w['lgkmcnt']] = lgkmcnt_dist.get(w['lgkmcnt'], 0) + 1
        vmcnt_dist[w['vmcnt']] = vmcnt_dist.get(w['vmcnt'], 0) + 1
    
    print("\nlgkmcnt distribution:", dict(sorted(lgkmcnt_dist.items())))
    print("vmcnt distribution:", dict(sorted(vmcnt_dist.items())))
    
    # Get baseline
    print(f"\n=== Running baseline (batch={args.batch_size}) ===")
    save_kernel(kernel_data, ORIGINAL_KERNEL)
    baseline = run_benchmark(args.batch_size, args.num_iters)
    if baseline is None:
        print("Failed to get baseline!")
        return
    print(f"Baseline: {baseline:.2f} μs")
    
    # Define strategies
    strategies = []
    
    # Strategy 1: vmcnt tuning (already tested, include best)
    strategies.append(("vmcnt_reduce25", 
        lambda d, w: apply_waitcnt_patch(d, w['offset'], 
            max(0, w['vmcnt'] - w['vmcnt']//4), w['lgkmcnt'], w['expcnt'])
        if w['vmcnt'] > 0 else d))
    
    # Strategy 2: lgkmcnt tuning
    strategies.append(("lgkmcnt_reduce25",
        lambda d, w: apply_waitcnt_patch(d, w['offset'],
            w['vmcnt'], max(0, w['lgkmcnt'] - w['lgkmcnt']//4), w['expcnt'])
        if w['lgkmcnt'] > 0 else d))
    
    strategies.append(("lgkmcnt_to_0",
        lambda d, w: apply_waitcnt_patch(d, w['offset'],
            w['vmcnt'], 0, w['expcnt'])
        if w['lgkmcnt'] > 0 else d))
    
    # Strategy 3: Combined vmcnt and lgkmcnt
    strategies.append(("both_reduce25",
        lambda d, w: apply_waitcnt_patch(d, w['offset'],
            max(0, w['vmcnt'] - w['vmcnt']//4),
            max(0, w['lgkmcnt'] - w['lgkmcnt']//4),
            w['expcnt'])
        if w['vmcnt'] > 0 or w['lgkmcnt'] > 0 else d))
    
    strategies.append(("both_reduce50",
        lambda d, w: apply_waitcnt_patch(d, w['offset'],
            max(0, w['vmcnt'] // 2),
            max(0, w['lgkmcnt'] // 2),
            w['expcnt'])
        if w['vmcnt'] > 0 or w['lgkmcnt'] > 0 else d))
    
    # Strategy 4: Cap both at specific values
    strategies.append(("cap_vmcnt8_lgkmcnt4",
        lambda d, w: apply_waitcnt_patch(d, w['offset'],
            min(w['vmcnt'], 8) if w['vmcnt'] > 0 else w['vmcnt'],
            min(w['lgkmcnt'], 4) if w['lgkmcnt'] > 0 else w['lgkmcnt'],
            w['expcnt'])))
    
    # Test strategies
    results = []
    best_time = baseline
    best_strategy = "baseline"
    best_kernel = kernel_data
    
    for name, transform in strategies:
        print(f"\n--- Testing: {name} ---")
        
        patched = kernel_data
        for w in instructions['waitcnt']:
            patched = transform(patched, w)
        
        save_kernel(patched, ORIGINAL_KERNEL)
        time_result = run_benchmark(args.batch_size, args.num_iters)
        
        if time_result is None:
            print("  CRASHED")
            save_kernel(kernel_data, ORIGINAL_KERNEL)
            continue
        
        speedup = baseline / time_result
        results.append((name, time_result, speedup))
        print(f"  Time: {time_result:.2f} μs (speedup: {speedup:.3f}x)")
        
        if time_result < best_time:
            best_time = time_result
            best_strategy = name
            best_kernel = patched
            print("  NEW BEST!")
    
    # Restore original
    save_kernel(kernel_data, ORIGINAL_KERNEL)
    
    # Summary
    print("\n" + "=" * 60)
    print("ADVANCED OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Baseline: {baseline:.2f} μs")
    print(f"Best: {best_time:.2f} μs ({baseline/best_time:.3f}x) - {best_strategy}")
    print("\nAll results:")
    for name, time_val, speedup in sorted(results, key=lambda x: x[1]):
        marker = " ★" if name == best_strategy else ""
        print(f"  {name:25s}: {time_val:8.2f} μs ({speedup:.3f}x){marker}")


if __name__ == '__main__':
    main()
