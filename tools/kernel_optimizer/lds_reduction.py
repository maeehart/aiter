#!/usr/bin/env python3
"""
LDS Reduction Experiment

The kernel uses 64KB LDS but maybe we can reduce this in the metadata
and see if it still works. If the kernel actually uses less than 64KB,
reducing the allocation could allow more workgroups per CU.
"""

import os
import shutil
import subprocess
import sys
import struct
import re

KERNEL_DIR = "/workspace/dev/aiter_long_context2/hsa/gfx942/fmoe/silu"
KERNEL_NAME = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co"
ORIGINAL_PATH = f"{KERNEL_DIR}/{KERNEL_NAME}"
BACKUP_PATH = f"{ORIGINAL_PATH}.backup"

def ensure_backup():
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(ORIGINAL_PATH, BACKUP_PATH)

def analyze_lds_actual_usage():
    """Find the actual maximum LDS offset used."""
    result = subprocess.run(['llvm-objdump', '-d', ORIGINAL_PATH], 
                           capture_output=True, text=True)
    disasm = result.stdout
    
    # Find all ds_* offsets
    offsets = []
    for match in re.finditer(r'offset:(\d+)', disasm):
        offsets.append(int(match.group(1)))
    
    if offsets:
        max_offset = max(offsets)
        print(f"Maximum LDS offset found: {max_offset} bytes ({max_offset/1024:.1f} KB)")
        return max_offset
    return None

def find_lds_size_location():
    """Find where group_segment_fixed_size is stored in the binary."""
    with open(ORIGINAL_PATH, 'rb') as f:
        data = f.read()
    
    # The value 65536 = 0x10000 should appear in the kernel descriptor
    # In little-endian: 00 00 01 00
    pattern = b'\x00\x00\x01\x00'
    
    locations = []
    start = 0
    while True:
        pos = data.find(pattern, start)
        if pos == -1:
            break
        locations.append(pos)
        start = pos + 1
    
    print(f"\nFound {len(locations)} occurrences of 0x10000 (65536)")
    for loc in locations[:5]:
        context = data[max(0, loc-8):loc+8]
        print(f"  Offset 0x{loc:x}: ...{context.hex()}...")
    
    return locations

def modify_lds_size(target_size):
    """Attempt to modify the LDS size in the kernel descriptor."""
    ensure_backup()
    
    with open(ORIGINAL_PATH, 'rb') as f:
        data = bytearray(f.read())
    
    # The group_segment_fixed_size is at offset 0x00 in kernel descriptor
    # Kernel descriptor starts in .rodata section
    # From our analysis: .rodata is at 0x1d00, KD is at start
    # group_segment_fixed_size (4 bytes) is at offset 0 of KD
    
    # Find 0x00010000 (65536 in LE) in likely locations
    kd_offset = 0x1d00  # From earlier analysis
    
    # Read current value at KD start
    current = struct.unpack('<I', data[kd_offset:kd_offset+4])[0]
    print(f"\nValue at KD offset 0x{kd_offset:x}: 0x{current:x} ({current})")
    
    if current == 65536:
        print(f"Found group_segment_fixed_size at 0x{kd_offset:x}")
        
        # Modify it
        new_size = target_size
        data[kd_offset:kd_offset+4] = struct.pack('<I', new_size)
        
        output_path = f"{KERNEL_DIR}/{KERNEL_NAME[:-3]}_lds{target_size//1024}k.co"
        with open(output_path, 'wb') as f:
            f.write(data)
        
        print(f"Created: {output_path} with LDS={target_size} bytes")
        return output_path
    else:
        print(f"Unexpected value at KD offset - searching...")
        # Search for 65536 value
        for i in range(len(data) - 4):
            val = struct.unpack('<I', data[i:i+4])[0]
            if val == 65536:
                print(f"  Found 65536 at offset 0x{i:x}")
    
    return None

def test_kernel(kernel_path, batch_size=1024, num_iters=5):
    """Test if modified kernel works."""
    script = f'''
import os
os.environ['VLLM_ROCM_USE_AITER'] = '1'
import torch
import shutil

shutil.copy("{kernel_path}", "{ORIGINAL_PATH}")

from aiter.fused_moe import fused_moe, QuantType, ActivationType

torch.manual_seed(42)
torch.cuda.manual_seed(42)

hidden = torch.randn({batch_size}, 7168, device="cuda", dtype=torch.bfloat16)
w1 = torch.randn(256, 512, 7168, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
w2 = torch.randn(256, 7168, 256, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
w1_scale = torch.ones((256, 4, 56), device="cuda", dtype=torch.float32) * 0.1
w2_scale = torch.ones((256, 56, 2), device="cuda", dtype=torch.float32) * 0.1
weights = torch.rand({batch_size}, 8, device="cuda", dtype=torch.float32)
weights = weights / weights.sum(dim=-1, keepdim=True)
ids = torch.randint(0, 256, ({batch_size}, 8), device="cuda", dtype=torch.int32)

try:
    out = fused_moe(hidden.clone(), w1, w2, weights, ids,
                   activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
                   w1_scale=w1_scale, w2_scale=w2_scale)
    torch.cuda.synchronize()
    print(f"SUCCESS: sum={{out.sum().item():.2f}}")
except Exception as e:
    print(f"FAILED: {{str(e)[:200]}}")

shutil.copy("{BACKUP_PATH}", "{ORIGINAL_PATH}")
'''
    
    result = subprocess.run([sys.executable, '-c', script],
                           capture_output=True, text=True, timeout=60)
    print(f"\nTest output: {result.stdout.strip()}")
    if result.stderr and 'Error' in result.stderr:
        print(f"Errors: {result.stderr[-300:]}")
    
    return "SUCCESS" in result.stdout

if __name__ == "__main__":
    ensure_backup()
    
    print("=" * 70)
    print("LDS Reduction Experiment")
    print("=" * 70)
    
    # Analyze actual LDS usage
    print("\n=== Analyzing actual LDS usage ===")
    max_lds = analyze_lds_actual_usage()
    
    # Find LDS size location
    print("\n=== Finding LDS size in binary ===")
    find_lds_size_location()
    
    # Try reducing LDS
    for target_kb in [56, 48, 32]:
        print(f"\n{'=' * 70}")
        print(f"Testing {target_kb}KB LDS")
        print('=' * 70)
        
        kernel_path = modify_lds_size(target_kb * 1024)
        if kernel_path:
            success = test_kernel(kernel_path)
            if success:
                print(f"✓ {target_kb}KB LDS WORKS!")
            else:
                print(f"✗ {target_kb}KB LDS FAILED")
    
    # Restore
    shutil.copy(BACKUP_PATH, ORIGINAL_PATH)
    print("\n\nRestored original kernel")
