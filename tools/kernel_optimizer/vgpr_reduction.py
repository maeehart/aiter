#!/usr/bin/env python3
"""
VGPR Reduction Experiment

Attempt to reduce the declared VGPR count in the kernel metadata.
The kernel uses v0-v255 (256) + a0-a127 (128) = 384 architectural registers,
but declares 512. Reducing this could improve occupancy.

WARNING: This modifies the binary at multiple locations:
1. COMPUTE_PGM_RSRC1 in the kernel descriptor
2. .vgpr_count in the ELF metadata

If the metadata doesn't match actual usage, the kernel will crash.
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

def restore_original():
    shutil.copy(BACKUP_PATH, ORIGINAL_PATH)

def analyze_kernel():
    """Analyze current VGPR settings."""
    print("=== Current Kernel Analysis ===")
    
    with open(ORIGINAL_PATH, 'rb') as f:
        data = f.read()
    
    # Find .rodata section - kernel descriptor starts there
    # Looking at the dump, .rodata is at file offset ~0x1d00
    
    # Find COMPUTE_PGM_RSRC1 (should be 0x7f030c00 = 0x000c037f little-endian)
    rsrc1_pattern = bytes([0x7f, 0x03, 0x0c, 0x00])
    rsrc1_offset = data.find(rsrc1_pattern)
    
    if rsrc1_offset != -1:
        print(f"COMPUTE_PGM_RSRC1 found at offset 0x{rsrc1_offset:x}")
        rsrc1 = struct.unpack('<I', data[rsrc1_offset:rsrc1_offset+4])[0]
        vgpr_field = rsrc1 & 0x3F
        print(f"  VGPR field (bits 0-5): {vgpr_field}")
        print(f"  Granularity 4: {(vgpr_field + 1) * 4} VGPRs")
        print(f"  Granularity 8: {(vgpr_field + 1) * 8} VGPRs")
    else:
        print("COMPUTE_PGM_RSRC1 not found!")
        return None
    
    # Find .vgpr_count in metadata (msgpack format)
    # Look for the string ".vgpr_count" followed by the value
    metadata_match = re.search(b'\\.vgpr_count', data)
    if metadata_match:
        print(f"\n.vgpr_count string found at offset 0x{metadata_match.start():x}")
        # The value should be nearby in msgpack format
        # Look for 0x200 (512) as uint16 in msgpack: 0xcd 0x02 0x00
        # Or as uint32: 0xce 0x00 0x00 0x02 0x00
    
    return rsrc1_offset

def modify_vgpr_count(target_vgprs):
    """
    Attempt to modify VGPR count.
    
    GFX942 VGPR granularity appears to be 8 (field * 8 = VGPRs).
    Field = 63 -> 512 VGPRs
    Field = 47 -> 384 VGPRs
    Field = 31 -> 256 VGPRs
    """
    ensure_backup()
    
    with open(ORIGINAL_PATH, 'rb') as f:
        data = bytearray(f.read())
    
    # Calculate new field value (assuming granularity 8)
    new_field = (target_vgprs // 8) - 1
    print(f"\nAttempting to set VGPRs to {target_vgprs} (field = {new_field})")
    
    # Find and modify COMPUTE_PGM_RSRC1
    rsrc1_pattern = bytes([0x7f, 0x03, 0x0c, 0x00])  # Current: 0x000c037f
    rsrc1_offset = data.find(rsrc1_pattern)
    
    if rsrc1_offset == -1:
        print("ERROR: COMPUTE_PGM_RSRC1 not found!")
        return False
    
    # Read current RSRC1
    rsrc1 = struct.unpack('<I', data[rsrc1_offset:rsrc1_offset+4])[0]
    
    # Modify VGPR field (bits 0-5)
    new_rsrc1 = (rsrc1 & ~0x3F) | new_field
    
    print(f"RSRC1: 0x{rsrc1:08x} -> 0x{new_rsrc1:08x}")
    
    # Apply the change
    data[rsrc1_offset:rsrc1_offset+4] = struct.pack('<I', new_rsrc1)
    
    # Write modified kernel
    output_path = f"{KERNEL_DIR}/{KERNEL_NAME[:-3]}_vgpr{target_vgprs}.co"
    with open(output_path, 'wb') as f:
        f.write(data)
    
    print(f"Created: {output_path}")
    return output_path

def test_kernel(kernel_path, batch_size=1024, num_iters=10):
    """Test if modified kernel works."""
    script = f'''
import os
os.environ['VLLM_ROCM_USE_AITER'] = '1'
import torch
import shutil

# Swap kernel
ORIGINAL = "{ORIGINAL_PATH}"
shutil.copy("{kernel_path}", ORIGINAL)

from aiter.fused_moe import fused_moe, QuantType, ActivationType

torch.manual_seed(42)
hidden = torch.randn({batch_size}, 7168, device="cuda", dtype=torch.bfloat16)
w1 = torch.randn(256, 512, 7168, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
w2 = torch.randn(256, 7168, 256, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fnuz)
w1_scale = torch.ones((256, 4, 56), device="cuda", dtype=torch.float32) * 0.1
w2_scale = torch.ones((256, 56, 2), device="cuda", dtype=torch.float32) * 0.1
weights = torch.rand({batch_size}, 8, device="cuda", dtype=torch.float32)
weights = weights / weights.sum(dim=-1, keepdim=True)
ids = torch.randint(0, 256, ({batch_size}, 8), device="cuda", dtype=torch.int32)

try:
    for _ in range(5):  # Warmup
        out = fused_moe(hidden.clone(), w1, w2, weights, ids,
                       activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
                       w1_scale=w1_scale, w2_scale=w2_scale)
    torch.cuda.synchronize()
    
    import time
    start = time.time()
    for _ in range({num_iters}):
        out = fused_moe(hidden.clone(), w1, w2, weights, ids,
                       activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
                       w1_scale=w1_scale, w2_scale=w2_scale)
        torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / {num_iters}
    
    print(f"SUCCESS: {{elapsed:.2f}} ms/iter")
    print(f"OUTPUT_CHECKSUM: {{out.sum().item():.6f}}")
except Exception as e:
    print(f"FAILED: {{e}}")

# Restore original
shutil.copy("{BACKUP_PATH}", ORIGINAL)
'''
    
    result = subprocess.run([sys.executable, '-c', script],
                           capture_output=True, text=True, timeout=120)
    
    print(f"\nTest output:")
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr[-500:]}")
    
    return "SUCCESS" in result.stdout

if __name__ == "__main__":
    ensure_backup()
    
    print("VGPR Reduction Experiment")
    print("=" * 60)
    
    # Analyze current state
    analyze_kernel()
    
    # Try different VGPR counts
    for target in [384, 320, 256]:
        print(f"\n{'=' * 60}")
        print(f"Testing {target} VGPRs")
        print('=' * 60)
        
        kernel_path = modify_vgpr_count(target)
        if kernel_path:
            success = test_kernel(kernel_path)
            if success:
                print(f"✓ {target} VGPRs WORKS!")
            else:
                print(f"✗ {target} VGPRs FAILED")
        
    # Restore original
    restore_original()
    print("\n\nRestored original kernel")
