# 64x256 Kernel Pattern Analysis

## Key Discovery: 2-Pass Internal Structure

The 64x256 kernel processes **32 tokens at a time internally**, using a 2-pass approach:

### Register Usage
- `s101`: Pass flag (0 = first pass, 1 = second pass)
- `s3`: Block ID (incremented for second pass)
- `s100`: Outer loop counter
- `s98`, `s99`: Persistent scheduling state

### 2-Pass Control Flow

```assembly
; At kernel start (0x2ED8)
s_mov_b32 s101, 0                    ; Initialize pass flag = 0

; Main loop processes 32 tokens
; ...kernel work...

; At end of iteration (0xA458-0xA468)
s_cmp_eq_u32 s101, 1                 ; Check if second pass done
s_cbranch_scc1 label_1D9B            ; If yes, go to outer loop
s_add_u32 s3, s3, 1                  ; Increment block ID
s_mov_b32 s101, 1                    ; Mark second pass
s_branch label_005C                  ; Branch to inner loop (skip arg loading)

; Outer loop (label_1D9B at 0xA46C)
s_add_u32 s100, s100, 1              ; Increment outer counter
s_cmp_eq_u32 s96, 0                  ; Check if done
s_cbranch_scc0 kernel_start          ; Loop back
s_endpgm                             ; Exit
```

### The Sorting Mismatch Problem

The kernel uses `s_mul_i32 s60, s3, 32` for block start calculation, meaning:
- It expects `sorted_expert_ids[s3]` to contain 32-token block entries
- But 64-block sorting produces entries that cover 64 tokens each

**This is why the kernel crashes with 64-block sorting but works with 32-block sorting!**

## Correct Usage Pattern

The 64x256 kernel should be used with **32-block sorting**, not 64-block sorting:

```python
# CORRECT: Use 32-block sorting
sorted_ids, sorted_weights, sorted_expert_ids, num_valid, out = (
    moe_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_size=32)
)

# Call 64x256 kernel
aiter.fmoe_fp8_blockscale_g1u1(
    out, input, w1, w2,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid,
    topk, input_scale, w1_scale, w2_scale, 
    "_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_64x256E",
    ...
)
```

## Why 64x256 Could Be Faster

With 32-block sorting, the 64x256 kernel:
1. Amortizes kernel launch overhead across 2x tokens per outer loop
2. Keeps more state in registers between passes
3. Reduces the number of times expert weights are reloaded

## What We Need to Fix

To properly use the 64x256 kernel:
1. Always use `block_size=32` in `moe_sorting()`
2. The kernel will internally process pairs of 32-token blocks
3. The 2-pass flag `s101` handles the internal iteration

## Building a New Kernel

To create a truly flexible kernel, we would need to:
1. Parameterize the block size at compile time
2. Or create a runtime-configurable version
3. Ensure sorting and kernel block sizes match

