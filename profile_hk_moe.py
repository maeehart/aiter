#!/usr/bin/env python
"""Profile HipKittens MoE kernels for bottleneck analysis."""
import torch
from aiter.fused_moe import fused_topk
from aiter.hipkittens_moe import hipkittens_fused_moe_fp8_fused_act
from aiter import dtypes
from einops import rearrange
import math

def quantize_weights_blockscale(w, scale_blk_n=128, scale_blk_k=128):
    """Quantize weights to FP8 with blockscale."""
    from aiter import pertoken_quant
    E, N, K = w.shape
    num_blk_n = math.ceil(N / scale_blk_n)
    num_blk_k = math.ceil(K / scale_blk_k)
    N_padded = num_blk_n * scale_blk_n
    K_padded = num_blk_k * scale_blk_k
    if N_padded != N or K_padded != K:
        w_padded = torch.zeros((E, N_padded, K_padded), dtype=w.dtype, device=w.device)
        w_padded[:, :N, :K] = w
        w = w_padded
    tmp = rearrange(w.view(E, num_blk_n, scale_blk_n, num_blk_k, scale_blk_k), 
                    'e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)').contiguous()
    w_q, w_scale = pertoken_quant(tmp, quant_dtype=dtypes.fp8)
    w_q = rearrange(w_q.view(E, num_blk_n, num_blk_k, scale_blk_n, scale_blk_k), 
                    'e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)').contiguous()
    w_q = w_q[:, :N, :K].contiguous()
    w_scale = w_scale.view(E, num_blk_n, num_blk_k)
    return w_q, w_scale


if __name__ == "__main__":
    # DeepSeek R1 config
    cfg = {'model_dim': 7168, 'inter_dim': 256, 'num_experts': 256, 'topk': 8}
    bs = 8192

    print(f"Profiling HipKittens MoE FP8: batch={bs}, model_dim={cfg['model_dim']}, "
          f"inter_dim={cfg['inter_dim']}, experts={cfg['num_experts']}, topk={cfg['topk']}")

    # Create inputs
    hidden = torch.randn((bs, cfg['model_dim']), dtype=torch.bfloat16, device='cuda') / 10
    w1_bf16 = torch.randn((cfg['num_experts'], cfg['inter_dim']*2, cfg['model_dim']), 
                          dtype=torch.bfloat16, device='cuda') / 10
    w2_bf16 = torch.randn((cfg['num_experts'], cfg['model_dim'], cfg['inter_dim']), 
                          dtype=torch.bfloat16, device='cuda') / 10

    # Quantize to FP8
    w1_fp8, w1_scale = quantize_weights_blockscale(w1_bf16)
    w2_fp8, w2_scale = quantize_weights_blockscale(w2_bf16)
    
    scores = torch.randn((bs, cfg['num_experts']), dtype=torch.float32, device='cuda')
    topk_w, topk_ids = fused_topk(hidden, scores, cfg['topk'], True)

    # Warmup
    print("Warmup...")
    for _ in range(3):
        _ = hipkittens_fused_moe_fp8_fused_act(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)
        torch.cuda.synchronize()

    # Profile runs
    print("Running 10 iterations for profiling...")
    for _ in range(10):
        _ = hipkittens_fused_moe_fp8_fused_act(hidden, w1_fp8, w2_fp8, w1_scale, w2_scale, topk_w, topk_ids)
    torch.cuda.synchronize()
    print("Done")

