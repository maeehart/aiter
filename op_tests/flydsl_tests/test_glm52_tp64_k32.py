# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for GLM-5.2 TP64 native A4W4 MoE kernels."""

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)

_MODEL_DIM = 6144
_INTER_DIM = 32
_BLOCK_M = 32


def _make_problem(token, model_dim, experts, topk, *, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    hidden = (
        torch.randn(
            (token, model_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    w1 = (
        torch.randn(
            (experts, _INTER_DIM * 2, model_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    w2 = (
        torch.randn(
            (experts, model_dim, _INTER_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    routes = torch.arange(
        token * topk,
        dtype=torch.int32,
        device="cuda",
    ).view(token, topk)
    topk_ids = routes % experts
    topk_weights = torch.rand(
        (token, topk),
        dtype=torch.float32,
        device="cuda",
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    return hidden, w1, w2, topk_weights, topk_ids


def _quantize_problem(hidden, w1, w2):
    experts, double_inter, model_dim = w1.shape
    a1_q, a1_scale = per_1x32_f4_quant(hidden, quant_dtype=dtypes.fp4x2)
    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(experts, double_inter, model_dim // 2)
    w2_q = w2_q.view(experts, model_dim, _INTER_DIM // 2)
    return a1_q, a1_scale, w1_q, w1_scale, w2_q, w2_scale


def _reference(
    a1_q,
    a1_scale,
    w1_q,
    w1_scale,
    w2_q,
    w2_scale,
    topk_weights,
    topk_ids,
):
    stage1 = torch_moe_stage1(
        a1_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=False,
    )
    a2_q, a2_scale = per_1x32_f4_quant(stage1, quant_dtype=dtypes.fp4x2)
    output = torch_moe_stage2(
        a2_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )
    return stage1, a2_q, a2_scale, output


def _sort_routes(topk_ids, topk_weights, experts, model_dim):
    return moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        model_dim,
        torch.bfloat16,
        _BLOCK_M,
    )[:4]


def _sort_payload(
    payload,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    *,
    token,
    topk,
):
    sorted_rows = sorted_expert_ids.numel() * _BLOCK_M
    sorted_u8 = torch.zeros(
        (sorted_rows, _INTER_DIM // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    packed = sorted_ids.to(torch.int64)
    row = torch.arange(packed.numel(), dtype=torch.int64, device="cuda")
    sorted_token = packed & 0x00FFFFFF
    sorted_slot = packed >> 24
    valid_rows = int(num_valid_ids.flatten()[0].item())
    valid = (row < valid_rows) & (sorted_token < token) & (sorted_slot < topk)
    source_row = sorted_token * topk + sorted_slot
    sorted_u8[row[valid]] = payload.view(torch.uint8).reshape(
        token * topk, _INTER_DIM // 2
    )[source_row[valid]]
    return sorted_u8.view(payload.dtype)


def _run_compact_stage2(
    *,
    payload,
    payload_scale,
    w2_q,
    w2_scale,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    token,
    model_dim,
    experts,
    topk,
    block_n,
):
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2

    output = torch.zeros(
        (token, model_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2_scale_shuffled = e8m0_shuffle(w2_scale.view(experts * model_dim, -1))
    mxfp4_moe_gemm2(
        inter_sorted_quant=payload,
        inter_sorted_shuffled_scale=payload_scale,
        w2_u8=w2_q.view(torch.uint8).contiguous(),
        w2_scale_u8=w2_scale_shuffled.view(torch.uint8).contiguous(),
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_weights,
        out=output,
        M_logical=token,
        max_sorted=payload.shape[0],
        NE=experts,
        D_HIDDEN=model_dim,
        D_INTER=_INTER_DIM,
        topk=topk,
        BM=_BLOCK_M,
        BN=block_n,
        BK=128,
        use_nt=False,
        a_dtype="fp4",
        b_dtype="fp4",
        epilog="atomic",
        SBM=_BLOCK_M,
        persist=False,
        g2_bf16_lds=False,
        g2_spart=0,
    )
    torch.cuda.synchronize()
    return output


def _assert_close(reference, output, *, minimum_fraction):
    nan_mask = output.isnan()
    close = torch.isclose(
        reference.float(),
        output.float(),
        atol=1.0,
        rtol=0.05,
    )
    assert close.float().mean().item() >= minimum_fraction
    assert not nan_mask.any()
    assert not output.isinf().any()


@pytest.mark.parametrize(
    ("token", "model_dim", "experts", "topk", "block_n"),
    [
        pytest.param(32, 128, 1, 1, 128, id="single-expert"),
        pytest.param(16, _MODEL_DIM, 256, 8, 256, id="glm52-t16"),
        pytest.param(64, _MODEL_DIM, 256, 8, 256, id="glm52-t64"),
        pytest.param(256, _MODEL_DIM, 256, 8, 256, id="glm52-t256"),
        pytest.param(64, _MODEL_DIM, 257, 9, 256, id="glm52-fused-shared"),
    ],
)
def test_glm52_tp64_compact_k32_stage2(token, model_dim, experts, topk, block_n):
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    a2 = (
        torch.randn(
            (token, topk, _INTER_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    w1 = torch.zeros(
        (experts, _INTER_DIM * 2, model_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2 = (
        torch.randn(
            (experts, model_dim, _INTER_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    routes = torch.arange(
        token * topk,
        dtype=torch.int32,
        device="cuda",
    ).view(token, topk)
    topk_ids = routes % experts
    topk_weights = torch.rand(
        (token, topk),
        dtype=torch.float32,
        device="cuda",
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = _sort_routes(
        topk_ids, topk_weights, experts, model_dim
    )

    a2_q, a2_scale = per_1x32_f4_quant(a2, quant_dtype=dtypes.fp4x2)
    w1_q, _ = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(experts, _INTER_DIM * 2, model_dim // 2)
    w2_q = w2_q.view(experts, model_dim, _INTER_DIM // 2)
    reference = torch_moe_stage2(
        a2_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )

    payload = _sort_payload(
        a2_q,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        token=token,
        topk=topk,
    )
    payload_scale = moe_mxfp4_sort(
        a2_scale.view(token, topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=_BLOCK_M,
    )
    output = _run_compact_stage2(
        payload=payload,
        payload_scale=payload_scale,
        w2_q=w2_q,
        w2_scale=w2_scale,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        token=token,
        model_dim=model_dim,
        experts=experts,
        topk=topk,
        block_n=block_n,
    )
    _assert_close(reference, output, minimum_fraction=0.95)


@pytest.mark.parametrize("token", [16, 64, 256])
def test_glm52_tp64_compact_k32_e2e(token):
    experts = 256
    topk = 8
    hidden, w1, w2, topk_weights, topk_ids = _make_problem(
        token, _MODEL_DIM, experts, topk, seed=321
    )
    a1_q, a1_scale, w1_q, w1_scale, w2_q, w2_scale = _quantize_problem(hidden, w1, w2)
    reference_stage1, _, _, reference = _reference(
        a1_q,
        a1_scale,
        w1_q,
        w1_scale,
        w2_q,
        w2_scale,
        topk_weights,
        topk_ids,
    )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = _sort_routes(
        topk_ids, topk_weights, experts, _MODEL_DIM
    )
    a1_scale_sorted = moe_mxfp4_sort(
        a1_scale.view(token, 1, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=_BLOCK_M,
    )
    w1_shuffled = shuffle_weight(w1_q, (16, 16))
    w1_scale_shuffled = e8m0_shuffle(w1_scale.view(experts * _INTER_DIM * 2, -1))

    stage1_bf16 = flydsl_moe_stage1(
        a=a1_q,
        w1=w1_shuffled,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk,
        tile_m=_BLOCK_M,
        tile_n=32,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype="bf16",
        w1_scale=w1_scale_shuffled,
        a1_scale=a1_scale_sorted,
        sorted_weights=None,
        persist_m=1,
        waves_per_eu=2,
        b_nt=0,
        gate_mode="separated",
    )
    torch.testing.assert_close(
        stage1_bf16,
        reference_stage1,
        atol=0.01,
        rtol=0.01,
    )

    payload, payload_scale = flydsl_moe_stage1(
        a=a1_q,
        w1=w1_shuffled,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk,
        tile_m=_BLOCK_M,
        tile_n=32,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype="fp4",
        w1_scale=w1_scale_shuffled,
        a1_scale=a1_scale_sorted,
        sorted_weights=None,
        persist_m=1,
        waves_per_eu=2,
        b_nt=0,
        gate_mode="separated",
        v2_output_layout=True,
    )
    output = _run_compact_stage2(
        payload=payload,
        payload_scale=payload_scale,
        w2_q=w2_q,
        w2_scale=w2_scale,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        token=token,
        model_dim=_MODEL_DIM,
        experts=experts,
        topk=topk,
        block_n=256,
    )
    _assert_close(reference, output, minimum_fraction=0.90)


@pytest.mark.parametrize(("experts", "topk"), [(256, 8), (257, 9)])
@pytest.mark.parametrize("token", [1, 16, 64, 256])
def test_glm52_tp64_fused_moe_uses_compact_k32(token, experts, topk):
    hidden, w1, w2, topk_weights, topk_ids = _make_problem(
        token, _MODEL_DIM, experts, topk, seed=777
    )
    a1_q, a1_scale, w1_q, w1_scale, w2_q, w2_scale = _quantize_problem(hidden, w1, w2)
    _, _, _, reference = _reference(
        a1_q,
        a1_scale,
        w1_q,
        w1_scale,
        w2_q,
        w2_scale,
        topk_weights,
        topk_ids,
    )

    w1_shuffled = shuffle_weight(w1_q, (16, 16))
    w1_shuffled.is_shuffled = True
    w2_compact = w2_q.contiguous()
    w2_compact.compact_k32 = True
    w1_scale_shuffled = e8m0_shuffle(w1_scale.view(experts * _INTER_DIM * 2, -1))
    w2_scale_shuffled = e8m0_shuffle(w2_scale.view(experts * _MODEL_DIM, -1))
    output = fused_moe(
        hidden,
        w1_shuffled,
        w2_compact,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale_shuffled,
        w2_scale=w2_scale_shuffled,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
        doweight_stage1=False,
    )
    torch.cuda.synchronize()
    _assert_close(reference, output, minimum_fraction=0.90)
