"""Behavioral coverage for SiTUv2 tuner forwarding and config dispatch."""

import importlib
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import aiter.fused_moe as fused_moe
from aiter import ActivationType, QuantType, dtypes


_TUNER = importlib.import_module("csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune")
_MODEL_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "aiter"
    / "configs"
    / "model_configs"
    / "kimik3_fp4_tuned_fmoe.csv"
)


class _TensorStub:
    def __init__(self, shape):
        self.shape = shape


class TestFmoeSiTUv2Forwarding(unittest.TestCase):
    def test_kernel_and_torch_reference_receive_requested_pair(self):
        requested = (4.0, 25.0)
        kernel_calls = []
        reference_calls = []

        def fake_kernel(**kwargs):
            kernel_calls.append(kwargs)
            return "kernel-result"

        def fake_reference(*args, **kwargs):
            reference_calls.append(kwargs)
            return "reference-result"

        a1_qt = _TensorStub((1, 2))
        w1_qt = _TensorStub((1, 4, 2))
        with (
            mock.patch.object(_TUNER, "flydsl_moe_stage1", side_effect=fake_kernel),
            mock.patch.object(_TUNER, "torch_moe_stage1", side_effect=fake_reference),
        ):
            kernel_result = _TUNER.FmoeTuner.run_flydsl_stage1_out(
                a1_qt,
                w1_qt,
                "sorted_ids",
                "sorted_expert_ids",
                None,
                "num_valid_ids",
                "w1_scale",
                "a1_scale",
                None,
                dtypes.bf16,
                1,
                {
                    "out_dtype": "bf16",
                    "tile_m": 32,
                    "tile_n": 128,
                    "tile_k": 128,
                    "a_dtype": "bf16",
                    "b_dtype": "int4",
                },
                32,
                dtypes.bf16,
                QuantType.per_1x32,
                ActivationType.Situv2,
                *requested,
            )
            reference_result = _TUNER.FmoeTuner.run_torch_moe_stage1(
                a1_qt,
                w1_qt,
                "w2",
                "topk_weights",
                "topk_ids",
                None,
                None,
                activation=ActivationType.Situv2,
                quant_type=QuantType.No,
                situ_beta=requested[0],
                situ_linear_beta=requested[1],
            )

        self.assertEqual(kernel_result, "kernel-result")
        self.assertEqual(reference_result, "reference-result")
        self.assertEqual(
            (kernel_calls[0]["situ_beta"], kernel_calls[0]["situ_linear_beta"]),
            requested,
        )
        self.assertEqual(
            (reference_calls[0]["situ_beta"], reference_calls[0]["situ_linear_beta"]),
            requested,
        )

    def test_generated_stage2_reference_receives_requested_pair(self):
        requested = (4.0, 25.0)
        captured = []
        a1_qt = torch.zeros((1, 2), dtype=torch.bfloat16)
        w1_qt = torch.zeros((1, 4, 2), dtype=torch.bfloat16)
        w2_qt = torch.zeros((1, 2, 2), dtype=torch.bfloat16)
        data = {
            "input": a1_qt,
            "a1_qt": a1_qt,
            "w1_qt": w1_qt,
            "w2_qt": w2_qt,
            "w1_qt_shffle": w1_qt,
            "w2_qt_shffle": w2_qt,
            "sorted_ids": torch.zeros((1,), dtype=torch.int32),
            "sorted_weights": torch.ones((1,), dtype=torch.float32),
            "sorted_expert_ids": torch.zeros((1,), dtype=torch.int32),
            "num_valid_ids": torch.tensor(1, dtype=torch.int32),
            "topk_ids": torch.zeros((1, 1), dtype=torch.int64),
            "topk_weights": torch.ones((1, 1), dtype=torch.float32),
            "moe_buf": torch.empty((1,), dtype=torch.uint8),
            "a1_scale": torch.ones((1, 1), dtype=torch.bfloat16),
            "w1_scale": torch.ones((1, 1), dtype=torch.bfloat16),
            "w2_scale": torch.ones((1, 1), dtype=torch.bfloat16),
        }

        def fake_reference(*args, **kwargs):
            captured.append(kwargs)
            return torch.zeros((1, 2), dtype=torch.bfloat16)

        with (
            mock.patch.object(_TUNER.FmoeTuner, "generate_data", return_value=data),
            mock.patch.object(
                _TUNER.FmoeTuner, "run_torch_moe_stage1", side_effect=fake_reference
            ),
            mock.patch.object(_TUNER.fp4_utils, "e8m0_shuffle", side_effect=lambda x: x),
            mock.patch.object(
                _TUNER.aiter,
                "get_torch_quant",
                return_value=lambda x, **kwargs: (x, torch.ones((1, 1))),
            ),
        ):
            _TUNER.FmoeTuner.generate_data_2stages(
                1,
                2,
                2,
                1,
                1,
                ActivationType.Situv2,
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
                QuantType.No,
                True,
                False,
                32,
                stage=2,
                situ_beta=requested[0],
                situ_linear_beta=requested[1],
                device="cpu",
            )

        self.assertEqual(
            (captured[0]["situ_beta"], captured[0]["situ_linear_beta"]), requested
        )

    def test_committed_gfx942_rows_and_unlisted_control_select_expected_kernels(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = Path(tempdir) / "tuned_fmoe.csv"
            shutil.copyfile(_MODEL_CONFIG, config)
            original_cfg = fused_moe.cfg_2stages
            fused_moe.cfg_2stages = None
            fused_moe.get_2stage_cfgs.cache_clear()
            try:
                with (
                    mock.patch.object(
                        fused_moe,
                        "AITER_CONFIGS",
                        SimpleNamespace(AITER_CONFIG_FMOE_FILE=str(config)),
                    ),
                    mock.patch.object(fused_moe, "get_gfx_runtime", return_value="gfx942"),
                    mock.patch.object(fused_moe, "get_cu_num", return_value=304),
                    mock.patch.object(fused_moe, "is_flydsl_available", return_value=True),
                ):
                    selected = {
                        topk: fused_moe.get_2stage_cfgs(
                            8192,
                            3584,
                            3072,
                            112,
                            topk,
                            torch.bfloat16,
                            torch.bfloat16,
                            torch.int4,
                            QuantType.per_1x32,
                            True,
                            ActivationType.Situv2,
                            False,
                            0,
                            0,
                        )
                        for topk in (15, 16, 14)
                    }
            finally:
                fused_moe.cfg_2stages = original_cfg
                fused_moe.get_2stage_cfgs.cache_clear()

        expected_tuned = (
            "flydsl_moe1_abf16_wint4_bf16_t64x128x128",
            "flydsl_moe2_abf16_wint4_bf16_t64x128x128_atomic",
        )
        for topk in (15, 16):
            self.assertEqual(selected[topk].block_m, 64)
            self.assertEqual(
                (
                    selected[topk].stage1.keywords["kernelName"],
                    selected[topk].stage2.keywords["kernelName"],
                ),
                expected_tuned,
            )

        self.assertEqual(selected[14].block_m, 32)
        self.assertEqual(
            (
                selected[14].stage1.keywords["kernelName"],
                selected[14].stage2.keywords["kernelName"],
            ),
            (
                "flydsl_moe1_abf16_wint4_bf16_t32x128x128",
                "flydsl_moe2_abf16_wint4_bf16_t32x128x128_atomic",
            ),
        )


if __name__ == "__main__":
    unittest.main()
