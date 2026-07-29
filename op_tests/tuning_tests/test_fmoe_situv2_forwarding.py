"""Dependency-free coverage for SiTUv2 tuner parameter forwarding."""

import ast
import copy
from pathlib import Path
import unittest


_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "csrc"
    / "ck_gemm_moe_2stages_codegen"
    / "gemm_moe_tune.py"
)


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _calls(function, callee):
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == callee
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == callee
        )
    ]


def _keyword_value(call, name):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"missing {name} keyword")


class TestFmoeSiTUv2Forwarding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(_SOURCE.read_text())

    def test_none_normalizes_to_runtime_default_and_requested_values_survive(self):
        constants = [
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id
                in {"SITUV2_DEFAULT_BETA", "SITUV2_DEFAULT_LINEAR_BETA"}
                for target in node.targets
            )
        ]
        resolver = copy.deepcopy(_function(self.tree, "resolve_situv2_betas"))
        resolver.decorator_list = []
        module = ast.Module(body=constants + [resolver], type_ignores=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(module), str(_SOURCE), "exec"), namespace)

        self.assertEqual(namespace["resolve_situv2_betas"](None, None), (1.0, 1.0))
        self.assertEqual(namespace["resolve_situv2_betas"](4.0, 25.0), (4.0, 25.0))

    def test_kernel_and_reference_receive_the_same_named_pair(self):
        kernel = _calls(_function(self.tree, "run_flydsl_stage1_out"), "flydsl_moe_stage1")
        reference = _calls(_function(self.tree, "run_torch_moe_stage1"), "torch_moe_stage1")
        self.assertEqual(len(kernel), 1)
        self.assertEqual(len(reference), 1)

        for call in (kernel[0], reference[0]):
            self.assertIsInstance(_keyword_value(call, "situ_beta"), ast.Name)
            self.assertEqual(_keyword_value(call, "situ_beta").id, "situ_beta")
            self.assertIsInstance(_keyword_value(call, "situ_linear_beta"), ast.Name)
            self.assertEqual(
                _keyword_value(call, "situ_linear_beta").id, "situ_linear_beta"
            )

    def test_run_config_and_generated_stage2_reference_share_the_pair(self):
        run_config = _function(self.tree, "run_config")
        fused_calls = [
            call
            for call in _calls(run_config, "run_perftest")
            if call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "fused_moe"
        ]
        self.assertEqual(len(fused_calls), 1)
        self.assertEqual(_keyword_value(fused_calls[0], "beta").id, "situ_beta")
        self.assertEqual(
            _keyword_value(fused_calls[0], "linear_beta").id, "situ_linear_beta"
        )

        stage2_data = _function(self.tree, "generate_data_2stages")
        argument_names = [arg.arg for arg in stage2_data.args.args]
        self.assertLess(
            argument_names.index("situ_beta"), argument_names.index("device")
        )
        self.assertLess(
            argument_names.index("situ_linear_beta"), argument_names.index("device")
        )
        reference_calls = _calls(stage2_data, "run_torch_moe_stage1")
        self.assertEqual(len(reference_calls), 1)
        self.assertEqual(
            _keyword_value(reference_calls[0], "situ_beta").id, "situ_beta"
        )
        self.assertEqual(
            _keyword_value(reference_calls[0], "situ_linear_beta").id,
            "situ_linear_beta",
        )


if __name__ == "__main__":
    unittest.main()
