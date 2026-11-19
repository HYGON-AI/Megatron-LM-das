# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests executing production functions with mocked dependencies."""

import argparse
import ast
import importlib.util
import sys
import unittest
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


base = load("linear_ce_features.feature", "hcu_megatron/features_manager/feature.py")
with mock.patch.dict(sys.modules, {"linear_ce_features.feature": base}):
    feature_module = load(
        "linear_ce_features.fusions.linear_cross_entropy_feature",
        "hcu_megatron/features_manager/fusions/linear_cross_entropy_feature.py",
    )


class TestLinearCEAdaptor(unittest.TestCase):
    def setUp(self):
        self.inference = SimpleNamespace(is_active=lambda: False)
        self.args = SimpleNamespace(enable_vocab_parallel=False)
        self.ordinary_ce = mock.Mock()
        self.ordinary_ce.return_value.transpose.return_value.contiguous.return_value = "ordinary-loss"
        self.namespace = load_functions(
            "hcu_megatron/core/models/gpt/gpt_model.py",
            {"gpt_model_postprocess"},
            {
                "InferenceMode": self.inference,
                "get_args": lambda: self.args,
                "tensor_parallel": SimpleNamespace(vocab_parallel_cross_entropy=self.ordinary_ce),
                "has_config_logger_enabled": lambda config: False,
            },
        )
        self.native = mock.Mock(return_value="fused-loss")
        context = mock.patch.dict(
            sys.modules,
            {
                "hcu_megatron.core.fusions.fused_linear_cross_entropy": SimpleNamespace(
                    linear_cross_entropy_for_training=self.native
                ),
            },
        )
        context.start()
        self.addCleanup(context.stop)
        self.feature = feature_module.LinearCrossEntropyFeature()
        self.logits = mock.Mock()
        self.logits.transpose.return_value.contiguous.return_value = "inference-logits"
        self.model = SimpleNamespace(
            post_process=True,
            training=True,
            config=SimpleNamespace(
                cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="linear", mtp_num_layers=None, use_mup=False
            ),
            share_embeddings_and_output_weights=False,
            output_layer=mock.Mock(return_value=(self.logits, None)),
            shared_embedding_or_output_weight=mock.Mock(return_value=object()),
            _scale_logits=lambda logits: logits,
            compute_language_model_loss=mock.Mock(return_value="ordinary-loss"),
        )

    def call(self, labels="labels", **kwargs):
        return self.namespace["gpt_model_postprocess"](
            self.model, "hidden", "ids", "positions", labels, None, None, None, **({"loss_mask": "mask"} | kwargs)
        )

    def test_official_parser_choices_and_feature_registration(self):
        for choices in (["native", "te"], ("native", "te"), ["native", "te", "linear"]):
            parser = argparse.ArgumentParser()
            parser.add_argument("--cross-entropy-loss-fusion", action="store_true")
            action = parser.add_argument("--cross-entropy-fusion-impl", choices=choices, default="native")
            original_actions = list(parser._actions)
            self.feature.register_args(parser)
            self.feature.register_args(parser)
            self.assertEqual(parser._actions, original_actions)
            self.assertFalse(parser.parse_args([]).cross_entropy_loss_fusion)
            self.assertEqual(parser.parse_args([]).cross_entropy_fusion_impl, "native")
            args = parser.parse_args(["--cross-entropy-loss-fusion", "--cross-entropy-fusion-impl", "linear"])
            self.assertTrue(args.cross_entropy_loss_fusion)
            self.assertEqual(args.cross_entropy_fusion_impl, "linear")
            self.assertEqual(action.choices.count("linear"), 1)
            self.assertNotIn("--use-hcu-linear-cross-entropy", parser._option_string_actions)
        early_parser = argparse.ArgumentParser()
        original_actions = list(early_parser._actions)
        self.feature.register_args(early_parser)
        self.assertEqual(early_parser._actions, original_actions)
        recorder = mock.Mock()
        self.feature.register_patches(recorder, argparse.Namespace())
        recorder.register_patch.assert_not_called()

    def test_validation_requires_both_flags(self):
        for enabled, impl in [(False, "linear"), (True, "native"), (True, "te")]:
            args = argparse.Namespace(cross_entropy_loss_fusion=enabled, cross_entropy_fusion_impl=impl)
            self.assertIs(self.feature.validate_args(args), args)
        args = argparse.Namespace(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="linear")
        with self.assertRaisesRegex(ValueError, "bf16"):
            self.feature.validate_args(args)
        args.bf16 = True
        self.assertIs(self.feature.validate_args(args), args)
        for name, value in [
            ("tensor_model_parallel_size", 2),
            ("context_parallel_size", 2),
            ("sequence_parallel", True),
            ("mtp_num_layers", 1),
            ("use_mup", True),
            ("enable_vocab_parallel", True),
            ("defer_embedding_wgrad_compute", True),
        ]:
            with self.subTest(name=name):
                setattr(args, name, value)
                with self.assertRaisesRegex(ValueError, name):
                    self.feature.validate_args(args)
                delattr(args, name)

    def test_validation_runs_with_empty_early_adaptor_args(self):
        namespace = load_functions(
            "hcu_megatron/training/arguments.py",
            {"validate_args_func_decorator"},
            {
                "wraps": wraps,
                "ADAPTOR_FEATURES": [self.feature],
                "ORIGIN_ARG_VALUES": {},
                "get_adaptor_args": lambda: argparse.Namespace(),
                "_print_env_vars": lambda *a, **kw: None,
            },
        )
        args = argparse.Namespace(
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl="linear",
            schedule_method=None,
            delay_wgrad_compute=False,
            cuda_graph_impl="none",
            sync_free_moe_backend=None,
            use_primus_deepep=False,
            bf16=True,
        )

        def upstream(args, defaults):
            self.assertEqual(args.cross_entropy_fusion_impl, "native")
            return args

        validate = namespace["validate_args_func_decorator"](upstream)
        self.assertIs(validate(args), args)
        self.assertEqual(args.cross_entropy_fusion_impl, "linear")
        args.bf16 = False
        with self.assertRaisesRegex(ValueError, "bf16"):
            validate(args)

    def test_fused_training_eval_and_weight_selection(self):
        for shared in (False, True):
            for training in (False, True):
                self.model.training = training
                self.model.share_embeddings_and_output_weights = shared
                self.assertEqual(self.call(), "fused-loss")
                weight = (
                    self.model.shared_embedding_or_output_weight.return_value
                    if shared
                    else self.model.output_layer.weight
                )
                self.native.assert_called_with("hidden", weight, "labels", "mask")
        self.model.output_layer.assert_not_called()
        self.model.compute_language_model_loss.assert_not_called()

    def test_other_backends_keep_original_path(self):
        for enabled, impl in [(False, "linear"), (False, "native"), (True, "native"), (True, "te")]:
            self.model.config.cross_entropy_loss_fusion = enabled
            self.model.config.cross_entropy_fusion_impl = impl
            self.assertEqual(self.call(), "ordinary-loss")
        self.native.assert_not_called()
        self.assertEqual(self.model.output_layer.call_count, 4)

    def test_non_output_and_inference_keep_original_path(self):
        self.model.post_process = False
        self.assertEqual(self.call(), "hidden")
        self.model.post_process = True
        self.assertEqual(self.call(labels=None), "inference-logits")
        self.inference.is_active = lambda: True
        labels = mock.Mock()
        self.assertEqual(self.call(labels=labels, runtime_gather_output=True), "ordinary-loss")
        self.ordinary_ce.assert_called_once_with(self.logits, labels.transpose.return_value.contiguous.return_value)
        self.native.assert_not_called()

    def test_rejections_happen_before_output_or_native(self):
        for extra, message in [
            ({"loss_mask": None}, "loss_mask"),
            ({"packed_seq_params": object()}, "packed_seq_params"),
            ({"output_processor": object()}, "output_processor"),
            ({"mtp_in_postprocess": True}, "mtp_in_postprocess"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.call(**extra)
        self.model.output_layer.assert_not_called()
        self.native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
