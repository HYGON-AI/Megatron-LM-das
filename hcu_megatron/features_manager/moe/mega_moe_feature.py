# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Feature registration for the Primus-Turbo MegaMoE layer."""

from argparse import ArgumentParser
from functools import wraps

from ..feature import AbstractFeature


def advance_mega_moe_weight_generation() -> None:
    """Invalidate Primus-Turbo's cached MXFP8 expert weights."""
    from primus_turbo.pytorch.kernels.fused_mega_moe import advance_weight_generation

    advance_weight_generation()


def mega_moe_train_step_wrapper(train_step):
    """Advance the MXFP8 weight generation after a completed optimizer step."""

    @wraps(train_step)
    def wrapper(*args, **kwargs):
        result = train_step(*args, **kwargs)
        advance_mega_moe_weight_generation()
        return result

    return wrapper


class MegaMoeFeature(AbstractFeature):
    def __init__(self):
        super().__init__("mega-moe")

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--enable-mega-moe",
            "--use-turbo-mega-moe",
            action="store_true",
            default=False,
            dest="mega_moe",
            help="Replace Megatron MoELayer with Primus-Turbo MegaMoE",
        )
        group.add_argument(
            "--turbo-mega-moe-precision",
            type=str,
            default="bf16",
            choices=["bf16", "mxfp8"],
            help="MegaMoE expert precision. Model parameters remain bf16.",
        )

    def validate_args(self, args):
        if not getattr(args, "mega_moe", False):
            return args

        assert getattr(args, "bf16", False), "--enable-mega-moe requires --bf16"
        tp_size = int(getattr(args, "tensor_model_parallel_size", 1))
        ep_size = int(getattr(args, "expert_model_parallel_size", 1))
        assert tp_size == 1, "--enable-mega-moe requires tensor model parallel size 1"
        assert 1 < ep_size <= 8, (
            "--enable-mega-moe requires expert model parallel size in [2, 8]"
        )
        return args

    def register_patches(self, patch_manager, args):
        from hcu_megatron.core.extensions.mega_moe import PrimusTurboMegaMoELayer

        # Both symbols must be replaced because transformer_layer.py compares the
        # spec module against MoELayer by identity.
        patch_manager.register_patch(
            "megatron.core.transformer.moe.moe_layer.MoELayer",
            PrimusTurboMegaMoELayer,
        )
        patch_manager.register_patch(
            "megatron.core.models.gpt.moe_module_specs.MoELayer",
            PrimusTurboMegaMoELayer,
        )

        if args.turbo_mega_moe_precision == "mxfp8":
            patch_manager.register_patch(
                "megatron.training.training.train_step",
                mega_moe_train_step_wrapper,
                apply_wrapper=True,
            )
