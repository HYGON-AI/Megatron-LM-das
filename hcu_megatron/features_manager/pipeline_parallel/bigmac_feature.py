# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Opt-in BigMac integration; existing DAS schedules stay unchanged by default.
from functools import partial

from ..feature import AbstractFeature
from hcu_megatron.core.pipeline_parallel.bigmac.config import executor_enabled, encoder_enabled


class BigMacFeature(AbstractFeature):
    def __init__(self):
        super().__init__("use-pp-executor", optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group("BigMac pipeline executor")
        group.add_argument("--bigmac-data-compat", action="store_true")
        group.add_argument("--use-pp-executor", action="store_true", default=executor_enabled())
        group.add_argument("--use-bigmac-style", action="store_true",
                           default=executor_enabled() and encoder_enabled(),
                           help="Distribute Qwen3-VL vision encoder units across PP ranks.")

    def validate_args(self, args):
        if not args.use_pp_executor:
            return args
        if getattr(args, "schedule_method", "vanilla") != "vanilla":
            raise ValueError("BigMac requires --schedule-method vanilla")
        for name in (
            "enable_vocab_parallel", "delay_1f1b_cooldown_wgrad_compute",
            "overlap_moe_expert_parallel_comm", "use_megatron_fsdp", "use_torch_fsdp2",
            "delay_wgrad_compute", "defer_embedding_wgrad_compute",
            "fine_grained_activation_offloading",
        ):
            if getattr(args, name, False):
                raise ValueError(f"BigMac does not support --{name.replace('_', '-')}")
        if getattr(args, "context_parallel_size", 1) != 1:
            raise ValueError("The open-source BigMac executor requires context parallel size 1")
        if getattr(args, "cuda_graph_impl", "none") == "full_iteration":
            raise ValueError("BigMac does not support full-iteration CUDA graphs")
        if getattr(args, "overlap_param_gather", False) and (
            args.use_bigmac_style or getattr(args, "align_param_gather", False)
        ):
            raise ValueError(
                "BigMac requires disabling --overlap-param-gather for encoder scheduling "
                "or aligned parameter gathering"
            )
        if args.use_bigmac_style:
            if getattr(args, "bridge_language_model_only", False):
                raise ValueError("BigMac vision scheduling requires the full Qwen3-VL model")
            if getattr(args, "save_wgrads_interval", None):
                raise ValueError("BigMac vision scheduling does not support --save-wgrads-interval")
            if getattr(args, "model_arch", None) != "qwen3vl" or not getattr(args, "use_bridge", False):
                raise ValueError("--use-bigmac-style requires the existing --model-arch qwen3vl --use-bridge entry")
            if getattr(args, "virtual_pipeline_model_parallel_size", None) is not None:
                raise ValueError("BigMac Qwen3-VL currently supports non-interleaved PP only")
        return args

    def register_patches(self, patch_manager, args):
        if getattr(args, "bigmac_data_compat", False):
            from hcu_megatron.core.pipeline_parallel.bigmac.data import register_patches
            register_patches(patch_manager)
        if not args.use_pp_executor:
            return
        from hcu_megatron.core.pipeline_parallel.bigmac.integration import (
            get_forward_backward_func_wrapper,
        )
        patch_manager.register_patch(
            "megatron.core.pipeline_parallel.schedules.get_forward_backward_func",
            partial(get_forward_backward_func_wrapper, with_ve=args.use_bigmac_style),
            apply_wrapper=True,
        )
        if args.use_bigmac_style:
            from hcu_megatron.core.pipeline_parallel.bigmac.qwen import register_patches
            register_patches(patch_manager)
