# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser, Action
from megatron.core.utils import is_te_min_version

from ..feature import AbstractFeature


class NewtonSchulzTpImplAction(Action):
    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, value)
        if value == "transformer_engine":
            namespace.use_optimizer_feature = True


class OptimizerFeature(AbstractFeature):

    def __init__(self):
        super().__init__('use-optimizer-feature')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--use-optimizer-feature', action='store_true',
                           help='whether to use optimizer related feature.')
        group.add_argument('--reuse-fp32-param', action='store_true',
                           help='The distributed training optimizer frees up '
                                'param copies of FP32 to save memory.')
        group.add_argument('--newton-schulz-tp-impl',
                           type=str,
                           default="native",
                           action=NewtonSchulzTpImplAction,
                           choices=["native", "transformer_engine"],
                           help="Implementation of newton_schulz_tp when tp mode is 'distributed'.",
                           )

    def validate_args(self, args):
        if args.reuse_fp32_param and not args.bf16:
            raise AssertionError('--reuse-fp32-param only support for `bf16`')
        return args

    def register_patches(self, patch_manager, args):
        if args.reuse_fp32_param:
            from hcu_megatron.core.memory.reuse_param.adaptor import (
                step_with_ready_grads,
                prepare_grads,
                reuse_fp32_param_init_wrapper,
                optimizer_config_init_wrapper
            )
            from hcu_megatron.core.memory.reuse_param.adaptor import reuse_fp32_param_distrib_optimizer_init_wrapper
            from hcu_megatron.core.memory.reuse_param.adaptor import reuse_fp32_param_param_and_grad_buffer_init_wrapper

            patch_manager.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.prepare_grads',
                                        prepare_grads)
            patch_manager.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step_with_ready_grads',
                                        step_with_ready_grads)
            patch_manager.register_patch('megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                                        reuse_fp32_param_init_wrapper)
            patch_manager.register_patch('megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                                        optimizer_config_init_wrapper)

            patch_manager.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                                        reuse_fp32_param_distrib_optimizer_init_wrapper)
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBuffer.__init__',
                                        reuse_fp32_param_param_and_grad_buffer_init_wrapper)

        if args.newton_schulz_tp_impl == "transformer_engine":
            if not is_te_min_version("2.15.0"):
                raise AssertionError("Must have at least transformer-engine version of 2.15.0")

            from hcu_megatron.optimizer.muon_utils import newton_schulz_tp

            patch_manager.register_patch('megatron.core.optimizer.emerging_optimizers.newton_schulz_tp',
                                         newton_schulz_tp)