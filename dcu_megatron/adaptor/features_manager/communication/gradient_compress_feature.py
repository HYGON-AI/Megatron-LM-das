from argparse import ArgumentParser

from ..feature import AbstractFeature


class GradientCompressFeature(AbstractFeature):
    def __init__(self):
        super().__init__('enable-dynamic-grad-comp')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group = parser.add_argument_group(title='grad comp args')
        group.add_argument('--enable-dynamic-grad-comp',
                           dest='enable_dynamic_grad_comp',
                           action='store_true',
                           help='Enable dynamic gradient compression (e.g., adaptive rank/sparsity based on training phase or gradient statistics).')
        group.add_argument('--grad-comp',
                           dest='grad_comp', action='store_true', help='use grad comp algorithm for data parallel.')
        group.add_argument('--grad-comp-warm-up', type=float, default=0.1,
                            help='PwerSGD warm up period for accuracy gain.')
        group.add_argument('--rank-adjust-window-size',
                            type=int, default=1000,
                            help='the window size of adjust rank')
        group.add_argument('--iteration-sample-ratio',
                            type=float, default=0.01,
                            help='iteration_sample_ratio')
        group.add_argument('--gradient-sample-ratio',
                            type=float, default=1.0,
                            help='gradient_sample_ratio')
        group.add_argument('--collect-log-path', type=str, default='./logs',
                           help='If set, collect some data during the iteration process, such as the time and loss of each iteration')

    def register_patches(self, patch_manager, args):
        from dcu_megatron.core.distributed.finalize_model_grads import finalize_model_grads
        from dcu_megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup, _ParamAndGradBuffer, \
            _ParamAndGradBucket
        from dcu_megatron.training.training import save_checkpoint_and_time_wrapper
        from dcu_megatron.training.initialize import initialize_megatron_wrapper

        # edgc相关功能函数替换
        if args.enable_dynamic_grad_comp:
            patch_manager.register_patch('megatron.core.distributed.finalize_model_grads.finalize_model_grads',
                                        finalize_model_grads)
            patch_manager.register_cls_funcs('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup',
                                             [_ParamAndGradBucketGroup.__init__,
                                              _ParamAndGradBucketGroup.start_grad_sync,
                                              _ParamAndGradBucketGroup.finish_grad_sync,])
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBuffer._new_bucket',
                                        _ParamAndGradBuffer._new_bucket)
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucket',
                                        _ParamAndGradBucket)

            patch_manager.register_patch('megatron.training.training.save_checkpoint_and_time',
                                        save_checkpoint_and_time_wrapper,
                                        apply_wrapper=True)

            patch_manager.register_patch('megatron.training.initialize.initialize_megatron',
                                        initialize_megatron_wrapper,
                                        apply_wrapper=True)
