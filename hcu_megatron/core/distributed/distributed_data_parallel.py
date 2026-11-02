# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from functools import wraps

import torch

from megatron.core.transformer.cuda_graphs import is_graph_capturing

from hcu_megatron.training import get_args


def ultraep_backward_post_hook_wrapper(original):
    """Wrap the selected Megatron/HCU hook for UltraEP; preserve baseline behavior."""
    @wraps(original)
    def wrapper(self, param):
        original_hook = original(self, param)
        if not getattr(param, 'is_eplb_master', False):
            return original_hook
        param._ddp_bucket_group = self.param_to_bucket_group[param]
        if getattr(self.ddp_config, 'overlap_param_gather', False):
            param._eplb_forward_pre_hook = self._make_forward_pre_hook()

        def hook(*args):
            if not getattr(getattr(param, '_eplb_manager', None), 'ultraep_active', True):
                return original_hook(*args)
            if is_graph_capturing():
                return
            if param.grad is not None:
                if not getattr(param, 'grad_added_to_main_grad', False) or getattr(param, 'zero_out_wgrad', False):
                    param.main_grad.add_(param.grad.data)
                param.grad = None
            # Replica reduction will signal this parameter's bucket readiness.

        return hook

    return wrapper


class DistributedDataParallel():
    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """
        Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
        ready (i.e., when all grads in a bucket have been computed in all microbatches
        in a batch).
        """

        def hook(*unused):
            args = get_args()
            if is_graph_capturing():
                return

            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    # param.grad can temporarily be None in the following cases:
                    # (1) using dualpipev/ZB_H1 schedule.
                    # (2) using ripipe schedule.
                    is_ripipe = getattr(args, 'recompute_in_advance', False) or getattr(args, 'recompute_in_bubble', False)
                    if (
                        not (args.gradient_accumulation_fusion and args.delay_wgrad_compute)
                        and not is_ripipe
                    ):
                        assert (
                            param.grad is not None
                        ), 'param.grad being None is not safe when overlap_grad_reduce is True'

                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(
                        param, self.force_all_reduce
                    )

        return hook
