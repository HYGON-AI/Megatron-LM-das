# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch

from megatron.training import get_args
from megatron.core.transformer.cuda_graphs import is_graph_capturing


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
