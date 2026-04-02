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
            if is_graph_capturing():
                return

            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    # support dualpipev
                    if not get_args().gradient_accumulation_fusion or not get_args().delay_wgrad_compute:
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
