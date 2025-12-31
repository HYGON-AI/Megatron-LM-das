from functools import wraps
from typing import Callable, Iterator, List, Optional, Union

import torch

from megatron.training import get_args

from .dualpipev.dualpipev_schedules import forward_backward_pipelining_with_cutinhalf
from .fine_grained_activation_offload import fine_grained_offloading_reset


def get_forward_backward_func_wrapper(fn):
    @wraps(fn)
    def wrapper():
        """Retrieves the appropriate forward_backward function given the
        configuration of parallel_state.

        Returns a function that will perform all of the forward and
        backward passes of the model given the pipeline model parallel
        world size and virtual pipeline model parallel world size in the
        global parallel_state.

        """

        args = get_args()
        if args.schedule_method == "vanilla":
            return fn()
        elif args.schedule_method == "dualpipev":
            return forward_backward_pipelining_with_cutinhalf
        else:
            raise ValueError(f"schedule_method {args.schedule_method} is not supported")

    return wrapper


def forward_backward_pipelining_wrapper(fn):
    @wraps(fn)
    def wrapper(
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: Optional[int] = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: Optional[bool] = None,
        adjust_tensor_shapes_fn: Optional[Callable] = None,
    ):

        args = get_args()

        if not forward_only and args.fine_grained_activation_offloading:
            fine_grained_offloading_reset()

        return fn(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            forward_only=forward_only,
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn
        )

    return wrapper
