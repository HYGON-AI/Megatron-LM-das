import contextlib
from functools import wraps
from typing import Callable, Iterator, List, Optional, Union

import torch

from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
)
from megatron.core.pipeline_parallel.schedules import set_current_microbatch
from megatron.core.timers import Timer

from .dualpipev.dualpipev_schedules import forward_backward_pipelining_with_cutinhalf
from .ripipe_schedules import forward_backward_ripipe_pipelining
from .fine_grained_activation_offload import fine_grained_offloading_reset
from .seq1f1b.schedules import seq1f1b_forward_backward_pipelining_without_interleaving, seq1f1b_forward_backward_pipelining_with_interleaving
from dcu_megatron.core.pipeline_parallel.schedule_timers import ScheduleTimers
from dcu_megatron.core.parallel_state import get_dualpipe_chunk


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
            if args.enable_vocab_parallel:
                from dcu_megatron.core.pipeline_parallel.vocab_parallel_schedule import (
                    forward_backward_pipelining_with_vocab_parallel
                )
                return forward_backward_pipelining_with_vocab_parallel

            return fn()
        elif args.schedule_method == "dualpipev":
            return forward_backward_pipelining_with_cutinhalf
        elif args.schedule_method == "seq1f1b":
            return seq1f1b_forward_backward_pipelining_without_interleaving
        elif args.schedule_method == "interleaved_seq1f1b":
            return seq1f1b_forward_backward_pipelining_with_interleaving
        elif args.schedule_method == "ripipe":
            return forward_backward_ripipe_pipelining
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


def forward_step_calc_loss(
    model,
    output_tensor,
    loss_func,
    config,
    vp_stage,
    collect_non_loss_data,
    num_microbatches,
    forward_data_store,
    cp_group_size=None,
    is_last_stage=None,
    skip_loss_compute=False,
    force_loss_compute=False,
    run_timer=False,
    mem_before=None,
):
    """Calculate the loss and number of tokens for forward_step()"""

    model_vp_stage = getattr(model, "vp_stage", None)
    if vp_stage is not None and model_vp_stage is not None:
        assert (
            vp_stage == model_vp_stage
        ), f"vp_stage ({vp_stage}) doesn't match model_vp_stage ({model_vp_stage})"

    if cp_group_size is None and is_last_stage is None:
        # fallback to parallel state
        cp_group_size = parallel_state.get_context_parallel_world_size()
        if get_args().schedule_method == "dualpipev":
            is_last_stage = parallel_state.is_pipeline_first_stage() and get_dualpipe_chunk() == 1
        else:
            is_last_stage = parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=vp_stage
            )
    else:
        assert (
            cp_group_size is not None and is_last_stage is not None
        ), "cp_group_size and is_last_stage must be provided"

    # support vocab parallel
    is_last_stage = (is_last_stage and (not skip_loss_compute)) or force_loss_compute

    num_tokens = torch.tensor(0, dtype=torch.int)
    if is_last_stage:
        if get_args().enable_vocab_parallel:
            output_tensor = output_tensor.transpose(0, 1).contiguous()

        if not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    # Protect against division by zero when all tokens are masked
                    #   in a microbatch.
                    output_tensor /= torch.clamp(num_tokens, min=1)
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor *= cp_group_size
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    if run_timer:
        assert mem_before is not None
        ScheduleTimers.for_chunk(0).f.stop()
        ScheduleTimers.for_chunk(0).f_mem += torch.cuda.memory_allocated() - mem_before

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # Set the loss scale for Multi-Token Prediction (MTP) loss.
    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    return output_tensor, num_tokens


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    cp_group_size,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    vp_stage=None,
    is_last_stage=True,
    skip_loss_compute=False,
    force_loss_compute=False,
    run_timer=False,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.
        vp_stage (int, optional):
            The virtual pipeline stage. Defaults to None.
        is_last_stage (bool, optional):
            Whether it is the last stage. Defaults to True.
            Also considering virtual stages.
            In case of PP/VPP, is_last_stage/is_vp_last_stage.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """

    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    mem_before = None
    if run_timer:
        ScheduleTimers.for_chunk(0).f_cnt += 1
        ScheduleTimers.for_chunk(0).f.start()
        mem_before = torch.cuda.memory_allocated()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model, microbatch_id=current_microbatch)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch, microbatch_id=current_microbatch
            )
    output_tensor, num_tokens = forward_step_calc_loss(
        model,
        output_tensor,
        loss_func,
        config,
        vp_stage,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size,
        is_last_stage,
        skip_loss_compute=skip_loss_compute,
        force_loss_compute=force_loss_compute,
        run_timer=run_timer,
        mem_before=mem_before,
    )

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, run_timer=False):
    from megatron.core.pipeline_parallel.schedules import backward_step as _backward_step

    if run_timer:
        ScheduleTimers.for_chunk(0).b_cnt += 1
        ScheduleTimers.for_chunk(0).b.start()
        mem_before = torch.cuda.memory_allocated()

    input_tensor_grad = _backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    if run_timer:
        ScheduleTimers.for_chunk(0).b.stop()
        ScheduleTimers.for_chunk(0).b_mem += torch.cuda.memory_allocated() - mem_before

    return input_tensor_grad


def bootstrap_and_profile_p2p_communication(
    p2p_communicator, send_tensor_shapes, recv_tensor_shapes
):
    if ScheduleTimers.iter_counter == 1:
        nccl_init_tensor = [torch.Tensor([0]).cuda()]
        shape = [(1,)]
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            p2p_communicator.recv_forward(shape, is_pp_first_stage(p2p_communicator.pp_group))
        if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            p2p_communicator.send_forward(nccl_init_tensor, is_pp_last_stage(p2p_communicator.pp_group))
            p2p_communicator.recv_backward(shape, is_pp_last_stage(p2p_communicator.pp_group))
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            p2p_communicator.send_backward(nccl_init_tensor, is_pp_first_stage(p2p_communicator.pp_group))

        send_data = [torch.zeros(*shape, dtype=p2p_communicator.config.pipeline_dtype).cuda() for
                     shape in send_tensor_shapes]
        recv_data = [torch.zeros(*shape, dtype=p2p_communicator.config.pipeline_dtype).cuda() for
                     shape in recv_tensor_shapes]
        torch.distributed.barrier()
        t = Timer('comm-benchmark')
        t.start()
        for _ in range(10):
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                p2p_communicator.recv_forward(recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group))
            if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                p2p_communicator.send_forward(send_data, is_pp_last_stage(p2p_communicator.pp_group))
                p2p_communicator.recv_backward(send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group))
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                p2p_communicator.send_backward(recv_data, is_pp_first_stage(p2p_communicator.pp_group))
        t.stop()
        per_communication = torch.cuda.FloatTensor([t.elapsed() / (
            p2p_communicator.pp_group.size() - 1) / 10])
        torch.distributed.all_reduce(per_communication, torch.distributed.ReduceOp.MAX)
        ScheduleTimers.comm_time = per_communication.item()
