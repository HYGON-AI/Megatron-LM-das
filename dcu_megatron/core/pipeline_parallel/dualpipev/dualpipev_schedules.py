# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import contextlib
from functools import wraps
from typing import Iterator, List, Union

from megatron.training import print_rank_0

import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.training import get_args
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
)
from megatron.core.pipeline_parallel.schedules import clear_embedding_activation_buffer, deallocate_output_tensor
from megatron.core import ModelParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import _communicate
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    set_current_microbatch,
    check_first_val_step,
    finish_embedding_wgrad_compute
)

from dcu_megatron.core.pipeline_parallel.combined_1f1b import forward_backward_step, set_streams, wrap_forward_func
from dcu_megatron.core.parallel_state import set_dualpipe_chunk
from dcu_megatron.training.utils import print_rank_message
# from mindspeed.core.pipeline_parallel.fb_overlap.modules.weight_grad_store import WeightGradStore


# Types
Shape = Union[List[int], torch.Size]
LOSS_BACKWARD_SCALE = torch.tensor(1.0)


def is_dualpipev_last_stage(model_chunk_id):
    return parallel_state.is_pipeline_first_stage(ignore_virtual=True) and model_chunk_id == 1


def send_forward(output_tensor: torch.Tensor, tensor_shape, config: ModelParallelConfig, model_chunk_id, async_op=False) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """
    tensor_send_next, tensor_send_prev = None, None
    if model_chunk_id == 0:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            return None
        tensor_send_next = output_tensor
    else:
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            return None
        tensor_send_prev = output_tensor

    if config.timers is not None:
        config.timers('forward-send', log_level=2).start()

    _, _, fwd_wait_handles = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=False,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
        wait_on_reqs=(not async_op)
    )
    if config.timers is not None:
        config.timers('forward-send').stop()

    return fwd_wait_handles


def send_backward(input_tensor_grad: torch.Tensor, tensor_shape, config: ModelParallelConfig, model_chunk_id, async_op=False) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    tensor_send_next, tensor_send_prev = None, None
    if model_chunk_id == 0:
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            return None
        tensor_send_prev = input_tensor_grad
    else:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            return None
        tensor_send_next = input_tensor_grad

    if config.timers is not None:
        config.timers('backward-send', log_level=2).start()
    _, _, reqs = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=False,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
        wait_on_reqs=(not async_op)
    )
    if config.timers is not None:
        config.timers('backward-send').stop()
    return reqs


def recv_forward(tensor_shape: Shape, config: ModelParallelConfig, model_chunk_id, async_op=False) -> torch.Tensor:
    """ Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """
    recv_prev, recv_next = False, False
    if model_chunk_id == 0:
        recv_prev = True
    else:
        recv_next = True

    if (
        (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and recv_prev)
        or (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and recv_next)
    ):
        fwd_wait_handles = None
        return None, fwd_wait_handles
    else:
        if config.timers is not None:
            config.timers('forward-recv', log_level=2).start()
        tensor_recv_prev, tensor_recv_next, fwd_wait_handles = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            config=config,
            wait_on_reqs=(not async_op),
        )
        if config.timers is not None:
            config.timers('forward-recv').stop()

    if recv_prev:
        return tensor_recv_prev, fwd_wait_handles
    else:
        return tensor_recv_next, fwd_wait_handles


def recv_backward(tensor_shape: Shape, config: ModelParallelConfig, model_chunk_id, async_op=False) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    recv_prev, recv_next = False, False
    if model_chunk_id == 0:
        recv_next = True
    else:
        recv_prev = True

    if (
        (parallel_state.is_pipeline_first_stage(ignore_virtual=True) and recv_prev)
        or (parallel_state.is_pipeline_last_stage(ignore_virtual=True) and recv_next)
    ):
        output_tensor_grad = None
        bwd_wait_handles = None
        return output_tensor_grad, bwd_wait_handles
    else:

        if config.timers is not None:
            config.timers('backward-recv', log_level=2).start()
        tensor_recv_prev, tensor_recv_next, bwd_wait_handles = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            config=config,
            wait_on_reqs=(not async_op)
        )
        if config.timers is not None:
            config.timers('backward-recv').stop()

    if recv_prev:
        return tensor_recv_prev, bwd_wait_handles
    else:
        return tensor_recv_next, bwd_wait_handles


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    model_chunk_id,
    async_op=False
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    recv_prev, recv_next = False, False
    tensor_send_next, tensor_send_prev = None, None
    if model_chunk_id == 0:
        if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            tensor_send_next = output_tensor
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            recv_prev = True
    if model_chunk_id == 1:
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            tensor_send_prev = output_tensor
        if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            recv_next = True

    if config.timers is not None:
        config.timers('forward-send-forward-recv', log_level=2).start()
    tensor_recv_prev, tensor_recv_next, fwd_wait_handles = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not async_op),
        config=config
    )
    if config.timers is not None:
        config.timers('forward-send-forward-recv').stop()

    if model_chunk_id == 0:
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            return tensor_recv_prev, fwd_wait_handles
        else:
            return None, fwd_wait_handles
    else:
        if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            return tensor_recv_next, fwd_wait_handles
        else:
            return None, fwd_wait_handles


def send_forward_recv_slave_forward(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    fwd_model_chunk_id,
    async_op=False,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.
    See _communicate for argument details.
    """
    recv_prev, recv_next = False, False
    tensor_send_next, tensor_send_prev = None, None
    if fwd_model_chunk_id == 0:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            return None, None
        tensor_send_next = output_tensor
        recv_next = True
    if fwd_model_chunk_id == 1:
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            return None, None
        tensor_send_prev = output_tensor
        recv_prev = True
    if config.timers is not None:
        config.timers('forward-send-slave-forward-recv', log_level=2).start()
    tensor_recv_prev, tensor_recv_next, fwd_wait_handles = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not async_op),
        config=config,
    )
    if config.timers is not None:
        config.timers('forward-send-slave-forward-recv').stop()

    if fwd_model_chunk_id == 0:
        return tensor_recv_next, fwd_wait_handles
    else:
        return tensor_recv_prev, fwd_wait_handles


def send_backward_recv_slave_backward(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    fwd_model_chunk_id,
    async_op=False,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.
    See _communicate for argument details.
    """
    recv_prev, recv_next = False, False
    tensor_send_next, tensor_send_prev = None, None
    if fwd_model_chunk_id == 0:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            return None, None
        tensor_send_next = input_tensor_grad
        recv_next = True
    if fwd_model_chunk_id == 1:
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            return None, None
        tensor_send_prev = input_tensor_grad
        recv_prev = True
    if config.timers is not None:
        config.timers('forward-send-slave-forward-recv', log_level=2).start()
    tensor_recv_prev, tensor_recv_next, fwd_wait_handles = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not async_op),
        config=config,
    )
    if config.timers is not None:
        config.timers('forward-send-slave-forward-recv').stop()

    if fwd_model_chunk_id == 0:
        return tensor_recv_next, fwd_wait_handles
    else:
        return tensor_recv_prev, fwd_wait_handles


def get_send_handle(handles, model_chunk_id, forward=True):
    send_handle = None
    if handles is None:
        return send_handle

    if forward:
        send_direction = "send_next" if model_chunk_id == 0 else "send_prev"
        if send_direction in handles:
            send_handle = handles.pop(send_direction)
    else:
        send_direction = "send_prev" if model_chunk_id == 0 else "send_next"
        if send_direction in handles:
            send_handle = handles.pop(send_direction)

    return send_handle


def get_recv_handle(handles, model_chunk_id, forward=True):
    recv_handle = None
    if handles is None:
        return recv_handle

    if forward:
        recv_direction = "recv_prev" if model_chunk_id == 0 else "recv_next"
        if recv_direction in handles:
            recv_handle = handles.pop(recv_direction)
    else:
        recv_direction = "recv_next" if model_chunk_id == 0 else "recv_prev"
        if recv_direction in handles:
            recv_handle = handles.pop(recv_direction)

    return recv_handle


def generate_dualpipev_schedule(pp_size, num_microbatches):
    num_microbatches = num_microbatches * 2
    num_warmup_stages = [0] * pp_size
    num_interleaved_forward_stages = [0] * pp_size
    num_1b1w1f_stages = [0] * pp_size
    num_overlap_stages = [0] * pp_size
    num_1b1overlap_stages = [0] * pp_size
    num_interleaved_backward_stages = [0] * pp_size
    num_cooldown_stages = [0] * pp_size

    pp_size *= 2
    for i in range(pp_size // 2):
        num_warmup_stages[i] = pp_size - 2 - i * 2

        num_interleaved_forward_stages[i] = i + 1  # 1f1f

        num_1b1w1f_stages[i] = pp_size // 2 - i - 1

        num_overlap_stages[i] = num_microbatches - pp_size * 2 + i * 2 + 2

        num_1b1overlap_stages[i] = (pp_size // 2 - i - 1) * 2

        num_interleaved_backward_stages[i] = i + 1

        num_cooldown_stages[i] = [i, pp_size // 2 - i, i]

    schedule_all_stages = {
        'warmup': num_warmup_stages,
        'interleaved_forward': num_interleaved_forward_stages,
        '1b1w1f': num_1b1w1f_stages,
        'overlap': num_overlap_stages,
        '1b1overlap': num_1b1overlap_stages,
        'interleaved_backward': num_interleaved_backward_stages,
        'cooldown': num_cooldown_stages
    }

    return schedule_all_stages


def forward_step_no_model_graph(
    forward_step_func,
    model_chunk_id,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
):
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

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
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int)
    if is_dualpipev_last_stage(model_chunk_id):
        if not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor *= parallel_state.get_context_parallel_world_size()
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor *= parallel_state.get_context_parallel_world_size()
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

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

    # If T5 model and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


shared_embedding = None


def get_shared_embedding_from_dual_chunk():
    assert shared_embedding is not None
    return shared_embedding


def set_shared_embedding_from_dual_chunk(model1, model2):
    global shared_embedding
    if shared_embedding is not None:
        return
    if model1.module.module.pre_process:
        shared_embedding = model1.module.module.embedding.word_embeddings.weight
    elif model2.module.module.pre_process:
        shared_embedding = model2.module.module.embedding.word_embeddings.weight


def forward_backward_pipelining_with_cutinhalf(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    args = get_args()
    args.moe_fb_overlap = True
    args.dualpipe_no_dw_detach = True

    set_shared_embedding_from_dual_chunk(model[0], model[1])
    assert (
        isinstance(model, list) and len(model) == 2
    ), 'Dualpipe Schedule expects two model chunks'

    assert (
        isinstance(data_iterator, list) and len(data_iterator) == 2
    ), 'Dualpipe Schedule expects two data_iterators'

    config = get_model_config(model[0])
    config.batch_p2p_comm = False

    if (
        not forward_only
        and config.combined_1f1b
    ):
        set_streams()
        forward_step_func = wrap_forward_func(config, forward_step_func)

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync

    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
    # They will be re-enabled at the end of this function.
    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of steps for each stage
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    rank = parallel_state.get_pipeline_model_parallel_rank()
    schedule = generate_dualpipev_schedule(pp_size, num_microbatches)

    model_type = get_model_type(model[0])

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()
    forward_data_store = []
    input_tensors = [[], []]
    output_tensors = [[], []]
    output_tensor_grads = [[], []]

    master_chunk_id = 0
    slave_chunk_id = 1
    cur_fwd_chunk_microbatch = [0, num_microbatches]
    cur_bwd_chunk_microbatch = [0, num_microbatches]
    num_chunk_max_microbatch = [num_microbatches, num_microbatches * 2]

    def wait_comm_handle(comm_handle):
        if comm_handle is not None:
            comm_handle.wait()
        comm_handle = None

    def forward_step_helper(model_chunk_id, cur_microbatch, checkpoint_activations_microbatch=False):
        set_dualpipe_chunk(model_chunk_id)
        if not forward_only:
            offset = cur_bwd_chunk_microbatch[model_chunk_id]
            input_tensor = input_tensors[model_chunk_id][cur_microbatch - offset]
        else:
            input_tensor = input_tensors[model_chunk_id][0]

        is_first_microbatch = check_first_val_step(
            first_val_step,
            forward_only,
            cur_fwd_chunk_microbatch[model_chunk_id],
        ),
        output_tensor, num_tokens = forward_step_no_model_graph(
            forward_step_func,
            model_chunk_id,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            is_first_microbatch=is_first_microbatch,
            current_microbatch=cur_microbatch
        )
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens.item()
        if forward_only:
            input_tensors[model_chunk_id].pop(0)
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(model_chunk_id, compute_wgrad=False):
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        if compute_wgrad:
            model[model_chunk_id].backward_dw()

        return input_tensor_grad

    def combined_forward_backward_helper(
        fwd_model_chunk_id=None,
        bwd_model_chunk_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        block_level_wgrad_compute=False,
    ):
        """Helper method to run combined forward and backward step"""
        # forward prepare
        f_context = contextlib.nullcontext()
        fwd_input_tensor = None
        fwd_microbatch_id = None
        if fwd_model_chunk_id is not None:
            fwd_microbatch_id = cur_fwd_chunk_microbatch[fwd_model_chunk_id]
            set_dualpipe_chunk(fwd_model_chunk_id)
            offset = cur_bwd_chunk_microbatch[fwd_model_chunk_id]
            fwd_input_tensor = input_tensors[fwd_model_chunk_id][fwd_microbatch_id - offset]

        # backward prepare
        b_context = contextlib.nullcontext()
        bwd_input_tensor = None
        bwd_output_tensor = None
        bwd_output_tensor_grad = None
        if bwd_model_chunk_id is not None:
            bwd_input_tensor = input_tensors[bwd_model_chunk_id].pop(0)
            bwd_output_tensor = output_tensors[bwd_model_chunk_id].pop(0)
            bwd_output_tensor_grad = output_tensor_grads[bwd_model_chunk_id].pop(0)

        output_tensor, num_tokens, input_tensor_grad, chunk_backward_dw_func = forward_backward_step(
            forward_step_func,
            data_iterator[fwd_model_chunk_id] if fwd_model_chunk_id is not None else None,
            model[fwd_model_chunk_id] if fwd_model_chunk_id is not None else None,
            num_microbatches,
            fwd_input_tensor,
            forward_data_store,
            model[bwd_model_chunk_id] if bwd_model_chunk_id is not None else None,
            bwd_input_tensor,
            bwd_output_tensor,
            bwd_output_tensor_grad,
            config,
            f_context=f_context,
            b_context=b_context,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=False,
            current_microbatch=fwd_microbatch_id,
            block_level_wgrad_compute=block_level_wgrad_compute,
        )

        # forward post process
        if fwd_model_chunk_id is not None:
            cur_fwd_chunk_microbatch[fwd_model_chunk_id] += 1

            output_tensors[fwd_model_chunk_id].append(output_tensor)
            nonlocal total_num_tokens
            total_num_tokens += num_tokens.item()

            if forward_only:
                input_tensors[fwd_model_chunk_id].pop(0)
                output_tensors[fwd_model_chunk_id].pop()

        # backward post process
        if bwd_model_chunk_id is not None:
            cur_bwd_chunk_microbatch[bwd_model_chunk_id] += 1

        return output_tensor, input_tensor_grad, chunk_backward_dw_func

    def forward_backward_helper_wrapper(
        fwd_model_chunk_id=None,
        bwd_model_chunk_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        checkpoint_activations_microbatch=None,
        block_level_wgrad_compute=False,
    ):
        """
        wrap forward_helper、backward_helper、combined_forward_backward_helper in a unified way
        """

        if config.combined_1f1b and config.combined_1f1b_recipe == "ep_a2a" and not forward_only:
            assert (
                checkpoint_activations_microbatch is None
            ), "checkpoint_activations_microbatch not supported when combined_1f1b is true"
            return combined_forward_backward_helper(
                fwd_model_chunk_id=fwd_model_chunk_id,
                bwd_model_chunk_id=bwd_model_chunk_id,
                pre_forward=pre_forward,
                pre_backward=pre_backward,
                post_forward=post_forward,
                post_backward=post_backward,
                block_level_wgrad_compute=block_level_wgrad_compute,
            )
        else:
            output_tensor = None
            input_tensor_grad = None
            if fwd_model_chunk_id is not None:
                # forward pass
                if pre_forward is not None:
                    pre_forward()

                output_tensor = forward_step_helper(
                    fwd_model_chunk_id,
                    cur_fwd_chunk_microbatch[fwd_model_chunk_id],
                    checkpoint_activations_microbatch
                )
                cur_fwd_chunk_microbatch[fwd_model_chunk_id] += 1
                if post_forward is not None:
                    output_tensor = post_forward(output_tensor)

            if bwd_model_chunk_id is not None:
                # backward pass
                if pre_backward is not None:
                    pre_backward()

                input_tensor_grad = backward_step_helper(bwd_model_chunk_id, compute_wgrad=(not block_level_wgrad_compute))
                cur_bwd_chunk_microbatch[bwd_model_chunk_id] += 1
                if post_backward is not None:
                    input_tensor_grad = post_backward(input_tensor_grad)

            if bwd_model_chunk_id is not None and block_level_wgrad_compute:
                def chunk_backward_dw():
                    model[bwd_model_chunk_id].backward_dw()
                return output_tensor, input_tensor_grad, chunk_backward_dw

            return output_tensor, input_tensor_grad, None

    output_tensor = None
    fwd_recv_buffer = [None]
    bwd_recv_buffer = [None]
    fwd_wait_recv_handles = [None, None]
    fwd_wait_send_handles = [None, None]
    bwd_wait_recv_handles = [None, None]
    bwd_wait_send_handles = [None, None]
    checkpoint_activations_microbatch = None

    # Run warmup forward passes
    input_tensor, _ = recv_forward(tensor_shape, config, master_chunk_id)
    input_tensors[master_chunk_id].append(input_tensor)
    is_slave_only = False
    for _ in range(schedule['warmup'][rank]):
        wait_comm_handle(fwd_wait_recv_handles[master_chunk_id])

        # recv for next iteration
        if cur_fwd_chunk_microbatch[master_chunk_id] + 1 < num_chunk_max_microbatch[master_chunk_id]:
            fwd_recv_buffer[0], fwd_wait_handles = recv_forward(tensor_shape, config, master_chunk_id, async_op=True)
            fwd_wait_recv_handles[master_chunk_id] = get_recv_handle(fwd_wait_handles, master_chunk_id, forward=True)
            input_tensors[master_chunk_id].append(fwd_recv_buffer[0])
            fwd_recv_buffer[0] = None

        output_tensor, _, _ = forward_backward_helper_wrapper(
            fwd_model_chunk_id=master_chunk_id,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        wait_comm_handle(fwd_wait_send_handles[master_chunk_id])
        fwd_wait_handles = send_forward(output_tensor, tensor_shape, config, master_chunk_id, async_op=True)
        fwd_wait_send_handles[master_chunk_id] = get_send_handle(fwd_wait_handles, master_chunk_id, forward=True)
        if not forward_only:
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        is_slave_only = (cur_fwd_chunk_microbatch[master_chunk_id] == num_chunk_max_microbatch[master_chunk_id])
        if is_slave_only:
            break

    # Run interleaved forward passes for two model chunk
    for i in range(schedule['interleaved_forward'][rank]):
        # recv input for slave chunk
        if not parallel_state.is_pipeline_last_stage():
            fwd_recv_buffer[0], fwd_wait_handles = recv_forward(tensor_shape, config, slave_chunk_id, async_op=True)
            fwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(fwd_wait_handles, slave_chunk_id, forward=True)
            input_tensors[slave_chunk_id].append(fwd_recv_buffer[0])
            fwd_recv_buffer[0] = None

        # master forward
        is_slave_only = (cur_fwd_chunk_microbatch[master_chunk_id] == num_chunk_max_microbatch[master_chunk_id])
        if not is_slave_only:
            wait_comm_handle(fwd_wait_recv_handles[master_chunk_id])
            output_tensor, _, _ = forward_backward_helper_wrapper(
                fwd_model_chunk_id=master_chunk_id,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )

            if not parallel_state.is_pipeline_last_stage():
                wait_comm_handle(fwd_wait_send_handles[master_chunk_id])
                fwd_wait_handles = send_forward(output_tensor, tensor_shape, config, master_chunk_id, async_op=True)
                if not forward_only:
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                fwd_wait_send_handles[master_chunk_id] = get_send_handle(fwd_wait_handles, master_chunk_id, forward=True)

        # recv input for master clunk
        if cur_fwd_chunk_microbatch[master_chunk_id] < num_chunk_max_microbatch[master_chunk_id]:
            fwd_recv_buffer[0], fwd_wait_handles = recv_forward(tensor_shape, config, master_chunk_id, async_op=True)
            fwd_wait_recv_handles[master_chunk_id] = get_recv_handle(fwd_wait_handles, master_chunk_id, forward=True)
            input_tensors[master_chunk_id].append(fwd_recv_buffer[0])
            fwd_recv_buffer[0] = None

        # prepare input for slave chunk
        if parallel_state.is_pipeline_last_stage():
            if not forward_only:
                input_tensor = output_tensor.detach()
                input_tensor.requires_grad = True
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            else:
                input_tensor = output_tensor
            input_tensors[slave_chunk_id].append(input_tensor)
        else:
            wait_comm_handle(fwd_wait_recv_handles[slave_chunk_id])

        # slave forward
        output_tensor, _, _ = forward_backward_helper_wrapper(
            fwd_model_chunk_id=slave_chunk_id,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        wait_comm_handle(fwd_wait_send_handles[slave_chunk_id])
        fwd_wait_handles = send_forward(output_tensor, tensor_shape, config, slave_chunk_id, async_op=True)
        if not forward_only:
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        fwd_wait_send_handles[slave_chunk_id] = get_send_handle(fwd_wait_handles, slave_chunk_id, forward=True)

    # check whether data transmission is completed.
    wait_comm_handle(fwd_wait_send_handles[master_chunk_id])

    # Run 1b1w1f stages for slave chunk
    if not forward_only:
        if parallel_state.is_pipeline_last_stage():
            bwd_recv_buffer[0], bwd_wait_handles = recv_backward(tensor_shape, config, slave_chunk_id, async_op=True)
            bwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(bwd_wait_handles, slave_chunk_id, forward=False)
            output_tensor_grads[slave_chunk_id].append(bwd_recv_buffer[0])
            bwd_recv_buffer[0] = None
        else:
            output_tensor_grad, _ = recv_backward(tensor_shape, config, slave_chunk_id)
            output_tensor_grads[slave_chunk_id].append(output_tensor_grad)

    for i in range(schedule['1b1w1f'][rank]):
        fwd_recv_buffer[0], fwd_wait_handles = recv_forward(tensor_shape, config, slave_chunk_id, async_op=True)
        fwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(fwd_wait_handles, slave_chunk_id, forward=True)
        input_tensors[slave_chunk_id].append(fwd_recv_buffer[0])
        fwd_recv_buffer[0] = None

        if not forward_only:
            _, input_tensor_grad, chunk_backward_dw_func = forward_backward_helper_wrapper(
                bwd_model_chunk_id=slave_chunk_id,
                block_level_wgrad_compute=True,
            )
            bwd_wait_handles = send_backward(input_tensor_grad, tensor_shape, config, slave_chunk_id, async_op=True)
            bwd_wait_send_handles[slave_chunk_id] = get_send_handle(bwd_wait_handles, slave_chunk_id, forward=False)
            if chunk_backward_dw_func is not None:
                chunk_backward_dw_func()
                del chunk_backward_dw_func
            wait_comm_handle(bwd_wait_send_handles[slave_chunk_id])

        # foward
        wait_comm_handle(fwd_wait_recv_handles[slave_chunk_id])
        output_tensor, _, _ = forward_backward_helper_wrapper(
            fwd_model_chunk_id=slave_chunk_id,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        wait_comm_handle(fwd_wait_send_handles[slave_chunk_id])
        fwd_wait_handles = send_forward(output_tensor, tensor_shape, config, slave_chunk_id, async_op=True)
        if not forward_only:
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        fwd_wait_send_handles[slave_chunk_id] = get_send_handle(fwd_wait_handles, slave_chunk_id, forward=True)

        if not forward_only:
            output_tensor_grad, _ = recv_backward(tensor_shape, config, slave_chunk_id)
            output_tensor_grads[slave_chunk_id].append(output_tensor_grad)

    # check whether forward data transmission is completed.
    wait_comm_handle(fwd_wait_send_handles[slave_chunk_id])

    # Run overlaping f&bw stages
    prev_step_backward_only = False
    fwd_model_chunk_id = master_chunk_id
    bwd_model_chunk_id = slave_chunk_id
    num_overlap_steps = schedule['overlap'][rank] + schedule['1b1overlap'][rank]
    if not forward_only:
        num_overlap_steps += schedule['interleaved_backward'][rank]
    for step_id in range(num_overlap_steps):
        # print_rank_0(f"num_overlap_steps: {step_id}")
        only_bwd = False
        if cur_fwd_chunk_microbatch[fwd_model_chunk_id] == num_chunk_max_microbatch[fwd_model_chunk_id]:
            only_bwd = True

        def pp_pre_forward():
            nonlocal fwd_wait_recv_handles

            # wait input for current step
            wait_comm_handle(fwd_wait_recv_handles[fwd_model_chunk_id])
            if not forward_only:
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        def pp_post_forward(output_tensor):
            nonlocal cur_fwd_chunk_microbatch
            nonlocal num_chunk_max_microbatch
            nonlocal fwd_wait_send_handles

            # Check whether the forward data transmission is completed.
            if not prev_step_backward_only:
                wait_comm_handle(fwd_wait_send_handles[bwd_model_chunk_id])

            if fwd_model_chunk_id == master_chunk_id:
                fwd_send_only = False
            else:
                fwd_send_only = (cur_fwd_chunk_microbatch[master_chunk_id] == num_chunk_max_microbatch[master_chunk_id])

            if fwd_send_only:
                fwd_wait_handles = send_forward(output_tensor, tensor_shape, config, fwd_model_chunk_id, async_op=True)
                fwd_wait_send_handles[fwd_model_chunk_id] = get_send_handle(fwd_wait_handles, fwd_model_chunk_id, forward=True)
            else:
                if parallel_state.is_pipeline_last_stage() and fwd_model_chunk_id == master_chunk_id:
                    if not forward_only:
                        input_tensor = output_tensor.detach()
                        input_tensor.requires_grad = True
                    else:
                        input_tensor = output_tensor
                    input_tensors[1 - fwd_model_chunk_id].append(input_tensor)
                else:
                    fwd_recv_buffer[0], fwd_wait_send_recv_handles = send_forward_recv_slave_forward(
                        output_tensor, tensor_shape, config, fwd_model_chunk_id, async_op=True)
                    fwd_wait_send_handles[fwd_model_chunk_id] = get_send_handle(fwd_wait_send_recv_handles, fwd_model_chunk_id, forward=True)
                    fwd_wait_recv_handles[bwd_model_chunk_id] = get_recv_handle(fwd_wait_send_recv_handles, bwd_model_chunk_id, forward=True)
                    input_tensors[1 - fwd_model_chunk_id].append(fwd_recv_buffer[0])
                    fwd_recv_buffer[0] = None

            return output_tensor

        def pp_pre_backward():
            nonlocal bwd_wait_recv_handles

            if not forward_only:
                wait_comm_handle(bwd_wait_recv_handles[bwd_model_chunk_id])

        def pp_post_backward(input_tensor_grad):
            nonlocal output_tensor_grads
            nonlocal bwd_wait_send_handles
            nonlocal bwd_wait_recv_handles

            if not forward_only:
                if parallel_state.is_pipeline_last_stage() and fwd_model_chunk_id == master_chunk_id:
                    output_tensor_grad = input_tensor_grad
                    output_tensor_grads[fwd_model_chunk_id].append(output_tensor_grad)
                    input_tensor_grad = None
                else:
                    if parallel_state.is_pipeline_first_stage() and fwd_model_chunk_id == slave_chunk_id:
                        input_tensor_grad = None

                    wait_comm_handle(bwd_wait_send_handles[fwd_model_chunk_id])
                    bwd_recv_buffer[0], bwd_wait_send_recv_handles = send_backward_recv_slave_backward(
                        input_tensor_grad,
                        tensor_shape,
                        config,
                        fwd_model_chunk_id,
                        async_op=True
                    )
                    bwd_wait_send_handles[bwd_model_chunk_id] = get_send_handle(bwd_wait_send_recv_handles, bwd_model_chunk_id, forward=False)
                    bwd_wait_recv_handles[fwd_model_chunk_id] = get_recv_handle(bwd_wait_send_recv_handles, fwd_model_chunk_id, forward=False)
                    output_tensor_grads[fwd_model_chunk_id].append(bwd_recv_buffer[0])
                    bwd_recv_buffer[0] = None

            return input_tensor_grad

        if not only_bwd:
            if step_id == 0 and parallel_state.is_pipeline_last_stage():
                if cur_fwd_chunk_microbatch[master_chunk_id] < num_chunk_max_microbatch[master_chunk_id]:
                    wait_comm_handle(fwd_wait_recv_handles[master_chunk_id])
                    output_tensor, _, _ = forward_backward_helper_wrapper(
                        fwd_model_chunk_id=master_chunk_id,
                        checkpoint_activations_microbatch=checkpoint_activations_microbatch,
                    )
                    if not forward_only:
                        input_tensor = output_tensor.detach()
                        input_tensor.requires_grad = True
                    else:
                        input_tensor = output_tensor
                    input_tensors[slave_chunk_id].append(input_tensor)

                if not forward_only:
                    _, input_tensor_grad, _ = forward_backward_helper_wrapper(
                        bwd_model_chunk_id=bwd_model_chunk_id,
                        pre_backward=pp_pre_backward,
                        post_backward=pp_post_backward,
                    )
            else:
                output_tensor, input_tensor_grad, _ = forward_backward_helper_wrapper(
                    fwd_model_chunk_id=fwd_model_chunk_id,
                    bwd_model_chunk_id=None if forward_only else bwd_model_chunk_id,
                    pre_forward=pp_pre_forward,
                    pre_backward=pp_pre_backward,
                    post_forward=pp_post_forward,
                    post_backward=pp_post_backward,
                    checkpoint_activations_microbatch=checkpoint_activations_microbatch,
                )

        # only run backward
        else:
            # Check whether the forward data transmission is completed.
            if not prev_step_backward_only:
                wait_comm_handle(fwd_wait_send_handles[bwd_model_chunk_id])
                if not forward_only:
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            if bwd_model_chunk_id == slave_chunk_id and cur_fwd_chunk_microbatch[slave_chunk_id] < num_chunk_max_microbatch[slave_chunk_id]:
                fwd_recv_buffer[0], fwd_wait_handles = recv_forward(tensor_shape, config, slave_chunk_id, async_op=True)
                fwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(fwd_wait_handles, slave_chunk_id, forward=True)
                input_tensors[slave_chunk_id].append(fwd_recv_buffer[0])
                fwd_recv_buffer[0] = None

            if not forward_only:
                _, input_tensor_grad, chunk_backward_dw_func = forward_backward_helper_wrapper(
                    bwd_model_chunk_id=bwd_model_chunk_id,
                    pre_backward=pp_pre_backward,
                    post_backward=pp_post_backward,
                )

        # swap fwd & bwd chunks
        fwd_model_chunk_id, bwd_model_chunk_id = bwd_model_chunk_id, fwd_model_chunk_id
        prev_step_backward_only = only_bwd

    # Run cooldown phases
    if not forward_only:
        if rank == 0:
            # launch grad reductions.
            if config.grad_sync_func is not None:
                enable_grad_sync()
                config.grad_sync_func[slave_chunk_id](model[slave_chunk_id].parameters())
                disable_grad_sync()

        chunk_backward_dw_funcs = []
        for i in range(schedule['cooldown'][rank][0]):
            wait_comm_handle(bwd_wait_recv_handles[bwd_model_chunk_id])

            _, input_tensor_grad, chunk_backward_dw_func = forward_backward_helper_wrapper(
                bwd_model_chunk_id=bwd_model_chunk_id,
                block_level_wgrad_compute=True,
            )
            chunk_backward_dw_funcs.append((chunk_backward_dw_func, i == schedule['cooldown'][rank][0] - 1))

            if parallel_state.is_pipeline_last_stage() and bwd_model_chunk_id == slave_chunk_id:
                output_tensor_grad = input_tensor_grad
                output_tensor_grads[1 - bwd_model_chunk_id].append(output_tensor_grad)
            else:
                wait_comm_handle(bwd_wait_send_handles[1 - bwd_model_chunk_id])
                bwd_recv_buffer[0], bwd_wait_send_recv_handles = send_backward_recv_slave_backward(
                    input_tensor_grad,
                    tensor_shape,
                    config,
                    1 - bwd_model_chunk_id,
                    async_op=False,
                )
                bwd_wait_send_handles[bwd_model_chunk_id] = get_send_handle(bwd_wait_send_recv_handles, bwd_model_chunk_id, forward=False)
                bwd_wait_recv_handles[1 - bwd_model_chunk_id] = get_recv_handle(bwd_wait_send_recv_handles, 1 - bwd_model_chunk_id, forward=False)
                output_tensor_grads[1 - bwd_model_chunk_id].append(bwd_recv_buffer[0])
                bwd_recv_buffer[0] = None

            # swap bwd chunks
            bwd_model_chunk_id = 1 - bwd_model_chunk_id

        wait_comm_handle(bwd_wait_send_handles[1 - bwd_model_chunk_id])
        wait_comm_handle(bwd_wait_recv_handles[bwd_model_chunk_id])
        # nB0W
        for i in range(schedule['cooldown'][rank][1]):
            _, input_tensor_grad, chunk_backward_dw_func = forward_backward_helper_wrapper(
                bwd_model_chunk_id=bwd_model_chunk_id,
                block_level_wgrad_compute=True,
            )
            chunk_backward_dw_funcs.append((chunk_backward_dw_func, False))

            bwd_wait_handles = send_backward(input_tensor_grad, tensor_shape, config, master_chunk_id, async_op=True)
            bwd_wait_send_handles[master_chunk_id] = get_send_handle(bwd_wait_handles, master_chunk_id, forward=False)
            # weight backward
            chunk_backward_dw_func, is_last_slave_chunk = chunk_backward_dw_funcs.pop(0)
            if chunk_backward_dw_func is not None:
                chunk_backward_dw_func()
                del chunk_backward_dw_func
            wait_comm_handle(bwd_wait_send_handles[master_chunk_id])

            if is_last_slave_chunk:
                # launch grad reductions.
                if config.grad_sync_func is not None:
                    enable_grad_sync()
                    config.grad_sync_func[slave_chunk_id](model[slave_chunk_id].parameters())
                    disable_grad_sync()

            if i < schedule['cooldown'][rank][1] - 1:
                output_tensor_grad, _ = recv_backward(tensor_shape, config, master_chunk_id, async_op=False)
                output_tensor_grads[master_chunk_id].append(output_tensor_grad)

        # nW
        for i in range(schedule['cooldown'][rank][2]):
            chunk_backward_dw_func, is_last_slave_chunk = chunk_backward_dw_funcs.pop(0)
            if chunk_backward_dw_func is not None:
                chunk_backward_dw_func()
                del chunk_backward_dw_func
            if is_last_slave_chunk:
                # Launch grad reductions.
                if config.grad_sync_func is not None:
                    enable_grad_sync()
                    config.grad_sync_func[slave_chunk_id](model[slave_chunk_id].parameters())
                    disable_grad_sync()

    # launch any remaining grad reductions.
    if config.grad_sync_func is not None:
        enable_grad_sync()
        config.grad_sync_func[master_chunk_id](model[master_chunk_id].parameters())

    if config.finalize_model_grads_func is not None and not forward_only:
        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )

    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    return forward_data_store
