# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import contextlib
from typing import Iterator, List, Union, Optional, Callable

import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
)
from megatron.core.pipeline_parallel.schedules import clear_embedding_activation_buffer, deallocate_output_tensor
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    set_current_microbatch,
    check_first_val_step,
    finish_embedding_wgrad_compute
)
from megatron.core.pipeline_parallel.utils import (
    set_streams,
    is_pp_first_stage,
    is_pp_last_stage,
)
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.paged_stash import paged_stash_reset
from megatron.training import get_args

from ..combined_1f1b import combined_forward_backward_step
from hcu_megatron.core.models.common.language_module.language_module import set_shared_embedding_from_dual_chunk
from hcu_megatron.core.parallel_state import set_dualpipe_chunk
from hcu_megatron.core.pipeline_parallel.schedules import forward_step_calc_loss
from hcu_megatron.core.tensor_parallel.vocab_input_store import VocabInputStore
from hcu_megatron.core.transformer.enums import DualpipeVChunkType


# Types
Shape = Union[List[int], torch.Size]


def is_dualpipev_last_stage(model_chunk_id, is_first_stage):
    loss_chunk = DualpipeVChunkType.loss if get_args().enable_vocab_parallel else DualpipeVChunkType.second_block
    return is_first_stage and model_chunk_id == loss_chunk.value


class DualpipeVP2PCommunicator(P2PCommunicator):
    def send_forward(self, output_tensor: torch.Tensor, tensor_shape, model_chunk_id, async_op=False):
        """Send tensor to next rank in pipeline (forward send).

        See _communicate for argument details.
        """
        config = self.config
        tensor_send_next, tensor_send_prev = None, None
        if model_chunk_id == 0:
            if is_pp_last_stage(self.pp_group):
                return None
            tensor_send_next = output_tensor
        else:
            if is_pp_first_stage(self.pp_group):
                return None
            tensor_send_prev = output_tensor

        if config.timers is not None:
            config.timers('forward-send', log_level=2).start()

        _, _, fwd_wait_handles = self._communicate(
            tensor_send_next=tensor_send_next,
            tensor_send_prev=tensor_send_prev,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not async_op)
        )
        if config.timers is not None:
            config.timers('forward-send').stop()

        return fwd_wait_handles

    def send_forward_recv_forward(
        self,
        output_tensor: torch.Tensor,
        tensor_shape: Shape,
        model_chunk_id,
        async_op=False,
    ) -> torch.Tensor:
        """Batched recv from previous rank and send to next rank in pipeline.

        See _communicate for argument details.
        """
        config = self.config
        recv_prev, recv_next = False, False
        tensor_send_next, tensor_send_prev = None, None
        if model_chunk_id == 0:
            if not is_pp_last_stage(self.pp_group):
                tensor_send_next = output_tensor
            if not is_pp_first_stage(self.pp_group):
                recv_prev = True

        if model_chunk_id == 1:
            if not is_pp_first_stage(self.pp_group):
                tensor_send_prev = output_tensor
            if not is_pp_last_stage(self.pp_group):
                recv_next = True

        if config.timers is not None:
            config.timers('forward-send-forward-recv', log_level=2).start()
        tensor_recv_prev, tensor_recv_next, fwd_wait_handles = self._communicate(
            tensor_send_next=tensor_send_next,
            tensor_send_prev=tensor_send_prev,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not async_op),
        )
        if config.timers is not None:
            config.timers('forward-send-forward-recv').stop()

        if model_chunk_id == 0:
            if not is_pp_first_stage(self.pp_group):
                return tensor_recv_prev, fwd_wait_handles
            elif get_args().enable_vocab_parallel:
                tensor_recv_prev = VocabInputStore.forward_get(remove=False)
                return tensor_recv_prev, fwd_wait_handles
            else:
                return None, fwd_wait_handles
        else:
            if not is_pp_last_stage(self.pp_group):
                return tensor_recv_next, fwd_wait_handles
            else:
                return None, fwd_wait_handles

    def send_backward(self, input_tensor_grad: torch.Tensor, tensor_shape, model_chunk_id, async_op=False):
        """Send tensor to next rank in pipeline (forward send).

        See _communicate for argument details.
        """
        config = self.config
        tensor_send_next, tensor_send_prev = None, None
        if model_chunk_id == 0:
            if is_pp_first_stage(self.pp_group):
                return None
            tensor_send_prev = input_tensor_grad
        else:
            if is_pp_last_stage(self.pp_group):
                return None
            tensor_send_next = input_tensor_grad

        if config.timers is not None:
            config.timers('backward-send', log_level=2).start()
        _, _, reqs = self._communicate(
            tensor_send_next=tensor_send_next,
            tensor_send_prev=tensor_send_prev,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not async_op)
        )
        if config.timers is not None:
            config.timers('backward-send').stop()
        return reqs

    def recv_forward(self, tensor_shape: Shape, model_chunk_id, async_op=False):
        """ Receive tensor from previous rank in pipeline (forward receive).

        See _communicate for argument details.
        """
        config = self.config
        recv_prev, recv_next = False, False
        if model_chunk_id == 0:
            recv_prev = True
        else:
            recv_next = True

        if is_pp_first_stage(self.pp_group) and recv_prev:
            tensor_recv_prev = None
            if get_args().enable_vocab_parallel:
                tensor_recv_prev = VocabInputStore.forward_get(remove=False)
            return tensor_recv_prev, None
        elif is_pp_last_stage(self.pp_group) and recv_next:
            fwd_wait_handles = None
            return None, fwd_wait_handles
        else:
            if config.timers is not None:
                config.timers('forward-recv', log_level=2).start()
            tensor_recv_prev, tensor_recv_next, fwd_wait_handles = self._communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                wait_on_reqs=(not async_op),
            )
            if config.timers is not None:
                config.timers('forward-recv').stop()

        if recv_prev:
            return tensor_recv_prev, fwd_wait_handles
        else:
            return tensor_recv_next, fwd_wait_handles

    def recv_backward(self, tensor_shape: Shape, model_chunk_id, async_op=False):
        """Receive tensor from next rank in pipeline (backward receive).

        See _communicate for argument details.
        """
        config = self.config
        recv_prev, recv_next = False, False
        if model_chunk_id == 0:
            recv_next = True
        else:
            recv_prev = True

        if (
            (is_pp_first_stage(self.pp_group) and recv_prev)
            or (is_pp_last_stage(self.pp_group) and recv_next)
        ):
            output_tensor_grad = None
            bwd_wait_handles = None
            return output_tensor_grad, bwd_wait_handles
        else:

            if config.timers is not None:
                config.timers('backward-recv', log_level=2).start()
            tensor_recv_prev, tensor_recv_next, bwd_wait_handles = self._communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                wait_on_reqs=(not async_op)
            )
            if config.timers is not None:
                config.timers('backward-recv').stop()

        if recv_prev:
            return tensor_recv_prev, bwd_wait_handles
        else:
            return tensor_recv_next, bwd_wait_handles

    def send_forward_recv_slave_forward(
        self,
        output_tensor: torch.Tensor,
        tensor_shape: Shape,
        fwd_model_chunk_id,
        async_op=False,
    ) -> torch.Tensor:
        """Batched recv from previous rank and send to next rank in pipeline.
        See _communicate for argument details.
        """
        config = self.config
        recv_prev, recv_next = False, False
        tensor_send_next, tensor_send_prev = None, None
        if fwd_model_chunk_id == 0:
            if is_pp_last_stage(self.pp_group):
                return None, None
            tensor_send_next = output_tensor
            recv_next = True
        if fwd_model_chunk_id == 1:
            if is_pp_first_stage(self.pp_group):
                tensor_recv_prev = None
                if get_args().enable_vocab_parallel:
                    tensor_recv_prev = VocabInputStore.forward_get(remove=False)
                return tensor_recv_prev, None
            tensor_send_prev = output_tensor
            recv_prev = True
        if config.timers is not None:
            config.timers('forward-send-slave-forward-recv', log_level=2).start()
        tensor_recv_prev, tensor_recv_next, fwd_wait_handles = self._communicate(
            tensor_send_next=tensor_send_next,
            tensor_send_prev=tensor_send_prev,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not async_op),
        )
        if config.timers is not None:
            config.timers('forward-send-slave-forward-recv').stop()

        if fwd_model_chunk_id == 0:
            return tensor_recv_next, fwd_wait_handles
        else:
            return tensor_recv_prev, fwd_wait_handles

    def send_backward_recv_slave_backward(
        self,
        input_tensor_grad: torch.Tensor,
        tensor_shape: Shape,
        fwd_model_chunk_id,
        async_op=False,
    ) -> torch.Tensor:
        """Batched recv from previous rank and send to next rank in pipeline.
        See _communicate for argument details.
        """
        config = self.config
        recv_prev, recv_next = False, False
        tensor_send_next, tensor_send_prev = None, None
        if fwd_model_chunk_id == 0:
            if is_pp_last_stage(self.pp_group):
                return None, None
            tensor_send_next = input_tensor_grad
            recv_next = True
        if fwd_model_chunk_id == 1:
            if is_pp_first_stage(self.pp_group):
                return None, None
            tensor_send_prev = input_tensor_grad
            recv_prev = True
        if config.timers is not None:
            config.timers('forward-send-slave-forward-recv', log_level=2).start()
        tensor_recv_prev, tensor_recv_next, fwd_wait_handles = self._communicate(
            tensor_send_next=tensor_send_next,
            tensor_send_prev=tensor_send_prev,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not async_op),
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

    args = get_args()
    pp_size *= 2
    for i in range(pp_size // 2):
        num_warmup_stages[i] = pp_size - 2 - i * 2

        num_interleaved_forward_stages[i] = i + 1  # 1f1f

        num_1b1w1f_stages[i] = [pp_size // 2 - i - 1, False] # True: 1B1W1F. False: 1B1F, finally Ws

        num_overlap_stages[i] = num_microbatches - pp_size * 2 + i * 2 + 2

        num_1b1overlap_stages[i] = (pp_size // 2 - i - 1) * 2

        num_interleaved_backward_stages[i] = i + 1

        num_cooldown_stages[i] = [i, pp_size // 2 - i, i]

        if args.enable_vocab_parallel:
            if args.disable_backward_fusion:
                num_interleaved_forward_stages[i] += 1
                num_overlap_stages[i] -= 2
                num_interleaved_backward_stages[i] += 2

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
    cp_group_size,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    is_first_stage=False,
    skip_loss_compute=False,
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
        if model_chunk_id == 1:
            current_microbatch -= num_microbatches

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
        None,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size,
        is_dualpipev_last_stage(model_chunk_id, is_first_stage),
        skip_loss_compute=skip_loss_compute,
    )

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


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
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    config = get_model_config(model[0])
    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = DualpipeVP2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )

    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model[0])
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection"
            " provide none or provide all the process groups"
        )

    if is_pp_first_stage(p2p_communicator.pp_group):
        set_shared_embedding_from_dual_chunk(model[0], model[1])

    assert (
        isinstance(model, list) and len(model) == 2
    ), 'Dualpipe Schedule expects two model chunks'

    assert (
        isinstance(data_iterator, list) and len(data_iterator) == 2
    ), 'Dualpipe Schedule expects two data_iterators'

    config = get_model_config(model[0])
    config.batch_p2p_comm = False

    if getattr(config, "moe_paged_stash", False):
        paged_stash_reset(enabled=not forward_only, config=config)

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_first_stage(p2p_communicator.pp_group)
        )

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
    pp_size = p2p_communicator.pp_group.size()
    rank = p2p_communicator.pp_group.rank()
    schedule = generate_dualpipev_schedule(pp_size, num_microbatches)

    model_type = get_model_type(model[0])

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // cp_group.size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // tp_group.size()

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

        first_microbatch_id = 0 if model_chunk_id == master_chunk_id else num_microbatches
        is_first_microbatch = check_first_val_step(
            first_val_step,
            forward_only,
            cur_microbatch == first_microbatch_id,
        )
        output_tensor, num_tokens = forward_step_no_model_graph(
            forward_step_func,
            model_chunk_id,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=is_first_microbatch,
            current_microbatch=cur_microbatch,
            is_first_stage=is_pp_first_stage(pp_group),
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
            input_tensor, output_tensor, output_tensor_grad, config
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

        set_streams(high_priority=config.high_priority_a2a_comm_stream)
        # forward prepare
        fwd_input_tensor = None
        fwd_microbatch_id = None
        if fwd_model_chunk_id is not None:
            fwd_microbatch_id = cur_fwd_chunk_microbatch[fwd_model_chunk_id]
            set_dualpipe_chunk(fwd_model_chunk_id)
            offset = cur_bwd_chunk_microbatch[fwd_model_chunk_id]
            fwd_input_tensor = input_tensors[fwd_model_chunk_id][fwd_microbatch_id - offset]

        # backward prepare
        bwd_input_tensor = None
        bwd_output_tensor = None
        bwd_output_tensor_grad = None
        if bwd_model_chunk_id is not None:
            bwd_input_tensor = input_tensors[bwd_model_chunk_id].pop(0)
            bwd_output_tensor = output_tensors[bwd_model_chunk_id].pop(0)
            bwd_output_tensor_grad = output_tensor_grads[bwd_model_chunk_id].pop(0)

        output_tensor, num_tokens, input_tensor_grad, chunk_backward_dw_func = combined_forward_backward_step(
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
        disable_overlap_moe_expert_parallel_comm=False,
    ):
        """
        wrap forward_helper、backward_helper、combined_forward_backward_helper in a unified way
        """

        if config.overlap_moe_expert_parallel_comm and not forward_only:
            assert (
                checkpoint_activations_microbatch is None
            ), "checkpoint_activations_microbatch not supported when overlap_moe_expert_parallel_comm is true"

            if disable_overlap_moe_expert_parallel_comm:
                output_tensor = None
                if fwd_model_chunk_id is not None:
                    output_tensor, _, _ = combined_forward_backward_helper(
                        fwd_model_chunk_id=fwd_model_chunk_id,
                        pre_forward=pre_forward,
                        post_forward=post_forward,
                    )

                input_tensor_grad, chunk_backward_dw_func = None, None
                if bwd_model_chunk_id is not None:
                    _, input_tensor_grad, chunk_backward_dw_func = combined_forward_backward_helper(
                        bwd_model_chunk_id=bwd_model_chunk_id,
                        pre_backward=pre_backward,
                        post_backward=post_backward,
                        block_level_wgrad_compute=block_level_wgrad_compute,
                    )

                return output_tensor, input_tensor_grad, chunk_backward_dw_func

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
    input_tensor, _ = p2p_communicator.recv_forward(tensor_shape, master_chunk_id)
    input_tensors[master_chunk_id].append(input_tensor)
    is_slave_only = False
    fwd_wait_handles_warmup = None
    for i in range(schedule['warmup'][rank]):
        wait_comm_handle(fwd_wait_recv_handles[master_chunk_id])

        output_tensor, _, _ = forward_backward_helper_wrapper(
            fwd_model_chunk_id=master_chunk_id,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        is_last_warmup_step = (i == schedule['warmup'][rank] - 1)
        if cur_fwd_chunk_microbatch[master_chunk_id] < num_chunk_max_microbatch[master_chunk_id]:
            if not is_last_warmup_step:
                input_tensor, _ = p2p_communicator.send_forward_recv_forward(
                    output_tensor, tensor_shape, master_chunk_id)
            else:
                input_tensor, _ = p2p_communicator.recv_forward(tensor_shape, master_chunk_id)
                fwd_wait_handles = p2p_communicator.send_forward(output_tensor, tensor_shape, master_chunk_id, async_op=True)
                fwd_wait_handles_warmup = get_send_handle(fwd_wait_handles, master_chunk_id, forward=True)

            input_tensors[master_chunk_id].append(input_tensor)

            if not forward_only:
                deallocate_output_tensor(
                    output_tensor, config.deallocate_pipeline_outputs)

            continue

        fwd_wait_handles = p2p_communicator.send_forward(output_tensor, tensor_shape, master_chunk_id, async_op=is_last_warmup_step)
        if is_last_warmup_step:
            fwd_wait_handles_warmup = get_send_handle(fwd_wait_handles, master_chunk_id, forward=True)
        if not forward_only:
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        break

    # Run interleaved forward passes for two model chunk
    for i in range(schedule['interleaved_forward'][rank]):
        # master forward
        is_slave_only = (cur_fwd_chunk_microbatch[master_chunk_id] == num_chunk_max_microbatch[master_chunk_id])
        if not is_slave_only:
            wait_comm_handle(fwd_wait_recv_handles[master_chunk_id])
            output_tensor, _, _ = forward_backward_helper_wrapper(
                fwd_model_chunk_id=master_chunk_id,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )

        # prepare input for slave chunk
        if not is_pp_last_stage(p2p_communicator.pp_group):
            input_tensor, _ = p2p_communicator.recv_forward(tensor_shape, slave_chunk_id, async_op=False)
            input_tensors[slave_chunk_id].append(input_tensor)
        else:
            if not forward_only:
                input_tensor = output_tensor.detach()
                input_tensor.requires_grad = True
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            else:
                input_tensor = output_tensor
            input_tensors[slave_chunk_id].append(input_tensor)

        # recv input for master clunk
        if cur_fwd_chunk_microbatch[master_chunk_id] < num_chunk_max_microbatch[master_chunk_id]:
            fwd_recv_buffer[0], fwd_wait_handles = p2p_communicator.recv_forward(tensor_shape, master_chunk_id, async_op=True)
            fwd_wait_recv_handles[master_chunk_id] = get_recv_handle(fwd_wait_handles, master_chunk_id, forward=True)
            input_tensors[master_chunk_id].append(fwd_recv_buffer[0])
            fwd_recv_buffer[0] = None

        if fwd_wait_handles_warmup is not None:
            wait_comm_handle(fwd_wait_handles_warmup)
            fwd_wait_handles_warmup = None

        # slave forward
        output_tensor_slave, _, _ = forward_backward_helper_wrapper(
            fwd_model_chunk_id=slave_chunk_id,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        wait_comm_handle(fwd_wait_send_handles[slave_chunk_id])
        fwd_wait_handles = p2p_communicator.send_forward(output_tensor_slave, tensor_shape, slave_chunk_id, async_op=True)
        fwd_wait_send_handles[slave_chunk_id] = get_send_handle(fwd_wait_handles, slave_chunk_id, forward=True)
        if not forward_only:
            deallocate_output_tensor(output_tensor_slave, config.deallocate_pipeline_outputs)

        if not is_slave_only:
            if not parallel_state.is_pipeline_last_stage():
                wait_comm_handle(fwd_wait_send_handles[master_chunk_id])
                fwd_wait_handles = p2p_communicator.send_forward(output_tensor, tensor_shape, master_chunk_id, async_op=True)
                fwd_wait_send_handles[master_chunk_id] = get_send_handle(fwd_wait_handles, master_chunk_id, forward=True)
                if not forward_only:
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # check whether data transmission is completed.
    wait_comm_handle(fwd_wait_send_handles[master_chunk_id])

    # Run 1b1w1f stages for slave chunk
    if not forward_only and not is_pp_last_stage(p2p_communicator.pp_group):
        output_tensor_grad, _ = p2p_communicator.recv_backward(tensor_shape, slave_chunk_id)
        output_tensor_grads[slave_chunk_id].append(output_tensor_grad)

    for i in range(schedule['1b1w1f'][rank][0]):
        if not forward_only:
            _, input_tensor_grad, chunk_backward_dw_func = forward_backward_helper_wrapper(
                bwd_model_chunk_id=slave_chunk_id,
                block_level_wgrad_compute=True,
            )

            bwd_wait_handles = p2p_communicator.send_backward(input_tensor_grad, tensor_shape, slave_chunk_id, async_op=True)
            bwd_wait_send_handles[slave_chunk_id] = get_send_handle(bwd_wait_handles, slave_chunk_id, forward=False)

            fwd_recv_buffer[0], fwd_wait_handles = p2p_communicator.recv_forward(tensor_shape, slave_chunk_id, async_op=True)
            fwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(fwd_wait_handles, slave_chunk_id, forward=True)
            input_tensors[slave_chunk_id].append(fwd_recv_buffer[0])
            fwd_recv_buffer[0] = None

            if chunk_backward_dw_func is not None:
                chunk_backward_dw_func()
                del chunk_backward_dw_func

            wait_comm_handle(bwd_wait_send_handles[slave_chunk_id])
        else:
            fwd_recv_buffer[0], fwd_wait_handles = p2p_communicator.recv_forward(tensor_shape, slave_chunk_id, async_op=True)
            fwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(fwd_wait_handles, slave_chunk_id, forward=True)
            input_tensors[slave_chunk_id].append(fwd_recv_buffer[0])
            fwd_recv_buffer[0] = None

        # foward
        wait_comm_handle(fwd_wait_recv_handles[slave_chunk_id])
        output_tensor, _, _ = forward_backward_helper_wrapper(
            fwd_model_chunk_id=slave_chunk_id,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        if not forward_only:
            output_tensor_grad, _ = p2p_communicator.recv_backward(tensor_shape, slave_chunk_id)
            output_tensor_grads[slave_chunk_id].append(output_tensor_grad)

        wait_comm_handle(fwd_wait_send_handles[slave_chunk_id])
        fwd_wait_handles = p2p_communicator.send_forward(output_tensor, tensor_shape, slave_chunk_id, async_op=True)
        if not forward_only:
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        fwd_wait_send_handles[slave_chunk_id] = get_send_handle(fwd_wait_handles, slave_chunk_id, forward=True)

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
        from megatron.training import get_args
        disable_overlap_moe_expert_parallel_comm = get_args().enable_a2a_overlap_only_in_1f1b_phase and (step_id>=schedule['overlap'][rank])

        only_bwd = False
        if cur_fwd_chunk_microbatch[fwd_model_chunk_id] == num_chunk_max_microbatch[fwd_model_chunk_id]:
            only_bwd = True

        def pp_pre_forward(vp_stage=None):
            nonlocal fwd_wait_recv_handles

            # wait input for current step
            wait_comm_handle(fwd_wait_recv_handles[fwd_model_chunk_id])
            if not forward_only:
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        def pp_post_forward(output_tensor, vp_stage=None):
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
                fwd_wait_handles = p2p_communicator.send_forward(output_tensor, tensor_shape, fwd_model_chunk_id, async_op=True)
                fwd_wait_send_handles[fwd_model_chunk_id] = get_send_handle(fwd_wait_handles, fwd_model_chunk_id, forward=True)
            else:
                if is_pp_last_stage(p2p_communicator.pp_group) and fwd_model_chunk_id == master_chunk_id:
                    if not forward_only:
                        input_tensor = output_tensor.detach()
                        input_tensor.requires_grad = True
                    else:
                        input_tensor = output_tensor
                    input_tensors[1 - fwd_model_chunk_id].append(input_tensor)
                else:
                    fwd_recv_buffer[0], fwd_wait_send_recv_handles = p2p_communicator.send_forward_recv_slave_forward(
                        output_tensor, tensor_shape, fwd_model_chunk_id, async_op=True)
                    fwd_wait_send_handles[fwd_model_chunk_id] = get_send_handle(fwd_wait_send_recv_handles, fwd_model_chunk_id, forward=True)
                    fwd_wait_recv_handles[bwd_model_chunk_id] = get_recv_handle(fwd_wait_send_recv_handles, bwd_model_chunk_id, forward=True)
                    input_tensors[1 - fwd_model_chunk_id].append(fwd_recv_buffer[0])
                    fwd_recv_buffer[0] = None

            return output_tensor

        def pp_pre_backward(vp_stage=None):
            nonlocal bwd_wait_recv_handles

            if not forward_only:
                wait_comm_handle(bwd_wait_recv_handles[bwd_model_chunk_id])

        def pp_post_backward(input_tensor_grad, vp_stage=None):
            nonlocal output_tensor_grads
            nonlocal bwd_wait_send_handles
            nonlocal bwd_wait_recv_handles

            if not forward_only:
                if is_pp_last_stage(p2p_communicator.pp_group) and fwd_model_chunk_id == master_chunk_id:
                    output_tensor_grad = input_tensor_grad
                    output_tensor_grads[fwd_model_chunk_id].append(output_tensor_grad)
                    input_tensor_grad = None
                else:
                    if is_pp_first_stage(p2p_communicator.pp_group) and fwd_model_chunk_id == slave_chunk_id:
                        input_tensor_grad = None

                    wait_comm_handle(bwd_wait_send_handles[fwd_model_chunk_id])
                    bwd_recv_buffer[0], bwd_wait_send_recv_handles = p2p_communicator.send_backward_recv_slave_backward(
                        input_tensor_grad,
                        tensor_shape,
                        fwd_model_chunk_id,
                        async_op=True,
                    )
                    bwd_wait_send_handles[bwd_model_chunk_id] = get_send_handle(bwd_wait_send_recv_handles, bwd_model_chunk_id, forward=False)
                    bwd_wait_recv_handles[fwd_model_chunk_id] = get_recv_handle(bwd_wait_send_recv_handles, fwd_model_chunk_id, forward=False)
                    output_tensor_grads[fwd_model_chunk_id].append(bwd_recv_buffer[0])
                    bwd_recv_buffer[0] = None

            return input_tensor_grad

        if not only_bwd:
            if step_id == 0 and is_pp_last_stage(p2p_communicator.pp_group):
                if cur_fwd_chunk_microbatch[master_chunk_id] < num_chunk_max_microbatch[master_chunk_id]:
                    output_tensor, _, _ = forward_backward_helper_wrapper(
                        fwd_model_chunk_id=master_chunk_id,
                        checkpoint_activations_microbatch=checkpoint_activations_microbatch,
                        pre_forward=pp_pre_forward,
                        post_forward=pp_post_forward,
                    )

                if not forward_only:
                    bwd_recv_buffer[0], bwd_wait_handles = p2p_communicator.recv_backward(tensor_shape, slave_chunk_id, async_op=True)
                    bwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(bwd_wait_handles, slave_chunk_id, forward=False)
                    output_tensor_grads[slave_chunk_id].append(bwd_recv_buffer[0])
                    bwd_recv_buffer[0] = None
                    _, input_tensor_grad, _ = forward_backward_helper_wrapper(
                        bwd_model_chunk_id=bwd_model_chunk_id,
                        pre_backward=pp_pre_backward,
                        post_backward=pp_post_backward,
                    )
            else:
                output_tensor, input_tensor_grad, chunk_backward_dw_func = forward_backward_helper_wrapper(
                    fwd_model_chunk_id=fwd_model_chunk_id,
                    bwd_model_chunk_id=None if forward_only else bwd_model_chunk_id,
                    pre_forward=pp_pre_forward,
                    pre_backward=pp_pre_backward,
                    post_forward=pp_post_forward,
                    post_backward=pp_post_backward,
                    checkpoint_activations_microbatch=checkpoint_activations_microbatch,
                    block_level_wgrad_compute=(bwd_model_chunk_id == 1 and is_pp_first_stage(p2p_communicator.pp_group)),
                    disable_overlap_moe_expert_parallel_comm=disable_overlap_moe_expert_parallel_comm,
                )

                if chunk_backward_dw_func is not None:
                    chunk_backward_dw_func()
                    del chunk_backward_dw_func

        # only run backward
        else:
            # Check whether the forward data transmission is completed.
            if not prev_step_backward_only:
                wait_comm_handle(fwd_wait_send_handles[bwd_model_chunk_id])
                if not forward_only:
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            if bwd_model_chunk_id == slave_chunk_id and cur_fwd_chunk_microbatch[slave_chunk_id] < num_chunk_max_microbatch[slave_chunk_id]:
                fwd_recv_buffer[0], fwd_wait_handles = p2p_communicator.recv_forward(tensor_shape, slave_chunk_id, async_op=True)
                fwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(fwd_wait_handles, slave_chunk_id, forward=True)
                input_tensors[slave_chunk_id].append(fwd_recv_buffer[0])
                fwd_recv_buffer[0] = None

            if not forward_only:
                if step_id == 0 and is_pp_last_stage(p2p_communicator.pp_group):
                    bwd_recv_buffer[0], bwd_wait_handles = p2p_communicator.recv_backward(tensor_shape, slave_chunk_id, async_op=True)
                    bwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(bwd_wait_handles, slave_chunk_id, forward=False)
                    output_tensor_grads[slave_chunk_id].append(bwd_recv_buffer[0])
                    bwd_recv_buffer[0] = None

                _, input_tensor_grad, _ = forward_backward_helper_wrapper(
                    bwd_model_chunk_id=bwd_model_chunk_id,
                    pre_backward=pp_pre_backward,
                    post_backward=pp_post_backward,
                    block_level_wgrad_compute=False,
                    disable_overlap_moe_expert_parallel_comm=disable_overlap_moe_expert_parallel_comm,
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

            if is_pp_last_stage(p2p_communicator.pp_group) and bwd_model_chunk_id == slave_chunk_id:
                output_tensor_grad = input_tensor_grad
                output_tensor_grads[1 - bwd_model_chunk_id].append(output_tensor_grad)
            else:
                wait_comm_handle(bwd_wait_send_handles[1 - bwd_model_chunk_id])
                bwd_recv_buffer[0], bwd_wait_send_recv_handles = p2p_communicator.send_backward_recv_slave_backward(
                    input_tensor_grad,
                    tensor_shape,
                    1 - bwd_model_chunk_id,
                    async_op=True,
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

            bwd_wait_handles = p2p_communicator.send_backward(input_tensor_grad, tensor_shape, master_chunk_id, async_op=True)
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
                output_tensor_grad, _ = p2p_communicator.recv_backward(tensor_shape, master_chunk_id, async_op=False)
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
        finish_embedding_wgrad_compute(config, embedding_module, is_pp_first_stage(p2p_communicator.pp_group), tp_group)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if not forward_only and config.fine_grained_activation_offloading:
        off_interface.reset()

    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    return forward_data_store
