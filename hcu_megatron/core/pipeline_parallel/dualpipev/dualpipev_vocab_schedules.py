# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import contextlib
from typing import Iterator, List, Union, Optional, Callable

import torch

from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.utils import (
    get_model_config,
    get_model_type,
)
from megatron.core.pipeline_parallel.schedules import clear_embedding_activation_buffer, deallocate_output_tensor
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    check_first_val_step,
    finish_embedding_wgrad_compute
)
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.pipeline_parallel.utils import (
    set_streams,
    is_pp_first_stage,
    is_pp_last_stage,
)
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.paged_stash import paged_stash_reset

from ..combined_1f1b import combined_forward_backward_step
from hcu_megatron.core.parallel_state import set_dualpipe_chunk
from hcu_megatron.core.models.common.language_module.language_module import set_shared_embedding_from_dual_chunk
from .dualpipev_schedules import (
    DualpipeVP2PCommunicator,
    forward_step_no_model_graph,
    get_send_handle,
    get_recv_handle,
    generate_dualpipev_schedule,
)
from hcu_megatron.core.parallel_state import (
    get_lm_head_model_parallel_group,
    set_virtual_vocab_parallel_chunk,
)
from ..utils import get_lm_head_res_reduce_stream, set_lm_head_res_reduce_stream
from hcu_megatron.core.tensor_parallel.vocab_output_store import VocabOutputStore
from hcu_megatron.core.tensor_parallel.vocab_input_store import VocabInputStore
from hcu_megatron.core.transformer.enums import DualpipeVChunkType


# Types
Shape = Union[List[int], torch.Size]


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
    set_lm_head_res_reduce_stream()

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
        set_shared_embedding_from_dual_chunk(model[2], model[3], enable_vocab_parallel=True)

    assert (
        isinstance(model, list) and len(model) == 5 if is_pp_first_stage(p2p_communicator.pp_group) else 4
    ), 'Dualpipe Schedule expects 4 or 5 model chunks'

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
    input_tensors = [[], [], [], []]
    output_tensors = [[], [], [], []]
    output_tensor_grads = [[], [], [], []]

    master_chunk_id = DualpipeVChunkType.first_block.value
    slave_chunk_id = DualpipeVChunkType.second_block.value
    cur_fwd_chunk_microbatch = [0, num_microbatches]
    cur_bwd_chunk_microbatch = [0, num_microbatches]
    num_chunk_max_microbatch = [num_microbatches, num_microbatches * 2]

    def wait_comm_handle(comm_handle):
        if comm_handle is not None:
            comm_handle.wait()
        comm_handle = None

    input_embedding_output_shape = None

    comm_wait_tensor = torch.Tensor([0]).cuda()
    comm_wait_tensor.record_stream(get_lm_head_res_reduce_stream())

    embedding_model_chunk_id = DualpipeVChunkType.embedding.value
    output_model_chunk_id = DualpipeVChunkType.output.value
    loss_model_chunk_id = DualpipeVChunkType.loss.value

    def input_embedding_forward_step_helper(
        microbatch_id,
    ):
        set_virtual_vocab_parallel_chunk(embedding_model_chunk_id)

        input_tensor = [None]

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"IF{microbatch_id}")

        output_tensor, _ = forward_step_no_model_graph(
            forward_step_func,
            embedding_model_chunk_id,
            data_iterator[0],
            model[embedding_model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            is_first_stage=is_pp_first_stage(p2p_communicator.pp_group),
            skip_loss_compute=True,
        )

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        nonlocal input_embedding_output_shape
        input_embedding_output_shape = output_tensor[0].shape

        reduced_output_tensor = output_tensor[0].clone().detach().to(dtype=config.pipeline_dtype).requires_grad_(True)

        input_tensors[embedding_model_chunk_id].append(input_tensor)
        output_tensors[embedding_model_chunk_id].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        def callback():
            nonlocal reduced_output_tensor

            handle = torch.distributed.all_reduce(
                reduced_output_tensor,
                torch.distributed.ReduceOp.SUM,
                group=get_lm_head_model_parallel_group(),
                async_op=True,
            )

            if is_pp_first_stage(p2p_communicator.pp_group):
                VocabInputStore.forward_store(reduced_output_tensor, handle)

            return

        return callback

    def input_embedding_backward_step_helper(
        microbatch_id
    ):
        set_virtual_vocab_parallel_chunk(embedding_model_chunk_id)

        input_tensor = input_tensors[embedding_model_chunk_id].pop(0)
        output_tensor = output_tensors[embedding_model_chunk_id].pop(0)

        if is_pp_first_stage(p2p_communicator.pp_group):
            output_tensor_grad = [VocabInputStore.backward_get()]
        else:
            output_tensor_grad = [
                torch.empty(
                    input_embedding_output_shape,
                    dtype=config.pipeline_dtype,
                    device=torch.cuda.current_device(),
                )
            ]

        handle = torch.distributed.broadcast(
            output_tensor_grad[0],
            parallel_state.get_pipeline_model_parallel_first_rank(),
            group=get_lm_head_model_parallel_group(),
            async_op=True,
        )

        def callback():
            handle.wait()

            if get_args().profile:
                torch.cuda.nvtx.range_push(f"IB{microbatch_id}")

            try:
                backward_step(
                    input_tensor, output_tensor, output_tensor_grad, config,
                )
            except Exception as e:
                raise Exception(f"{e} {input_tensor=}, {output_tensor=}, {output_tensor_grad=}")

            if get_args().profile:
                torch.cuda.nvtx.range_pop()

        return callback

    # Storing grad output of the loss reduce stage from B step to the next F step.
    last_stage_forward_input_store = None
    last_stage_backward_input_store = None
    lm_head_reduce_output_store = None

    broadcast_lm_head_output_handle = None
    broadcast_lm_head_grad_input_handle = None

    def _broadcast(item):
        handle = None
        if item is not None:
            handle = torch.distributed.broadcast(
                item,
                parallel_state.get_pipeline_model_parallel_first_rank(),
                group=get_lm_head_model_parallel_group(),
                async_op=True,
            )

        return handle

    def broadcast_lm_head_input(microbatch_id, output_tensor, grad_output):
        """
        Assumes `output_tensor` is retrieved from `last_stage_forward_input_store`.
        We do not store it into `last_stage_forward_input_store` again.
        """
        nonlocal last_stage_backward_input_store, broadcast_lm_head_output_handle, broadcast_lm_head_grad_input_handle

        assert is_pp_first_stage(p2p_communicator.pp_group), \
            "lm head input must be broadcasted from the first stage"
        assert not config.variable_seq_lengths, 'not supported yet'

        # get_lm_head_res_reduce_stream().wait_stream(torch.cuda.current_stream())
        if microbatch_id < num_microbatches:
            if broadcast_lm_head_output_handle is not None:
                broadcast_lm_head_output_handle.wait()

            # with torch.cuda.stream(get_lm_head_res_reduce_stream()):
            broadcast_lm_head_output_handle = _broadcast(output_tensor[0])

        if not forward_only and microbatch_id > 0:
            if broadcast_lm_head_grad_input_handle is not None:
                broadcast_lm_head_grad_input_handle.wait()

            # with torch.cuda.stream(get_lm_head_res_reduce_stream()):
            broadcast_lm_head_grad_input_handle = _broadcast(grad_output[0])
            last_stage_backward_input_store = grad_output[0]

    def receive_lm_head_input(microbatch_id):
        def _create_broadcast_tensor(shape, dtype):
            return torch.empty(
                shape,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=True,
            )

        output_tensor = None
        grad_output = None
        if not is_pp_first_stage(p2p_communicator.pp_group):
            handles = []

            if microbatch_id < num_microbatches:
                output_tensor = _create_broadcast_tensor(tensor_shape, config.pipeline_dtype)
                handle = _broadcast(output_tensor)
                handles.append(handle)

            if not forward_only and microbatch_id > 0:
                gathered_tensor_shapes = list(tensor_shape[:-1])
                gathered_tensor_shapes[0] *= tp_group.size()
                grad_output = _create_broadcast_tensor(gathered_tensor_shapes, torch.float32)
                handle = _broadcast(grad_output)
                handles.append(handle)

        def callback():
            nonlocal output_tensor, grad_output, handles, last_stage_backward_input_store, last_stage_forward_input_store

            if not is_pp_first_stage(p2p_communicator.pp_group):

                for handle in handles:
                    if handle is not None:
                        handle.wait()

            if microbatch_id < num_microbatches:
                if is_pp_first_stage(p2p_communicator.pp_group):
                    output_tensor = last_stage_forward_input_store
                    last_stage_forward_input_store = None
            else:
                output_tensor = None

            if microbatch_id > 0:
                # Ensure that the reduction is complete.
                # torch.cuda.current_stream().wait_stream(get_lm_head_res_reduce_stream())

                logits_max, sum_exp_logits, _, _ = lm_head_reduce_output_store

                if not forward_only and is_pp_first_stage(p2p_communicator.pp_group):
                    grad_output = last_stage_backward_input_store
                    last_stage_backward_input_store = None
            else:
                sum_exp_logits = None
                logits_max = None
                grad_output = None

            return [output_tensor], sum_exp_logits, logits_max, [grad_output]

        return callback

    def lm_head_step_helper(
        microbatch_id,
        lm_head_inputs,
    ):
        if get_args().profile:
            torch.cuda.nvtx.range_push(f"S{microbatch_id}")

        lm_head_input_tensor, sum_exp_logits, logits_max, grad_output = lm_head_inputs

        set_virtual_vocab_parallel_chunk(output_model_chunk_id)
        VocabOutputStore.microbatch_id = microbatch_id

        grad_input = [None]
        if not forward_only and microbatch_id > 0:
            input_tensor = input_tensors[output_model_chunk_id].pop(0)
            output_tensor = output_tensors[output_model_chunk_id].pop(0)

            # Only for weight grad updates, input grad returned is ignored.
            VocabOutputStore.backward_store(sum_exp_logits, logits_max, grad_output[0])

            grad_input = backward_step(
                input_tensor, output_tensor, [grad_output[0].transpose(0, 1)], config,
            )

        if microbatch_id < num_microbatches:
            output_tensor, _ = forward_step_no_model_graph(
                forward_step_func,
                output_model_chunk_id,
                data_iterator[0],
                model[output_model_chunk_id],
                num_microbatches,
                lm_head_input_tensor,
                forward_data_store,
                config,
                cp_group_size=pg_collection.cp.size(),
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
                current_microbatch=microbatch_id,
                is_first_stage=is_pp_first_stage(p2p_communicator.pp_group),
                skip_loss_compute=True,
            )
            output_tensor = [output_tensor[0].clone()]
            sum_exp_logits, logits_max, predicted_logits, target_mask, softmax_grad_input, \
                ground_truth_grad_input = VocabOutputStore.forward_get()

            input_tensors[output_model_chunk_id].append(lm_head_input_tensor)
            output_tensors[output_model_chunk_id].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            if get_args().disable_backward_fusion:
                lm_head_res = (logits_max, sum_exp_logits, predicted_logits, target_mask,
                               grad_input[0])
            else:
                lm_head_res = (logits_max, sum_exp_logits, predicted_logits, target_mask,
                           softmax_grad_input, ground_truth_grad_input)
        else:
            if get_args().disable_backward_fusion:
                lm_head_res = (None, None, None, None, grad_input[0])
            else:
                lm_head_res = None

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return lm_head_res

    def reduce_lm_head_res_alg1(microbatch_id, logits_max, sum_exp_logits, predicted_logits, target_mask, grad_input):
        """
        Reduces `logits_max`, `sum_exp_logits`, `predicted_logits` and
        `grad_input` among all pipeline parallel ranks.
        """
        # get_lm_head_res_reduce_stream().wait_stream(torch.cuda.current_stream())
        # with torch.cuda.stream(get_lm_head_res_reduce_stream()):
        if microbatch_id < num_microbatches:

            local_logits_max = logits_max.clone()
            torch.distributed.all_reduce(
                logits_max,
                torch.distributed.ReduceOp.MAX,
                group=get_lm_head_model_parallel_group(),
                async_op=False,
            )
            local_logits_max -= logits_max

            predicted_logits += local_logits_max
            predicted_logits[target_mask] = 0.0
            torch.distributed.all_reduce(
                predicted_logits,
                torch.distributed.ReduceOp.SUM,
                group=get_lm_head_model_parallel_group(),
                async_op=False,
            )

            local_logits_max.exp_()
            sum_exp_logits.mul_(local_logits_max)
            torch.distributed.all_reduce(
                sum_exp_logits,
                torch.distributed.ReduceOp.SUM,
                group=get_lm_head_model_parallel_group(),
                async_op=False,
            )

            # for tensor in (logits_max, sum_exp_logits, predicted_logits, target_mask):
            #     tensor.record_stream(get_lm_head_res_reduce_stream())

        if not forward_only and microbatch_id > 0:
            torch.distributed.all_reduce(
                grad_input,
                torch.distributed.ReduceOp.SUM,
                group=get_lm_head_model_parallel_group(),
                async_op=False,
            )

            # grad_input.record_stream(get_lm_head_res_reduce_stream())

        return logits_max, sum_exp_logits, predicted_logits, grad_input

    def loss_calculation_helper(
        microbatch_id,
    ):
        if not is_pp_first_stage(p2p_communicator.pp_group):
            return

        nonlocal lm_head_reduce_output_store, lm_head_reduce_output_store

        # Ensure that the reduction is complete.
        # torch.cuda.current_stream().wait_stream(get_lm_head_res_reduce_stream())

        _, sum_exp_logits, predicted_logits, _ = lm_head_reduce_output_store

        # Calculate the loss. Then, execute the function that reduces the losses.

        input_tensor = torch.log(sum_exp_logits) - predicted_logits
        input_tensor = [input_tensor.detach().requires_grad_(True)]

        set_virtual_vocab_parallel_chunk(loss_model_chunk_id)

        output_tensor, _ = forward_step_no_model_graph(
            forward_step_func,
            loss_model_chunk_id,
            data_iterator[0],
            model[loss_model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            is_first_stage=is_pp_first_stage(p2p_communicator.pp_group),
        )

        if not forward_only:
            output_tensor_grad = backward_step(
                input_tensor, output_tensor, [None], config,
            )
        else:
            output_tensor_grad = [None]

        if get_args().disable_backward_fusion:
            if microbatch_id < num_microbatches:
                nonlocal last_stage_forward_input_store
                broadcast_lm_head_input(microbatch_id + 1, [last_stage_forward_input_store],
                                        [output_tensor_grad[0]])
            elif not forward_only:
                broadcast_lm_head_input(microbatch_id + 1, None,
                                        [output_tensor_grad[0]])
        else:
            return output_tensor_grad

    def forward_step_helper(model_chunk_id, cur_microbatch, checkpoint_activations_microbatch=False):
        set_dualpipe_chunk(model_chunk_id)
        set_virtual_vocab_parallel_chunk(model_chunk_id)

        if not forward_only:
            offset = cur_bwd_chunk_microbatch[model_chunk_id]
            input_tensor = input_tensors[model_chunk_id][cur_microbatch - offset]
        else:
            input_tensor = input_tensors[model_chunk_id][0]

        if model_chunk_id == master_chunk_id and rank == 0:
            input_tensor = None

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
            skip_loss_compute=True,
        )
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens.item()
        if forward_only:
            input_tensors[model_chunk_id].pop(0)
            output_tensors[model_chunk_id].pop()

        if (
            is_pp_first_stage(p2p_communicator.pp_group)
            and model_chunk_id == slave_chunk_id
        ):
            nonlocal last_stage_forward_input_store
            last_stage_forward_input_store = output_tensor.clone().detach().to(config.pipeline_dtype).requires_grad_(True)
            if cur_microbatch == num_microbatches:
                broadcast_lm_head_input(cur_microbatch - num_microbatches, [last_stage_forward_input_store], None)

        return output_tensor

    def backward_step_helper(model_chunk_id, compute_wgrad=False):
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        if rank == 0 and model_chunk_id == slave_chunk_id:
            # Ensure that the reduction is complete.
            # torch.cuda.current_stream().wait_stream(get_lm_head_res_reduce_stream())    # worse performance

            _, _, _, grad_input = lm_head_reduce_output_store
            # Calculate the input grads of the lm head layer, without calling backward.
            input_tensor_grad = [grad_input]
            output_tensor_grad = input_tensor_grad

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, config
        )

        if rank == 0 and model_chunk_id == master_chunk_id:
            VocabInputStore.backward_store(input_tensor_grad)

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
    ):
        """
        wrap forward_helper、backward_helper、combined_forward_backward_helper in a unified way
        """

        if config.overlap_moe_expert_parallel_comm and not forward_only:
            assert (
                checkpoint_activations_microbatch is None
            ), "checkpoint_activations_microbatch not supported when overlap_moe_expert_parallel_comm is true"
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

    num_input_embedding_forward_steps_remaining = num_microbatches
    num_input_embedding_backward_steps_remaining = num_microbatches

    for i in range(rank + 1):
        input_embedding_forward_step_helper(i)()
        num_input_embedding_forward_steps_remaining -= 1

    # Run warmup forward passes
    input_tensor, _ = p2p_communicator.recv_forward(tensor_shape, master_chunk_id)
    input_tensors[master_chunk_id].append(input_tensor)
    is_slave_only = False
    fwd_wait_handles_warmup = None
    for i in range(schedule['warmup'][rank]):

        input_embedding_forward_step_helper(
            num_microbatches - num_input_embedding_forward_steps_remaining
        )()
        num_input_embedding_forward_steps_remaining -= 1

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
    num_forward_backward_calculated = 0
    num_remaining_s_pass = num_microbatches
    for i in range(schedule['interleaved_forward'][rank]):
        if num_forward_backward_calculated <= rank:
            input_embedding_forward_step_helper(
                num_microbatches - num_input_embedding_forward_steps_remaining
            )()
            num_input_embedding_forward_steps_remaining -= 1

        # master forward
        is_slave_only = (cur_fwd_chunk_microbatch[master_chunk_id] == num_chunk_max_microbatch[master_chunk_id])
        if not is_slave_only:
            wait_comm_handle(fwd_wait_recv_handles[master_chunk_id])
            output_tensor, _, _ = forward_backward_helper_wrapper(
                fwd_model_chunk_id=master_chunk_id,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )

        num_forward_backward_calculated += 1

        if (
            num_forward_backward_calculated > rank
            and (num_forward_backward_calculated - rank) % 2 == 0
        ):
            if num_remaining_s_pass < num_microbatches:
                loss_calculation_helper(num_microbatches - num_remaining_s_pass - 1)

            lm_head_inputs = receive_lm_head_input(num_microbatches - num_remaining_s_pass)()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs)

            if get_args().disable_backward_fusion:
                lm_head_res = reduce_lm_head_res_alg1(num_microbatches - num_remaining_s_pass, *lm_head_res)

            lm_head_reduce_output_store = lm_head_res

            if num_remaining_s_pass == num_microbatches:
                input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )()
                num_input_embedding_forward_steps_remaining -= 1

            num_remaining_s_pass -= 1

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

        if num_forward_backward_calculated <= rank:
            input_embedding_forward_step_helper(
                num_microbatches - num_input_embedding_forward_steps_remaining
            )()
            num_input_embedding_forward_steps_remaining -= 1

        # slave forward
        output_tensor_slave, _, _ = forward_backward_helper_wrapper(
            fwd_model_chunk_id=slave_chunk_id,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        num_forward_backward_calculated += 1
        if (
            num_forward_backward_calculated > rank
            and (num_forward_backward_calculated - rank) % 2 == 0
        ):
            if num_remaining_s_pass < num_microbatches:
                loss_calculation_helper(num_microbatches - num_remaining_s_pass - 1)

            lm_head_inputs = receive_lm_head_input(num_microbatches - num_remaining_s_pass)()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs)

            if get_args().disable_backward_fusion:
                lm_head_res = reduce_lm_head_res_alg1(num_microbatches - num_remaining_s_pass, *lm_head_res)

            lm_head_reduce_output_store = lm_head_res

            if num_remaining_s_pass == num_microbatches:
                input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )()
                num_input_embedding_forward_steps_remaining -= 1

            num_remaining_s_pass -= 1

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

    chunk_backward_dw_funcs = []
    for i in range(schedule['1b1w1f'][rank][0]):
        if not forward_only:
            _, input_tensor_grad, chunk_backward_dw_func = forward_backward_helper_wrapper(
                bwd_model_chunk_id=slave_chunk_id,
                block_level_wgrad_compute=True,
            )

            if not schedule['1b1w1f'][rank][1]:
                chunk_backward_dw_funcs.append((chunk_backward_dw_func, False))

            bwd_wait_handles = p2p_communicator.send_backward(input_tensor_grad, tensor_shape, slave_chunk_id, async_op=True)
            bwd_wait_send_handles[slave_chunk_id] = get_send_handle(bwd_wait_handles, slave_chunk_id, forward=False)

            fwd_recv_buffer[0], fwd_wait_handles = p2p_communicator.recv_forward(tensor_shape, slave_chunk_id, async_op=True)
            fwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(fwd_wait_handles, slave_chunk_id, forward=True)
            input_tensors[slave_chunk_id].append(fwd_recv_buffer[0])
            fwd_recv_buffer[0] = None

            if (
                schedule['1b1w1f'][rank][1]
                and chunk_backward_dw_func is not None
            ):
                chunk_backward_dw_func()
                del chunk_backward_dw_func
        else:
            fwd_recv_buffer[0], fwd_wait_handles = p2p_communicator.recv_forward(tensor_shape, slave_chunk_id, async_op=True)
            fwd_wait_recv_handles[slave_chunk_id] = get_recv_handle(fwd_wait_handles, slave_chunk_id, forward=True)
            input_tensors[slave_chunk_id].append(fwd_recv_buffer[0])
            fwd_recv_buffer[0] = None

        num_forward_backward_calculated += 1
        if (
            not forward_only
            and not schedule['1b1w1f'][rank][1]
            and num_forward_backward_calculated > pp_size + rank + 3
        ):
            chunk_backward_dw_func, _ = chunk_backward_dw_funcs.pop(0)
            if chunk_backward_dw_func is not None:
                chunk_backward_dw_func()
                del chunk_backward_dw_func

        if (num_forward_backward_calculated - rank) % 2 == 0:
            loss_calculation_helper(num_microbatches - num_remaining_s_pass - 1)

            lm_head_inputs = receive_lm_head_input(num_microbatches - num_remaining_s_pass)()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs)

            if get_args().disable_backward_fusion:
                lm_head_res = reduce_lm_head_res_alg1(num_microbatches - num_remaining_s_pass, *lm_head_res)

            lm_head_reduce_output_store = lm_head_res
            num_remaining_s_pass -= 1

        if not forward_only:
            wait_comm_handle(bwd_wait_send_handles[slave_chunk_id])

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

        num_forward_backward_calculated += 1
        if (
            not forward_only
            and not schedule['1b1w1f'][rank][1]
            and num_forward_backward_calculated > pp_size + rank + 3
        ):
            chunk_backward_dw_func, _ = chunk_backward_dw_funcs.pop(0)
            if chunk_backward_dw_func is not None:
                chunk_backward_dw_func()
                del chunk_backward_dw_func

        if (
            i != schedule['1b1w1f'][rank][0] - 1
            and (num_forward_backward_calculated - rank) % 2 == 0
        ):
            loss_calculation_helper(num_microbatches - num_remaining_s_pass - 1)

            lm_head_inputs = receive_lm_head_input(num_microbatches - num_remaining_s_pass)()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs)

            if get_args().disable_backward_fusion:
                lm_head_res = reduce_lm_head_res_alg1(num_microbatches - num_remaining_s_pass, *lm_head_res)

            lm_head_reduce_output_store = lm_head_res
            num_remaining_s_pass -= 1

    # check whether forward data transmission is completed.
    wait_comm_handle(fwd_wait_send_handles[slave_chunk_id])

    # Run overlaping f&bw stages
    prev_step_backward_only = False
    fwd_model_chunk_id = master_chunk_id
    bwd_model_chunk_id = slave_chunk_id
    num_overlap_steps = schedule['overlap'][rank] + schedule['1b1overlap'][rank] + schedule['interleaved_backward'][rank]
    num_bw_steps = 0
    for step_id in range(num_overlap_steps):
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

        if (
            num_remaining_s_pass >= 0
            and (num_bw_steps + rank) % 2 == 0
        ):
            loss_calculation_helper(num_microbatches - num_remaining_s_pass - 1)

            if forward_only and num_remaining_s_pass == 0:
                break

            receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

            if (
                not forward_only
                and (num_bw_steps - rank) // 2 > 2
            ):
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )

                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None

            if num_bw_steps >= rank and num_input_embedding_forward_steps_remaining > 0:
                input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )()
                num_input_embedding_forward_steps_remaining -= 1

            input_embedding_backward_callback()

            lm_head_inputs = receive_lm_head_input_callback()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs)

            if get_args().disable_backward_fusion:
                lm_head_res = reduce_lm_head_res_alg1(num_microbatches - num_remaining_s_pass, *lm_head_res)

            lm_head_reduce_output_store = lm_head_res
            num_remaining_s_pass -= 1

            if num_remaining_s_pass == 0:
                if config.grad_sync_func is not None:
                    enable_grad_sync()
                    config.grad_sync_func[2](model[2].parameters())
                    disable_grad_sync()

        if forward_only and step_id >= num_overlap_steps - schedule['interleaved_backward'][rank]:
            num_bw_steps += 1
            continue

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
                )

        # swap fwd & bwd chunks
        fwd_model_chunk_id, bwd_model_chunk_id = bwd_model_chunk_id, fwd_model_chunk_id
        prev_step_backward_only = only_bwd
        num_bw_steps += 1

    # Run cooldown phases
    if not forward_only:
        if rank == 0:
            # launch grad reductions.
            if config.grad_sync_func is not None:
                enable_grad_sync()
                config.grad_sync_func[slave_chunk_id](model[slave_chunk_id].parameters())
                disable_grad_sync()

        for i in range(schedule['cooldown'][rank][0]):
            wait_comm_handle(bwd_wait_recv_handles[bwd_model_chunk_id])

            _, input_tensor_grad, chunk_backward_dw_func = forward_backward_helper_wrapper(
                bwd_model_chunk_id=bwd_model_chunk_id,
                block_level_wgrad_compute=True,
            )
            chunk_backward_dw_funcs.append((chunk_backward_dw_func, i == schedule['cooldown'][rank][0] - 1))

            num_bw_steps += 1
            if (num_bw_steps + rank) % 2 == 0:
                input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )()
                num_input_embedding_backward_steps_remaining -= 1

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

            num_bw_steps += 1
            if (num_bw_steps + rank) % 2 == 0:
                input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )()
                num_input_embedding_backward_steps_remaining -= 1

            bwd_wait_handles = p2p_communicator.send_backward(input_tensor_grad, tensor_shape, master_chunk_id, async_op=True)
            bwd_wait_send_handles[master_chunk_id] = get_send_handle(bwd_wait_handles, master_chunk_id, forward=False)
            # weight backward
            chunk_backward_dw_func, is_last_slave_chunk = chunk_backward_dw_funcs.pop(0)
            if chunk_backward_dw_func is not None:
                chunk_backward_dw_func()
                del chunk_backward_dw_func

            num_bw_steps += 1
            if (num_bw_steps + rank) % 2 == 0:
                input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )()
                num_input_embedding_backward_steps_remaining -= 1

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

            num_bw_steps += 1
            if (num_bw_steps + rank) % 2 == 0:
                input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )()
                num_input_embedding_backward_steps_remaining -= 1

            if is_last_slave_chunk:
                # Launch grad reductions.
                if config.grad_sync_func is not None:
                    enable_grad_sync()
                    config.grad_sync_func[slave_chunk_id](model[slave_chunk_id].parameters())
                    disable_grad_sync()

        if num_input_embedding_backward_steps_remaining > 0:
            input_embedding_backward_step_helper(
                num_microbatches - num_input_embedding_backward_steps_remaining
            )()

    # launch any remaining grad reductions.
    if config.grad_sync_func is not None:
        enable_grad_sync()
        config.grad_sync_func[master_chunk_id](model[master_chunk_id].parameters())
        config.grad_sync_func[3](model[3].parameters())

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
