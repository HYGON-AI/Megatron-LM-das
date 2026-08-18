# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import contextlib
from typing import Iterator, List, Union, Optional, Callable

import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.pipeline_parallel.schedules import (
    get_tensor_shapes,
    check_first_val_step,
    deallocate_output_tensor,
    clear_embedding_activation_buffer,
    finish_embedding_wgrad_compute,
)
from megatron.core.utils import (
    get_model_config,
    get_model_type,
    get_model_xattn,
)
from megatron.training import get_args

from hcu_megatron.core.pipeline_parallel.schedules import (
    forward_step,
    backward_step,
)
from hcu_megatron.core.parallel_state import (
    get_lm_head_model_parallel_group,
    set_virtual_vocab_parallel_chunk,
)
from hcu_megatron.core.pipeline_parallel.schedules import bootstrap_and_profile_p2p_communication
from hcu_megatron.core.pipeline_parallel.schedule_timers import ScheduleTimers
from hcu_megatron.core.tensor_parallel.vocab_output_store import VocabOutputStore
from hcu_megatron.core.tensor_parallel.vocab_input_store import VocabInputStore
from .utils import get_lm_head_res_reduce_stream, set_lm_head_res_reduce_stream


def forward_backward_pipelining_with_vocab_parallel(
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
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Run non-interleaved 1F1B schedule with Vocabulary Parallelism.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    set_lm_head_res_reduce_stream()

    assert isinstance(model, list)

    config = get_model_config(model[0])
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(
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

    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model[0])
        assert model_type != ModelType.encoder_and_decoder, (
            "encoder PP stages not yet supported when passing custom process groups. "
            "support coming soon!"
        )
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have a tp_group"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have a cp_group"
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it is used default "
            "`parallel_state.default_embedding_ranks` to create the process group. If you are "
            "using the default process group, please use `parallel_state.get_embedding_group()` "
            "to get the process group. If you don't need explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have a pos_embd. In previous version, it is used default "
            "`parallel_state.default_position_embedding_ranks` to create the process group."
            " If you are using the default process group, please use "
            "`parallel_state.get_position_embedding_group()` "
            "If you don't need pos_embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pp'), "pg_collection must have a pp_group"
        assert hasattr(pg_collection, 'dp_cp'), "pg_collection must have a dp_cp_group"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection"
            " provide none or provide all the process groups"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model[0], is_pp_last_stage(p2p_communicator.pp_group))

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

    grad_sync_func = None
    if forward_only:
        grad_sync_func = config.grad_sync_func
        config.grad_sync_func = None

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

    pipeline_parallel_size = p2p_communicator.pp_group.size()
    pipeline_parallel_rank = p2p_communicator.pp_group.rank()

    # Increment iter_counter in ScheduleTimers
    ScheduleTimers.iter_counter += 1

    if ScheduleTimers.iter_counter == get_args().schedule_timer_end + 1:
        ScheduleTimers.sync_timer = False

    if ScheduleTimers.iter_counter == get_args().schedule_timer_end + 6:
        conclusion = ScheduleTimers.joint_conclusion(sync_timer=False, global_reduce=False)
        print(f"rank {torch.distributed.get_rank()} profiling conclusion: {conclusion}")

    if ScheduleTimers.iter_counter >= get_args().schedule_timer_end + 1:
        conclusion = ScheduleTimers.joint_conclusion()
        f = conclusion[0][0][0]
        b = conclusion[0][0][1]
        c = conclusion[0][0][3]
    else:
        f = 1
        b = 2
        c = 0

    assert f >= c, 'vocab parallel schedules assume f >= c, ' \
        'f < c will lead to additional pipeline bubbles due to incorrect ' \
        'placements for the S pass'
    assert b >= f, 'vocab parallel schedules assume b >= f for S pass placements'

    offset = 0
    is_bsf = [True]
    offsets = [0]
    while len(is_bsf) < pipeline_parallel_size:
        # we can either subtract f from the offset, or add (b - f) to the offset
        # FSM:
        # - BSF --[ - f ]--> BSF
        # - BSF --[ 0 ]--> BFS --[ (b - f) ]--> BSF
        if offset - f >= -b:
            offset = offset - f
            is_bsf.append(True)
            offsets.append(offset)
        else:
            is_bsf.append(False)
            offsets.append(offset)
            offset = offset + (b - f)
            is_bsf.append(True)
            offsets.append(offset)
    if len(is_bsf) > pipeline_parallel_size:
        is_bsf.pop()
        offsets.pop()

    is_bsf.reverse()
    offsets.reverse()

    num_warmup_s_pass = [0 for _ in range(pipeline_parallel_size)]   # [3, 2, 1, 0]
    for rank in range(pipeline_parallel_size - 2, -1, -1):
        if (not is_bsf[rank + 1]) and (is_bsf[rank]):
            num_warmup_s_pass[rank] = num_warmup_s_pass[rank + 1]
        else:
            num_warmup_s_pass[rank] = num_warmup_s_pass[rank + 1] + 1

    run_timer = (
        get_args().schedule_timer_end + 5
        >= ScheduleTimers.iter_counter
        >= get_args().schedule_timer_start
    )

    # Compute number of warmup microbatches.
    num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank
    if forward_only:
        num_warmup_microbatches -= 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    if forward_only:
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches
        first_stage_num_warmup_microbatches = min(pipeline_parallel_size - 2, num_microbatches)
    else:
        first_stage_num_warmup_microbatches = min(pipeline_parallel_size, num_microbatches)
        num_microbatches_remaining = max(
            0,
            num_microbatches - num_warmup_microbatches - 1
        )
        if get_args().disable_backward_fusion:
            # Add one more warm-up microbatch.
            num_microbatches_remaining = max(0, num_microbatches_remaining - 1)

    assert config.num_microbatches_with_partial_activation_checkpoints is None, 'not supported'

    model_type = get_model_type(model[0])
    encoder_decoder_xattn = get_model_xattn(model[0])

    rank = pipeline_parallel_rank
    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    lm_head_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )

    bootstrap_and_profile_p2p_communication(
        p2p_communicator, send_tensor_shapes, recv_tensor_shapes)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    input_tensors = [[], [], []]
    output_tensors = [[], [], []]
    forward_data_store = []

    # Storing grad output of the loss reduce stage from B step to the next F step.
    last_stage_forward_input_store = None
    last_stage_backward_input_store = None
    lm_head_reduce_output_store = None

    comm_wait_tensor = torch.Tensor([0]).cuda()
    comm_wait_tensor.record_stream(get_lm_head_res_reduce_stream())

    broadcast_lm_head_output_handle = None
    broadcast_lm_head_grad_input_handle = None

    def _broadcast(item):
        handle = None
        if item is not None:
            handle = torch.distributed.broadcast(
                item,
                parallel_state.get_pipeline_model_parallel_last_rank(),
                group=get_lm_head_model_parallel_group(),
                async_op=True,
            )

        return handle

    def broadcast_lm_head_input(microbatch_id, output_tensor, grad_output):
        """
        Assumes `output_tensor` is retrieved from `last_stage_forward_input_store`.
        We do not store it into `last_stage_forward_input_store` again.
        """
        nonlocal config, last_stage_backward_input_store, num_microbatches, \
                 broadcast_lm_head_output_handle, broadcast_lm_head_grad_input_handle
        assert is_pp_last_stage(p2p_communicator.pp_group), \
            "lm head input must be broadcasted from the last stage"
        assert not config.variable_seq_lengths, 'not supported yet'

        # get_lm_head_res_reduce_stream().wait_stream(torch.cuda.current_stream())
        if microbatch_id < num_microbatches:
            if broadcast_lm_head_output_handle is not None:
                broadcast_lm_head_output_handle.wait()

            # with torch.cuda.stream(get_lm_head_res_reduce_stream()):
            broadcast_lm_head_output_handle = _broadcast(output_tensor[0])
            output_tensor[0].record_stream(get_lm_head_res_reduce_stream())

        if not forward_only and microbatch_id > 0:
            if broadcast_lm_head_grad_input_handle is not None:
                broadcast_lm_head_grad_input_handle.wait()

            # with torch.cuda.stream(get_lm_head_res_reduce_stream()):
            broadcast_lm_head_grad_input_handle = _broadcast(grad_output[0])
            grad_output[0].record_stream(get_lm_head_res_reduce_stream())
            last_stage_backward_input_store = grad_output[0]

    def receive_lm_head_input(microbatch_id):
        nonlocal config, num_microbatches, last_stage_forward_input_store, \
                 last_stage_backward_input_store, lm_head_tensor_shapes, \
                 lm_head_reduce_output_store

        def _create_broadcast_tensor(shape, dtype):
            return torch.empty(
                shape,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=True,
            )

        output_tensor = None
        grad_output = None
        if not is_pp_last_stage(p2p_communicator.pp_group):
            handles = []

            # get_lm_head_res_reduce_stream().wait_stream(torch.cuda.current_stream())
            # with torch.cuda.stream(get_lm_head_res_reduce_stream()):
            if microbatch_id < num_microbatches:
                output_tensor = _create_broadcast_tensor(lm_head_tensor_shapes[0], config.pipeline_dtype)
                handle = _broadcast(output_tensor)
                output_tensor.record_stream(get_lm_head_res_reduce_stream())
                handles.append(handle)

            if not forward_only and microbatch_id > 0:
                gathered_tensor_shapes = list(lm_head_tensor_shapes[0][:-1])
                gathered_tensor_shapes[0] *= tp_group.size()
                grad_output = _create_broadcast_tensor(gathered_tensor_shapes, torch.float32)
                handle = _broadcast(grad_output)
                grad_output.record_stream(get_lm_head_res_reduce_stream())
                handles.append(handle)

        def callback():
            nonlocal output_tensor, grad_output, handles, microbatch_id, num_microbatches, \
                     config, last_stage_forward_input_store, last_stage_backward_input_store, \
                     lm_head_tensor_shapes, lm_head_reduce_output_store

            if not is_pp_last_stage(p2p_communicator.pp_group):

                for handle in handles:
                    if handle is not None:
                        handle.wait()

            if microbatch_id < num_microbatches:
                if is_pp_last_stage(p2p_communicator.pp_group):
                    output_tensor = last_stage_forward_input_store
                    last_stage_forward_input_store = None
            else:
                output_tensor = None

            if microbatch_id > 0:
                # Ensure that the reduction is complete.
                torch.cuda.current_stream().wait_stream(get_lm_head_res_reduce_stream())

                logits_max, sum_exp_logits, _, _ = lm_head_reduce_output_store

                if not forward_only and is_pp_last_stage(p2p_communicator.pp_group):
                    grad_output = last_stage_backward_input_store
                    last_stage_backward_input_store = None
            else:
                sum_exp_logits = None
                logits_max = None
                grad_output = None

            return [output_tensor], sum_exp_logits, logits_max, [grad_output]

        return callback

    def sequence_shard(t: torch.Tensor, *, dim: int = 0):
        nonlocal config
        if t is None:
            return None

        if not config.sequence_parallel:
            return t
        world_size = tp_group.size()
        rank = parallel_state.get_tensor_model_parallel_rank()
        dim_size = t.size(dim=dim) // world_size
        slices = [slice(None)] * t.dim()
        slices[dim] = slice(rank * dim_size, (rank + 1) * dim_size)
        return t[tuple(slices)]

    def reduce_lm_head_res_alg1(microbatch_id, logits_max, sum_exp_logits, predicted_logits, target_mask, grad_input):
        """
        Reduces `logits_max`, `sum_exp_logits`, `predicted_logits` and
        `grad_input` among all pipeline parallel ranks.
        """
        get_lm_head_res_reduce_stream().wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(get_lm_head_res_reduce_stream()):
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

                for tensor in (logits_max, sum_exp_logits, predicted_logits, target_mask):
                    tensor.record_stream(get_lm_head_res_reduce_stream())

            if not forward_only and microbatch_id > 0:
                torch.distributed.all_reduce(
                    grad_input,
                    torch.distributed.ReduceOp.SUM,
                    group=get_lm_head_model_parallel_group(),
                    async_op=False,
                )

                grad_input.record_stream(get_lm_head_res_reduce_stream())

        return logits_max, sum_exp_logits, predicted_logits, grad_input

    def reduce_lm_head_res_alg2(logits_max, sum_exp_logits, predicted_logits, target_mask, softmax_grad_input, ground_truth_grad_input):
        """
        Reduces `logits_max`, `sum_exp_logits`, `predicted_logits` and
        `grad_input` among all pipeline parallel ranks.
        """

        logits_max = sequence_shard(logits_max)
        sum_exp_logits = sequence_shard(sum_exp_logits)
        predicted_logits = sequence_shard(predicted_logits)
        target_mask = sequence_shard(target_mask)

        for tensor in (logits_max, sum_exp_logits, predicted_logits, target_mask, softmax_grad_input,
                       ground_truth_grad_input):
            tensor.record_stream(get_lm_head_res_reduce_stream())

        get_lm_head_res_reduce_stream().wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(get_lm_head_res_reduce_stream()):
            local_logits_max = logits_max.clone()
            handle = torch.distributed.all_reduce(
                logits_max,
                torch.distributed.ReduceOp.MAX,
                group=get_lm_head_model_parallel_group(),
                async_op=True,
            )
            handle.wait()
            local_logits_max -= logits_max

            predicted_logits += local_logits_max
            predicted_logits[target_mask] = 0.0
            handle = torch.distributed.all_reduce(
                predicted_logits,
                torch.distributed.ReduceOp.SUM,
                group=get_lm_head_model_parallel_group(),
                async_op=True,
            )
            handle.wait()

            local_logits_max.exp_()
            sum_exp_logits.mul_(local_logits_max)
            local_sum_exp_logits = sum_exp_logits.clone()
            handle = torch.distributed.all_reduce(
                sum_exp_logits,
                torch.distributed.ReduceOp.SUM,
                group=get_lm_head_model_parallel_group(),
                async_op=True,
            )
            handle.wait()

            local_sum_exp_logits.div_(sum_exp_logits)
            softmax_grad_input.mul_(local_sum_exp_logits.unsqueeze(-1))
            softmax_grad_input -= ground_truth_grad_input
            handle = torch.distributed.all_reduce(
                softmax_grad_input,
                torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_lm_head_model_parallel_group(),
                async_op=True,
            )
            handle.wait()

        return logits_max, sum_exp_logits, predicted_logits, softmax_grad_input

    def forward_step_helper(
        microbatch_id,
        input_tensor,
        run_timer
    ):
        """
        Executes forward step and completes language model head communication (if any). Returns
        the output tensor.

        Note: This function does not push the input and output tensors into `input_tensors` and
        `output_tensors`. The caller should do this after sending the output tensor.
        """
        nonlocal forward_step_func, data_iterator, model, num_microbatches, forward_data_store, \
                 config, collect_non_loss_data, encoder_decoder_xattn, total_num_tokens, forward_only, \
                 first_val_step, forward_only

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"F{microbatch_id}")

        set_virtual_vocab_parallel_chunk(0)

        if parallel_state.is_pipeline_first_stage():
            input_tensor = [None]

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model[0],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            is_last_stage=is_pp_last_stage(p2p_communicator.pp_group),
            skip_loss_compute=True,
            run_timer=run_timer
        )

        total_num_tokens += num_tokens.item()

        if is_pp_last_stage(p2p_communicator.pp_group):
            nonlocal last_stage_forward_input_store
            last_stage_forward_input_store = output_tensor[0].clone().detach() \
                                             .to(config.pipeline_dtype).requires_grad_(True)
            if forward_only or microbatch_id == 0:
                broadcast_lm_head_input(microbatch_id, [last_stage_forward_input_store], None)

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return output_tensor

    input_embedding_backward_callback = lambda: None

    def loss_calculation_helper(
        microbatch_id,
    ):
        if not is_pp_last_stage(p2p_communicator.pp_group):
            return

        nonlocal lm_head_reduce_output_store, num_microbatches, config, \
                 model_type, forward_step_func, data_iterator, model, forward_data_store, \
                 collect_non_loss_data, encoder_decoder_xattn, lm_head_reduce_output_store, \
                 first_val_step, forward_only, rank

        # Ensure that the reduction is complete.
        torch.cuda.current_stream().wait_stream(get_lm_head_res_reduce_stream())

        _, sum_exp_logits, predicted_logits, _ = lm_head_reduce_output_store

        # Calculate the loss. Then, execute the function that reduces the losses.

        input_tensor = torch.log(sum_exp_logits) - predicted_logits
        input_tensor = [input_tensor.detach().requires_grad_(True)]

        set_virtual_vocab_parallel_chunk(3)

        output_tensor, _ = forward_step(
            forward_step_func,
            data_iterator,
            model[3],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            is_last_stage=is_pp_last_stage(p2p_communicator.pp_group),
            run_timer=False
        )

        if forward_only:
            return

        output_tensor_grad = backward_step(
            input_tensor, output_tensor, [None], config,
            run_timer=False
        )

        if get_args().disable_backward_fusion:
            if microbatch_id < num_microbatches:
                nonlocal last_stage_forward_input_store
                broadcast_lm_head_input(microbatch_id + 1, [last_stage_forward_input_store],
                                        [output_tensor_grad[0]])
            else:
                broadcast_lm_head_input(microbatch_id + 1, None,
                                        [output_tensor_grad[0]])
        else:
            return output_tensor_grad

    def backward_step_helper(
        microbatch_id,
        output_tensor_grad,
        run_timer,
    ):
        nonlocal input_tensors, output_tensors, num_microbatches, config, rank, enable_grad_sync, \
                 model_type, forward_step_func, data_iterator, model, forward_data_store, \
                 collect_non_loss_data, encoder_decoder_xattn, lm_head_reduce_output_store, \
                 first_val_step, forward_only

        post_process = lambda: None

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"B{microbatch_id}")

        if parallel_state.is_pipeline_last_stage():
            # Ensure that the reduction is complete.
            torch.cuda.current_stream().wait_stream(get_lm_head_res_reduce_stream())    # worse performance

            _, _, _, grad_input = lm_head_reduce_output_store

            if not get_args().disable_backward_fusion:
                output_tensor_grad = loss_calculation_helper(microbatch_id)

                if microbatch_id < num_microbatches - 1:
                    nonlocal last_stage_forward_input_store
                    broadcast_lm_head_input(microbatch_id + 1, [last_stage_forward_input_store],
                                            [sequence_shard(output_tensor_grad[0])])
                else:
                    output_tensor_grad_store = output_tensor_grad
                    post_process = lambda: broadcast_lm_head_input(microbatch_id + 1, None,
                                                                [sequence_shard(output_tensor_grad_store[0])])

            # Calculate the input grads of the lm head layer, without calling backward.
            input_tensor_grad = [grad_input]

            if not get_args().disable_backward_fusion:
                input_tensor_grad[0].mul_(sequence_shard(output_tensor_grad[0]).unsqueeze(dim=-1))

            output_tensor_grad = input_tensor_grad

        input_tensor = input_tensors[0].pop(0)
        output_tensor = output_tensors[0].pop(0)

        set_virtual_vocab_parallel_chunk(0)

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, config,
            run_timer=run_timer
        )

        if is_pp_first_stage(p2p_communicator.pp_group):
            VocabInputStore.backward_store(input_tensor_grad[0])

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return input_tensor_grad, post_process

    def lm_head_step_helper(
        microbatch_id,
        lm_head_inputs,
        run_timer
    ):
        nonlocal input_tensors, output_tensors, model_type, config, num_microbatches, \
                 forward_step_func, data_iterator, model, forward_data_store, \
                 collect_non_loss_data, encoder_decoder_xattn, first_val_step, forward_only

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"S{microbatch_id}")

        lm_head_input_tensor, sum_exp_logits, logits_max, grad_output = lm_head_inputs

        set_virtual_vocab_parallel_chunk(1)
        VocabOutputStore.microbatch_id = microbatch_id

        if (run_timer) and (0 < microbatch_id < num_microbatches):
            ScheduleTimers.for_chunk(0).s_cnt += 1
            ScheduleTimers.for_chunk(0).s.start()

        grad_input = [None]
        if not forward_only and microbatch_id > 0:
            input_tensor = input_tensors[1].pop(0)
            output_tensor = output_tensors[1].pop(0)

            # Only for weight grad updates, input grad returned is ignored.
            VocabOutputStore.backward_store(sum_exp_logits, logits_max, grad_output[0])

            grad_input = backward_step(
                input_tensor, output_tensor, [grad_output[0].transpose(0, 1)], config,
                run_timer=False
            )

        if microbatch_id < num_microbatches:
            output_tensor, _ = forward_step(
                forward_step_func,
                data_iterator,
                model[1],
                num_microbatches,
                lm_head_input_tensor,
                forward_data_store,
                config,
                cp_group_size=pg_collection.cp.size(),
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
                current_microbatch=microbatch_id,
                is_last_stage=is_pp_last_stage(p2p_communicator.pp_group),
                skip_loss_compute=True,
                run_timer=False
            )
            output_tensor = [output_tensor[0].clone()]
            sum_exp_logits, logits_max, predicted_logits, target_mask, softmax_grad_input, \
                ground_truth_grad_input = VocabOutputStore.forward_get()

            input_tensors[1].append(lm_head_input_tensor)
            output_tensors[1].append(output_tensor)
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

        if (run_timer) and (0 < microbatch_id < num_microbatches):
            ScheduleTimers.for_chunk(0).s.stop()

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        return lm_head_res

    input_embedding_output_shape = None

    def input_embedding_forward_step_helper(
        microbatch_id,
    ):
        nonlocal forward_step_func, data_iterator, model, num_microbatches, forward_data_store, \
                 config, collect_non_loss_data, encoder_decoder_xattn, forward_only, first_val_step, \
                 run_timer

        set_virtual_vocab_parallel_chunk(2)

        input_tensor = [None]

        if get_args().profile:
            torch.cuda.nvtx.range_push(f"IF{microbatch_id}")

        if run_timer:
            ScheduleTimers.for_chunk(0).input_f_cnt += 1
            ScheduleTimers.for_chunk(0).input_f.start()

        output_tensor, _ = forward_step(
            forward_step_func,
            data_iterator,
            model[2],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, microbatch_id == 0),
            current_microbatch=microbatch_id,
            is_last_stage=is_pp_last_stage(p2p_communicator.pp_group),
            skip_loss_compute=True,
            run_timer=False
        )

        if run_timer:
            ScheduleTimers.for_chunk(0).input_f.stop()

        if get_args().profile:
            torch.cuda.nvtx.range_pop()

        nonlocal input_embedding_output_shape
        input_embedding_output_shape = output_tensor[0].shape

        reduced_output_tensor = output_tensor[0].clone().detach().to(dtype=config.pipeline_dtype).requires_grad_(True)

        input_tensors[2].append(input_tensor)
        output_tensors[2].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        def callback():
            nonlocal reduced_output_tensor

            torch.distributed.all_reduce(
                comm_wait_tensor,
                torch.distributed.ReduceOp.MAX,
                group=get_lm_head_model_parallel_group(),
                async_op=True,
            )

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
        set_virtual_vocab_parallel_chunk(2)

        input_tensor = input_tensors[2].pop(0)
        output_tensor = output_tensors[2].pop(0)

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

        get_lm_head_res_reduce_stream().wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(get_lm_head_res_reduce_stream()):
            torch.distributed.all_reduce(
                    comm_wait_tensor,
                    torch.distributed.ReduceOp.MAX,
                    group=get_lm_head_model_parallel_group(),
                    async_op=True,
            )

            handle = torch.distributed.broadcast(
                output_tensor_grad[0],
                parallel_state.get_pipeline_model_parallel_first_rank(),
                group=get_lm_head_model_parallel_group(),
                async_op=True,
            )
            output_tensor_grad[0].record_stream(get_lm_head_res_reduce_stream())

        def callback():
            nonlocal input_tensor, output_tensor, output_tensor_grad, model_type, \
                     config, handle, run_timer

            handle.wait()

            torch.cuda.current_stream().wait_stream(get_lm_head_res_reduce_stream())

            if get_args().profile:
                torch.cuda.nvtx.range_push(f"IB{microbatch_id}")

            if run_timer:
                ScheduleTimers.for_chunk(0).input_b_cnt += 1
                ScheduleTimers.for_chunk(0).input_b.start()

            backward_step(
                input_tensor, output_tensor, output_tensor_grad, config,
                run_timer=False
            )

            if run_timer:
                ScheduleTimers.for_chunk(0).input_b.stop()

            if get_args().profile:
                torch.cuda.nvtx.range_pop()

        return callback

    num_input_embedding_forward_steps_remaining = num_microbatches
    num_input_embedding_backward_steps_remaining = num_microbatches

    for i in range(first_stage_num_warmup_microbatches - num_warmup_microbatches + 1):
        input_embedding_forward_step_helper(i)()
        num_input_embedding_forward_steps_remaining -= 1

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        if not forward_only:
            input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group))

        input_embedding_forward_step_helper(
            num_microbatches - num_input_embedding_forward_steps_remaining
        )()
        num_input_embedding_forward_steps_remaining -= 1

        if forward_only:
            input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group))

        output_tensor = forward_step_helper(
            i,
            input_tensor,
            run_timer,
        )

        # The communication for the last stage should be deferred until after the first S pass.
        if forward_only:
            p2p_communicator.send_forward(output_tensor, is_pp_last_stage(p2p_communicator.pp_group))
        elif i < num_warmup_microbatches - 1:
            p2p_communicator.send_forward(output_tensor, is_pp_last_stage(p2p_communicator.pp_group))
            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    if forward_only:
        for i in range(num_microbatches_remaining):
            if num_input_embedding_forward_steps_remaining > 0:
                input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )()
                num_input_embedding_forward_steps_remaining -= 1

            input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group))

            output_tensor = forward_step_helper(num_warmup_microbatches + i, input_tensor, False)

            lm_head_inputs = receive_lm_head_input(i)()
            lm_head_res = lm_head_step_helper(i, lm_head_inputs, False)

            if get_args().disable_backward_fusion:
                lm_head_res = reduce_lm_head_res_alg1(i, *lm_head_res)
            lm_head_reduce_output_store = lm_head_res

            if get_args().disable_backward_fusion:
                loss_calculation_helper(i)

            p2p_communicator.send_forward(output_tensor, is_pp_last_stage(p2p_communicator.pp_group))

        for i in range(num_warmup_microbatches):
            lm_head_inputs = receive_lm_head_input(num_microbatches_remaining + i)()
            lm_head_res = lm_head_step_helper(num_microbatches_remaining + i, lm_head_inputs, False)

            if get_args().disable_backward_fusion:
                lm_head_res = reduce_lm_head_res_alg1(num_microbatches_remaining + i, *lm_head_res)
            lm_head_reduce_output_store = lm_head_res

            if get_args().disable_backward_fusion:
                loss_calculation_helper(i)

        # Restore config.grad_sync_func and config.param_sync_func.
        if forward_only:
            config.grad_sync_func = grad_sync_func

        return forward_data_store

    num_remaining_s_pass = num_microbatches
    lm_head_inputs = receive_lm_head_input(0)()
    lm_head_res = lm_head_step_helper(0, lm_head_inputs, run_timer)
    num_remaining_s_pass -= 1

    if get_args().disable_backward_fusion:
        lm_head_res = reduce_lm_head_res_alg1(0, *lm_head_res)
    else:
        lm_head_res = reduce_lm_head_res_alg2(*lm_head_res)
    lm_head_reduce_output_store = lm_head_res

    if get_args().disable_backward_fusion:
        input_embedding_forward_step_helper(
            num_microbatches - num_input_embedding_forward_steps_remaining
        )()
        num_input_embedding_forward_steps_remaining -= 1

    if num_warmup_microbatches > 0:
        p2p_communicator.send_forward(output_tensor, is_pp_last_stage(p2p_communicator.pp_group))
        if not forward_only:
            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    if num_warmup_microbatches + 1 <= num_microbatches:
        # Decide to checkpoint all layers' activations of the current micro-batch
        input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group))

    if num_warmup_microbatches + 1 <= num_microbatches:
        output_tensor = forward_step_helper(
            num_warmup_microbatches,
            input_tensor,
            run_timer,
        )

        if (not get_args().disable_backward_fusion) and is_pp_last_stage(p2p_communicator.pp_group):
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
            )
            input_tensors[0].append(input_tensor)
            output_tensors[0].append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    if get_args().disable_backward_fusion:
        loss_calculation_helper(0)

        lm_head_inputs = receive_lm_head_input(1)()
        lm_head_res = lm_head_step_helper(1, lm_head_inputs, run_timer)
        num_remaining_s_pass -= 1

        lm_head_res = reduce_lm_head_res_alg1(1, *lm_head_res)
        lm_head_reduce_output_store = lm_head_res

        if num_warmup_microbatches + 1 <= num_microbatches:
            p2p_communicator.send_forward(output_tensor, is_pp_last_stage(p2p_communicator.pp_group))
            if not forward_only:
                input_tensors[0].append(input_tensor)
                output_tensors[0].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        if num_warmup_microbatches + 2 <= num_microbatches:
            input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group))

        if num_warmup_microbatches + 2 <= num_microbatches:
            output_tensor = forward_step_helper(
                num_warmup_microbatches + 1,
                input_tensor,
                run_timer,
            )
            if is_pp_last_stage(p2p_communicator.pp_group):
                if not forward_only:
                    output_tensor_grad = None
                    input_tensors[0].append(input_tensor)
                    output_tensors[0].append(output_tensor)
                    deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    num_warmup_s_pass_rank = num_warmup_s_pass[pipeline_parallel_rank]

    if get_args().disable_backward_fusion:
        warmup_offset = 1
    else:
        warmup_offset = 0

    for i in range(num_warmup_s_pass_rank):
        if num_remaining_s_pass >= 1 - warmup_offset:
            lm_head_inputs = receive_lm_head_input(i + 1 + warmup_offset)()
            lm_head_res = lm_head_step_helper(i + 1 + warmup_offset, lm_head_inputs, run_timer)
            if (i + 1 >= num_warmup_s_pass[0]) and (num_input_embedding_forward_steps_remaining > 0):
                input_embedding_forward_callback = input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )
                num_input_embedding_forward_steps_remaining -= 1
            else:
                input_embedding_forward_callback = lambda: None
            if (
                (pipeline_parallel_rank == pipeline_parallel_size - 2)
                or (not is_bsf[pipeline_parallel_rank + 1])
            ):
                input_embedding_forward_callback()
                if get_args().disable_backward_fusion:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg1(i + 2, *lm_head_res)
                else:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)
            if i == num_warmup_s_pass_rank - 1:
                output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                    output_tensor, send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
                )
                input_tensors[0].append(input_tensor)
                output_tensors[0].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
            if (
                (pipeline_parallel_rank != pipeline_parallel_size - 2)
                and (is_bsf[pipeline_parallel_rank + 1])
            ):
                input_embedding_forward_callback()
                if get_args().disable_backward_fusion:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg1(i + 2, *lm_head_res)
                else:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)
            num_remaining_s_pass -= 1

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        if (
            (is_bsf[pipeline_parallel_rank])
            or (offsets[pipeline_parallel_rank] - f > -b)
        ):
            if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 2 + warmup_offset:
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )
                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None
            receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

        if get_args().disable_backward_fusion:
            loss_calculation_helper(i + 1)

        # print_rank_message(f"{num_microbatches_remaining=} {i=} backward_step_helper begin", rank_id=0)
        input_tensor_grad, _ = backward_step_helper(
            i, output_tensor_grad, run_timer,
        )

        if is_bsf[pipeline_parallel_rank]:
            lm_head_inputs = receive_lm_head_input_callback()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
            if (
                (num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + warmup_offset)
                and (num_input_embedding_forward_steps_remaining > 0)
            ):
                input_embedding_forward_callback = input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )
                num_input_embedding_forward_steps_remaining -= 1
            else:
                input_embedding_forward_callback = lambda: None

            input_embedding_backward_callback()
            num_remaining_s_pass -= 1

        if not is_pp_last_stage(p2p_communicator.pp_group):
            input_tensor = p2p_communicator.send_backward_recv_forward(
                input_tensor_grad, recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group),
            )

        if (
            (is_bsf[pipeline_parallel_rank])
            and (offsets[pipeline_parallel_rank] + f >= 0)
        ):
            input_embedding_forward_callback()
            if get_args().disable_backward_fusion:
                lm_head_reduce_output_store = reduce_lm_head_res_alg1(
                    num_microbatches - num_remaining_s_pass - 1, *lm_head_res
                )
            else:
                lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)

        if parallel_state.is_pipeline_last_stage():
            input_tensor = p2p_communicator.send_backward_recv_forward(
                input_tensor_grad, recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group),
            )

        if (
            (not is_bsf[pipeline_parallel_rank])
            and (offsets[pipeline_parallel_rank] - f <= -b)
        ):
            if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 2 + warmup_offset:
                input_embedding_backward_callback = input_embedding_backward_step_helper(
                    num_microbatches - num_input_embedding_backward_steps_remaining
                )
                num_input_embedding_backward_steps_remaining -= 1
            else:
                input_embedding_backward_callback = lambda: None
            receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

        # print_rank_message(f"{num_microbatches_remaining=} {i=} forward_step_helper begin", rank_id=0)
        output_tensor = forward_step_helper(
            i + num_warmup_microbatches + 1 + warmup_offset,
            input_tensor,
            run_timer,
        )

        if not is_bsf[pipeline_parallel_rank]:
            lm_head_inputs = receive_lm_head_input_callback()
            lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
            if (
                (num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + warmup_offset)
                and (num_input_embedding_forward_steps_remaining > 0)
            ):
                input_embedding_forward_callback = input_embedding_forward_step_helper(
                    num_microbatches - num_input_embedding_forward_steps_remaining
                )
                num_input_embedding_forward_steps_remaining -= 1
            else:
                input_embedding_forward_callback = lambda: None
            input_embedding_backward_callback()
            num_remaining_s_pass -= 1

        if (
            pipeline_parallel_rank
            != pipeline_parallel_size - 2
        ):
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
            )

        if (
            (not is_bsf[pipeline_parallel_rank])
            or (offsets[pipeline_parallel_rank] + f < 0)
        ):
            input_embedding_forward_callback()
            if get_args().disable_backward_fusion:
                lm_head_reduce_output_store = reduce_lm_head_res_alg1(
                    num_microbatches - num_remaining_s_pass - 1, *lm_head_res
                )
            else:
                lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)

        if pipeline_parallel_rank == pipeline_parallel_size - 2:
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
            )

        input_tensors[0].append(input_tensor)
        output_tensors[0].append(output_tensor)
        deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_microbatches - num_microbatches_remaining):
            if (
                (is_bsf[pipeline_parallel_rank]
                 or (offsets[pipeline_parallel_rank] - f > -b))
            ):
                if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 2 + warmup_offset:
                    input_embedding_backward_callback = input_embedding_backward_step_helper(
                        num_microbatches - num_input_embedding_backward_steps_remaining
                    )
                    num_input_embedding_backward_steps_remaining -= 1
                else:
                    input_embedding_backward_callback = lambda: None
                if num_remaining_s_pass >= 1 - warmup_offset:
                    receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

            if (
                get_args().disable_backward_fusion
                and is_pp_last_stage(p2p_communicator.pp_group)
                and (i + num_microbatches_remaining + 1 < num_microbatches)
            ):
                loss_calculation_helper(i + num_microbatches_remaining + 1)

            input_tensor_grad, post_process = backward_step_helper(
                i + num_microbatches_remaining, output_tensor_grad, run_timer,
            )

            s_executed = False

            if is_bsf[pipeline_parallel_rank]:
                if num_remaining_s_pass >= 1 - warmup_offset:
                    lm_head_inputs = receive_lm_head_input_callback()
                    lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
                    num_remaining_s_pass -= 1
                    s_executed = True
                input_embedding_backward_callback()

            if not is_pp_last_stage(p2p_communicator.pp_group):
                p2p_communicator.send_backward(input_tensor_grad, is_pp_first_stage(p2p_communicator.pp_group))

            if (
                s_executed
                and (offsets[pipeline_parallel_rank] + f >= 0)
            ):
                if get_args().disable_backward_fusion:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg1(
                        num_microbatches - num_remaining_s_pass - 1, *lm_head_res
                    )
                else:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)
                s_executed = False

            if is_pp_last_stage(p2p_communicator.pp_group):
                p2p_communicator.send_backward(input_tensor_grad, is_pp_first_stage(p2p_communicator.pp_group))

            if (
                (not is_bsf[pipeline_parallel_rank]
                 and (offsets[pipeline_parallel_rank] - f <= -b))
            ):
                if num_microbatches - num_remaining_s_pass >= num_warmup_s_pass[0] + 2 + warmup_offset:
                    input_embedding_backward_callback = input_embedding_backward_step_helper(
                        num_microbatches - num_input_embedding_backward_steps_remaining
                    )
                    num_input_embedding_backward_steps_remaining -= 1
                else:
                    input_embedding_backward_callback = lambda: None
                if num_remaining_s_pass >= 1 - warmup_offset:
                    receive_lm_head_input_callback = receive_lm_head_input(num_microbatches - num_remaining_s_pass)

            if not is_bsf[pipeline_parallel_rank]:
                if num_remaining_s_pass >= 1 - warmup_offset:
                    lm_head_inputs = receive_lm_head_input_callback()
                    lm_head_res = lm_head_step_helper(num_microbatches - num_remaining_s_pass, lm_head_inputs, run_timer)
                    num_remaining_s_pass -= 1
                    s_executed = True
                input_embedding_backward_callback()

            if (
                pipeline_parallel_rank
                != pipeline_parallel_size - 2
            ):
                if i + 1 < num_microbatches - num_microbatches_remaining:
                    output_tensor_grad = p2p_communicator.recv_backward(
                        send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
                    )

            if s_executed:
                if get_args().disable_backward_fusion:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg1(
                        num_microbatches - num_remaining_s_pass - 1, *lm_head_res
                    )
                else:
                    lm_head_reduce_output_store = reduce_lm_head_res_alg2(*lm_head_res)

            if (
                pipeline_parallel_rank
                == pipeline_parallel_size - 2
            ):
                if i + 1 < num_microbatches - num_microbatches_remaining:
                    output_tensor_grad = p2p_communicator.recv_backward(
                        send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
                    )

        # Launch any remaining grad reductions
        enable_grad_sync()
        if config.grad_sync_func is not None:
            config.grad_sync_func[0](model[0].parameters())
            config.grad_sync_func[1](model[1].parameters())
        disable_grad_sync()

        while num_input_embedding_backward_steps_remaining > 0:
            input_embedding_backward_step_helper(
                num_microbatches - num_input_embedding_backward_steps_remaining
            )()
            num_input_embedding_backward_steps_remaining -= 1

        if not get_args().disable_backward_fusion:
            post_process()

            lm_head_inputs = receive_lm_head_input(num_microbatches)()
            lm_head_step_helper(num_microbatches, lm_head_inputs, run_timer)

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            config.grad_sync_func[2](model[2].parameters())

        if config.finalize_model_grads_func is not None and not forward_only:

            # If defer_embedding_wgrad_compute is enabled we need to do the
            # weight gradient GEMM's here.
            finish_embedding_wgrad_compute(
                config, embedding_module, is_pp_last_stage(p2p_communicator.pp_group), tp_group
            )

            # Finalize model grads (perform full grad all-reduce / reduce-scatter for
            # data parallelism, layernorm all-reduce for sequence parallelism, and
            # embedding all-reduce for pipeline parallelism).
            config.finalize_model_grads_func(
                model,
                total_num_tokens if config.calculate_per_token_loss else None,
                pg_collection=pg_collection,
            )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store
