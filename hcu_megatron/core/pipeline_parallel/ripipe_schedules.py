# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""RI-PIPE schedules: Recompute Independent Pipelining

This implementation extends the base pipeline schedules with recompute-independent
pipelining strategies that optimize the trade-off between computation and communication
by scheduling recompute operations more efficiently.
"""

import collections
import torch
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.utils import get_model_config
from megatron.training import get_args
from megatron.core.pipeline_parallel.schedules import (
    forward_step,
    backward_step,
    check_first_val_step,
    clear_embedding_activation_buffer,
    finish_embedding_wgrad_compute,
    get_pp_rank_microbatches,
    get_schedule_table,
    deallocate_output_tensor,
    get_model_type,
    contextlib,
    partial,
    nvtx_range_push,
    nvtx_range_pop,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from typing import Optional, Callable, Union, Iterator, List
from hcu_megatron.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager


def forward_backward_ripipe_pipelining(
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
    first_val_step: bool = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed, enhanced with RI-PIPE
    recompute scheduling strategies.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    # Convention used in this function:
    # num_microbatches for number of microbatches per pipeline stage;
    # num_model_chunks for virtual pipeline size;
    # then total_num_microbatches = num_microbatches * num_model_chunks.
    # Their corresponding index variables are
    # microbatch_id in [0, num_microbatches)
    # model_chunk_id in [0, num_model_chunks)
    # virtual_microbatch_id in [0, total_num_microbatches)

    config = get_model_config(model[0])
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

    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for interleaved pipeline parallelism"

    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Initialize RI-PIPE specific components
    args = get_args()
    pipeline_checkpoint_manager = None
    if args.recompute_in_bubble or args.recompute_in_advance:
        pipeline_checkpoint_manager = get_pipeline_checkpoint_manager(
            num_of_chunks=len(model))
        pipeline_checkpoint_manager.open_ri_pipe = True
        pipeline_checkpoint_manager.do_pre_recompute = True

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        # vp is ignored for clear_embedding_activation_buffer
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
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

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    forward_data_store = []
    output_tensor_grads = None
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
    else:
        output_tensor_grads = None

    pipeline_parallel_size = p2p_communicator.pp_group.size()
    pipeline_parallel_rank = p2p_communicator.pp_group.rank()

    if (
        config.microbatch_group_size_per_vp_stage > num_microbatches
        or config.microbatch_group_size_per_vp_stage < pipeline_parallel_size
    ):
        msg = (
            'The number of contiguous micro-batches in a virtual pipeline stage'
            f'should range in [PP={pipeline_parallel_size} , M={num_microbatches}]'
        )
        raise ValueError(msg)

    # If the final micro-batch group has fewer micro-batches than pipeline-parallel size,
    # the pipeline will have dependency bubbles.
    final_microbatch_group_size = num_microbatches % config.microbatch_group_size_per_vp_stage
    if 0 < final_microbatch_group_size < pipeline_parallel_size:
        msg = 'The remainder of M (the total micro-batches) divided by N (number of '
        msg += 'contiguous micro-batches in a virtual pipeline stage) should be 0, '
        msg += 'or larger than or equal to the pipeline-parallel size, but it is '
        msg += f'{final_microbatch_group_size}. '
        msg += 'Otherwise, it introduces dependency bubbles in the pipeline '
        msg += 'and reduces throughput.'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // cp_group.size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // tp_group.size()

    # Compute number of warmup and remaining microbatches.
    # seems only used for vpp
    num_model_chunks = len(model)
    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches,
        num_model_chunks,
        config.microbatch_group_size_per_vp_stage,
        forward_only=forward_only,
        overlap_moe_expert_parallel_comm=config.overlap_moe_expert_parallel_comm,
        p2p_communicator=p2p_communicator,
    )

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    # Create a tunable schedule lookup table.
    # The schedule lookup table uses the virtual_microbatch_id to find the corresponding
    # microbatch_id and model_chunk_id. For example, the tunable schedule table for
    # PP2 N3M5 with VP2 is constructed as below:
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    schedule_table = get_schedule_table(
        num_microbatches, len(model), config.microbatch_group_size_per_vp_stage
    )

    # Decouple individual lookup table for microbatch_id and model_chunk_id.
    # For example, the micro-batch table for PP2 N3M5 with VP2 is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # Similarly, the model chunk table is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    # Both tables are indexed with virtual_microbatch_id.
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    def get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def get_chunk_batch_id(microbatch_id, forward):
        """ripipe related, needed by recompute_in_bubble function."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        group_id = microbatch_id // (pipeline_parallel_size * num_model_chunks)
        intra_chunk_batch_id = (microbatch_id_in_group % pipeline_parallel_size)
        return group_id, intra_chunk_batch_id, model_chunk_id


    def should_recompute(fk):
        """ripipe related, needed by recompute_in_bubble function, used to determine
        whether a mircobatch needs to be recomputed in the 1f1b stage."""
        gid, intro_group_bid, chunk_id = get_chunk_batch_id(fk, forward=True)
        # Fix: Consider all chunk IDs, not just chunk_id == 0, to match the old version behavior
        if gid < 2:
            return False
        elif gid < 2 + num_microbatches_recompute_steady_groups:
            if intro_group_bid >= (1 + 2 * pipeline_parallel_rank):
                return True
        else:
            if intro_group_bid >= pipeline_parallel_size - num_microbatches_recompute_tail:
                return True
        return False

    def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
        """Helper method to count number of released (i.e. popped from input_tensors)
        microbatches for a model chunk."""
        if forward_only:  # Micro-batch is released after forward prop.
            return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
        else:  # Micro-batch is released after backward prop.
            # Zero backward prop in warmup.
            if virtual_microbatch_id < num_warmup_microbatches:
                return 0
            else:
                backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
                model_chunk_id = num_model_chunks - model_chunk_id - 1
                return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)

    def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
        else:
            return False

    def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
        """Determine if peers are sending, and where in data structure
        to put received tensors.
        Return a boolean if the pipeline stage expects to recv from peers, and the
        corresponding model_chunk_id for the received tensor.
        """
        recv = True
        # The leading pipeline stage is the first rank in fwd and the last rank in bwd.
        is_leading_pipeline_stage = (
            is_pp_first_stage(p2p_communicator.pp_group)
            if forward
            else is_pp_last_stage(p2p_communicator.pp_group)
        )

        last_model_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading_pipeline_stage:
            # The leading pipeline stage is ahead of the ending pipeline stage
            # (i.e. last rank in fwd and first rank in bwd) by (pipeline_parallel_size - 1).
            # Let's consider bwd as an example with PP 4:
            #       0 1 2 3 ...
            #     0 1 2 3 ...
            #   0 1 2 3 ...
            # 0 1 2 3 ...
            if virtual_microbatch_id < (pipeline_parallel_size - 1):
                # The ending stage has not produced any tensors, so no recv will be initiated.
                recv = False
                next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
            else:
                # Find the model chunk of the aligned microbatches in the ending stage.
                # For example, microbatch 0 in the ending stage is aligned with microbatch 3
                # in the leading stage.
                next_model_chunk_id = get_model_chunk_id(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )
            # Last model chunk in the final stage does not produce tensors.
            if next_model_chunk_id == last_model_chunk:
                recv = False
            if forward:
                # Model chunk id increases in forward.
                next_model_chunk_id += 1
            else:
                # Model chunk id decreases in backward.
                next_model_chunk_id -= 1
        else:
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)

        return recv, next_model_chunk_id

    def forward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id, microbatch_id):
        """Preprocess for forward_step_helper"""
        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
            if (
                param_sync_virtual_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
            ):
                param_sync_chunk_id = (
                    get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
                )
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if _is_vp_first_stage(vp_stage=model_chunk_id) and is_pp_first_stage(pp_group):
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)

        # For non-depth-first pipeline schedules, the first rank would buffer multiple received
        # activation tensors for a model chunk until accessed during warmup.
        # This input buffering is needed to overlap the computation with the receipt of
        # the next inputs. To index the proper buffered inputs for forword_step, we use
        # microbatch_id offset with number of released microbatches that have completed backprop.
        offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
        input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

        return input_tensor

    def forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens):
        """Postprocess for forward_step_helper"""
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens

        # If forward-only, no need to save tensors for a backward pass.
        if forward_only:
            # Release the tensor that have completed forward step.
            input_tensors[model_chunk_id].pop(0)
            output_tensors[model_chunk_id].pop()

        return

    def forward_step_helper(virtual_microbatch_id, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks"""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
        microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id, forward=True)

        input_tensor = forward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id, microbatch_id
        )

        # Set the virtual pipeline model parallel rank before forward pass
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=pg_collection.cp.size(),
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
            vp_stage=model_chunk_id,
            is_last_stage=_is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group),
        )

        # RI-PIPE: When a microbatch finish its forward pass, save needed recomputation
        # functions for this microbatch.
        if args.recompute_in_bubble or args.recompute_in_advance:
            pipeline_checkpoint_manager.batch_fin(model_chunk_id)

        forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens)

        return output_tensor

    def backward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id):
        """Preprocess for backward_step_helper"""
        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
            virtual_microbatch_id
        ):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        # pylint: disable=E0606
        if _is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group):
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        return input_tensor, output_tensor, output_tensor_grad

    def backward_step_helper_postprocess(virtual_microbatch_id):
        """Postprocess for backward_step_helper"""
        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
            if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_virtual_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(
                    grad_sync_virtual_microbatch_id, forward=False
                )
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

    def backward_step_helper(virtual_microbatch_id):
        """Helper method to run backward step with model split into chunks"""
        nonlocal output_tensor_grads
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)

        # Set the virtual pipeline model parallel rank before processing recompute functions
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # RI-PIPE: Ensure recompute operations are completed before backward pass
        # This is critical to avoid the "recompute is not done" error
        if pipeline_checkpoint_manager.open_ri_pipe and pipeline_checkpoint_manager.do_pre_recompute:
            # Process any recompute functions in the chunk list before backward pass
            # Only process recompute functions if the chunk_list for this model_chunk_id is not empty
            # and chunk_do_recompute is enabled
            if pipeline_checkpoint_manager.chunk_do_recompute and \
               len(pipeline_checkpoint_manager.chunk_list[model_chunk_id]) > 0:
                pipeline_checkpoint_manager.recompute_next(model_chunk_id)

        input_tensor, output_tensor, output_tensor_grad = backward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id
        )

        # Use the original backward_step function without additional grad_sync manipulation
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        backward_step_helper_postprocess(virtual_microbatch_id)

        return input_tensor_grad
                
    def forward_backward_helper_wrapper(
        f_virtual_microbatch_id=None,
        b_virtual_microbatch_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        checkpoint_activations_microbatch=None,
    ):
        """
        wrap forward_helper, backward_helper, and combined_forward_backward_helper in a unified way
        """
        from megatron.core.pipeline_parallel.combined_1f1b import (
            combined_1f1b_schedule_for_interleaved_pipelining,
        )
        if config.overlap_moe_expert_parallel_comm and not forward_only:  # Combined 1F1B path
            return combined_1f1b_schedule_for_interleaved_pipelining(
                config,
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                forward_data_store,
                forward_step_helper_preprocess,
                forward_step_helper_postprocess,
                backward_step_helper_preprocess,
                backward_step_helper_postprocess,
                get_microbatch_id_in_model_chunk,
                get_model_chunk_id,
                partial(check_first_val_step, first_val_step, forward_only),
                is_first_microbatch_for_model_chunk,
                collect_non_loss_data,
                f_virtual_microbatch_id=f_virtual_microbatch_id,
                b_virtual_microbatch_id=b_virtual_microbatch_id,
                pre_forward=pre_forward,
                pre_backward=pre_backward,
                post_forward=post_forward,
                post_backward=post_backward,
            )
        else:  # Conventional interleaved 1F1B path
            forward_output_tensor = None
            backward_input_tensor_grad = None
            # forward pass
            if f_virtual_microbatch_id is not None:
                forward_model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
                if pre_forward is not None:
                    pre_forward()
                forward_output_tensor = forward_step_helper(
                    f_virtual_microbatch_id, checkpoint_activations_microbatch
                )
                if post_forward is not None:
                    forward_output_tensor = post_forward(forward_output_tensor)

            # Backward pass.
            if b_virtual_microbatch_id is not None:
                backward_model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
                if pre_backward is not None:
                    pre_backward()
                backward_input_tensor_grad = backward_step_helper(b_virtual_microbatch_id)
                if post_backward is not None:
                    backward_input_tensor_grad = post_backward(backward_input_tensor_grad)
            return forward_output_tensor, backward_input_tensor_grad

    # ==============================main logic=========================================
    _is_vp_first_stage = partial(
        is_vp_first_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    _is_vp_last_stage = partial(
        is_vp_last_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    pp_group = p2p_communicator.pp_group

    # ripipe related, calculate the variables needed by the recompute_in_bubble function
    num_microbatches_recompute, num_microbatches_recompute_forward, num_microbatches_recompute_steady_groups, \
        num_microbatches_recompute_tail = get_ripipe_recompute_count_params(num_microbatches,
                                                                            num_model_chunks,
                                                                            num_warmup_microbatches)

    # Run warmup forward passes.
    nvtx_range_push(suffix="warmup")
    input_tensors[0].append(
        p2p_communicator.recv_forward(
            tensor_shape, _is_vp_first_stage(vp_stage=0) and is_pp_first_stage(pp_group)
        )
    )

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if is_pp_first_stage(p2p_communicator.pp_group):
        fwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if is_pp_last_stage(p2p_communicator.pp_group):
        bwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        bwd_recv_buffer_size = 1
    fwd_recv_buffer = [None] * fwd_recv_buffer_size
    bwd_recv_buffer = [None] * bwd_recv_buffer_size
    recv_prev_wait_handles = []
    send_next_wait_handle = None
    send_prev_wait_handle = None
    recv_next_wait_handles = []

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)

        # ripipe related, when use recompute_in_bubble function, do not do recompute
        # for the first pp * vp microbatches.
        if args.recompute_in_bubble:
            if k < pipeline_parallel_size * num_model_chunks:
                pipeline_checkpoint_manager.disable_recompute()
            else:
                num_microbatches_recompute_forward -= 1
        if args.recompute_in_bubble or args.recompute_in_advance:
            pipeline_checkpoint_manager.enable_recompute()

        if config.overlap_p2p_comm_warmup_flush:
            if (
                not (
                    _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group)
                )
                and k != 0
            ):
                assert recv_prev_wait_handles, (
                    f'pp rank {pipeline_parallel_rank}, iteration {k},'
                    'should have registered recv handle'
                )
                recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                recv_prev_wait_handle.wait()

        # Determine if tensor should be received from previous stage.
        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

        # No receive in last iteration when recv iteration k+1.
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Prefetch recv for iteration k+1 for non-first ranks.
        if config.overlap_p2p_comm_warmup_flush and not is_pp_first_stage(
            p2p_communicator.pp_group
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communicator.send_forward_recv_forward(
                    output_tensor=None,  # No output_tensor to send.
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )
            )

            if fwd_wait_recv_handles:
                recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        output_tensor, _ = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=k,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # Don't send tensor downstream if on last stage.
        if _is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group):
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm_warmup_flush:
            if (
                k == (num_warmup_microbatches - 1)
                and not config.overlap_p2p_comm
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communicator.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                    )
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape
                )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not is_pp_first_stage(p2p_communicator.pp_group):
                # Send only since recv prefetched.
                _, fwd_wait_handles = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=False, tensor_shape=tensor_shape, overlap_p2p_comm=True
                )
            else:  # No prefetch for first rank, so both send and recv initiated.
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(k + 1) % fwd_recv_buffer_size] = None

        if config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                if recv_next:
                    output_tensor_grads[num_model_chunks - 1].append(bwd_recv_buffer[-1])
    nvtx_range_pop(suffix="warmup")

    # Run 1F1B in steady state.
    nvtx_range_push(suffix="steady")
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        
        # ripipe related, when use recompute_in_bubble function, do not do recompute
        # for the first pp * vp microbatches.
        if args.recompute_in_bubble:
            if forward_k < pipeline_parallel_size * num_model_chunks:
                pipeline_checkpoint_manager.disable_recompute()
            else:
                num_microbatches_recompute_forward -= 1
        if args.recompute_in_bubble or args.recompute_in_advance:
            pipeline_checkpoint_manager.enable_recompute()
        
        if config.overlap_p2p_comm:

            backward_k = k

            # Sync forward recv
            def pp_pre_forward(vp_stage=None):
                nonlocal num_microbatches_recompute_forward
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                if not (_is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_prev_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, fwd iteration {forward_k}, '
                            'should have registered recv handle'
                        )
                        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                        recv_prev_wait_handle.wait()
                    else:
                        if recv_prev_wait_handles is not None and recv_prev_wait_handles:
                            recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                            recv_prev_wait_handle.wait()

                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

                # ripipe related, determine whether this microbatch should be recomputed
                # when using recompute_in_bubble function.
                if args.recompute_in_bubble:
                    if num_microbatches_recompute_forward > 0:
                        num_microbatches_recompute_forward -= 1
                    elif num_microbatches_recompute > 0 and should_recompute(forward_k):
                        pass
                    else:
                        pipeline_checkpoint_manager.disable_recompute()
                
                if args.recompute_in_bubble or args.recompute_in_advance:
                    pipeline_checkpoint_manager.enable_recompute()

            # Async forward send / receive
            def pp_post_forward(output_tensor, vp_stage=None):
                nonlocal send_next_wait_handle
                nonlocal fwd_recv_buffer
                nonlocal fwd_wait_handles
                nonlocal recv_prev_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                # Last virtual stage no activation tensor to send.
                if _is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group):
                    output_tensor = None

                recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                    forward_k, forward=True
                )

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_next_wait_handle is not None:
                    send_next_wait_handle.wait()
                if fwd_wait_handles is not None:
                    send_next_wait_handle = (
                        fwd_wait_handles.pop("send_next")
                        if "send_next" in fwd_wait_handles
                        else None
                    )
                    if "recv_prev" in fwd_wait_handles:
                        recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
                # assert fwd_wait_handles is not None

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.
                if recv_prev:
                    input_tensors[next_forward_model_chunk_id].append(
                        fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                    )
                    fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None

                return output_tensor

            # Sync backward recv
            def pp_pre_backward(vp_stage=None):
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                if not (_is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_next_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, bwd iteration {backward_k}, '
                            'should have registered recv next handle'
                        )
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()
                    else:
                        if recv_next_wait_handles is not None and recv_next_wait_handles:
                            recv_next_wait_handle = recv_next_wait_handles.pop(0)
                            recv_next_wait_handle.wait()

            # Async backward send / receive
            def pp_post_backward(input_tensor_grad, vp_stage=None):
                nonlocal send_prev_wait_handle
                nonlocal bwd_wait_handles
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                # First virtual stage no activation gradient tensor to send.
                if _is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group):
                    input_tensor_grad = None

                recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                    backward_k, forward=False
                )

                (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[backward_k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(backward_k + 1) % bwd_recv_buffer_size] = None
                return input_tensor_grad

            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                pre_forward=pp_pre_forward,
                pre_backward=pp_pre_backward,
                post_forward=pp_post_forward,
                post_backward=pp_post_backward,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )
            
            # ripipe related, actually do the recomputation.
            if args.recompute_in_advance or args.recompute_in_bubble:
                vpp_rank = get_model_chunk_id(k, forward=False)
                # Set the virtual pipeline model parallel rank as in the original implementation
                parallel_state.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                # TODO: remove chunk_list[0]) > 0
                if len(pipeline_checkpoint_manager.chunk_list[vpp_rank]) > 0:
                    pipeline_checkpoint_manager.recompute_next(vpp_rank)
                
        else:  # No p2p overlap.
            backward_k = k
            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )
            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            if _is_vp_last_stage(vp_stage=forward_model_chunk_id) and is_pp_last_stage(pp_group):
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            if _is_vp_first_stage(vp_stage=backward_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (input_tensor, output_tensor_grad) = (
                p2p_communicator.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
            
        # ripipe related, actually do the recomputation.
        # Need to recompute in both p2p comm overlap and non-overlap cases
        if args.recompute_in_advance or args.recompute_in_bubble:
            # Fix: Use the correct model chunk ID for recompute
            vpp_rank = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            # TODO: remove chunk_list[0]) > 0
            if len(pipeline_checkpoint_manager.chunk_list[vpp_rank]) > 0:
                pipeline_checkpoint_manager.recompute_next(vpp_rank)
                
    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
    nvtx_range_pop(suffix="steady")

    # Run cooldown backward passes (flush out pipeline) for the last model chunk.
    nvtx_range_push(suffix="cooldown")
    curr_vp_stage = config.virtual_pipeline_model_parallel_size - 1
    if not forward_only:
        # ripipe related, actually do the recomputation.
        if args.recompute_in_advance:
            # Fix: Use correct virtual pipeline rank for cooldown
            vpp_rank = get_model_chunk_id(total_num_microbatches - 1, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            # TODO: remove chunk_list[0]) > 0
            if len(pipeline_checkpoint_manager.chunk_list[vpp_rank]) > 0:
                pipeline_checkpoint_manager.recompute_next(vpp_rank)
        if args.recompute_in_bubble and num_microbatches_recompute > 0:
            old_vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            # TODO: remove chunk_list[0]) > 0
            if len(pipeline_checkpoint_manager.chunk_list[0]) > 0:
                pipeline_checkpoint_manager.recompute_next_force(0)
            parallel_state.set_virtual_pipeline_model_parallel_rank(old_vpp_rank)
        
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if are_all_microbatches_in_warmup:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communicator.recv_backward(
                    tensor_shape,
                    is_last_stage=(
                        _is_vp_last_stage(vp_stage=curr_vp_stage) and is_pp_last_stage(pp_group)
                    ),
                )
            )
            
        # ripipe related
        if args.recompute_in_bubble:
            num_microbatches_recompute_forward = 1
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            if (
                not (_is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group))
                and k != 0
            ):
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, backward iteration {k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                k, forward=False
            )

            if k == (total_num_microbatches - 1):
                recv_next = False

            # Prefetch recv for backward iteration k+1 for non last ranks.
            if config.overlap_p2p_comm_warmup_flush and not is_pp_last_stage(
                p2p_communicator.pp_group
            ):
                # bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                out_tensor, bwd_wait_recv_handles = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad=None,  # No input_tensor_grad to send.
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            _, input_tensor_grad = forward_backward_helper_wrapper(b_virtual_microbatch_id=k)

            # First virtual stage no activation gradient tensor to send.
            if _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            # ripipe related, use async communication
            out_tensor, bwd_wait_handles = p2p_communicator.send_backward_recv_backward(
                input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, 
                overlap_p2p_comm=True
            )
            output_tensor_grads[next_backward_model_chunk_id].append(
                out_tensor
            )

            # ripipe related, actually do the recomputation
            if args.recompute_in_bubble and num_microbatches_recompute > 0 and \
                    num_microbatches_recompute_forward < num_microbatches_recompute:
                old_vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
                # TODO: remove chunk_list[0]) > 0
                if len(pipeline_checkpoint_manager.chunk_list[0]) > 0:
                    pipeline_checkpoint_manager.recompute_next_force(0)
                parallel_state.set_virtual_pipeline_model_parallel_rank(old_vpp_rank)
                num_microbatches_recompute_forward += 1
            if args.recompute_in_advance and k != (total_num_microbatches - 1):
                vpp_rank = get_model_chunk_id(k + 1, forward=False)
                parallel_state.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                # TODO: remove chunk_list[0]) > 0
                if len(pipeline_checkpoint_manager.chunk_list[vpp_rank]) > 0:
                    pipeline_checkpoint_manager.recompute_next(vpp_rank)
                
            # ripipe related, use async communication
            if config.overlap_p2p_comm and bwd_wait_handles is not None:
                if isinstance(bwd_wait_handles, dict):
                    for wait_handle in bwd_wait_handles.values():
                        if hasattr(wait_handle, 'wait'):
                            wait_handle.wait()
                else:
                    for wait_handle in bwd_wait_handles:
                        if hasattr(wait_handle, 'wait'):
                            wait_handle.wait()

        if send_prev_wait_handle is not None:
            send_prev_wait_handle.wait()

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
    nvtx_range_pop(suffix="cooldown")

    nvtx_range_push(suffix="misc")
    assert (
        not recv_prev_wait_handles
    ), 'recv_prev_wait_handles should be cleared at the end of a step'
    assert (
        not recv_next_wait_handles
    ), 'recv_next_wait_handles should be cleared at the end of a step'

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

    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and config.cuda_graph_scope != "full_iteration"
    ):
        create_cudagraphs()
    nvtx_range_pop(suffix="misc")

    # Apply RI-PIPE specific logic during the execution
    # This includes managing recompute scheduling based on the checkpoint manager
    if args.recompute_in_bubble or args.recompute_in_advance:
        # Check that all recompute operations are completed at the end of iteration
        pipeline_checkpoint_manager.iter_fin()
    
    return forward_data_store


def get_ripipe_recompute_count_params(num_microbatches, num_model_chunks, num_warmup_microbatches):
    """ripipe related, calculate the variables needed by the recompute_in_bubble function"""
    args = get_args()
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    num_microbatches_recompute_steady_groups = 0
    num_microbatches_recompute_tail = 0
    num_microbatches_recompute = 0
    num_microbatches_recompute_forward = 0
    if args.recompute_in_bubble and num_microbatches // pipeline_parallel_size > 1:
        num_microbatches_recompute = num_warmup_microbatches + 1 - num_model_chunks * pipeline_parallel_size
        if num_microbatches_recompute < 0:
            num_microbatches_recompute = 0

        num_microbatches_recompute_forward = num_microbatches_recompute
        if num_microbatches_recompute > 0 and num_microbatches // pipeline_parallel_size >= 3:
            num_microbatches_recompute_steady_groups = (num_microbatches // pipeline_parallel_size) - 3
            num_microbatches_recompute_tail = 2 + 2 * pipeline_parallel_rank
            if num_microbatches_recompute_steady_groups == 0:
                if num_microbatches_recompute_tail >= pipeline_parallel_size - 1 - 2 * pipeline_parallel_rank:
                    num_microbatches_recompute_tail = 0
                    num_microbatches_recompute_steady_groups = 1
            else:
                num_microbatches_recompute_tail = 1

    params = collections.namedtuple('RecomputeCountParams',
                                    ['num_microbatches_recompute', 'num_microbatches_recompute_forward',
                                     'num_microbatches_recompute_steady_groups', 'num_microbatches_recompute_tail'])
    return params(num_microbatches_recompute, num_microbatches_recompute_forward,
                  num_microbatches_recompute_steady_groups, num_microbatches_recompute_tail)
