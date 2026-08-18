# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import contextlib
import queue
from functools import wraps

import torch
from functools import partial
from typing import Callable, Iterator, List, Optional, Union

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    backward_step_multimodule,
    clear_embedding_activation_buffer,
    check_first_val_step,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    get_tensor_device,
    set_current_microbatch
)
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.timers import Timer
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.moe.paged_stash import paged_stash_reset
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
)
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_model_config,
)

from .ripipe_schedules import forward_backward_ripipe_pipelining
from .seq1f1b.schedules import seq1f1b_forward_backward_pipelining_without_interleaving, seq1f1b_forward_backward_pipelining_with_interleaving
from hcu_megatron.core.pipeline_parallel.schedule_timers import ScheduleTimers
from hcu_megatron.core.parallel_state import get_dualpipe_chunk
from hcu_megatron.training.arguments import get_adaptor_args


def get_forward_backward_func_wrapper(fn):
    @wraps(fn)
    def wrapper(pp_size=None, vp_size=None):
        """Retrieves the appropriate forward_backward function given the
        configuration of parallel_state.

        Returns a function that will perform all of the forward and
        backward passes of the model given the pipeline model parallel
        world size and virtual pipeline model parallel world size in the
        global parallel_state.

        """

        args = get_adaptor_args()
        if args.schedule_method == "vanilla":
            if args.enable_vocab_parallel:
                from hcu_megatron.core.pipeline_parallel.vocab_parallel_schedule import (
                    forward_backward_pipelining_with_vocab_parallel
                )
                return forward_backward_pipelining_with_vocab_parallel

            return fn(pp_size=pp_size, vp_size=vp_size)
        elif args.schedule_method == "dualpipev":
            if args.enable_vocab_parallel:
                from .dualpipev.dualpipev_vocab_schedules import forward_backward_pipelining_with_cutinhalf
            else:
                from .dualpipev.dualpipev_schedules import forward_backward_pipelining_with_cutinhalf
            return forward_backward_pipelining_with_cutinhalf
        elif args.schedule_method == "seq1f1b":
            return seq1f1b_forward_backward_pipelining_without_interleaving
        elif args.schedule_method == "interleaved_seq1f1b":
            return seq1f1b_forward_backward_pipelining_with_interleaving
        elif args.schedule_method == "ripipe":
            return forward_backward_ripipe_pipelining
        elif args.schedule_method == "zb_h1":
            return forward_backward_pipelining_zbh1
        else:
            raise ValueError(f"schedule_method {args.schedule_method} is not supported")

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
        if get_adaptor_args().schedule_method == "dualpipev":
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
        if get_adaptor_args().enable_vocab_parallel:
            output_tensor = output_tensor.transpose(0, 1).contiguous()

        if loss_func is None:
            forward_data_store.append(output_tensor)
        elif not collect_non_loss_data:
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
        device = get_tensor_device(output_tensor)
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale * cp_group_size / num_microbatches)

    # Set the loss scale for Multi-Token Prediction (MTP) loss.
    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        device = get_tensor_device(output_tensor)
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=device)
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


def backward_step(input_tensor, output_tensor, output_tensor_grad, config, run_timer=False):
    from megatron.core.pipeline_parallel.schedules import backward_step as _backward_step

    if run_timer:
        ScheduleTimers.for_chunk(0).b_cnt += 1
        ScheduleTimers.for_chunk(0).b.start()
        mem_before = torch.cuda.memory_allocated()

    input_tensor_grad = _backward_step(input_tensor, output_tensor, output_tensor_grad, config)

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


def get_tensor_shapes(
    *,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
    is_recv: bool = True,
):
    """Determine tensor shapes for pipeline communication.

    For hyper connections (mHC), intermediate pipeline stages communicate n-stream tensors
    with dimension hidden_size * num_residual_streams.

    Args:
        is_recv: If True, compute shape for receiving; if False, for sending.
                 This matters for hyper connections where first/last stages have different
                 send/recv dimensions.

    Returns [()] for variable_seq_lengths mode (shapes exchanged dynamically),
    or computed shapes for fixed sequence length mode.
    """
    tensor_shapes = []

    if config.variable_seq_lengths:
        # Shapes exchanged dynamically during P2P communication
        tensor_shapes.append(())
        return tensor_shapes

    # Fixed sequence lengths - compute shape
    effective_seq_length = decoder_seq_length if decoder_seq_length is not None else seq_length
    effective_seq_length = effective_seq_length // cp_group.size()

    if config.sequence_parallel:
        effective_seq_length = effective_seq_length // tp_group.size()

    # Determine hidden dimension based on hyper connections and pipeline stage
    hidden_size = config.hidden_size
    # TODO: make this more robust, including flexible VPP layout
    if getattr(config, 'enable_hyper_connections', False) and pp_group is not None:
        pp_rank = pp_group.rank()
        pp_size = pp_group.size()
        # For hyper connections:
        # - recv: stages with rank > 0 receive n-stream (n*C) from previous stage
        # - send: stages with rank < pp_size-1 send n-stream (n*C) to next stage
        use_nstream = False
        if is_recv and pp_rank > 0:
            # Receiving from previous stage (which sends n*C)
            use_nstream = True
        elif not is_recv and pp_rank < pp_size - 1:
            # Sending to next stage (send n*C)
            use_nstream = True

        if use_nstream:
            hidden_size = hidden_size * getattr(config, 'num_residual_streams', 1)

    tensor_shapes.append((effective_seq_length, micro_batch_size, hidden_size))
    return tensor_shapes


def forward_backward_pipelining_without_interleaving(
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
    pg_collection: Optional[
        Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]
    ] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    config.batch_p2p_comm = False
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    tp_group, cp_group, cp_size = None, None, None

    # Determine if this is a multi-module pipeline
    # (used for validation and backward function selection)
    is_multimodule = isinstance(pg_collection, MultiModuleProcessGroupCollection) or isinstance(
        p2p_communicator, MultiModulePipelineCommunicator
    )

    if p2p_communicator is None and pg_collection is None:
        # Default: single-module with parallel_state groups
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )

    elif p2p_communicator is not None and pg_collection is not None:
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"

        if is_multimodule:
            # Multi-module: use language model's CP size for loss scaling
            if not config.variable_seq_lengths:
                raise ValueError(
                    "config.variable_seq_lengths=True required for multi-module pipelines"
                )
            if pg_collection.has_language_model():
                cp_size = pg_collection.get_language_model_cp_size()
            else:
                # Encoder-only ranks should not use CP loss scaling.
                cp_size = None

        elif isinstance(pg_collection, ProcessGroupCollection):
            # Single-module: extract tp/cp groups and cp_size
            assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
            assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
            tp_group = pg_collection.tp
            cp_group = pg_collection.cp
            cp_size = cp_group.size()

        else:
            raise TypeError(
                f"pg_collection must be ProcessGroupCollection or "
                f"MultiModuleProcessGroupCollection, got {type(pg_collection)}"
            )
    else:
        raise ValueError("Provide both p2p_communicator and pg_collection, or neither")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, p2p_communicator.is_pp_last_stage
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if getattr(config, "moe_paged_stash", False):
        paged_stash_reset(enabled=not forward_only, config=config)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

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

    # Compute number of warmup microbatches.
    num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

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

    # Select backward function based on whether multi-module or single-module
    if is_multimodule:
        backward_func = partial(
            backward_step_multimodule,
            language_model_module_name=pg_collection.language_model_module_name,
        )
    else:
        backward_func = backward_step

    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_group=getattr(p2p_communicator, "pp_group", None),
        is_recv=True,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_group=getattr(p2p_communicator, "pp_group", None),
        is_recv=False,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
        else:
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if (
                not get_adaptor_args().delay_1f1b_cooldown_wgrad_compute
                and num_warmup_microbatches == 0
                and last_iteration
            ):
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            if get_adaptor_args().delay_1f1b_cooldown_wgrad_compute and not last_iteration:
                model.backward_dw()

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )

            if get_adaptor_args().delay_1f1b_cooldown_wgrad_compute and last_iteration:
                model.backward_dw()

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if not get_adaptor_args().delay_1f1b_cooldown_wgrad_compute and i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communicator.recv_backward(
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            p2p_communicator.send_backward(input_tensor_grad, p2p_communicator.is_pp_first_stage)
            if get_adaptor_args().delay_1f1b_cooldown_wgrad_compute:
                model.backward_dw()

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'cuda_graph_impl') and config.cuda_graph_impl == "local":
        create_cudagraphs()

    return forward_data_store


def forward_backward_pipelining_zbh1(
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
    pg_collection: Optional[
        Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]
    ] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    tp_group, cp_group, cp_size = None, None, None

    # Determine if this is a multi-module pipeline
    # (used for validation and backward function selection)
    is_multimodule = isinstance(pg_collection, MultiModuleProcessGroupCollection) or isinstance(
        p2p_communicator, MultiModulePipelineCommunicator
    )

    if p2p_communicator is None and pg_collection is None:
        # Default: single-module with parallel_state groups
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )

    elif p2p_communicator is not None and pg_collection is not None:
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"

        if is_multimodule:
            # Multi-module: use language model's CP size for loss scaling
            if not config.variable_seq_lengths:
                raise ValueError(
                    "config.variable_seq_lengths=True required for multi-module pipelines"
                )
            if pg_collection.has_language_model():
                cp_size = pg_collection.get_language_model_cp_size()
            else:
                # Encoder-only ranks should not use CP loss scaling.
                cp_size = None

        elif isinstance(pg_collection, ProcessGroupCollection):
            # Single-module: extract tp/cp groups and cp_size
            assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
            assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
            tp_group = pg_collection.tp
            cp_group = pg_collection.cp
            cp_size = cp_group.size()

        else:
            raise TypeError(
                f"pg_collection must be ProcessGroupCollection or "
                f"MultiModuleProcessGroupCollection, got {type(pg_collection)}"
            )
    else:
        raise ValueError("Provide both p2p_communicator and pg_collection, or neither")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, p2p_communicator.is_pp_last_stage
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if getattr(config, "moe_paged_stash", False):
        paged_stash_reset(enabled=not forward_only, config=config)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

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

    # Compute number of warmup microbatches.
    num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

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

    # Select backward function based on whether multi-module or single-module
    if is_multimodule:
        backward_step_func = partial(
            backward_step_multimodule,
            language_model_module_name=pg_collection.language_model_module_name,
        )
    else:
        backward_step_func = backward_step

    def backward_step_helper(input_tensor, output_tensor, output_tensor_grad, config):
        input_tensor_grad = backward_step_func(
            input_tensor, output_tensor, output_tensor_grad, config
        )

        def backward_dw():
            model.backward_dw()

        return input_tensor_grad, backward_dw

    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_group=getattr(p2p_communicator, "pp_group", None),
        is_recv=True,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_group=getattr(p2p_communicator, "pp_group", None),
        is_recv=False,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )

    # Run 1F1B in steady state.
    chunk_backward_dw_funcs = queue.Queue()
    pipeline_parallel_rank = p2p_communicator.pp_group.rank()
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
        else:
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            input_tensor_grad, cur_backward_dw_func = backward_step_helper(
                input_tensor, output_tensor, output_tensor_grad, config
            )
            chunk_backward_dw_funcs.put(cur_backward_dw_func)

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )

            if chunk_backward_dw_funcs.qsize() > pipeline_parallel_rank:
                backward_dw_func = chunk_backward_dw_funcs.get()
                backward_dw_func()

            if not last_iteration:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communicator.recv_backward(
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            input_tensor_grad, cur_backward_dw_func = backward_step_helper(
                input_tensor, output_tensor, output_tensor_grad, config
            )
            chunk_backward_dw_funcs.put(cur_backward_dw_func)

            p2p_communicator.send_backward(input_tensor_grad, p2p_communicator.is_pp_first_stage)

            backward_dw_func = chunk_backward_dw_funcs.get()
            backward_dw_func()

        # comupte wgrad for remaining micro-batches
        if not forward_only:
            for i in range(p2p_communicator.current_stage):
                backward_dw_func = chunk_backward_dw_funcs.get()
                backward_dw_func()

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'cuda_graph_impl') and config.cuda_graph_impl == "local":
        create_cudagraphs()

    return forward_data_store
