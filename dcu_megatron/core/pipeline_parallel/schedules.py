import contextlib
import torch

from functools import wraps, partial
from typing import Iterator, List, Union, Optional, Callable

from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.utils import (
    get_model_config,
    get_model_type,
    get_model_xattn,
)
from megatron.core.pipeline_parallel.schedules import (
    forward_step,
    backward_step,
    get_tensor_shapes,
    get_schedule_table,
    check_first_val_step,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    clear_embedding_activation_buffer,
)
from megatron.core.utils import (
    nvtx_range_pop,
    nvtx_range_push,
)

from .combined_1f1b import forward_backward_step
from .utils import VppContextManager, set_streams
from .dualpipev.dualpipev_schedules import forward_backward_pipelining_with_cutinhalf
from ..transformer import PipelineOffloadManager


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
        elif args.schedule_method == "interleaved_1f1b":
            return forward_backward_pipelining_with_interleaving
        else:
            raise ValueError(f"schedule_method {args.schedule_method} is not supported")

    return wrapper


def get_pp_rank_microbatches(
    num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage, forward_only=False
):
    """Get the number of total, warmup, and remaining microbatches in PP scheduling."""
    args = get_args()

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    virtual_pipeline_parallel_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    total_num_microbatches = num_microbatches * num_model_chunks
    are_all_microbatches_in_warmup = False

    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    elif pipeline_parallel_size > 1:
        if virtual_pipeline_parallel_size is None:
            # forward_backward_pipelining_without_interleaving
            num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
        else:
            # forward_backward_pipelining_with_interleaving
            # Run (num_model_chunks-1)*microbatch_group_size_per_vp_stage on
            # all workers, followed by more microbatches after depending on
            # stage ID (more forward passes for earlier stages, later stages can
            # immediately start with 1F1B).
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage

            if args.overlap_moe_expert_parallel_comm:
                num_warmup_microbatches = num_warmup_microbatches + 1
    else:
        # forward_backward_no_pipelining
        # This path is only used for cuda graph capturing compatibility for the PP=1 case.
        num_warmup_microbatches = 0

    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches
        are_all_microbatches_in_warmup = True
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    return (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    )


def forward_backward_pipelining_with_interleaving(
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
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    # Convention used in this function:
    # num_microbatches for number of microbatches per pipeline stage;
    # num_model_chunks for virtual pipeline size;
    # then total_num_microbatches = num_microbatches * num_model_chunks.
    # Their corresponding index variables are
    # microbatch_id in [0, num_microbatches)
    # model_chunk_id in [0, num_model_chunks)
    # virtual_microbatch_id in [0, total_num_microbatches)

    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for interleaved pipeline parallelism"

    config = get_model_config(model[0])

    set_streams()

    if not forward_only and config.fine_grained_activation_offloading:
        PipelineOffloadManager.get_instance().reset()

    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

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

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    forward_data_store = []
    output_tensor_grads = None
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
    else:
        output_tensor_grads = None

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

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

    if model_type == ModelType.encoder_and_decoder:
        xattn_needed = get_model_xattn(model)
        assert (
            not xattn_needed
        ), "Interleaving is not supported when xattn is required between encoder and decoder"
        tensor_shape = get_tensor_shapes(
            rank=parallel_state.get_pipeline_model_parallel_rank(),
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
            encoder_decoder_xattn=xattn_needed,
        )
        tensor_shape = list(tensor_shape[0])
    else:
        tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
        if config.sequence_parallel:
            tensor_shape[0] = (
                tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
            )

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches, num_model_chunks, config.microbatch_group_size_per_vp_stage, forward_only
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
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            if forward
            else parallel_state.is_pipeline_last_stage(ignore_virtual=True)
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
        if parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=model_chunk_id):
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

    def forward_step_helper(
        virtual_microbatch_id, microbatch_id, checkpoint_activations_microbatch
    ):
        """Helper method to run forward step with model split into chunks"""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)

        input_tensor = forward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id, microbatch_id
        )

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
            vp_stage=model_chunk_id,
        )

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
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=model_chunk_id):
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
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)

        input_tensor, output_tensor, output_tensor_grad = backward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id
        )

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        backward_step_helper_postprocess(virtual_microbatch_id)

        return input_tensor_grad

    def combined_forward_backward_helper(
        f_virtual_microbatch_id=None,
        b_virtual_microbatch_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """Helper method to run combined forward and backward step"""
        # forward prepare
        f_model_chunk_id = None
        f_microbatch_id = None
        if f_virtual_microbatch_id is not None:
            f_microbatch_id = get_microbatch_id_in_model_chunk(f_virtual_microbatch_id, True)
        f_context = contextlib.nullcontext()
        input_tensor = None
        if f_virtual_microbatch_id is not None:
            f_model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
            f_context = VppContextManager(f_model_chunk_id)
            with f_context:
                input_tensor = forward_step_helper_preprocess(
                    f_virtual_microbatch_id, f_model_chunk_id, f_microbatch_id
                )

        # backward prepare
        b_model_chunk_id = None
        b_context = contextlib.nullcontext()
        b_input_tensor = None
        b_output_tensor = None
        b_output_tensor_grad = None
        if b_virtual_microbatch_id is not None:
            model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
            b_model_chunk_id = model_chunk_id
            b_context = VppContextManager(b_model_chunk_id)
            with b_context:
                b_input_tensor, b_output_tensor, b_output_tensor_grad = (
                    backward_step_helper_preprocess(b_virtual_microbatch_id, b_model_chunk_id)
                )

        output_tensor, num_tokens, input_tensor_grad, _ = forward_backward_step(
            forward_step_func,
            data_iterator[f_model_chunk_id] if f_model_chunk_id is not None else None,
            model[f_model_chunk_id] if f_model_chunk_id is not None else None,
            num_microbatches,
            input_tensor,
            forward_data_store,
            model[b_model_chunk_id] if b_model_chunk_id is not None else None,
            b_input_tensor,
            b_output_tensor,
            b_output_tensor_grad,
            config,
            f_context=f_context,
            b_context=b_context,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=check_first_val_step(
                first_val_step,
                forward_only,
                (
                    is_first_microbatch_for_model_chunk(f_virtual_microbatch_id)
                    if f_virtual_microbatch_id is not None
                    else None
                ),
            ),
            current_microbatch=f_microbatch_id,
        )

        # forward post process
        if f_model_chunk_id is not None:
            with f_context:
                forward_step_helper_postprocess(f_model_chunk_id, output_tensor, num_tokens)

        # backward post process
        if b_model_chunk_id:
            with b_context:
                backward_step_helper_postprocess(b_virtual_microbatch_id)
                if input_tensor is not None:
                    assert input_tensor_grad is not None

        return output_tensor, input_tensor_grad

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
        wrap forward_helper、backward_helper、combined_forward_backward_helper in a unified way
        """

        if config.overlap_moe_expert_parallel_comm and not forward_only:
            assert (
                checkpoint_activations_microbatch is None
            ), "checkpoint_activations_microbatch not supported when overlap_moe_expert_parallel_comm is true"
            return combined_forward_backward_helper(
                f_virtual_microbatch_id=f_virtual_microbatch_id,
                b_virtual_microbatch_id=b_virtual_microbatch_id,
                pre_forward=pre_forward,
                pre_backward=pre_backward,
                post_forward=post_forward,
                post_backward=post_backward,
            )
        else:
            output_tensor = None
            input_tensor_grad = None
            if f_virtual_microbatch_id is not None:
                # forward pass
                if pre_forward is not None:
                    pre_forward()

                microbatch_id = get_microbatch_id_in_model_chunk(f_virtual_microbatch_id, forward=True)
                output_tensor = forward_step_helper(
                    f_virtual_microbatch_id, microbatch_id, checkpoint_activations_microbatch
                )
                if post_forward is not None:
                    output_tensor = post_forward(output_tensor)

            if b_virtual_microbatch_id is not None:
                # Backward pass.
                if pre_backward is not None:
                    pre_backward()
                input_tensor_grad = backward_step_helper(b_virtual_microbatch_id)
                if post_backward is not None:
                    input_tensor_grad = post_backward(input_tensor_grad)
            return output_tensor, input_tensor_grad

    is_vp_first_stage = partial(parallel_state.is_pipeline_first_stage, ignore_virtual=False)
    is_vp_last_stage = partial(parallel_state.is_pipeline_last_stage, ignore_virtual=False)

    # Run warmup forward passes.
    nvtx_range_push(suffix="warmup")
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(
        p2p_communication.recv_forward(tensor_shape, config, is_vp_first_stage(vp_stage=0))
    )

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        fwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
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

        if config.overlap_p2p_comm_warmup_flush:
            if not is_vp_first_stage(vp_stage=cur_model_chunk_id) and k != 0:
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
        if config.overlap_p2p_comm_warmup_flush and not parallel_state.is_pipeline_first_stage(
            ignore_virtual=True
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communication.send_forward_recv_forward(
                    output_tensor=None,  # No output_tensor to send.
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    config=config,
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
        if is_vp_last_stage(vp_stage=cur_model_chunk_id):
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
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                    )
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config
                )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # Send only since recv prefetched.
                _, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=False,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            else:  # No prefetch for first rank, so both send and recv initiated.
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communication.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        config=config,
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
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
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

        
        microbatch_id = get_microbatch_id_in_model_chunk(forward_k, forward=True)
        if config.overlap_p2p_comm:
            def pp_pre_forward(vp_stage=None):
                nonlocal recv_prev_wait_handles

                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)

                if not is_vp_first_stage(vp_stage=vp_stage):

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

            def pp_post_forward(output_tensor, vp_stage=None):
                nonlocal send_next_wait_handle
                nonlocal fwd_recv_buffer
                nonlocal fwd_wait_handles
                nonlocal recv_prev_wait_handles

                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)

                # Last virtual stage no activation tensor to send.
                if is_vp_last_stage(vp_stage=vp_stage):
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
                    p2p_communication.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        config=config,
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
                # assert fwd_wait_handles is not None

                if recv_prev:
                    input_tensors[next_forward_model_chunk_id].append(
                        fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                    )
                    fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None

                return output_tensor

            # Backward pass.
            backward_k = k
            # grad send receive sync
            def pp_pre_backward(vp_stage=None):
                nonlocal recv_next_wait_handles

                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                if not is_vp_last_stage(vp_stage=vp_stage):
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

            # async grad send receive
            def pp_post_backward(input_tensor_grad, vp_stage=None):
                nonlocal send_prev_wait_handle
                nonlocal bwd_wait_handles
                nonlocal recv_next_wait_handles
                nonlocal bwd_recv_buffer

                # First virtual stage no activation gradient tensor to send.
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                if is_vp_first_stage(vp_stage=vp_stage):
                    input_tensor_grad = None

                recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                    backward_k, forward=False
                )

                (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev") if "send_prev" in bwd_wait_handles else None
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
            if is_vp_last_stage(vp_stage=forward_model_chunk_id):
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            if is_vp_first_stage(vp_stage=backward_model_chunk_id):
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
                p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
    nvtx_range_pop(suffix="steady")

    # Run cooldown backward passes (flush out pipeline) for the last model chunk.
    nvtx_range_push(suffix="cooldown")
    curr_vp_stage = config.virtual_pipeline_model_parallel_size - 1
    if not forward_only:
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if are_all_microbatches_in_warmup:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(
                    tensor_shape,
                    config=config,
                    is_last_stage=is_vp_last_stage(vp_stage=curr_vp_stage),
                )
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            if not is_vp_last_stage(vp_stage=cur_model_chunk_id) and k != 0:
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
            if config.overlap_p2p_comm_warmup_flush and not parallel_state.is_pipeline_last_stage(
                ignore_virtual=True
            ):
                bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad=None,  # No input_tensor_grad to send.
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            _, input_tensor_grad = forward_backward_helper_wrapper(b_virtual_microbatch_id=k)

            # First virtual stage no activation gradient tensor to send.
            if is_vp_first_stage(vp_stage=cur_model_chunk_id):
                input_tensor_grad = None

            if config.overlap_p2p_comm_warmup_flush:
                if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    _, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=False,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                else:
                    bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_handles = (
                        p2p_communication.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            config=config,
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
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(k + 1) % bwd_recv_buffer_size] = None

            else:
                output_tensor_grad = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config
                )

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

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

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()
    nvtx_range_pop(suffix="misc")

    return forward_data_store
