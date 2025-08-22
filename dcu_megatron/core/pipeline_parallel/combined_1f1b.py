import contextlib
from contextlib import nullcontext

import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.pipeline_parallel.utils import ScheduleNode
from megatron.core.utils import get_attr_wrapped_model
from megatron.training.utils import unwrap_model

from dcu_megatron.core.parallel_state import get_dualpipe_chunk
from dcu_megatron.core.pipeline_parallel.utils import AbstractSchedulePlan


def forward_step_calc_loss(
    model,
    output_tensor,
    loss_func,
    config,
    vp_stage,
    collect_non_loss_data,
    num_microbatches,
    forward_data_store,
):
    """Calculate the loss and number of tokens for forward_step()"""
    model_vp_stage = getattr(model, "vp_stage", None)
    if vp_stage is not None and model_vp_stage is not None:
        assert (
            vp_stage == model_vp_stage
        ), f"vp_stage ({vp_stage}) doesn't match model_vp_stage ({model_vp_stage})"
    num_tokens = torch.tensor(0, dtype=torch.int)

    args = get_args()
    is_last_stage = False
    if args.schedule_method == "dualpipev":
        is_last_stage = parallel_state.is_pipeline_first_stage() and get_dualpipe_chunk() == 1
    else:
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    if is_last_stage:
        if not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
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

    return output_tensor, num_tokens


def forward_backward_step(
    forward_step_func,
    data_iterator,
    f_model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    b_model,
    b_input_tensor,
    b_output_tensor,
    b_output_tensor_grad,
    config,
    f_context=None,
    b_context=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
    block_level_wgrad_compute=False,
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
        block_level_wgrad_compute (bool, optional):
            Delay the wgrad compute for batch-level overlapping

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """

    assert (
        checkpoint_activations_microbatch is None
    ), "checkpoint_activations_microbatch is not supported for overlap_moe_expert_parallel_comm"

    from megatron.core.pipeline_parallel.schedules import set_current_microbatch

    if f_model is not None and config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    # forward preprocess, the same as the forward_step()
    unwrap_output_tensor = False
    f_schedule_plan = None
    if f_model is not None:
        with f_context:
            if is_first_microbatch and hasattr(f_model, 'set_is_first_microbatch'):
                f_model.set_is_first_microbatch()
            if current_microbatch is not None:
                set_current_microbatch(f_model, current_microbatch)

            if not isinstance(input_tensor, list):
                input_tensor = [input_tensor]
                unwrap_output_tensor = True

            set_input_tensor = get_attr_wrapped_model(f_model, "set_input_tensor")
            set_input_tensor(input_tensor)

            with context_manager:
                unwrapped_model = unwrap_model(f_model)
                f_schedule_plan, loss_func = forward_step_func(data_iterator, unwrapped_model, return_schedule_plan=True)
                assert isinstance(
                    f_schedule_plan, AbstractSchedulePlan
                ), "first output of forward_step_func must be one instance of AbstractSchedulePlan"

    # backward preprocess
    unwrap_input_tensor_grad = False
    b_schedule_plan = None
    if b_model is not None:
        # Retain the grad on the input_tensor.
        if not isinstance(b_input_tensor, list):
            b_input_tensor = [b_input_tensor]
            unwrap_input_tensor_grad = True

        for x in b_input_tensor:
            if x is not None:
                x.retain_grad()

        if not isinstance(b_output_tensor, list):
            b_output_tensor = [b_output_tensor]
        if not isinstance(b_output_tensor_grad, list):
            b_output_tensor_grad = [b_output_tensor_grad]

        # Backward pass for loss function
        b_schedule_plan = b_output_tensor[0].schedule_plan
        b_output_tensor[0].schedule_plan = None
        if b_output_tensor_grad[0] is None:
            # backward schedule plan
            loss_node = b_output_tensor[0].loss_func
            b_output_tensor[0].loss_func = None
            if config.grad_scale_func is not None:
                b_output_tensor[0] = config.grad_scale_func(b_output_tensor[0])

            torch.autograd.backward(b_output_tensor[0], grad_tensors=b_output_tensor_grad[0])
            b_output_tensor_grad[0] = loss_node.get_grad()

    # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
    # otherwise do nothing extra at the outer level
    # if we are using other fp8 recipes, then the context manager enter&exit are free
    # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
    # control which layer will be fp8 or bf16
    use_outer_fp8_context = config.fp8 and config.fp8_recipe == Fp8Recipe.delayed
    outer_fp8_context = get_fp8_context(config) if use_outer_fp8_context else nullcontext()

    b_grad = b_output_tensor_grad[0] if b_model else None
    with context_manager and outer_fp8_context:
        # schedule forward and backward
        output_tensor, chunk_backward_dw_func = type(f_schedule_plan or b_schedule_plan).run(
            f_schedule_plan,
            b_schedule_plan,
            b_grad=b_grad,
            f_context=f_context,
            b_context=b_context,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
            block_level_wgrad_compute=block_level_wgrad_compute,
        )

    # forward post process
    num_tokens = None
    if f_model is not None:
        with f_context:
            loss_node = ScheduleNode(
                loss_func, torch.cuda.current_stream(), f_schedule_plan.event, name="loss_func"
            )
            loss_func = loss_node.forward
            output_tensor, num_tokens = forward_step_calc_loss(
                f_model,
                output_tensor,
                loss_func,
                config,
                f_context.vpp_rank if hasattr(f_context, 'vpp_rank') else None,
                collect_non_loss_data,
                num_microbatches,
                forward_data_store,
            )
            # Set the schedule plan and loss function to the output tensor
            # This is used to get the schedule plan and loss function in the backward pass
            output_tensor.schedule_plan = f_schedule_plan
            output_tensor.loss_func = loss_node

            if not unwrap_output_tensor:
                output_tensor, num_tokens = [output_tensor], num_tokens

    # backward post process
    input_tensor_grad = None
    if b_model is not None:
        input_tensor_grad = [None]
        if b_input_tensor is not None:
            input_tensor_grad = []
            for x in b_input_tensor:
                if x is None:
                    input_tensor_grad.append(None)
                else:
                    input_tensor_grad.append(x.grad)

        if unwrap_input_tensor_grad:
            input_tensor_grad = input_tensor_grad[0]

    return output_tensor, num_tokens, input_tensor_grad, chunk_backward_dw_func
