import dataclasses

import torch
from functools import wraps
from typing import List, Optional
from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron.core.transformer.module import Float16Module
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.enums import ModelType
from megatron.training.global_vars import get_args, get_timers
from megatron.training.utils import unwrap_model
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.module import fp32_to_float16, float16_to_fp32
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state
from megatron.training.utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group
)

from dcu_megatron.core.parallel_state import get_dualpipe_chunk


def dualpipev_fp16forward(self, *inputs, **kwargs):
    dualpipe_first_stage = mpu.is_pipeline_first_stage() and get_dualpipe_chunk() == 0
    if dualpipe_first_stage:
        inputs = fp32_to_float16(inputs, self.float16_convertor)
    outputs = self.module(*inputs, **kwargs)
    dualpipe_last_stage = mpu.is_pipeline_first_stage() and get_dualpipe_chunk() == 1
    if dualpipe_last_stage:
        outputs = float16_to_fp32(outputs)
    return outputs


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    assert model_type != ModelType.encoder_and_decoder, \
        "dualpipev schedule not supported for model with both encoder and decoder"

    model = []
    args.dualpipev_first_chunk = True
    first_model = model_provider_func(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=False
    )
    first_model.model_type = model_type
    model.append(first_model)

    args.dualpipev_first_chunk = False
    second_model = model_provider_func(
        pre_process=False,
        post_process=mpu.is_pipeline_first_stage()
    )
    second_model.model_type = model_type
    model.append(second_model)

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                param)

    # Print number of parameters.
    num_parameters = sum(
        [sum([p.nelement() for p in model_module.parameters()])
         for model_module in model]
    )

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
                  mpu.get_tensor_model_parallel_rank(),
                  mpu.get_pipeline_model_parallel_rank(),
                  sum([sum([p.nelement() for p in model_module.parameters()])
                       for model_module in model])), flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        config = get_model_config(model[0])
        kwargs = {}
        for f in dataclasses.fields(DistributedDataParallelConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
        kwargs['check_for_nan_in_grad'] = args.check_for_nan_in_loss_and_grad
        kwargs['check_for_large_grads'] = args.check_for_large_grads
        if args.ddp_num_buckets is not None:
            assert args.ddp_bucket_size is None, \
                "Cannot specify both --ddp-num-buckets and --ddp-bucket-size"
            assert args.ddp_num_buckets > 0, \
                "--ddp-num-buckets must be greater than 0"
            kwargs['bucket_size'] = num_parameters // args.ddp_num_buckets
        else:
            kwargs['bucket_size'] = args.ddp_bucket_size
        kwargs['pad_buckets_for_high_nccl_busbw'] = args.ddp_pad_buckets_for_high_nccl_busbw
        kwargs['average_in_collective'] = args.ddp_average_in_collective
        if args.use_custom_fsdp and args.use_precision_aware_optimizer:
            kwargs["preserve_fp32_weights"] = False
        ddp_config = DistributedDataParallelConfig(**kwargs)

        if not getattr(args, "use_torch_fsdp2", False):
            # In the custom FSDP and DDP use path, we need to initialize the bucket size.

            # If bucket_size is not provided as an input, use sane default.
            # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
            # ring-reduce implementations are large enough to remain bandwidth-bound rather than
            # latency-bound.
            if ddp_config.bucket_size is None:
                ddp_config.bucket_size = max(
                    40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
                )
            # Set bucket_size to infinity if overlap_grad_reduce is False.
            if not ddp_config.overlap_grad_reduce:
                ddp_config.bucket_size = None

        model = [DDP(config,
                     ddp_config,
                     model_chunk,
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
                 for (model_chunk_idx, model_chunk) in enumerate(model)]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)
    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.

    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_first_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0].keys():
            numerator = 0
            denominator = 0
            for x in losses_reduced:
                val = x[key]
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):
                    numerator += val[0]
                    denominator += val[1]
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator
        return loss_reduced, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad


def get_num_layers_to_build(config: TransformerConfig) -> int:

    num_layers_per_pipeline_rank = (
        config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
    )

    num_layers_to_build = num_layers_per_pipeline_rank // 2

    return num_layers_to_build


def _allreduce_embedding_grads_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = get_args()
        if args.schedule_method == 'dualpipev':
            # dualpipev no need to do embedding allreduce
            # embedding and lm head are on save rank.
            if not args.untie_embeddings_and_output_weights:
                raise NotImplementedError
            else:
                return
        else:
            return fn(*args, **kwargs)

    return wrapper
