import dataclasses

import torch

from functools import wraps
from typing import Optional
from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig, TorchFullyShardedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.enums import ModelType
from megatron.training.global_vars import get_args
from megatron.core.transformer.module import fp32_to_float16, float16_to_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel as megatron_FSDP
from megatron.training.utils import to_empty_if_meta_device

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from dcu_megatron.core.parallel_state import get_dualpipe_chunk


def dualpipev_fp16forward(self, *inputs, fp32_output=True, **kwargs):
    dualpipe_first_stage = mpu.is_pipeline_first_stage() and get_dualpipe_chunk() == 0
    if dualpipe_first_stage:
        inputs = fp32_to_float16(inputs, self.float16_convertor)
    outputs = self.module(*inputs, **kwargs)
    dualpipe_last_stage = mpu.is_pipeline_first_stage() and get_dualpipe_chunk() == 1
    if dualpipe_last_stage and fp32_output is True:
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
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    num_parameters = sum(
        [sum([p.nelement() for p in model_module.parameters()]) for model_module in model]
    )
    if mpu.get_data_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
        print(
            ' > number of parameters on (tensor, pipeline) '
            'model parallel rank ({}, {}): {}'.format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                num_parameters,
            ),
            flush=True,
        )

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if (
        not (args.use_torch_fsdp2 and args.use_cpu_initialization)
        and not args.init_model_with_meta_device
    ):
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    # Materialize tensors on meta device (GPU allocation) if not using FSDP2 and not using Megatron FSDP.
    if args.init_model_with_meta_device and not args.use_torch_fsdp2 and not args.use_megatron_fsdp:
        #for model_module in model:
        model = [to_empty_if_meta_device(model_module, device=torch.device("cuda")) for model_module in model]

    # Before TE2.x: The model_module.bfloat16()/model_module.half() above will call the inplace
    #               copy of TE's Float8Tensor, which will write an unwanted value (amax calculated
    #               from the current fp8 param) to its amax_history. The below function will correct
    #               the amax_history back.
    # After TE2.x: Below function is an empty function and does nothing.
    correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        if args.use_torch_fsdp2:
            assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
            DP = torch_FSDP
        elif args.use_megatron_fsdp:
            DP = megatron_FSDP
        else:
            DP = DDP

        config = get_model_config(model[0])

        if getattr(args, "use_torch_fsdp2", False):
            reshard_after_forward = getattr(args, "torch_fsdp2_reshard_after_forward", True)
            ddp_config = TorchFullyShardedDataParallelConfig(reshard_after_forward=reshard_after_forward)
        else:
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
            if args.use_megatron_fsdp and args.use_precision_aware_optimizer:
                kwargs["preserve_fp32_weights"] = False
            ddp_config = DistributedDataParallelConfig(**kwargs)

            # In the Megatron FSDP and DDP use path, we need to initialize the bucket size.
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

        with torch.cuda.stream(torch.cuda.Stream()):
            model = [
                DP(
                    config=config,
                    ddp_config=ddp_config,
                    module=model_chunk,
                    # Turn off bucketing for model_chunk 2 onwards, since communication for these
                    # model chunks is overlapped with compute anyway.
                    disable_bucketing=(model_chunk_idx > 0)
                    or args.overlap_param_gather_with_optimizer_step,
                )
                for (model_chunk_idx, model_chunk) in enumerate(model)
            ]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def get_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        pp_rank (Optional[int]): Pipeline parallel rank.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    args = get_args()
    if args.num_layers_to_build is not None:
        if isinstance(args.num_layers_to_build, int):
            return args.num_layers_to_build

        if getattr(args, 'dualpipev_first_chunk', True):
            return args.num_layers_to_build[pp_rank]
        else:
            return args.num_layers_to_build[-1-pp_rank]

    if (
        config.num_layers_in_first_pipeline_stage is not None
        or config.num_layers_in_last_pipeline_stage is not None
    ):

        assert not (
            config.account_for_embedding_in_pipeline_split
            or config.account_for_loss_in_pipeline_split
        ), " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = config.pipeline_model_parallel_size
        if getattr(args, "schedule_method", None) == "dualpipev":
            pipeline_stages_left *= 2

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        assert (
            layers_to_distribute % pipeline_stages_left == 0
        ), "With uneven pipelineing the left over layers must be divisible by left over stages"
        num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.
        if (
            is_first_pp_stage
            and getattr(args, 'dualpipev_first_chunk', True)
            and config.num_layers_in_first_pipeline_stage is not None
        ):
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if (
            is_first_pp_stage
            and not getattr(args, 'dualpipev_first_chunk', True)
            and config.num_layers_in_last_pipeline_stage is not None
        ):
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert (
            num_layers % config.pipeline_model_parallel_size == 0
        ), "num_layers should be divisible by pipeline_model_parallel_size"
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size
        if getattr(args, "schedule_method", None) == "dualpipev":
            assert (
                num_layers_per_pipeline_rank % 2 == 0
            ), "num_layers should be divisible by pipeline_model_parallel_size * 2"
            num_layers_per_pipeline_rank = num_layers_per_pipeline_rank // 2

    # Non-interleaved pipeline parallelism:
    # Each stage gets a contiguous set of layers.
    num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if getattr(args, "schedule_method", None) == "dualpipev":
        if is_first_pp_stage:
            if  args.dualpipev_first_chunk and config.account_for_embedding_in_pipeline_split:
                num_layers_to_build -= 1
                assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"
            elif  not args.dualpipev_first_chunk and config.account_for_loss_in_pipeline_split:
                num_layers_to_build -= 1
                assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

        return num_layers_to_build

    if is_first_pp_stage and config.account_for_embedding_in_pipeline_split:
        num_layers_to_build -= 1
        assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

    if is_last_pp_stage and config.account_for_loss_in_pipeline_split:
        num_layers_to_build -= 1
        assert num_layers_to_build >= 0, "Not enough layers in the last virtual pipeline stage"

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
