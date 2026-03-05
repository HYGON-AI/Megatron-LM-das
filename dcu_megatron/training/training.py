import dataclasses
from datetime import datetime, timedelta
import functools
import gc
import inspect
import logging
import os
import sys
from typing import Optional

import torch.distributed

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.training.log_handler import CustomHandler

# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.core.fp8_utils import correct_amax_history_if_needed

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False
from megatron.core.distributed import DistributedDataParallelConfig, TorchFullyShardedDataParallelConfig

import time

import torch

try:
    from megatron.rl import rl_utils

    has_rl_utils = True
except ImportError:
    has_rl_utils = False
try:
    from megatron.post_training.algos.distillation import (
        get_tensor_shapes_adjust_fn_for_distillation,
    )

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

try:
    from nvidia_resiliency_ext.inprocess import CallWrapper
except ImportError:
    CallWrapper = type(None)

from megatron.core.enums import ModelType
from megatron.core import mpu, tensor_parallel
from megatron.core.transformer.module import Float16Module

from megatron.core.utils import (
    check_param_hashes_across_dp_replicas,
    get_model_config,
    StragglerDetector,
)
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint
from megatron.training.checkpointing import checkpoint_exists
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel as megatron_FSDP

from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.rerun_state_machine import (
    get_rerun_state_machine,
    RerunMode,
)
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.parallel_state import (
    update_pg_timeout
)

from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches
)

from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.utils import (
    append_to_progress_log,
    calc_params_l2_norm,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    is_rank0,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model,
    update_use_dist_ckpt,
    to_empty_if_meta_device,
)

from megatron.training.global_vars import (
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
    get_energy_monitor,
)
from megatron.training import one_logger_utils
from megatron.training import ft_integration
from megatron.training.training import (
    print_datetime,
    should_disable_forward_pre_hook,
    disable_forward_pre_hook,
    train_step,
    save_checkpoint_and_time,
    enable_forward_pre_hook,
    num_floating_point_operations,
    evaluate_and_print_results,
    post_training_step_callbacks,
    dummy_train_step,
    _TRAIN_START_TIME,
    build_train_valid_test_data_iterators,
    get_optimizer_param_scheduler,

    preprocess_common_state_dict,
    HAVE_FSDP2,
)
from megatron.core.parallel_state import get_pipeline_model_parallel_group, get_pipeline_model_parallel_last_rank

from .edgc_utils import Utils, append_time_to_csv, append_data_to_csv, read_data_from_csv
from ..core.distributed.power_sgd import EFLayoutManager

stimer = StragglerDetector()


def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
    store=None,
    inprocess_call_wrapper: Optional[CallWrapper] = None,
):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
        get_embedding_ranks (TODO):
        get_position_embedding_ranks (TODO):
        non_loss_data_func (callable): A custom function to call during evaluation.
            It can run e.g. benchmarks.
        store: an optional instance of torch.distributed.Store, to be used by
            torch.distributed.init_process_group
        inprocess_call_wrapper: an optional instance of inprocess.CallWrapper,
            it is automatically injected when in-process restart is in use
    """

    if inprocess_call_wrapper is not None:
        iteration = inprocess_call_wrapper.iteration
        store = torch.distributed.PrefixStore(str(iteration), store)

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        store=store,
    )

    args = get_args()
    timers = get_timers()

    def _initialize_additional_paths_and_state(args):
        args.is_loading_checkpoint = False
        args.latest_iteration = 0
        log_dir = args.collect_log_path
        os.makedirs(log_dir, exist_ok=True)
        args.loss_path = os.path.join(log_dir, 'loss.csv')
        mapped_rank_filename = f"mapped_rank_{torch.distributed.get_rank()}.csv"
        args.mapped_rank_path = os.path.join(log_dir, mapped_rank_filename)

    if args.enable_dynamic_grad_comp:
        _initialize_additional_paths_and_state(args)

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Initialize fault tolerance
    # NOTE: ft_integration functions other than `setup` are no-op if the FT is not initialized
    if args.enable_ft_package:
        ft_integration.setup(args)
        ft_integration.maybe_setup_simulated_fault()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME], dtype=torch.double, device='cuda')
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()

    app_metrics = {}
    app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
    app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

    print_rank_0(
        'time to initialize megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME)
    )
    print_datetime('after megatron is initialized')
    app_metrics['app_model_init_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    # Track E2E metrics on pretrain start
    one_logger_utils.on_pretrain_start()

    # Context used for persisting some state between checkpoint saves.
    if args.non_persistent_ckpt_type == 'local':
        try:
            from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (
                LocalCheckpointManager,
            )
            from nvidia_resiliency_ext.checkpointing.local.replication.group_utils import (
                parse_group_sequence,
                GroupWrapper,
            )
            from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (
                CliqueReplicationStrategy,
            )
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'nvidia_resiliency_ext' module is required for local "
                "checkpointing but was not found. Please ensure it is installed."
            )

        if args.replication:
            repl_strategy = CliqueReplicationStrategy.from_replication_params(
                args.replication_jump, args.replication_factor
            )
        else:
            repl_strategy = None

        checkpointing_context = {
            'local_checkpoint_manager': LocalCheckpointManager(
                args.non_persistent_local_ckpt_dir, repl_strategy=repl_strategy
            )
        }
    else:
        checkpointing_context = {}

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, checkpointing_context=checkpointing_context
    )

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate ' 'scheduler are built')
    config = get_model_config(model[0])

    # Data stuff.
    app_metrics['app_build_dataiters_start_time'] = one_logger_utils.get_timestamp_in_ms()
    timers('train/valid/test-data-iterators-setup', log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for vp_stage in range(len(model)):
            dataset_provider_parameters = inspect.signature(train_valid_test_dataset_provider).parameters
            assert "vp_stage" in dataset_provider_parameters, \
                "vp_stage must be a kwarg in train_valid_test_dataset_provider when using virtual pipeline parallelism"
            vp_stage_train_valid_test_dataset_provider = \
                functools.partial(train_valid_test_dataset_provider, vp_stage=vp_stage)
            if getattr(train_valid_test_dataset_provider, 'is_distributed', False):
                vp_stage_train_valid_test_dataset_provider.is_distributed = True
            iterators = build_train_valid_test_data_iterators(
                vp_stage_train_valid_test_dataset_provider
            )
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    elif args.schedule_method == 'dualpipev':
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for _ in range(2):
            iterators = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        )
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')
    app_metrics['app_build_dataiters_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    # Track if training is enabled. Can only be done once args.do_train is assigned after dataloader is built.
    one_logger_utils.track_config_flags(
        args.train_iters,
        args.skip_train,
        args.do_train,
        args.do_valid,
        args.do_test,
        args.dataloader_type,
        args.retro_project_dir,
        args.retro_cyclic_train_iters,
    )

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'], barrier=True)

    one_logger = get_one_logger()
    one_logger and one_logger.log_metrics(app_metrics)

    wandb_writer = get_wandb_writer()
    if wandb_writer:
        # Add job name to the wandb config to make it easier to run more singleton dependency jobs.
        wandb_writer.config.update({'slurm_job_name': os.getenv("SLURM_JOB_NAME", "N/A")})

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model,
                optimizer,
                opt_param_scheduler,
                train_data_iterator,
                valid_data_iterator,
                process_non_loss_data_func,
                config,
                checkpointing_context,
                non_loss_data_func,
            )

        print_datetime('after training is done')

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
                preprocess_common_state_dict_fn=preprocess_common_state_dict,
            )

        one_logger and one_logger.log_metrics(
            {'app_train_loop_finish_time': one_logger_utils.get_timestamp_in_ms()}
        )

    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        if getattr(args, 'perform_rl_step', False):
            rl_utils.evaluate_and_print_results_rl(
                valid_data_iterator, model, optimizer,
                iteration, write_to_tensorboard=not args.skip_train
            )
        else:
            evaluate_and_print_results(
                prefix, forward_step_func,
                valid_data_iterator, model,
                iteration, process_non_loss_data_func, config,
                verbose=True, write_to_tensorboard=not args.skip_train,
                non_loss_data_func=non_loss_data_func
            )

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
            non_loss_data_func=non_loss_data_func,
        )

    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    ft_integration.on_checkpointing_start()
    maybe_finalize_async_save(blocking=True, terminate=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)

    one_logger and one_logger.log_metrics(
        {'app_finish_time': one_logger_utils.get_timestamp_in_ms()}
    )

    ft_integration.shutdown()
    one_logger_utils.finish()


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    def build_model():
        if (
            mpu.get_pipeline_model_parallel_world_size() > 1
            and args.virtual_pipeline_model_parallel_size is not None
        ):
            model = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
                post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
                this_model = model_provider_func(
                    pre_process=pre_process, post_process=post_process, vp_stage=i)
                this_model.model_type = model_type
                this_model.vp_stage = i
                model.append(this_model)

        elif args.enable_vocab_parallel:
            pre_process = mpu.is_pipeline_first_stage(ignore_virtual=True)
            assert model_type != ModelType.encoder_and_decoder, \
                'vocab parallel is not yet supported in encoder-decoder models'
            model = [
                model_provider_func(
                    pre_process=pre_process,
                    post_process=False,
                    split_vocab_embedding=pre_process,
                    include_layer_norm=mpu.is_pipeline_last_stage(ignore_virtual=True),
                ),
                model_provider_func(
                    pre_process=False,
                    post_process=True,
                    noop_block=True,
                ),
                model_provider_func(
                    pre_process=False,
                    post_process=False,
                    split_vocab_embedding=True,
                    noop_block=True,
                )
            ]
            model[0].model_type = model_type
            model[1].model_type = model_type
            model[2].model_type = model_type
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                model.append(
                    model_provider_func(
                        pre_process=False,
                        post_process=False,
                        noop_block=True,
                    )
                )
                model[3].model_type = model_type

        elif args.schedule_method == "dualpipev":
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

        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
            model.model_type = model_type
        return model

    if args.init_model_with_meta_device:
        with torch.device('meta'):
            model = build_model()
    else:
        model = build_model()

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
        if args.enable_vocab_parallel:
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                loss_chunk = Float16Module(config, model[3])
            model = [
                Float16Module(config, model[0]),
                Float16Module(config, model[1], force_output_fp32=True),
                Float16Module(config, model[2], is_embedding_chunk=True),
            ]
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                model.append(loss_chunk)
        else:
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

        dp_stream = torch.cuda.Stream()
        dp_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(dp_stream):
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
        torch.cuda.current_stream().wait_stream(dp_stream)

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def setup_model_and_optimizer(
    model_provider_func,
    model_type,
    no_wd_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
    checkpointing_context=None,
):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    if has_nvidia_modelopt:
        from megatron.post_training.checkpointing import has_modelopt_state
        # [ModelOpt]: Check if the checkpoint is a ModelOpt checkpoint and
        # set a flag to use our model provider if so.
        if args.load is not None and has_modelopt_state(args.load):
            print_rank_0(f'ModelOpt checkpoint detected')
            args.modelopt_enabled = True
        elif getattr(args, "export_kd_teacher_load", None):
            # For distillation ckpts without ModelOpt state
            args.modelopt_enabled = True

    model = get_model(model_provider_func, model_type)
    unwrapped_model = unwrap_model(model)

    one_logger and one_logger.log_metrics({"app_build_optimzer_start_time": one_logger_utils.get_timestamp_in_ms()})
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = timers
    optimizer = get_megatron_optimizer(
        config,
        model,
        no_wd_decay_cond,
        scale_lr_cond,
        lr_mult,
        use_gloo_process_groups=args.enable_gloo_process_groups,
        # If the user is asking for a non-zero embedding init std, skip weight decay for embeddings
        #  to avoid embeddings from shrinking to zero as recommended in https://arxiv.org/abs/2312.16903
        default_skip_embedding_weight_decay=args.embedding_init_method_std is not None,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
    one_logger and one_logger.log_metrics({"app_build_optimzer_finish_time": one_logger_utils.get_timestamp_in_ms()})

    if args.moe_use_upcycling:
        torch.distributed.barrier()
        assert not checkpoint_exists(args.save), (
            "The upcycling destination directory already exists. "
            "Please check if --moe-use-upcycling is mistakenly enabled. "
            "Upcycling should only be set for the first run when converting the dense model. "
            "All subsequent runs should remove this flag. "
        )
        # before changing moe related global args, save them in local variables
        num_experts = args.num_experts
        expert_model_parallel_size = args.expert_model_parallel_size
        moe_ffn_hidden_size = args.ffn_hidden_size

        # set dense model related args in to global args before getting dense model
        args.num_experts = None
        args.expert_model_parallel_size = 1
        args.ffn_hidden_size = moe_ffn_hidden_size * args.moe_upcycling_granularity

        # get dense model
        dense_model_for_upcycling = get_model(model_provider_func, model_type)

        # recover moe upcycling related args in global args before executing upcycling
        args.num_experts = num_experts
        args.expert_model_parallel_size = expert_model_parallel_size
        args.ffn_hidden_size = moe_ffn_hidden_size

        # execute upcycling
        _, args.num_floating_point_operations_so_far = upcycling_utils.load_and_upcycle_model(
            load_checkpoint,
            unwrapped_model,
            dense_model_for_upcycling,
            load_kwargs={
                'model': dense_model_for_upcycling,
                'optimizer': None,
                'opt_param_scheduler': None,
            },
        )
        args.iteration = 1
        save_checkpoint(
            args.iteration, model, None, None, args.num_floating_point_operations_so_far
        )
        torch.distributed.barrier()
        del dense_model_for_upcycling
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
        print_rank_0(f'Upcycled checkpoint saved to {args.save}')

    if (
            args.load is not None or args.pretrained_checkpoint is not None
    ) and not args.moe_use_upcycling:
        one_logger and one_logger.log_metrics(
            {'load_checkpoint_start_time': one_logger_utils.get_timestamp_in_ms()}
        )
        timers('load-checkpoint', log_level=0).start(barrier=True)

        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model,
            optimizer,
            opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2
                                       and getattr(args, "use_torch_fsdp2", False)
                                       and args.ckpt_format == "torch_dist",
        )
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
        one_logger and one_logger.log_metrics(
            {
                'load_checkpoint_finish_time': one_logger_utils.get_timestamp_in_ms(),
                'load_checkpoint_time': timers('load-checkpoint').active_time(),
            }
        )
        if args.iteration != 0 and args.enable_dynamic_grad_comp:
            args.is_loading_checkpoint = True
            args.latest_iteration = args.iteration
            Utils.loss = read_data_from_csv(args.loss_path, args.latest_iteration)
            Utils.mapped_rank = read_data_from_csv(args.mapped_rank_path, args.latest_iteration)

        if is_rank0():
            # iter——log写文件
            iter_log_path = os.path.join(args.load, 'last_ckpt_iter_log.txt')
            try:
                with open(iter_log_path, 'r') as f:
                    content = f.read()
                    current_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                    updated_iter_log = current_time + content[20:]
                    print_rank_0(f"{updated_iter_log}")
            except FileNotFoundError:
                pass

    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    # get model without FP16 and/or DDP wrappers
    if (
            args.iteration == 0
            and len(unwrapped_model) == 1
            and hasattr(unwrapped_model[0], 'init_state_dict_from_bert')
    ):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # Convert checkpoint format.
    if args.ckpt_convert_format is not None:
        load_ckpt_format = args.ckpt_format
        args.ckpt_format = args.ckpt_convert_format
        args.save = os.path.join(args.ckpt_convert_save, args.ckpt_convert_format)
        update_use_dist_ckpt(args)

        save_checkpoint(
            args.iteration,
            model,
            optimizer,
            opt_param_scheduler,
            args.num_floating_point_operations_so_far,
            preprocess_common_state_dict_fn=preprocess_common_state_dict,
        )

        print_rank_0("> converted checkpoint: %s -> %s." % (load_ckpt_format, args.ckpt_format))
        torch.distributed.barrier()
        exit()

    return model, optimizer, opt_param_scheduler


def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, forward_backward_func):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    def check_warm_up_done(args):
        return args.curr_iteration > args.warm_up_train_iter

    def should_broadcast_current_iteration(args):
        return (args.curr_iteration % args.rank_adjust_window_size == 1 and
                args.curr_iteration != 1 and
                args.curr_iteration != (args.latest_iteration + 1) and
                args.curr_iteration != (args.warm_up_train_iter + 1))

    def broadcast_predict_time(first_stage_predict_time):
        torch.distributed.broadcast(first_stage_predict_time,
                                    src=mpu.get_pipeline_model_parallel_first_rank(),
                                    group=mpu.get_pipeline_model_parallel_group())

    def adjust_and_predict_rank(first_stage_predict_time, ):
        if first_stage_predict_time is not None:
            predict_comp_rank = Utils.use_time_predict_rank(first_stage_predict_time)
            return Utils.adjust_rank(predict_comp_rank)
        return None, None

    def update_mapped_rank(args, mapped_rank):
        if mpu.is_pipeline_first_stage():
            mapped_rank = Utils.map_loss_change_to_rank(
                min_rank=(args.max_rank / 4),
                max_rank=args.max_rank,
                window_size=args.rank_adjust_window_size
            )
            mapped_rank, first_stage_predict_time = Utils.adjust_rank(mapped_rank)
            first_stage_predict_time = torch.tensor(first_stage_predict_time, device="cuda", dtype=torch.float32)
        else:
            first_stage_predict_time = torch.zeros(1, device="cuda", dtype=torch.float32)
        broadcast_predict_time(first_stage_predict_time)
        if mpu.is_pipeline_first_stage():
            args.mapped_rank = mapped_rank
            Utils.mapped_rank[-1] = mapped_rank
        else:
            args.predict_comp_rank, args.predict_time = adjust_and_predict_rank(first_stage_predict_time)
            value = Utils.second_syn_data_parallel_group(args.predict_comp_rank)
            args.final_rank = value if value is not None else args.predict_comp_rank
            Utils.mapped_rank.append(args.final_rank)

    if args.enable_dynamic_grad_comp:
        if check_warm_up_done(args) and should_broadcast_current_iteration(args):
            update_mapped_rank(args, mapped_rank=None)

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        if has_nvidia_modelopt:
            # [ModelOpt]: Pipeline-parallel Distillation stacks student and teacher tensors
            adjust_tensor_shapes_fn = get_tensor_shapes_adjust_fn_for_distillation(
                model, args.seq_length, args.micro_batch_size, args.decoder_seq_length
            )
        else:
            adjust_tensor_shapes_fn = None

        # For the mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
        # we need to call the _copy_main_params_to_param_buffer() after the grad buffer
        # is zeroed by zero_grad_buffer() because param and grad buffer are shared.
        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            for optim_instance in optimizer.chained_optimizers:
                if isinstance(optim_instance, DistributedOptimizer):
                    optim_instance._copy_main_params_to_param_buffer()

        # Forward pass.
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
        )
    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    if args.curr_iteration % args.save_interval == 10 and args.enable_dynamic_grad_comp:
        total_time = config.timers('edgc-backward-compute', log_level=0).elapsed(reset=True) * 1000.0
        args.per_microbatch_time = total_time / get_num_microbatches()

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
        increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    is_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)
    if args.schedule_method == "dualpipev":
        is_last_stage = mpu.is_pipeline_first_stage(ignore_virtual=True)
    if is_last_stage:
        # Average loss across microbatches.
        loss_reduced = {}

        for key in losses_reduced[0].keys():
            val = [x[key].view(-1) for x in losses_reduced]
            if val[0].numel() == 2:
                if args.sft:
                    # in mcore the normalization happens on micro batch instead of global
                    val = torch.vstack(val)
                    val = val[:, 0] / val[:, 1]
                    val = val.mean()
                    torch.distributed.all_reduce(
                        val,
                        group=mpu.get_data_parallel_group(with_context_parallel=True)
                    )
                    val /= torch.distributed.get_world_size(
                        group=mpu.get_data_parallel_group(with_context_parallel=True)
                    )
                    loss_reduced[key] = val
                else:
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    val = torch.vstack(val).sum(dim=0)
                    torch.distributed.all_reduce(
                        val,
                        group=mpu.get_data_parallel_group(with_context_parallel=True)
                    )
                    loss_reduced[key] = val[0] / val[1]
            elif val[0].numel() == 1:
                # legacy behavior, we average over the number of microbatches
                val = torch.cat(val).mean()
                loss_reduced[key] = val
            else:
                raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")

        if args.enable_dynamic_grad_comp:
            loss = list(loss_reduced.values())[0]
            iter_sample_interval = int(1 / args.iteration_sample_ratio)
            if args.curr_iteration % iter_sample_interval == 0:
                loss_tensor = torch.tensor(loss, device=torch.cuda.current_device(), dtype=torch.float32)
                group = get_pipeline_model_parallel_group()
                torch.distributed.broadcast(tensor=loss_tensor, src=torch.distributed.get_rank(), group=group)
                broadcasted_loss = loss_tensor.item()
                Utils.loss.append(broadcasted_loss)
                if is_last_rank():
                    append_data_to_csv(args.loss_path, args.curr_iteration, loss)

        results = (
            loss_reduced,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
        )
        if args.enable_dynamic_grad_comp and args.all_reduce_time:
            results = results + (args.params_all_reduce_time,)
        return results

    if args.enable_dynamic_grad_comp:
        iter_sample_interval = int(1 / args.iteration_sample_ratio)
        if args.curr_iteration % iter_sample_interval == 0:
            group = get_pipeline_model_parallel_group()
            src_rank = get_pipeline_model_parallel_last_rank()
            loss_tensor = torch.tensor(0.0, device=torch.cuda.current_device(), dtype=torch.float32)
            torch.distributed.broadcast(tensor=loss_tensor, src=src_rank, group=group)
            broadcasted_loss = loss_tensor.item()
            Utils.loss.append(broadcasted_loss)
        if args.all_reduce_time:
            return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad, args.params_all_reduce_time
        else:
            return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
    else:
        return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad


def training_log(
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        iteration,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()
    energy_monitor = get_energy_monitor()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = (
                    total_loss_dict.get(key, torch.tensor([0.0], dtype=torch.float, device='cuda'))
                    + loss_dict[key]
            )
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or value == -float('inf') or value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer',
    ]
    # Add timers from RL loop if needed.
    if getattr(args, 'perform_rl_step', False):
        timers_to_log.extend(['rollout-collection', 'inference-setup', 'collect-rollouts', 'postrollout-gc-collect',
                              'sync-rollouts', 'prepare-data-for-update', 'compute-group-stats',
                              'prepare-trajectories', 'get-ltor-masks-and-position-ids', 'create-logprobs-dataloader',
                              'compute-logprobs', 'compute-ref-logprobs', 'compute-prob-stats',
                              'prepare-advantages', 'create-dataloader', 'log-wandb-tb',
                              'offload-optimizer-before-inference', 'onload-kv-cache-before-inference',
                              'wait-for-decode-only', 'build-cuda-graphs', 'suspend-engine',
                              'offload-kv-cache-after-inference', 'onload-optimizer-after-inference'])

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * get_num_microbatches()

    # Track app tag & app tag ID
    one_logger_utils.track_app_tag(batch_size, args.world_size, args.seq_length)

    total_iterations = total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]

    # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
    learning_rate = reduce_max_stat_across_model_parallel_group(learning_rate)
    # Tensorboard values.
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log({'samples vs steps': args.consumed_train_samples}, iteration)
        writer.add_scalar('learning-rate', learning_rate, iteration)
        writer.add_scalar('learning-rate vs samples', learning_rate, args.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({'learning-rate': learning_rate}, iteration)
        if args.decoupled_lr is not None:
            writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
        if args.skipped_train_samples > 0:
            writer.add_scalar('skipped-train-samples', args.skipped_train_samples, iteration)
            if wandb_writer:
                wandb_writer.log({'skipped-train-samples': args.skipped_train_samples}, iteration)
        writer.add_scalar('batch-size', batch_size, iteration)
        writer.add_scalar('batch-size vs samples', batch_size, args.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({'batch-size': batch_size}, iteration)
        # Log bins for packed mode
        if has_rl_utils and args.rl_use_sequence_packing:
            packing_metrics = rl_utils.get_sequence_packing_tensorboard_metrics(args)
            for metric_name, metric_value in packing_metrics.items():
                writer.add_scalar(metric_name, metric_value, iteration)
            if wandb_writer and packing_metrics:
                wandb_writer.log(packing_metrics, iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key], args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'loss-scale': loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'world-size': args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'grad-norm': grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar(
                'num-zeros vs samples', num_zeros_in_grad, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'params-norm': params_norm}, iteration)
        if getattr(args, 'perform_rl_step', False):
            grpo_collection_iteration = iteration // (
                        args.grpo_iterations * ((args.grpo_samples_per_iteration) // args.global_batch_size))
            writer.add_scalar('grpo_collection_iteration', grpo_collection_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'grpo_collection_iteration': grpo_collection_iteration}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes", mem_stats["reserved_bytes.all.current"], iteration
            )
            writer.add_scalar(
                "mem-allocated-bytes", mem_stats["allocated_bytes.all.current"], iteration
            )
            writer.add_scalar(
                "mem-max-allocated-bytes", mem_stats["allocated_bytes.all.peak"], iteration
            )
            writer.add_scalar("mem-allocated-count", mem_stats["allocation.all.current"], iteration)
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_names = []
        if "aux_loss" in args.moe_router_load_balancing_type:
            track_names.append("load_balancing_loss")
        if "seq_aux_loss" in args.moe_router_load_balancing_type:
            track_names.append("seq_load_balancing_loss")
        if "global_aux_loss" in args.moe_router_load_balancing_type:
            track_names.append("global_load_balancing_loss")
        if args.moe_z_loss_coeff is not None:
            track_names.append("z_loss")
        track_moe_metrics(
            loss_scale=moe_loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
            per_layer_logging=args.moe_per_layer_logging,
            force_initialize=True,
            track_names=track_names,
            num_layers=args.num_layers,
            moe_layer_freq=args.moe_layer_freq,
            mtp_num_layers=args.mtp_num_layers,
        )
    if args.mtp_num_layers is not None:
        mtp_loss_scale = 1 / get_num_microbatches()
        MTPLossLoggingHelper.track_mtp_metrics(
            mtp_loss_scale, iteration, writer, wandb_writer, total_loss_dict
        )
    if iteration % args.log_interval == 0:
        if args.record_memory_history and is_last_rank():
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            with open(args.memory_snapshot_path, 'wb') as f:
                dump(snapshot, f)

        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(args, batch_size) / (
                elapsed_time_per_iteration * 10 ** 12 * args.world_size
        )

        one_logger_utils.track_e2e_metrics(args.log_throughput, throughput)

        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar('iteration-time', elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration}, iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(args.consumed_train_samples)
        if has_rl_utils and args.rl_use_sequence_packing:
            log_string += rl_utils.get_sequence_packing_log_info(args)
        if args.skipped_train_samples > 0:
            log_string += ' skipped samples: {:12d} |'.format(args.skipped_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0
        )
        if args.log_throughput:
            log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('throughput', throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({'throughput': throughput}, iteration)
        if args.log_energy:
            energy = (energy_monitor.lap() / total_iterations) / args.world_size
            power = energy / elapsed_time_per_iteration
            log_string += f' energy per GPU (J/iter/GPU): {energy:.1f} |'
            log_string += f' power per GPU (W/GPU): {power:.1f} |'
            if writer:
                writer.add_scalar('iter-energy/gpu', energy, iteration)
                writer.add_scalar('power/gpu', power, iteration)
            if wandb_writer:
                wandb_writer.log({'iter-energy/gpu': energy}, iteration)
                wandb_writer.log({'power/gpu': power}, iteration)
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += f' learning rate: {learning_rate:.6E} |'
        if args.decoupled_lr is not None and (
                mpu.is_pipeline_first_stage(ignore_virtual=True)
                or mpu.is_pipeline_last_stage(ignore_virtual=True)
        ):
            assert decoupled_learning_rate is not None
            log_string += f' decoupled learning rate: {decoupled_learning_rate:.6E} |'
        else:
            assert decoupled_learning_rate is None
        log_string += f' global batch size: {batch_size:5d} |'
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(
                    max(1, total_loss_dict[advanced_iters_key])
                )
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        log_string += f' loss scale: {loss_scale:.1f} |'
        if grad_norm is not None:
            log_string += f' grad norm: {grad_norm:.3f} |'
        if num_zeros_in_grad is not None:
            log_string += f' num zeros: {num_zeros_in_grad} |'
        if params_norm is not None:
            log_string += f' params norm: {params_norm:.3f} |'
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key]
        )
        log_string += ' number of nan iterations: {:3d} |'.format(total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory(f'(after {iteration} iterations)')
            report_memory_flag = False
        # Write timers to wandb, don't reset the counts
        if args.log_timers_to_tensorboard:
            timers.write(timers_to_log, writer, iteration, normalizer=args.log_interval, reset=False)
            timers.write(timers_to_log, wandb_writer, iteration, normalizer=args.log_interval, reset=False)
        # Log timers to stdout
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag, log_string


def checkpoint_and_decide_exit(
        model,
        optimizer,
        opt_param_scheduler,
        iteration,
        num_floating_point_operations_so_far,
        checkpointing_context,
        train_data_iterator,
        iter_log
):
    """Save checkpoint and decide whether to exit based on arguments (e.g., if
    --exit-duration-in-mins is set). Actual exit happens in main training loop
    based on the return value of this function."""
    args = get_args()
    timers = get_timers()

    # Exit based on signal handler.
    saved_checkpoint = False
    if args.exit_signal_handler:
        signal_handler = get_signal_handler()
        if any(signal_handler.signals_received()):
            if args.save:
                save_checkpoint_and_time(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context,
                    train_data_iterator=train_data_iterator,
                )
            print_datetime('exiting program after receiving SIGTERM.')

            return True

    # Regular save (persistent and non-persistent).
    if args.save and args.save_interval and iteration % args.save_interval == 0:
        save_checkpoint_and_time(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            train_data_iterator=train_data_iterator,
        )
        saved_checkpoint = True

    elif (
            args.save
            and args.non_persistent_save_interval
            and iteration % args.non_persistent_save_interval == 0
    ):
        save_checkpoint_and_time(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            non_persistent_ckpt=True,
            train_data_iterator=train_data_iterator,
        )
        saved_checkpoint = True

    if is_rank0() and saved_checkpoint:
        # iter——log写文件
        iter_log_path = os.path.join(args.save, 'last_ckpt_iter_log.txt')
        os.makedirs(args.save, exist_ok=True)
        with open(iter_log_path, 'w') as f:
            f.write(str(iter_log))

    # Exit based on duration.
    if args.exit_duration_in_mins:
        train_time = (time.time() - _TRAIN_START_TIME) / 60.0
        done_cuda = torch.tensor(
            [train_time > args.exit_duration_in_mins], dtype=torch.int, device='cuda'
        )
        torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
        done = done_cuda.item()
        if done:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context,
                    train_data_iterator=train_data_iterator,
                )
            print_datetime(f'exiting program after {train_time} minutes')

            return True

    # Exit based on iterations.
    if args.exit_interval and iteration % args.exit_interval == 0:
        if args.save and not saved_checkpoint:
            save_checkpoint_and_time(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
            )
        print_datetime(f'exiting program at iteration {iteration}')

        return True

    return False


def train(
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        train_data_iterator,
        valid_data_iterator,
        process_non_loss_data_func,
        config,
        checkpointing_context,
        non_loss_data_func,
):
    """Training function: run train_step desired number of times, run validation, checkpoint."""
    args = get_args()
    timers = get_timers()

    if getattr(args, 'perform_rl_step', False):
        assert has_rl_utils, "RL cannot run without the megatron.rl package"

    # Additional variable initialization for RL training
    if getattr(args, 'perform_rl_step', False):
        print_rank_0("> Loading pretrained checkpoint for reference weights in RL training...")
        load, finetune, no_load_optim = args.load, args.finetune, args.no_load_optim
        args.no_load_optim = True

        # Load pretrained checkpoint
        args.load = None
        args.finetune = True
        load_checkpoint(
            model,
            None,  # Don't load optimizer state
            None,  # Don't load scheduler state
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2
                                       and getattr(args, "use_torch_fsdp2", False)
                                       and args.ckpt_format == "torch_dist",
        )
        ref_state_dict = {k: (v.cpu() if v is not None else v) for k, v in model[0].state_dict().items()}

        # Reload RL training checkpoint weights
        args.load = load
        args.finetune = finetune
        print_rank_0("> Reloading RL training checkpoint...")
        load_checkpoint(
            model,
            None,
            None,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2
                                       and getattr(args, "use_torch_fsdp2", False)
                                       and args.ckpt_format == "torch_dist",
        )

        args.no_load_optim = no_load_optim

    # IMPORTANT FIX: For RL training, reinitialize the microbatch calculator with the correct configuration
    if getattr(args, 'perform_rl_step', False):
        print_rank_0("> Reinitializing microbatch calculator for GRPO training...")
        from megatron.core.num_microbatches_calculator import (
            destroy_num_microbatches_calculator,
            init_num_microbatches_calculator
        )
        # First destroy the existing calculator
        destroy_num_microbatches_calculator()
        # Then initialize with the correct perform_rl_step=True context
        init_num_microbatches_calculator(
            args.rank,
            args.rampup_batch_size,
            args.global_batch_size,
            args.micro_batch_size,
            mpu.get_data_parallel_world_size(),
            args.decrease_batch_size_if_needed
        )
        print_rank_0(f"> GRPO training: num_microbatches set to {get_num_microbatches()}")

    energy_monitor = get_energy_monitor()
    one_logger = get_one_logger()

    def edgc_config_printer():
        print_rank_0('============= EDGC Configuration =============')
        print_rank_0(f' >> enable_dynamic_grad_comp: {args.enable_dynamic_grad_comp}')
        print_rank_0(f' >> grad_comp_warm_up: {args.grad_comp_warm_up:.3f}')
        print_rank_0(f' >> rank_adjust_window_size: {args.rank_adjust_window_size}')
        print_rank_0(f' >> iteration_sample_ratio: {args.iteration_sample_ratio:.4f}')
        print_rank_0(f' >> gradient_sample_ratio: {args.gradient_sample_ratio:.4f}')
        print_rank_0(f' >> collect_log_path: {args.collect_log_path}')
        print_rank_0('==============================================')

    if args.enable_dynamic_grad_comp:
        edgc_config_printer()

    def _initialize_training_flags(args):
        args.all_reduce_time = False
        args.max_rank = None
        args.find_rank_upper_limit = False
        args.final_rank = None
        args.mapped_rank = None
        args.begin_max_rank = True
        args.begin_warm_up = True
        args.grad_comp_enabled = False
        args.compressor = None
        args.pre_rank = None
        args.ef_manager = EFLayoutManager(ef_store_dtype=torch.bfloat16)

    def _initialize_log_paths(args):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"{args.collect_log_path}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        paths = {
            'checkpoint_date_path': 'checkpoint_date.csv',
            'loss_path': 'loss.csv',
            'loss_validation_path': 'loss_validation.csv',
            'ppl_validation_path': 'ppl_validation.csv',
            'max_error_path': 'max_error.csv',
        }
        for attr, filename in paths.items():
            setattr(args, attr, os.path.join(log_dir, filename))

    def _initialize_warmup_iterations(args):
        args.warm_up_train_iter = int(args.train_iters * args.grad_comp_warm_up)

    if args.enable_dynamic_grad_comp:
        _initialize_training_flags(args)
        _initialize_log_paths(args)
        _initialize_warmup_iterations(args)
        if is_rank0():
            append_time_to_csv(args.checkpoint_date_path, args.iteration)

        if args.use_distributed_optimizer:
            collective_group = mpu.get_data_parallel_group()
            intra_distributed_optimizer_instance_size = mpu.get_data_parallel_world_size()
            intra_distributed_optimizer_instance_rank = torch.distributed.get_rank(group=collective_group)
            args.ef_manager.build_ef_layout_with_distributed_optimizer(model, device=torch.device("cuda"),
                                                                       intra_distributed_optimizer_instance_size=intra_distributed_optimizer_instance_size,
                                                                       intra_distributed_optimizer_instance_rank=intra_distributed_optimizer_instance_rank)
        else:
            args.ef_manager.build_ef_layout(model, device=torch.device("cuda"))

    if args.run_workload_inspector_server:
        try:
            from workload_inspector.utils.webserver import run_server
            import threading

            threading.Thread(
                target=run_server, daemon=True, args=(torch.distributed.get_rank(),)
            ).start()
        except ModuleNotFoundError:
            print_rank_0("workload inspector module not found.")

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration
    # Make sure rerun_state_machine has the right iteration loaded from checkpoint.
    rerun_state_machine = get_rerun_state_machine()
    if rerun_state_machine.current_iteration != iteration:
        print_rank_0(f"Overwriting rerun_state_machine.current_iteration from "
                     f"{rerun_state_machine.current_iteration} to {iteration}...")
        rerun_state_machine.current_iteration = iteration

    # Track E2E metrics at the start of training.
    one_logger_utils.on_train_start(
        iteration=iteration,
        consumed_train_samples=args.consumed_train_samples,
        train_samples=args.train_samples,
        seq_length=args.seq_length,
        train_iters=args.train_iters,
        save=args.save,
        async_save=args.async_save,
        log_throughput=args.log_throughput,
        num_floating_point_operations_so_far=args.num_floating_point_operations_so_far,
    )

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params.
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], (megatron_FSDP, DDP)) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            'When overlap_grad_reduce is True, config.no_sync_func must be None; '
            'a custom no_sync_func is not supported when overlapping grad-reduce'
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    if args.log_energy:
        energy_monitor.setup()
        energy_monitor.resume()

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    pre_hook_enabled = False
    should_exit = False
    exit_code = 0

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert (
                args.manual_gc_interval >= 0
        ), 'Manual garbage collection interval should be larger than or equal to 0'
        gc.disable()
        gc.collect()

    # Singleton initialization of straggler detector.
    if args.log_straggler:
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(
            world,
            rank,
            mmcnt=mmcnt,
            enabled=not args.disable_straggler_on_startup,
            port=args.straggler_ctrlr_port,
        )
    num_floating_point_operations_since_last_log_event = 0.0

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0
    # Wrap forward_backward_func for Full iteration CUDA graph
    forward_backward_func = get_forward_backward_func()
    if args.cuda_graph_impl == "local" and args.cuda_graph_scope == "full_iteration":
        forward_backward_func = FullCudaGraphWrapper(forward_backward_func,
                                                     cuda_graph_warmup_steps=args.cuda_graph_warmup_steps)

    def get_e2e_base_metrics():
        """Get base metrics values for one-logger to calculate E2E tracking metrics."""
        num_floating_point_operations_since_current_train_start = (
                num_floating_point_operations_so_far - args.num_floating_point_operations_so_far
        )
        return {
            'iteration': iteration,
            'train_duration': timers('interval-time').active_time(),
            'eval_duration': eval_duration,
            'eval_iterations': eval_iterations,
            'total_flops_since_current_train_start': num_floating_point_operations_since_current_train_start,
            'num_floating_point_operations_so_far': num_floating_point_operations_so_far,
            'consumed_train_samples': args.consumed_train_samples,
            'world_size': args.world_size,
            'seq_length': args.seq_length,
        }

    # Cache into one-logger for callback.
    if one_logger:
        with one_logger.get_context_manager():
            one_logger.store_set('get_e2e_base_metrics', get_e2e_base_metrics)

    prof = None
    if (
            args.profile
            and torch.distributed.get_rank() in args.profile_ranks
            and args.use_pytorch_profiler
    ):
        def trace_handler(p):
            from pathlib import Path
            Path(f"{args.profile_dir}").mkdir(parents=True, exist_ok=True)
            if args.rank in [0]:
                print(p.key_averages(group_by_input_shape=True,
                                     group_by_stack_n=5).table(sort_by="self_cuda_time_total",
                                                               row_limit=-1,
                                                               max_src_column_width=100,
                                                               max_name_column_width=280,
                                                               max_shapes_column_width=200))

            p.export_chrome_trace("{path}/trace_rank{rank}_step{step}.json".format(
                path=args.profile_dir, rank=torch.distributed.get_rank(), step=p.step_num))

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=max(args.profile_step_start - 1, 0),
                warmup=1 if args.profile_step_start > 0 else 0,
                active=args.profile_step_end - args.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
        )
        prof.start()
    elif args.profile and torch.distributed.get_rank() in args.profile_ranks and args.use_hip_profiler:
        import ctypes
        roctracer = ctypes.cdll.LoadLibrary("/opt/dtk/roctracer/lib/libroctracer64.so")

    start_iteration = iteration
    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False
    # Also, check weight hash across DP replicas to be very pedantic.
    if args.check_weight_hash_across_dp_replicas_interval is not None:
        assert check_param_hashes_across_dp_replicas(
            model, cross_check=True
        ), "Parameter hashes not matching across DP replicas"
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {iteration} iterations...")

    # Initialize CUDA Graphs helper.
    if args.cuda_graph_impl == "transformer_engine":
        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=config,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            optimizers=[optimizer],
        )

    # Run training iterations till done.
    buffered_rollouts = None
    while iteration < args.train_iters:
        if args.profile and torch.distributed.get_rank() in args.profile_ranks:
            if args.use_pytorch_profiler:
                prof.step()
            elif args.use_hip_profiler:
                if iteration == args.profile_step_start: roctracer.roctracer_start()
                if iteration == args.profile_step_end: roctracer.roctracer_stop()
            elif iteration == args.profile_step_start:
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        ft_integration.on_checkpointing_start()
        maybe_finalize_async_save(blocking=False)
        ft_integration.on_checkpointing_end(is_async_finalization=True)
        # Update the timeout for all process groups after initialization
        # We update the timeout after the first successful iteration,
        # which takes longer than others usually
        if args.distributed_timeout_seconds_after_init is not None and iteration == start_iteration + 1:
            # TODO: some dynamic timeout setting is required
            # based on the iteration time considering interval-based steps (e.g. eval, checkpoint)
            # e.g. timeout for normal iterations vs timeout for iterations with checkpoint
            # this timeout is triggered when there's no collective communication
            # for the duration of timeout
            update_pg_timeout(timedelta(seconds=args.distributed_timeout_seconds_after_init))
        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        # Standard microbatch update (sequence packing overrides this in rl_utils.py)
        update_num_microbatches(args.consumed_train_samples, consistency_check=False, verbose=True)
        # Skip automatic checkpoint on microbatch changes when sequence packing is active
        # as it intentionally reconfigures microbatches
        if get_num_microbatches() != num_microbatches and iteration != 0:
            if args.rl_use_sequence_packing:
                print_rank_0(
                    f"[Sequence Packing] Skipping automatic checkpoint at iteration {iteration} "
                    f"(microbatch change: {num_microbatches} -> {get_num_microbatches()})"
                )
            else:
                assert get_num_microbatches() > num_microbatches, (
                    f"Number of microbatches should be increasing due to batch size rampup; "
                    f"instead going from {num_microbatches} to {get_num_microbatches()}"
                )
                if args.save is not None:
                    save_checkpoint_and_time(
                        iteration,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        num_floating_point_operations_so_far,
                        checkpointing_context,
                        train_data_iterator=train_data_iterator,
                    )
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True, verbose=True)

        # Capture CUDA Graphs.
        if (
                args.cuda_graph_impl == "transformer_engine"
                and iteration == args.cuda_graph_warmup_steps
        ):
            if iteration > start_iteration and should_disable_forward_pre_hook(args):
                disable_forward_pre_hook(model, param_sync=False)
            cuda_graph_helper.create_cudagraphs()
            if iteration > start_iteration and should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                cuda_graph_helper.cuda_graph_set_manual_hooks()

        # Completely skip iteration if needed.
        if iteration in args.iterations_to_skip:
            # Dummy train_step to fast forward train_data_iterator.
            dummy_train_step(train_data_iterator)
            if iteration == start_iteration:
                start_iteration = iteration + 1
            iteration += 1
            batch_size = (
                    mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
            )
            args.consumed_train_samples += batch_size
            args.skipped_train_samples += batch_size
            continue

        args.curr_iteration = iteration
        # For GRPO, we keep the data for a few epochs. DeepSeekMath paper calls this number $\mu$.
        # It is similar to a PPO epoch.

        if getattr(args, 'perform_rl_step', False):
            with torch.no_grad():
                train_data_iterator = rl_utils.setup_grpo_data_iterator(
                    model, optimizer, iteration, ref_state_dict, buffered_rollouts
                )
                buffered_rollouts = train_data_iterator

        ft_integration.on_training_step_start()
        if args.enable_dynamic_grad_comp:
            if args.find_rank_upper_limit:
                loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad = \
                    train_step(forward_step_func,
                               train_data_iterator,
                               model,
                               optimizer,
                               opt_param_scheduler,
                               config,
                               forward_backward_func)
            else:
                args.all_reduce_time = True
                loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad, params_all_reduce_time = \
                    train_step(forward_step_func,
                               train_data_iterator,
                               model,
                               optimizer,
                               opt_param_scheduler,
                               config,
                               forward_backward_func)
                args.find_rank_upper_limit, args.max_rank = Utils.is_find_rank_upper_limit(params_all_reduce_time)
                Utils.syn_tensor_parallel_group()
                Utils.syn_data_parallel_group()
                Utils.syn_pipeline_parallel_group()
                if args.find_rank_upper_limit:
                    args.max_rank, _ = Utils.syn_rank(args.max_rank)
                    args.all_reduce_time = False
                    Utils.mapped_rank.append(args.max_rank)
        else:
            (
                loss_dict,
                skipped_iter,
                should_checkpoint,
                should_exit,
                exit_code,
                grad_norm,
                num_zeros_in_grad,
            ) = train_step(
                forward_step_func, train_data_iterator, model, optimizer, opt_param_scheduler, config,
                forward_backward_func
            )
        ft_integration.on_training_step_end()
        if should_checkpoint:
            save_checkpoint_and_time(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
            )
        if should_exit:
            break

        # Enable forward pre-hooks after first set of forward and backward passes.
        # When running in fp16, skip all NaN iterations until steady-state loss scaling value
        # is reached.
        if iteration == start_iteration:
            if skipped_iter:
                # Only enable forward pre-hook after a training step has successfully run. Relevant
                # for fp16 codepath where first XX iterations are skipped until steady-state loss
                # scale value is reached.
                start_iteration = iteration + 1
            else:
                # Enable forward pre-hook after training step has successfully run. All subsequent
                # forward passes will use the forward pre-hook / `param_sync_func` in
                # `forward_backward_func`.
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(model)
                    config.param_sync_func = param_sync_func
                    pre_hook_enabled = True
                    # Set the manual hooks here since it's not set right after the capturing.
                    if (
                            args.cuda_graph_impl == "transformer_engine"
                            and iteration == args.cuda_graph_warmup_steps
                    ):
                        cuda_graph_helper.cuda_graph_set_manual_hooks()

        iteration += 1
        if getattr(args, 'perform_rl_step', False) and args.rl_use_sequence_packing:
            iteration_sequences = rl_utils.get_iteration_sequence_count(args)
            # Track bins separately for packed mode
            rl_utils.update_sequence_packing_metrics(args)
        else:
            batch_size = (
                    mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
            )
            iteration_sequences = batch_size

        # Update consumed samples (always means sequences now)
        args.consumed_train_samples += iteration_sequences

        # Use iteration_sequences as batch_size for floating point operations
        batch_size = iteration_sequences

        num_skipped_samples_in_batch = (
                get_current_global_batch_size() - get_current_running_global_batch_size()
        )
        if args.decrease_batch_size_if_needed:
            assert num_skipped_samples_in_batch >= 0
        else:
            assert num_skipped_samples_in_batch == 0
        args.skipped_train_samples += num_skipped_samples_in_batch
        num_floating_point_operations_in_batch = num_floating_point_operations(args, batch_size)
        num_floating_point_operations_so_far += num_floating_point_operations_in_batch
        num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

        # Logging.
        if not optimizer.is_stub_optimizer:
            loss_scale = optimizer.get_loss_scale().item()
        else:
            loss_scale = 1.0
        params_norm = None

        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if len(param_group['params']) == 0:
                continue
            if param_group['is_decoupled_lr']:
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']
        report_memory_flag, iter_log = training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
        )

        # Evaluation.
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            if args.log_energy:
                energy_monitor.pause()
            timers('interval-time').stop()
            if should_disable_forward_pre_hook(args):
                disable_forward_pre_hook(model)
                pre_hook_enabled = False
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f'iteration {iteration}'
            timers('eval-time', log_level=0).start(barrier=True)
            if getattr(args, 'perform_rl_step', False):
                rl_utils.evaluate_and_print_results_rl(valid_data_iterator, model, optimizer,
                                                       iteration, write_to_tensorboard=True)
            else:
                evaluate_and_print_results(prefix, forward_step_func,
                                           valid_data_iterator, model,
                                           iteration, process_non_loss_data_func,
                                           config, verbose=False, write_to_tensorboard=True,
                                           non_loss_data_func=non_loss_data_func)

            eval_duration += timers('eval-time').elapsed()
            eval_iterations += sum(args.eval_iters) if isinstance(args.eval_iters, list) else args.eval_iters
            timers('eval-time').stop()
            one_logger_utils.track_e2e_metrics()

            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                pre_hook_enabled = True
            timers('interval-time', log_level=0).start(barrier=True)
            if args.log_energy:
                energy_monitor.resume()

        # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
        # Some of these only happen at specific iterations.
        post_training_step_callbacks(
            model,
            optimizer,
            opt_param_scheduler,
            iteration,
            prof,
            num_floating_point_operations_since_last_log_event,
        )

        # Checkpoint and decide whether to exit.
        should_exit = checkpoint_and_decide_exit(
            model,
            optimizer,
            opt_param_scheduler,
            iteration,
            num_floating_point_operations_so_far,
            checkpointing_context,
            train_data_iterator,
            iter_log,
        )
        if should_exit:
            break

    one_logger_utils.track_e2e_metrics()

    # Flush TensorBoard, WandB writers and one-logger.
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    ft_integration.on_checkpointing_start()
    # This will finalize all unfinalized async request and terminate
    # a persistent async worker if persistent ckpt worker is enabled
    maybe_finalize_async_save(blocking=True, terminate=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)
    if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
        ft_integration.get_rank_monitor_client().shutdown_workload_monitoring()

    if args.log_energy:
        energy_monitor.lap()
        total_energy = energy_monitor.get_total()
        print_rank_0(f"Total training energy (GPU): {total_energy / 1e6} MJ")
        energy_monitor.shutdown()

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.finish()
        ft_integration.shutdown()
        one_logger_utils.finish()
        sys.exit(exit_code)

    return iteration, num_floating_point_operations_so_far


def evaluate(
        forward_step_func,
        data_iterator,
        model,
        process_non_loss_data_func,
        config,
        verbose=False,
        non_loss_data_func=None,
        eval_iters=None,
):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    timers('evaluate', log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        from megatron.legacy.model.vision.knn_monitor import compute_feature_bank

        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    # Disable result validation during evaluation
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // (args.micro_batch_size * args.data_parallel_size)
    forward_backward_func = get_forward_backward_func()
    if args.cuda_graph_impl == "local" and args.cuda_graph_scope == "full_iteration":
        forward_backward_func = FullCudaGraphWrapper(forward_backward_func,
                                                     cuda_graph_warmup_steps=args.cuda_graph_warmup_steps)

    if eval_iters is None:
        eval_iters = args.eval_iters

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f'Evaluating on {eval_iters * eval_batch_size} samples')
        while iteration < eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f'Evaluating iter {iteration}/{eval_iters}')

            # Don't care about timing during evaluation
            config.timers = None
            ft_integration.on_eval_step_start()
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
            )
            ft_integration.on_eval_step_end()
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if args.schedule_method == 'dualpipev':
                is_last_stage = mpu.is_pipeline_first_stage(ignore_virtual=True)
            else:
                is_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)

            if is_last_stage:
                # Reduce across processes.
                for key in loss_dicts[0].keys():
                    if key not in total_loss_dict:
                        total_loss_dict[key] = torch.tensor(
                            [0.0, 0.0], dtype=torch.float
                        ).cuda()
                    val = [x[key].view(-1) for x in loss_dicts]

                    if val[0].numel() == 2:
                        if args.sft:
                            # normalize over micro batch instead of global
                            val = torch.vstack(val)
                            val = val[:, 0] / val[:, 1]
                            val = val.mean()
                            torch.distributed.all_reduce(
                                val,
                                group=mpu.get_data_parallel_group(with_context_parallel=True)
                            )
                            val /= torch.distributed.get_world_size(
                                group=mpu.get_data_parallel_group(with_context_parallel=True)
                            )
                            total_loss_dict[key][0] += val
                            total_loss_dict[key][1] += 1
                        else:
                            val = torch.vstack(val).sum(dim=0)
                            torch.distributed.all_reduce(
                                val,
                                group=mpu.get_data_parallel_group(with_context_parallel=True)
                            )
                            total_loss_dict[key] += val
                    elif val[0].numel() == 1:
                        val = torch.cat(val).sum()
                        total_loss_dict[key][0] += val
                        total_loss_dict[key][1] += len(loss_dicts)
                    else:
                        raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")

            args.consumed_valid_samples += eval_batch_size

            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins], dtype=torch.int, device='cuda'
                )
                torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    rerun_state_machine.set_mode(rerun_mode)
                    print_rank_0('Exiting during evaluation, timelimit reached')
                    return None, None, True

        is_last_rank_func = is_rank0 if args.schedule_method == 'dualpipev' else is_last_rank
        collected_non_loss_data = None
        if non_loss_data_func is not None:
            collected_non_loss_data = non_loss_data_func(model)
        elif process_non_loss_data_func is not None and is_last_rank_func():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True,
            )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        numerator, denominator = total_loss_dict[key]
        total_loss_dict[key] = numerator / denominator

    timers('evaluate').stop()
    timers.log(['evaluate'])

    rerun_state_machine.set_mode(rerun_mode)

    rerun_state_machine.set_mode(rerun_mode)

    return total_loss_dict, collected_non_loss_data, False


def save_checkpoint_and_time_wrapper(fn):
    @wraps(fn)
    def wrapper(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            non_persistent_ckpt=False,
            train_data_iterator=None,
    ):
        args = get_args()

        if args.enable_dynamic_grad_comp:
            if torch.distributed.get_rank() == 0:
                append_time_to_csv(args.checkpoint_date_path, iteration)
                n = args.save_interval // (int(1 / args.iteration_sample_ratio))
                recent_loss = Utils.loss[-n:]
                append_data_to_csv(args.loss_path, iteration, recent_loss)
            append_data_to_csv(args.mapped_rank_path, iteration, Utils.mapped_rank)

        fn(iteration, model, optimizer, opt_param_scheduler,
           num_floating_point_operations_so_far, checkpointing_context,
           non_persistent_ckpt=non_persistent_ckpt, train_data_iterator=train_data_iterator)

    return wrapper
