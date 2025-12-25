# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Input/output checkpointing."""

import contextlib
import os
import random
import shutil
import sys
import threading
from enum import Enum, auto
from logging import getLogger
from pathlib import Path

import numpy as np
from time import time

import torch

from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import \
    FullyParallelSaveStrategyWrapper, FullyParallelLoadStrategyWrapper
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.core.utils import is_te_min_version
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.training.async_utils import schedule_async_save, is_empty_async_queue
from megatron.core.training.global_vars import get_args, get_one_logger
from megatron.core.training.utils import unwrap_model, print_rank_0, append_to_progress_log, is_last_rank
from megatron.core.dist_checkpointing.serialization import \
    get_default_save_sharded_strategy
from megatron.core.training.one_logger_utils import on_save_checkpoint_start, on_save_checkpoint_success
from megatron.core.training import wandb_utils

from megatron.core.training import ft_integration

# [ModelOpt]: Import
try:
    from modelopt.torch.opt.plugins import (
        save_modelopt_state,
        save_sharded_modelopt_state,
        restore_modelopt_state,
        restore_sharded_modelopt_state,
    )
    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False

_CHECKPOINT_VERSION = None

logger = getLogger(__name__)
_NON_PERSISTENT_CKPT_SUBDIR = 'non_persistent'
from .training import iter_log


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far,
                    checkpointing_context=None, pipeline_rank=None, expert_rank=None, tensor_rank=None, pipeline_parallel=None, expert_parallel=None, non_persistent_ckpt=False,
                    train_data_iterator=None, preprocess_common_state_dict_fn = None, release=False):
    """Save a model, optimizer and optionally dataloader checkpoint.

    Checkpointing context is used to persist some checkpointing state
    throughout a single job. Must be initialized externally (not used if None).

    If non_persistent_ckpt is True,
    the checkpoint will be saved with special functionality for removing old checkpoints.
    There are several types of non-persistent checkpoints:
    "global" - Saved as a standard checkpoint (e.g., on Lustre) with old checkpoints being removed.
    "local" - Each rank saves a portion of the checkpoint locally (e.g., on SSD/ramdisk).

    Dataloader checkpoint is only saved if the dataloader supports it. Currently this applies only
    to the Megatron Energon dataloader (multimodal) and not the built-in Megatron dataloader (text-only).
    """
    start_ckpt = time()
    args = get_args()

    if args.async_save and not is_empty_async_queue():
        print_rank_0('WARNING: Starting a checkpoint save before previous has finished. Consider increasing the checkpoint interval.')

    # Prepare E2E metrics at start of save checkpoint
    productive_metrics = on_save_checkpoint_start(args.async_save)

    # Monitor for the checkpointing timeout (no-op if FT is not enabled)
    ft_integration.on_checkpointing_start()

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    # Handle non_persistent_ckpt flag. Besides overwriting `args.save` and
    # `args.use_dist_ckpt`, non-persistent global ckpt requires no additional logic
    ckpt_type = CheckpointType.GLOBAL if args.use_dist_ckpt else CheckpointType.LEGACY
    save_dir = args.save
    if non_persistent_ckpt:
        if args.non_persistent_ckpt_type == 'global':
            ckpt_type = CheckpointType.GLOBAL
            save_dir = (
                args.non_persistent_global_ckpt_dir
                if args.non_persistent_global_ckpt_dir
                else os.path.join(save_dir, _NON_PERSISTENT_CKPT_SUBDIR)
            )
            # TODO Can we ensure the previous checkpoint is saved? We don't want to allow two saves in parallel.
            cleanup_old_non_persistent_checkpoint(
                save_dir, leave_ckpt_num=1, do_async=args.async_save
            )
        elif args.non_persistent_ckpt_type == 'local':
            ckpt_type = CheckpointType.LOCAL
            save_dir = checkpointing_context['local_checkpoint_manager'].local_ckpt_dir
        else:
            raise NotImplementedError(f"Please use local or global non-persistent checkpoints (got: {args.non_persistent_ckpt_type})")

    ckpt_format = args.ckpt_format if ckpt_type == CheckpointType.GLOBAL else 'torch'
    print_rank_0('saving checkpoint at iteration {:7d} to {} in {} format'.format(
        iteration, save_dir, ckpt_format))

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state(args.ckpt_format)

    # Collect rerun state across all ranks
    rerun_state_machine = get_rerun_state_machine()
    rerun_state = rerun_state_machine.state_dict(
        data_iterator=train_data_iterator, ckpt_format=args.ckpt_format,
    )

    # Checkpoint name.
    return_base_dir = (ckpt_type != CheckpointType.LEGACY)
    checkpoint_name = get_checkpoint_name(save_dir, iteration, release=release, pipeline_parallel=pipeline_parallel,
        tensor_rank=tensor_rank, pipeline_rank=pipeline_rank, expert_parallel=expert_parallel, expert_rank=expert_rank, return_base_dir=return_base_dir)

    # Save dataloader state if the dataloader supports it (currently only Megatron Energon).
    maybe_save_dataloader_state(train_data_iterator, iteration, getattr(args, "dataloader_save", None))

    # Save distributed optimizer's custom parameter state.
    if (
        args.use_distributed_optimizer
        and not args.no_save_optim
        and optimizer is not None
        and ckpt_type == CheckpointType.LEGACY
    ):
        optim_checkpoint_name = \
            get_distributed_optimizer_checkpoint_name(checkpoint_name)
        ensure_directory_exists(optim_checkpoint_name)
        if not optimizer.is_stub_optimizer:
            optimizer.save_parameter_state(optim_checkpoint_name)

    async_save_request = None
    if args.async_save:
        if ckpt_type == CheckpointType.LEGACY:
            raise NotImplementedError('Async checkpoint save not implemented for legacy checkpoints')
        elif ckpt_type == CheckpointType.GLOBAL and args.ckpt_format != 'torch_dist':
            raise NotImplementedError(f'Async checkpoint save not implemented for {args.ckpt_format} distributed checkpoint format')

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # Collect args, model, RNG.
    if not torch.distributed.is_initialized() \
            or mpu.get_expert_data_parallel_rank() == 0 \
            or ckpt_type != CheckpointType.LEGACY:
        optim_sd_kwargs = {}
        if ckpt_type != CheckpointType.LEGACY and args.use_distributed_optimizer:
            optim_sd_kwargs['sharding_type'] = ('fully_sharded_model_space'
                                                if args.ckpt_fully_parallel_save
                                                else 'dp_zero_gather_scatter')
            print_rank_0(f'Storing distributed optimizer sharded state of type {optim_sd_kwargs["sharding_type"]}')
        state_dict = generate_state_dict(
            args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            iteration=iteration,
            optim_sd_kwargs=optim_sd_kwargs,
            rerun_state=rerun_state,
        )

        state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
        if ckpt_type == CheckpointType.GLOBAL and ckpt_format == "torch_dist":
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                # TODO Handle non-empty directories (e.g., after a crash during saving).
                ensure_directory_exists(checkpoint_name, check_parent=False)
            if checkpointing_context is not None and 'save_strategy' in checkpointing_context:
                save_strategy = checkpointing_context['save_strategy']
                # Already saved once before - don't need to rerun sharding validation
                validate_sharding_integrity = not args.ckpt_assume_constant_structure
            else:
                validate_sharding_integrity = True
                save_strategy = get_default_save_sharded_strategy(args.ckpt_format)
                if args.ckpt_assume_constant_structure and args.ckpt_format == 'torch_dist':
                    save_strategy.use_cached_ckpt_structure = args.ckpt_assume_constant_structure
                    if checkpointing_context is not None and 'load_strategy' in checkpointing_context:
                        cached_global_metadata = getattr(checkpointing_context['load_strategy'], 'cached_global_metadata', None)
                        if cached_global_metadata is not None:
                            logger.debug("Plugging in the read metadata from the load strategy...")
                            save_strategy.cached_global_metadata = cached_global_metadata
                        else:
                            logger.debug("Failed to plug in the read metadata from the load strategy...")

                if args.ckpt_fully_parallel_save:
                    save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, mpu.get_data_parallel_group(with_context_parallel=True),
                                                                     args.ckpt_assume_constant_structure)
            # Store save strategy for future checkpoint saves
            if checkpointing_context is not None:
                checkpointing_context['save_strategy'] = save_strategy
            end_ckpt = time()
            logger.debug(f"rank: {rank}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt ")
            async_save_request = dist_checkpointing.save(state_dict, checkpoint_name, save_strategy,
                                                         async_sharded_save=args.async_save,
                                                         validate_access_integrity=validate_sharding_integrity,
                                                         preprocess_common_before_consistancy_check=preprocess_common_state_dict_fn)
            # [ModelOpt]: save sharded modelopt_state
            if has_nvidia_modelopt:
                save_sharded_modelopt_state(model, checkpoint_name, (args.ckpt_format, 1))
        elif ckpt_type == CheckpointType.GLOBAL and ckpt_format in ["torch_dcp", "fsdp_dtensor"]:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                # TODO Handle non-empty directories (e.g., after a crash during saving).
                ensure_directory_exists(checkpoint_name, check_parent=False)

            fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(checkpoint_name)
            torch.distributed.checkpoint.save(
                state_dict=state_dict,
                storage_writer=fs_storage_writer,
            )
        else:
            # [ModelOpt]: Inject modelopt_state into state_dict
            if has_nvidia_modelopt:
                if ckpt_type == CheckpointType.LOCAL:
                    print_rank_0('WARNING: Local checkpointing does not support nvidia_modelopt.')
                else:
                    save_modelopt_state(model, state_dict)

            end_ckpt = time()
            logger.debug(f"rank: {rank}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt ")
            if ckpt_type == CheckpointType.LOCAL:
                try:
                    from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict
                except ModuleNotFoundError:
                    raise RuntimeError("The 'nvidia_resiliency_ext' module is required for local "
                                       "checkpointing but was not found. Please ensure it is installed.")
                if (sharded_sd_metadata or {}).get('distrib_optim_sharding_type') in ['fully_reshardable', 'dp_zero_gather_scatter']:
                    # Note: Currently full reshardabilty is not supported when local checkpoints are used.
                    raise RuntimeError(
                        f"Local checkpointing does not support optimizer sharding type '{sharded_sd_metadata['distrib_optim_sharding_type']}'. "
                        "Don't use '--dist-ckpt-optim-fully-reshardable' when saving local checkpoints."
                    )

                algo = args.non_persistent_local_ckpt_algo
                cached_metadata = None
                if args.ckpt_assume_constant_structure and 'local_checkpoint_cache' in checkpointing_context:
                    cached_metadata = checkpointing_context['local_checkpoint_cache']
                state_dict_for_save, cacheable_metadata = MCoreTensorAwareStateDict.from_state_dict(
                    state_dict, algo=algo, cached_metadata=cached_metadata,
                    parallelization_group=mpu.get_data_parallel_group(with_context_parallel=True)
                )
                async_save_request = checkpointing_context['local_checkpoint_manager'].save(
                    state_dict_for_save, iteration, is_async=bool(args.async_save)
                )
                checkpointing_context['local_checkpoint_cache'] = cacheable_metadata
            else:
                assert ckpt_type == CheckpointType.LEGACY
                # Save.
                ensure_directory_exists(checkpoint_name)
                torch.save(state_dict, checkpoint_name)
    start_misc = time()
    if ckpt_type != CheckpointType.LOCAL:
        if not args.async_save:
            assert async_save_request is None
            # Wait so everyone is done (necessary)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    # And update the latest iteration
    if not torch.distributed.is_initialized() \
            or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(save_dir)

        if ckpt_type == CheckpointType.LOCAL:
            def iter_finalize_fn():
                print_rank_0('  successfully saved local checkpoint from iteration {:7d}'
                             .format(iteration))
                if args.log_progress and args.async_save:
                    append_to_progress_log(f'Saved async local checkpoint\tIteration: {iteration}',
                                           barrier=False)
        else:
            def iter_finalize_fn():
                prev_iteration = 0
                save_retain_interval = getattr(args, 'save_retain_interval', None)  # For backwards compatibility of tests.
                if save_retain_interval is not None:
                    if os.path.exists(tracker_filename):  # TODO: Make this work with MSC remote paths?
                        with open_file(tracker_filename, 'r') as f:
                            prev_iteration = int(f.read().strip())
                with open_file(tracker_filename, 'w') as f:
                    f.write("release" if release else str(iteration))
                tensor_rank_to_print = (tensor_rank if tensor_rank is not None else mpu.get_tensor_model_parallel_rank()) + 1
                pipeline_rank_to_print = (pipeline_rank if pipeline_rank is not None else mpu.get_pipeline_model_parallel_rank()) + 1
                print_rank_0(f'  successfully saved checkpoint from iteration {int(iteration):7d} to {args.save} '
                             f'[ t {tensor_rank_to_print}/{mpu.get_tensor_model_parallel_world_size()}, '
                             f'p {pipeline_rank_to_print}/{mpu.get_pipeline_model_parallel_world_size()} ]')
                if args.log_progress and args.async_save:
                    append_to_progress_log(f'Saved async checkpoint\tIteration: {iteration}',
                                           barrier=False)

                def delete_checkpoint(args, iteration_to_delete):
                    checkpoint_name = get_checkpoint_name(args.save, iteration=iteration_to_delete,
                                                          return_base_dir=True)
                    try:
                        shutil.rmtree(checkpoint_name)  # TODO: Make this work with MSC remote paths?
                        print_rank_0(f'  successfully deleted checkpoint from iteration {iteration_to_delete:7d} '
                                     f'at {args.save}')
                        if args.log_progress:
                            append_to_progress_log(f'Deleted checkpoint\tIteration: {iteration_to_delete}', barrier=False)
                    except Exception as e:
                        print_rank_0(f'  encountered exception "{e}" when trying to delete checkpoint from '
                                     f'iteration {iteration_to_delete:7d} at {args.save}')
                        # Any exception encountered in checkpoint deletion can be ignored and is not fatal.
                        pass

                if save_retain_interval is not None:
                    if prev_iteration > 0 and prev_iteration != iteration and prev_iteration % save_retain_interval != 0:
                        checkpoint_name = get_checkpoint_name(args.save, iteration=prev_iteration,
                                                              return_base_dir=True)
                        # Don't delete if `checkpoint_name` is a symbolic link.
                        if os.path.islink(checkpoint_name):  # TODO: Make this work with MSC remote paths?
                            print_rank_0(f'  skipping deleting checkpoint from iteration {prev_iteration:7d} '
                                         f'at {args.save} since it is a symbolic link')
                        else:
                            # Asynchronous version of delete_checkpoint(args, iteration_to_delete=prev_iteration).
                            threading.Thread(target=delete_checkpoint, args=(args, prev_iteration,)).start()

        if args.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(iter_finalize_fn)
        else:
            iter_finalize_fn()

    # Additional callback for one_logger (last rank)
    if not torch.distributed.is_initialized() \
       or is_last_rank():
        def onelogger_finalize_fn():
            on_save_checkpoint_success(productive_metrics, args.async_save)
        if args.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(onelogger_finalize_fn)
        else:
            onelogger_finalize_fn()

    # Additional callback for wandb (last rank)
    if not torch.distributed.is_initialized() \
       or is_last_rank():
        def wandb_finalize_fn():
            wandb_utils.on_save_checkpoint_success(checkpoint_name, get_checkpoint_tracker_filename(save_dir), save_dir, iteration)
        if args.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(wandb_finalize_fn)
        else:
            wandb_finalize_fn()

    if args.async_save:
        schedule_async_save(async_save_request)
        print_rank_0('  scheduled an async checkpoint save at iteration {:7d} to {}' \
                     .format(iteration, save_dir))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    end_misc = time()
    logger.debug(f"rank: {rank}, takes {end_misc - start_misc} to finalize ckpt save ")

    ft_integration.on_checkpointing_end(is_async_finalization=False)
