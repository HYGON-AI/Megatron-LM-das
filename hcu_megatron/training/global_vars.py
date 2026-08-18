# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import os

from megatron.training import global_vars
from megatron.training.global_vars import _ensure_var_is_not_initialized


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    _ensure_var_is_not_initialized(global_vars._GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and \
       args.tensorboard_dir and args.rank == 0 if args.schedule_method == "dualpipev" else (args.world_size - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            global_vars._GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir,
                max_queue=args.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _set_wandb_writer(args):
    _ensure_var_is_not_initialized(global_vars._GLOBAL_WANDB_WRITER,
                                   'wandb writer')
    if getattr(args, 'wandb_project', '') and args.rank == (0 if args.schedule_method == "dualpipev" else (args.world_size - 1)):
        if args.wandb_exp_name == '':
            raise ValueError("Please specify the wandb experiment name!")

        import wandb
        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, 'wandb')
        wandb_config = vars(args)
        if 'kitchen_config_file' in wandb_config and wandb_config['kitchen_config_file'] is not None:
            # Log the contents of the config for discovery of what the quantization
            # settings were.
            with open(wandb_config['kitchen_config_file'], "r") as f:
                wandb_config['kitchen_config_file_contents'] = f.read()
        wandb_kwargs = {
            'dir': save_dir,
            'name': args.wandb_exp_name,
            'project': args.wandb_project,
            'config': wandb_config}
        if args.wandb_entity:
            wandb_kwargs['entity'] = args.wandb_entity
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)
        wandb.init(**wandb_kwargs)
        global_vars._GLOBAL_WANDB_WRITER = wandb


def _set_one_logger(args):
    _ensure_var_is_not_initialized(global_vars._GLOBAL_ONE_LOGGER, 'one logger')

    if args.enable_one_logger and args.rank == (0 if args.schedule_method == "dualpipev" else (args.world_size - 1)):
        if args.one_logger_async or getattr(args, 'wandb_project', ''):
            one_logger_async = True
        else:
            one_logger_async = False
        try:
            from one_logger import OneLogger
            config = {
               'project': args.one_logger_project,
               'name': args.one_logger_run_name,
               'async': one_logger_async,
            }
            one_logger = OneLogger(config=config)
            global_vars._GLOBAL_ONE_LOGGER = one_logger
        except Exception:
            print('WARNING: one_logger package is required to enable e2e metrics '
                  'tracking. please go to '
                  'https://confluence.nvidia.com/display/MLWFO/Package+Repositories'
                  ' for details to install it')
