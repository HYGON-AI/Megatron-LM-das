# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Megatron initialization."""
import os
import random
import numpy as np
import torch

from datetime import timedelta
from functools import wraps

from megatron.training import get_args
from megatron.training import inprocess_restart
from megatron.core import mpu, tensor_parallel
from megatron.core.utils import is_torch_min_version
from megatron.training.utils import print_rank_0, warn_rank_0

from hcu_megatron.training.arguments import get_adaptor_args


def initialize_megatron_wrapper(initialize_megatron_func):
    @wraps(initialize_megatron_func)
    def wrapper(
        allow_no_cuda=False,
        skip_mpu_initialization=False,
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
        store=None,
    ):

        initialize_megatron_func(
            allow_no_cuda=allow_no_cuda,
            skip_mpu_initialization=skip_mpu_initialization,
            get_embedding_ranks=get_embedding_ranks,
            get_position_embedding_ranks=get_position_embedding_ranks,
            store=store,
        )

        args = get_args()
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

    return wrapper


def _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, store):
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        print_rank_0("torch distributed is already initialized, skipping initialization ...")
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        print_rank_0("> initializing torch distributed ...")
        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(args.local_rank)
            device_id = torch.device(f'cuda:{args.local_rank}')
        else:
            device_id = None

        # Set to non-default stream for cudagraph capturing.
        if args.cuda_graph_impl == "transformer_engine":
            torch.cuda.set_stream(torch.cuda.Stream())

        # Set flight recorder env vars if specified.
        # Priority: pre-existing environment variable > MLM argument.
        # All vars follow the same setdefault semantics: if already set in the
        # environment we warn and keep the user's value; otherwise we apply the
        # value derived from the MLM argument / flag.
        # The block is also triggered when either path env var is already set
        # so that the remaining defaults are applied consistently.
        _fr_path = (
            args.flight_recorder_dump_path
            or os.environ.get('TORCH_FR_DUMP_TEMP_FILE')
            or os.environ.get('TORCH_NCCL_DEBUG_INFO_TEMP_FILE')
        )
        if _fr_path is not None:
            _fr_dump_prefix = _fr_path
            if os.path.isdir(_fr_path):
                _fr_dump_prefix = os.path.join(_fr_path, '_dump_')
                warn_rank_0(
                    "Flight recorder: using directory "
                    f"'{_fr_path}' for dump path, appending per-rank prefix "
                    f"'{_fr_dump_prefix}'."
                )
            _fr_env_defaults = {
                'TORCH_FR_DUMP_TEMP_FILE': _fr_dump_prefix,
                'TORCH_NCCL_DEBUG_INFO_TEMP_FILE': _fr_dump_prefix,
                'TORCH_NCCL_TRACE_BUFFER_SIZE': str(args.flight_recorder_trace_buffer_size),
                'TORCH_NCCL_DUMP_ON_TIMEOUT': str(int(args.flight_recorder_dump_on_timeout)),
                'TORCH_INCLUDE_STACK_TRACE': str(int(args.flight_recorder_include_stack_trace)),
                'TORCH_INCLUDE_ONLY_ACTIVE': str(int(args.flight_recorder_include_only_active)),
                'TORCH_NCCL_EXTRA_DUMP_ON_EXEC': str(int(args.flight_recorder_extra_dump_on_exec)),
            }
            for _var, _default in _fr_env_defaults.items():
                if _var in os.environ:
                    warn_rank_0(
                        f"Flight recorder: environment variable {_var} is already set to "
                        f"'{os.environ[_var]}'; ignoring config value '{_default}'."
                    )
                else:
                    os.environ[_var] = _default
            print_rank_0(
                "Flight recorder env vars:\n"
                + "\n".join(f"  {k}={os.environ[k]}" for k in _fr_env_defaults)
            )

        # Call the init process
        init_process_group_kwargs = {
            'backend': args.distributed_backend,
            'store': store,
            'world_size': args.world_size,
            'rank': args.rank,
            'timeout': timedelta(minutes=args.distributed_timeout_minutes),
        }
        if os.getenv("LAUNCH_BACKEND", "mpirun") == "mpirun":
            init_process_group_kwargs.update({'init_method': args.dist_url})
        if args.fake_process_group:
            assert is_torch_min_version(
                "2.3.0"
            ), "Fake process group is only supported with PyTorch 2.3.0 and above."
            from torch.testing._internal.distributed.fake_pg import FakeStore

            store = FakeStore()
            init_process_group_kwargs['backend'] = 'fake'
            init_process_group_kwargs['store'] = store

        torch.distributed.init_process_group(**init_process_group_kwargs)
        inprocess_restart.maybe_force_nccl_backend_init(device_id)

    if args.overlap_moe_expert_parallel_comm and args.enable_pre_init_ep_overlap_streams:
        from megatron.core.pipeline_parallel.utils import set_streams, get_comp_stream, get_comm_stream
        set_streams()

        wait_tensor = torch.Tensor([0]).cuda()
        wait_tensor.record_stream(get_comp_stream())

        wait_tensor = torch.Tensor([0]).cuda()
        wait_tensor.record_stream(get_comm_stream())

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
                use_sharp=args.use_sharp,
                context_parallel_size=args.context_parallel_size,
                hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
                hybrid_context_parallel=args.hybrid_context_parallel,
                expert_model_parallel_size=args.expert_model_parallel_size,
                num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
                expert_tensor_parallel_size=args.expert_tensor_parallel_size,
                distributed_timeout_minutes=args.distributed_timeout_minutes,
                nccl_communicator_config_path=args.nccl_communicator_config_path,
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp',
                get_embedding_ranks=get_embedding_ranks,
                get_position_embedding_ranks=get_position_embedding_ranks,
                create_gloo_process_groups=args.use_gloo_process_groups,
                high_priority_stream_groups=args.high_priority_stream_groups,
                sharp_enabled_group=args.sharp_enabled_group,
            )
            print_rank_0(
                f"> initialized tensor model parallel with size "
                f"{mpu.get_tensor_model_parallel_world_size()}"
            )
            print_rank_0(
                f"> initialized pipeline model parallel with size "
                f"{mpu.get_pipeline_model_parallel_world_size()}"
            )


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(
                seed, te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng
            )
        if get_adaptor_args().reproduce:
            args = get_args()
            assert (args.attention_dropout > 0) is False, f"To utilize the reproduction function, args.attention_dropout = {args.attention_dropout} must be set to 0."
            assert (args.hidden_dropout > 0) is False, f"To utilize the reproduction function, args.hidden_dropout = {args.hidden_dropout} must be set to 0."
            torch.backends.cudnn.deterministic = True # 设置cudnn后端为确定性算法
            torch.backends.cudnn.benchmark = False # 固定卷积算法
            torch.use_deterministic_algorithms(True) # 使用torch的deterministic算子 避免不确定性
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed_))
