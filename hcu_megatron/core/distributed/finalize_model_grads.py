# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from typing import List, Optional
import torch

from megatron.core import mpu
from megatron.core import parallel_state
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_conditional_embedding_grads,
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_word_embedding_grads,
    _allreduce_position_embedding_grads,
    _allreduce_router_grads,
    reset_model_temporary_tensors,
    _update_router_expert_bias
)
from megatron.core.pipeline_parallel.utils import get_pp_last_rank
from megatron.core.process_groups_config import ProcessGroupCollection

from ...training.edgc_utils import Utils


def finalize_model_grads(
    model: List[torch.nn.Module],
    num_tokens: Optional[torch.Tensor] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    args = get_args()
    config = get_model_config(model[0])
    tp_dp_cp_group = None
    if pg_collection is not None:
        assert hasattr(pg_collection, 'tp')
        assert hasattr(pg_collection, 'pp')
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it is used default "
            "`parallel_state.default_embedding_ranks` to create the process group."
            " If you are using the default process group, please use"
            " `parallel_state.get_embedding_group()` "
            "If you don't need embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have a pos_embd. In previous version, it is used default "
            "`parallel_state.default_position_embedding_ranks` to create the process group."
            " If you are using the default process group, please use "
            " `parallel_state.get_position_embedding_group()` "
            "If you don't need pos_embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'dp_cp')
        if config.moe_router_enable_expert_bias:
            assert hasattr(pg_collection, 'tp_dp_cp') and pg_collection.tp_dp_cp is not None, (
                "pg_collection must have tp_dp_cp when " "moe_router_enable_expert_bias is enabled."
            )
            tp_dp_cp_group = pg_collection.tp_dp_cp
        tp_group = pg_collection.tp
        pp_group = pg_collection.pp
        embd_group = pg_collection.embd
        pos_emb_group = pg_collection.pos_embd
        dp_cp_group = pg_collection.dp_cp
    else:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)

        def _handle_all_reduce_time_start(args, config):
            if args.all_reduce_time:
                config.timers('DP_time', log_level=0).start()

        def _handle_all_reduce_time_end(args, config):
            if args.all_reduce_time:
                config.timers('DP_time').stop()

        def _update_gradient_compression_state(args):
            if args.max_rank is None:
                if args.is_loading_checkpoint:
                    if args.curr_iteration >= (args.latest_iteration + 12):
                        args.grad_comp_enabled = True
                else:
                    if args.curr_iteration >= 12:
                        args.grad_comp_enabled = True
            else:
                if args.curr_iteration > args.warm_up_train_iter:
                    if args.begin_max_rank:
                        args.grad_comp_enabled = not (args.is_loading_checkpoint and (
                                    len(Utils.mapped_rank) == 0 or Utils.mapped_rank[-1] is None))
                    elif (args.curr_iteration % args.rank_adjust_window_size == 1) and (
                            args.curr_iteration != (args.latest_iteration + 1)):
                        args.grad_comp_enabled = True
                        if not mpu.is_pipeline_first_stage():
                            _update_mapped_rank_based_on_final_rank(args)
                elif args.begin_warm_up:
                    args.grad_comp_enabled = False
                    args.begin_warm_up = False
            args.grad_comp = args.grad_comp_enabled

        def _update_mapped_rank_based_on_final_rank(args):
            if len(Utils.mapped_rank) >= 2:
                if args.final_rank is None:
                    args.grad_comp_enabled = False
                elif args.final_rank != Utils.mapped_rank[-2]:
                    if args.final_rank is not None:
                        args.mapped_rank = args.final_rank
                    else:
                        args.grad_comp_enabled = False
            else:
                args.mapped_rank = args.final_rank

        def _get_find_rank(args):
            """Helper to determine rank when finding rank upper limit."""
            if args.mapped_rank is not None:
                return int(args.mapped_rank)
            if args.is_loading_checkpoint:
                return int(Utils.mapped_rank[-1] if Utils.mapped_rank else args.max_rank)
            return int(args.max_rank)

        def _get_adaptive_rank(args):
            """Helper to determine rank during adaptive compression."""
            if args.is_loading_checkpoint:
                delta_iter = args.curr_iteration - args.latest_iteration
            else:
                delta_iter = args.curr_iteration
            return 2 ** int((delta_iter - 9) / 3)

        def compressor_update(args):
            if not args.enable_dynamic_grad_comp or not args.grad_comp:
                args.compressor = None
                return
            if args.fp16:
                compression_dtype = torch.float16
            elif args.bf16:
                compression_dtype = torch.bfloat16
            else:
                compression_dtype = torch.float32

            rank = _get_find_rank(args) if args.find_rank_upper_limit else _get_adaptive_rank(args)
            if args.pre_rank is not None:
                if args.pre_rank == rank:
                    args.compressor.begin_iteration(args.curr_iteration)
                    return
            args.pre_rank = rank
            from .power_sgd import PowerSGDCompressor
            args.compressor = PowerSGDCompressor(
                ef_layout_manager=args.ef_manager,
                rank=rank,
                compression_dtype=compression_dtype
            )
            args.compressor.begin_iteration(args.curr_iteration)

        if args.enable_dynamic_grad_comp and not args.overlap_grad_reduce:
            _handle_all_reduce_time_start(args, config)

        for model_chunk in model:
            if args.enable_dynamic_grad_comp:
                _update_gradient_compression_state(args)
                compressor_update(args)
            model_chunk.finish_grad_sync(force_all_reduce=force_all_reduce)
        if args.enable_dynamic_grad_comp:
            if args.begin_max_rank:
                args.begin_max_rank = False
            if not args.overlap_grad_reduce:
                _handle_all_reduce_time_end(args, config)
    else:
        for model_chunk in model:
            model_chunk.finish_grad_sync(force_all_reduce=force_all_reduce)

    if args.enable_dynamic_grad_comp:
        if args.all_reduce_time:
            args.params_all_reduce_time = config.timers('DP_time').elapsed(reset=True) * 1000.0

    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce t_embedder grads (for pp & vpp of DiT).
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_conditional_embedding_grads(model, config, pp_group)
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce').stop()

    if getattr(config, 'flextron', False):
        _allreduce_router_grads(model, config)

    # All-reduce layer-norm grads (for sequence parallelism) and non-tensor parallel modules.
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_non_tensor_model_parallel_grads(model, config, tp_group)
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_word_embedding_grads(model, config, embd_group, pp_group)
    _allreduce_position_embedding_grads(model, config, pos_emb_group, pp_group)

    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    if config.moe_router_enable_expert_bias:
        if pg_collection is None:
            tp_dp_cp_group = parallel_state.get_tensor_and_data_parallel_group(
                with_context_parallel=True
            )
        _update_router_expert_bias(model, config, tp_dp_cp_group=tp_dp_cp_group)

    reset_model_temporary_tensors(config, model)

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:

        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        assert not isinstance(pp_group, list)
        last_rank = get_pp_last_rank(pp_group)
        torch.distributed.broadcast(num_tokens, src=last_rank, group=pp_group)

        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=dp_cp_group)

        # Clamp to avoid div-by-zero without a host-side branch on a device tensor,
        # which would otherwise cause a sync that is illegal during CUDA graph capture.
        safe_num_tokens = torch.clamp(num_tokens, min=1)
        scaling = 1.0 / safe_num_tokens
        for model_chunk in model:
            model_chunk.scale_gradients(scaling)
