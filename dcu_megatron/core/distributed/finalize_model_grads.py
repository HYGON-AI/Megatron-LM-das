# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional
import torch

try:
    from torch.distributed._tensor import DTensor, distribute_tensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from megatron.core import mpu
from megatron.core import parallel_state
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from ...training.edgc_utils import Utils
from megatron.core.distributed.finalize_model_grads import _allreduce_conditional_embedding_grads, \
    _allreduce_layernorm_grads, _allreduce_embedding_grads, _update_router_expert_bias


def finalize_model_grads(model: List[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    args = get_args()
    config = get_model_config(model[0])

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
            model_chunk.finish_grad_sync()
        if args.enable_dynamic_grad_comp:
            if args.begin_max_rank:
                args.begin_max_rank = False
            if not args.overlap_grad_reduce:
                _handle_all_reduce_time_end(args, config)


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
    _allreduce_conditional_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce').stop()

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_layernorm_grads(model, config)
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    if config.moe_router_enable_expert_bias:
        _update_router_expert_bias(model, config)

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:

        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        if not isinstance(last_rank, list):
            assert not isinstance(last_rank, list)
            last_rank = [last_rank]
            assert not isinstance(pp_group, list)
            pp_group = [pp_group]

        # need to do a broadcast for every pp group, even though num_tokens should be the same.
        num_tokens_list = []
        for lr, group in zip(last_rank, pp_group):
            torch.distributed.broadcast(num_tokens, src=lr, group=group)
            num_tokens_list.append(torch.clone(num_tokens))
        assert all(x.item() == num_tokens_list[0] for x in num_tokens_list)

        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=parallel_state.get_data_parallel_group())
        for model_chunk in model:
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)