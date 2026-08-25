# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import dataclasses
import gc
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

import torch.distributed

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import get_canonical_lr_for_logging
from megatron.training.theoretical_memory_usage import report_theoretical_memory

import torch

try:
    from megatron.rl import rl_utils
    has_rl_utils = True
except ImportError:
    has_rl_utils = False

try:
    from modelopt.torch.distill.plugins.megatron import get_tensor_shapes_adjust_fn_for_distillation

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from megatron.core import mpu, nccl_allocator, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as megatron_FSDP,
)
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.optimizer import get_mup_config_overrides
from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper
from megatron.core.optimizer.qk_clip import clip_qk
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank
)
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.paged_stash import PagedStashRunner
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel as megatron_FSDP

from megatron.core.optimizer.qk_clip import clip_qk
from megatron.core.utils import (
    StragglerDetector,
    check_param_hashes_across_dp_replicas,
    configure_nvtx_profiling,
    get_attr_wrapped_model,
    get_model_config,
    get_pg_rank,
    get_pg_size,
)
from megatron.training.checkpointing import (
    checkpoint_exists,
    get_loaded_iteration,
    load_checkpoint,
    save_checkpoint,
    save_grads,
)

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from megatron.core.datasets.data_schedule import HybridCPDataLoaderWrapper
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.parallel_state import (
    create_all_gather_groups,
    update_pg_timeout,
)
from megatron.core.rerun_state_machine import (
    RerunMode,
    get_rerun_state_machine,
)
from megatron.core.resharding.refit import swap_model_weights
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.utils import unwrap_model
from megatron.training.config import FaultInjectorConfig
from megatron.training.initialize import write_args_to_tensorboard

try:
    from torch_memory_saver import torch_memory_saver
    torch_memory_saver.hook_mode = "torch"
    HAVE_TORCH_MEMORY_SAVER = True
except ImportError:
    HAVE_TORCH_MEMORY_SAVER = False

from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches,
)
from megatron.core.pipeline_parallel import get_forward_backward_func

from megatron.training import ft_integration, one_logger_utils
from megatron.training.activation_logging import (
    disable_activation_logging,
    disable_tokens_per_expert_logging,
    enable_activation_logging,
    enable_tokens_per_expert_logging,
    save_activations,
    save_tokens_per_expert,
)
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.dgrad_logging import disable_dgrad_logging, enable_dgrad_logging, save_dgrads
from megatron.training.global_vars import (
    get_energy_monitor,
    get_one_logger,
    get_signal_handler,
    get_tensorboard_writer,
    get_timers,
    get_wandb_writer,
)
from megatron.training.training import (
    consume_seqlen_stats_in_iteration,
    disable_forward_pre_hook,
    dummy_train_step,
    enable_forward_pre_hook,
    evaluate_and_print_results,
    get_megatron_ddp_config,
    get_megatron_optimizer_config,
    get_optimizer_param_scheduler,
    num_floating_point_operations as _upstream_num_floating_point_operations,
    post_training_step_callbacks,
    preprocess_common_state_dict,
    print_datetime,
    RL_LOGGABLE_TIMER_NAMES,
    should_disable_forward_pre_hook,
    save_checkpoint_and_time,
    update_train_iters,
    wrap_model_chunks_with_ddp,
    _TRAIN_START_TIME,
    _run_gpu_sniff_test,
)
from megatron.training.utils import (
    calc_params_l2_norm,
    is_last_rank,
    is_rank0,
    logical_and_across_model_parallel_group,
    print_rank_0,
    print_rank_last,
    reduce_max_stat_across_model_parallel_group,
    report_memory,
    to_empty_if_meta_device,
    update_use_dist_ckpt,
)
from .edgc_utils import Utils, append_time_to_csv, append_data_to_csv, read_data_from_csv
from ..core.distributed.power_sgd import EFLayoutManager
from hcu_megatron.training import get_args

stimer = StragglerDetector()


# core_transformer_config_from_args() derives a few TransformerConfig field names from
# differently named CLI args. If the source arg was explicitly present on the command
# line, treat the derived config field as CLI-owned too.
_BRIDGE_CLI_TO_CONFIG_FIELD_ALIASES = {
    "decoder_first_pipeline_num_layers": ("num_layers_in_first_pipeline_stage",),
    "decoder_last_pipeline_num_layers": ("num_layers_in_last_pipeline_stage",),
    "num_experts": ("num_moe_experts",),
    "fp8_param_gather": ("fp8_param",),
    "fp4_param_gather": ("fp4_param",),
    "no_persist_layer_norm": ("persist_layer_norm",),
    "params_dtype": ("pipeline_dtype",),
    "bf16": ("pipeline_dtype",),
    "fp16": ("pipeline_dtype",),
    "overlap_p2p_comm": ("batch_p2p_comm",),
    "swiglu": ("activation_func", "gated_linear_unit", "bias_activation_fusion"),
    "squared_relu": ("activation_func",),
    "quick_geglu": ("activation_func", "gated_linear_unit"),
    "bias_swiglu_fusion": ("bias_activation_fusion",),
    "bias_gelu_fusion": ("bias_activation_fusion",),
    "init_method_xavier_uniform": ("init_method", "scaled_init_method"),
    "group_query_attention": ("num_query_groups",),
    "num_query_groups": ("num_query_groups",),
    "cp_comm_type": ("cp_comm_type",),
    "hybrid_layer_pattern": ("is_hybrid_model", "experimental_attention_variant"),
    "seed": ("inference_sampling_seed",),
    "rotary_interleaved": ("rotary_interleaved",),
}


# The AutoBridge built in setup_model_and_optimizer(), kept so that checkpoint
# saves can export the model back to HuggingFace format without re-reading the
# HF model. None when --use-bridge is not set.
_BRIDGE = None


def _set_bridge(bridge):
    global _BRIDGE
    _BRIDGE = bridge


def get_bridge():
    """Return the AutoBridge created for --use-bridge, or None."""
    return _BRIDGE


def save_hf_checkpoint(iteration, model):
    """Export the model in HuggingFace format alongside the Megatron checkpoint.

    Writes ${save}/iter_XXXXXXX/hf/, containing config.json, the tokenizer files
    and safetensors weights, so the directory can be consumed directly by
    transformers' from_pretrained().

    This is collective: save_hf_pretrained() gathers weights across TP/PP/EP, so
    every rank must call it. Only rank 0 writes the config/tokenizer artifacts
    (unless distributed_save is on, where ranks share the weight shards).
    """
    args = get_args()
    bridge = get_bridge()
    if bridge is None:
        print_rank_0(
            "WARNING: --save-hf-weights is set but no bridge is available; "
            "skipping HuggingFace export."
        )
        return

    hf_path = os.path.join(
        args.save, 'iter_{:07d}'.format(iteration), 'hf'
    )
    print_rank_0(f'  saving HuggingFace format checkpoint to {hf_path}')
    start_time = time.time()

    # save_hf_pretrained() walks named_parameters() and unwraps DDP/Float16Module
    # itself, so the DDP-wrapped chunk list from setup_model_and_optimizer() is
    # exactly what it expects.
    bridge.save_hf_pretrained(
        model,
        hf_path,
        show_progress=False,
        distributed_save=getattr(args, "save_hf_weights_distributed", False),
    )

    torch.distributed.barrier()
    print_rank_0(
        f'  successfully saved HuggingFace format checkpoint to {hf_path} '
        f'({time.time() - start_time:.2f}s)'
    )


def _bridge_select_runtime_field_names(provider, transformer_config, args):
    """Return TransformerConfig field names explicitly requested by CLI.

    CLI-specified fields override the HF/provider config. Fields not explicitly
    specified stay provider-owned.
    """
    provider_names = {f.name for f in dataclasses.fields(provider)}
    tc_names = {f.name for f in dataclasses.fields(transformer_config)}
    common = provider_names & tc_names

    explicit_fields = set()
    for name in getattr(args, "_explicit_args", set()):
        if name in tc_names:
            explicit_fields.add(name)
        explicit_fields.update(_BRIDGE_CLI_TO_CONFIG_FIELD_ALIASES.get(name, ()))

    return common & explicit_fields


def _bridge_apply_runtime_overrides(provider, transformer_config, args):
    """Overlay Megatron args-derived runtime settings onto an AutoBridge provider.

    Mirrors the mechanism in
    `megatron.bridge.training.utils.omegaconf_utils.apply_overrides`: build a dict of
    override values, then hand it to that helper (which walks nested dataclasses and
    skips unknown fields). `_NO_COPY_KEYS` handles (e.g. `_pg_collection`) are re-bound
    by reference to preserve shared state; everything else is deep-copied so the
    ephemeral `transformer_config` and the provider don't alias.

    RoPE / TE-op-fuser switches are read from `args` rather than hardcoded.
    """
    import copy
    from megatron.bridge.training.utils.omegaconf_utils import apply_overrides

    runtime_names = _bridge_select_runtime_field_names(provider, transformer_config, args)
    no_copy_keys = getattr(provider, "_NO_COPY_KEYS", set())

    overrides = {}
    for name in runtime_names:
        value = getattr(transformer_config, name)
        if value is None:
            continue
        if name in no_copy_keys:
            # Preserve shared references — deep-copying would break already-initialized
            # process groups.
            setattr(provider, name, value)
            continue
        overrides[name] = copy.deepcopy(value)

    apply_overrides(provider, overrides, excluded_fields={})

    # RoPE kernel selection: TE fused RoPE does not support 3D positional encodings
    # (mRoPE), so leave those providers alone.
    pos_type = getattr(provider, "position_embedding_type", None)
    if pos_type != "mrope":
        if hasattr(args, "apply_rope_fusion"):
            provider.apply_rope_fusion = args.apply_rope_fusion
        if pos_type != "yarn":
            arg_pos_type = getattr(args, "position_embedding_type", None)
            if arg_pos_type:
                provider.position_embedding_type = arg_pos_type

    # TE FusedMLP: fuses chunk+SiLU/GeLU+multiply into one kernel for GLU-style FFN.
    provider.use_transformer_engine_op_fuser = getattr(
        args, "use_transformer_engine_op_fuser", True
    )

    # VLM-specific overrides (fields that live on the provider but not on
    # TransformerConfig, so they won't flow through the runtime-field overlay).
    _bridge_apply_vlm_overrides(provider, args)


# Fields declared on VLM providers (e.g. Qwen{2.5,3,3.5}VLModelProvider) but not on
# TransformerConfig / core_transformer_config_from_args. Each entry is applied only
# when both the CLI arg and the provider field exist, so pure-LLM providers are
# untouched.
_BRIDGE_VLM_OVERRIDE_FIELDS = (
    "freeze_language_model",
    "freeze_vision_model",
    "freeze_vision_projection",
)


def _bridge_apply_vlm_overrides(provider, args):
    """Overlay VLM-only training-strategy fields from CLI args onto a bridge provider.

    These fields (e.g. `freeze_language_model`) are defined on VLM providers such as
    Qwen2.5VL/Qwen3VL but do not exist on `TransformerConfig`, so the runtime-field
    intersection in `_bridge_select_runtime_field_names` cannot pick them up.
    We setattr each one individually, guarded on both sides so pure-LLM providers
    and older CLI args stay unaffected.
    """
    for name in _BRIDGE_VLM_OVERRIDE_FIELDS:
        if hasattr(provider, name) and hasattr(args, name):
            setattr(provider, name, getattr(args, name))


# ------------------------------------------------------------------------------
# VLM FLOPs accounting (wrapper over Megatron-LM's num_floating_point_operations)
# ------------------------------------------------------------------------------
# Rationale: upstream num_floating_point_operations only counts the language
# model. When training a VLM through the bridge (--use-bridge with a provider
# exposing vision_config), ViT + patch embed + vision->language projector FLOPs
# are omitted, biasing the reported throughput low. We stash the vision meta on
# `args` during setup_and_model_and_optimizer, then a same-name wrapper adds the
# vision term on top of the upstream LLM total. Local same-module functions
# (training_log, checkpoint_and_decide_exit, ...) automatically bind to this
# wrapper by Python name resolution; Megatron-LM's internal callers keep the
# upstream version, which is fine because hcu_megatron uses its own train loop.


def _bridge_extract_vision_meta(provider):
    """Extract the vision fields needed for FLOPs from a bridge provider.

    Returns a plain dict (not a config object) to keep args serialization safe
    and to avoid coupling the FLOPs code to Bridge classes. Returns None if the
    provider has no vision_config (pure LLM providers).
    """
    vision_cfg = getattr(provider, "vision_config", None)
    if vision_cfg is None:
        return None

    return {
        "num_layers": (
            getattr(vision_cfg, "num_hidden_layers", None)
            or getattr(vision_cfg, "depth", None)
        ),
        "hidden_size": (
            getattr(vision_cfg, "hidden_size", None)
            or getattr(vision_cfg, "embed_dim", None)
        ),
        "num_heads": (
            getattr(vision_cfg, "num_heads", None)
            or getattr(vision_cfg, "num_attention_heads", None)
        ),
        "intermediate_size": getattr(vision_cfg, "intermediate_size", None),
        "patch_size": (
            getattr(provider, "patch_size", None)
            or getattr(vision_cfg, "patch_size", 14)
        ),
        "spatial_merge_size": (
            getattr(provider, "spatial_merge_size", None)
            or getattr(vision_cfg, "spatial_merge_size", 1)
        ),
        "temporal_patch_size": getattr(provider, "temporal_patch_size", 1),
        # projector output dim = language hidden size (Qwen VL merger target)
        "llm_hidden_size": provider.hidden_size,
    }


def _vit_layer_flops(batch_size, num_image_tokens, hidden_size, num_heads,
                     intermediate_size, swiglu=False):
    """FLOPs for one ViT block: full self-attn (no causal factor) + MLP.

    Coefficients:
      - 3x for fwd+bwd (wgrad + dgrad)
      - 2x for GEMM mnk
    """
    del num_heads  # Not used; ViT self-attn is MHA and cost only depends on hidden_size.
    ffn_expansion = 3 if swiglu else 2
    attn_fwd = (
        num_image_tokens * hidden_size * (3 * hidden_size)          # qkv proj
        + 2 * hidden_size * num_image_tokens * num_image_tokens     # QK^T + AV
        + num_image_tokens * hidden_size * hidden_size              # o proj
    )
    mlp_fwd = ffn_expansion * num_image_tokens * hidden_size * intermediate_size
    return 2 * batch_size * 3 * (attn_fwd + mlp_fwd)


def _vision_module_flops(batch_size, vision_meta, image_tokens_per_sample):
    """Total FLOPs added by ViT + patch embed + vision->language projector.

    Returns 0 whenever any prerequisite is missing (no vision_meta, no image
    tokens), preserving the LLM-only code path bit-for-bit.
    """
    if vision_meta is None or not image_tokens_per_sample:
        return 0

    hs = vision_meta["hidden_size"]
    n_layers = vision_meta["num_layers"]
    n_heads = vision_meta["num_heads"]
    inter = vision_meta["intermediate_size"] or (4 * hs)
    patch = vision_meta["patch_size"]
    t_patch = vision_meta.get("temporal_patch_size", 1) or 1
    spatial = vision_meta.get("spatial_merge_size", 1) or 1
    llm_hs = vision_meta["llm_hidden_size"]

    vit_flops = n_layers * _vit_layer_flops(
        batch_size, image_tokens_per_sample, hs, n_heads, inter, swiglu=False,
    )
    # patch embed conv: per patch cost = 2 * (t_patch * patch^2 * in_channels=3) * hs,
    # 3x for fwd+bwd
    patch_embed_flops = (
        2 * 3 * batch_size * image_tokens_per_sample
        * (t_patch * patch * patch * 3) * hs
    )
    # vision->language projector (merger): merges spatial^2 vision tokens into one output token,
    # in_features = hs * spatial^2, out_features = llm_hs
    merged_tokens = (
        image_tokens_per_sample // (spatial * spatial) if spatial > 0
        else image_tokens_per_sample
    )
    projector_flops = (
        2 * 3 * batch_size * merged_tokens * (hs * spatial * spatial) * llm_hs
    )
    return vit_flops + patch_embed_flops + projector_flops


def _estimate_image_tokens_per_sample(args):
    """Preference order:
      (1) runtime signal on args (populated by forward_step, Phase 2 -- not wired yet)
      (2) --image-tokens-per-sample CLI (opt-in static value)
      (3) 0 (keeps LLM-only path)
    """
    runtime = getattr(args, "_bridge_image_tokens_per_sample", None)
    if runtime is not None:
        return int(runtime)
    static = getattr(args, "image_tokens_per_sample", None)
    if static is not None:
        return int(static)
    return 0


def num_floating_point_operations(
    args,
    batch_size,
    seqlen_squared_sum_in_batch=None,
    total_real_tokens_in_batch=None,
):
    """Wrapper around Megatron-LM's num_floating_point_operations that also
    accounts for VLM vision-side compute when the bridge exposes vision_config.

    Falls back to the upstream LLM total whenever `args._bridge_vision` is
    absent or the effective image-tokens-per-sample is 0 (both are true for
    every pure LLM run, so this is a bit-for-bit no-op there).
    """
    llm_flops = _upstream_num_floating_point_operations(
        args,
        batch_size,
        seqlen_squared_sum_in_batch=seqlen_squared_sum_in_batch,
        total_real_tokens_in_batch=total_real_tokens_in_batch
    )
    vision_meta = getattr(args, "_bridge_vision", None)
    image_tokens = _estimate_image_tokens_per_sample(args)
    return llm_flops + _vision_module_flops(batch_size, vision_meta, image_tokens)


def build_train_valid_test_data_iterators_wrapper(func):
    @wraps(func)
    def wrapper(build_train_valid_test_datasets_provider):
        args = get_args()

        if (
            args.virtual_pipeline_model_parallel_size is None
            and args.schedule_method == 'dualpipev'
        ):
            train_data_iterator = []
            valid_data_iterator = []
            test_data_iterator = []
            for _ in range(2):
                iterators = func(build_train_valid_test_datasets_provider)
                train_data_iterator.append(iterators[0])
                valid_data_iterator.append(iterators[1])
                test_data_iterator.append(iterators[2])
        else:
            train_data_iterator, valid_data_iterator, test_data_iterator = func(build_train_valid_test_datasets_provider)

        return train_data_iterator, valid_data_iterator, test_data_iterator

    return wrapper


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True, config=None, pg_collection=None):
    """Build the model."""
    args = get_args()
    args.model_type = model_type
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        if args.create_all_gather_group:
            timeout = timedelta(minutes=args.distributed_timeout_minutes) if args.distributed_timeout_minutes else None
            dp_cp_ag, expt_dp_ag = create_all_gather_groups(
                for_expert_parallelism=(args.expert_model_parallel_size > 1),
                timeout=timeout,
            )
            pg_collection.dp_cp_ag = dp_cp_ag
            pg_collection.expt_dp_ag = expt_dp_ag

            print_rank_0("> created all-gather process groups for AG/RS overlap")
            if expt_dp_ag is not None:
                print_rank_0(">   including expert parallelism AG group")

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

    # Build model.
    def build_model():
        args = get_args()
        if (
            get_pg_size(pg_collection.pp) > 1
            and args.virtual_pipeline_model_parallel_size is not None
        ):
            model = []
            vp_size = args.virtual_pipeline_model_parallel_size
            for i in range(vp_size):
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = is_pp_first_stage(pg_collection.pp) and is_vp_first_stage(
                    vp_stage=i, vp_size=vp_size
                )
                post_process = is_pp_last_stage(pg_collection.pp) and is_vp_last_stage(
                    vp_stage=i, vp_size=vp_size
                )
                this_model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                    vp_stage=i,
                    config=config,
                    pg_collection=pg_collection,
                )
                this_model.model_type = model_type
                this_model.vp_stage = i
                model.append(this_model)

        elif args.schedule_method == "dualpipev":
            model = []
            if args.enable_vocab_parallel:
                args.dualpipev_first_chunk = True
                first_model = model_provider_func(
                    pre_process=is_pp_first_stage(pg_collection.pp),
                    post_process=False,
                    vp_stage=0,
                    config=config,
                    pg_collection=pg_collection,
                    split_vocab_embedding=is_pp_first_stage(pg_collection.pp),
                )
                model.append(first_model)

                args.dualpipev_first_chunk = False
                second_model = model_provider_func(
                    pre_process=False,
                    post_process=False,
                    vp_stage=1,
                    config=config,
                    pg_collection=pg_collection,
                    include_layer_norm=is_pp_first_stage(pg_collection.pp),
                )
                model.append(second_model)

                output_chunk = model_provider_func(
                    pre_process=False,
                    post_process=True,
                    config=config,
                    pg_collection=pg_collection,
                    noop_block=True,
                )
                model.append(output_chunk)

                embedding_chunk = model_provider_func(
                    pre_process=False,
                    post_process=False,
                    config=config,
                    pg_collection=pg_collection,
                    split_vocab_embedding=True,
                    noop_block=True,
                )
                model.append(embedding_chunk)

                if is_pp_first_stage(pg_collection.pp):
                    loss_chunk = model_provider_func(
                        pre_process=False,
                        post_process=False,
                        config=config,
                        pg_collection=pg_collection,
                        noop_block=True,
                    )
                    model.append(loss_chunk)

                for chunk in model:
                    chunk.model_type = model_type

                return model

            args.dualpipev_first_chunk = True
            first_model = model_provider_func(
                pre_process=is_pp_first_stage(pg_collection.pp),
                post_process=False,
                vp_stage=0,
                config=config,
                pg_collection=pg_collection,
            )
            first_model.model_type = model_type
            model.append(first_model)

            args.dualpipev_first_chunk = False
            second_model = model_provider_func(
                pre_process=False,
                post_process=is_pp_first_stage(pg_collection.pp),
                vp_stage=1,
                config=config,
                pg_collection=pg_collection,
            )
            second_model.model_type = model_type
            model.append(second_model)

        else:
            if args.enable_vocab_parallel:
                pre_process = is_pp_first_stage(pg_collection.pp)

                model = [
                    model_provider_func(
                        pre_process=pre_process,
                        post_process=False,
                        config=config,
                        pg_collection=pg_collection,
                        split_vocab_embedding=pre_process,
                        include_layer_norm=is_pp_last_stage(pg_collection.pp),
                    ),
                    model_provider_func(
                        pre_process=False,
                        post_process=True,
                        config=config,
                        pg_collection=pg_collection,
                        noop_block=True,
                    ),
                    model_provider_func(
                        pre_process=False,
                        post_process=False,
                        config=config,
                        pg_collection=pg_collection,
                        split_vocab_embedding=True,
                        noop_block=True,
                    )
                ]
                model[0].model_type = model_type
                model[1].model_type = model_type
                model[2].model_type = model_type
                if is_pp_last_stage(pg_collection.pp):
                    model.append(
                        model_provider_func(
                            pre_process=False,
                            post_process=False,
                            config=config,
                            pg_collection=pg_collection,
                            noop_block=True,
                        )
                    )
                    model[3].model_type = model_type

                return model

            pre_process = is_pp_first_stage(pg_collection.pp)
            post_process = is_pp_last_stage(pg_collection.pp)
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                config=config,
                pg_collection=pg_collection,
            )
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
    if get_pg_rank(pg_collection.dp) == 0 and get_pg_rank(pg_collection.cp) == 0:
        print(
            ' > number of parameters on (tensor, pipeline) '
            'model parallel rank ({}, {}): {}'.format(
                get_pg_rank(pg_collection.tp),
                get_pg_rank(pg_collection.pp),
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
            is_dualpipev = args.schedule_method == "dualpipev"
            has_loss_chunk = False
            if (
                is_dualpipev and is_pp_first_stage(pg_collection.pp)
                or (not is_dualpipev and is_pp_last_stage(pg_collection.pp))
            ):
                has_loss_chunk = True
                loss_chunk = Float16Module(config, model[4] if is_dualpipev else model[3])

            if is_dualpipev:
                model = [
                    Float16Module(config, model[0]),
                    Float16Module(config, model[1]),
                    Float16Module(config, model[2], force_output_fp32=True),
                    Float16Module(config, model[3], is_embedding_chunk=True),
                ]
            else:
                model = [
                    Float16Module(config, model[0]),
                    Float16Module(config, model[1], force_output_fp32=True),
                    Float16Module(config, model[2], is_embedding_chunk=True),
                ]

            if has_loss_chunk:
                model.append(loss_chunk)
        else:
            model = [Float16Module(config, model_module) for model_module in model]

        if args.enable_hyper_connections and args.mhc_use_tilekernels:
            for model_module in model:
                for layer in model_module.module.decoder.layers:
                    if hasattr(layer, 'self_attention_hyper_connection'):
                        layer.self_attention_hyper_connection._maintain_float32_params()
                    if hasattr(layer, 'mlp_hyper_connection'):
                        layer.mlp_hyper_connection._maintain_float32_params()

        if args.enable_hyper_connections and args.mhc_use_tilekernels:
            for model_module in model:
                if hasattr(model_module.module, 'mtp'):
                    for layer in model_module.module.mtp.layers:
                        if hasattr(layer, 'mtp_model_layer'):
                            layer.mtp_model_layer.self_attention_hyper_connection._maintain_float32_params()
                            layer.mtp_model_layer.mlp_hyper_connection._maintain_float32_params()

    # Materialize tensors on meta device (GPU allocation) if not using FSDP2 and not using Megatron FSDP.
    if args.init_model_with_meta_device and not args.use_torch_fsdp2 and not args.use_megatron_fsdp:
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

        ddp_config = get_megatron_ddp_config(args)
        if not getattr(args, "use_torch_fsdp2", False):
            if ddp_config.num_buckets is not None:
                ddp_config.bucket_size = num_parameters // ddp_config.num_buckets

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

        # Compute per-chunk bucket sizes / disable_bucketing flags. Bucketing is
        # disabled for non-first chunks, when overlap_param_gather_with_optimizer_step
        # is on, or for non-zero pipeline-parallel ranks.
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        per_chunk_disable_bucketing = [
            (chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step
            for chunk_idx in range(len(model))
        ]
        per_chunk_bucket_sizes = [
            None if (disable or pp_rank > 0) else ddp_config.bucket_size
            for disable in per_chunk_disable_bucketing
        ]

        # Setup stream for ddp initialization. The side-stream may be necessary for cuda graph
        #  capture support with DDP, but we sync it with the current stream to avoid races.
        ddp_stream = torch.cuda.Stream()
        # Wait for the default stream to complete before starting ddp_stream
        ddp_stream.wait_stream(torch.cuda.current_stream())
        # Make ddp_stream start after whatever the default stream already queued
        with torch.cuda.stream(ddp_stream):
            model = wrap_model_chunks_with_ddp(
                model,
                config,
                ddp_config,
                use_layer_wise_distributed_optimizer=getattr(
                    args, 'use_layer_wise_distributed_optimizer', False
                ),
                use_layer_wise_param_layout=getattr(
                    args, 'use_layer_wise_param_layout', True
                ),
                DP=DP,
                pg_collection=pg_collection if args.use_megatron_fsdp else None,
                bucket_sizes=per_chunk_bucket_sizes,
                disable_bucketing_per_chunk=per_chunk_disable_bucketing,
            )
        # End of setup_stream
        # Critical: ensure side-stream work completes before touching params on default stream
        torch.cuda.current_stream().wait_stream(ddp_stream)

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def setup_model_and_optimizer(
    model_provider_func,
    model_type,
    checkpointing_context=None,
):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    # Typically, --skip-train is the only thing needed to disable the optimizer.
    has_normal_optimizer = not args.skip_train
    # Even with --skip-train, RL still creates an optimizer unless --no-load-optim is set.
    has_rl_optimizer = args.perform_rl_step and not args.no_load_optim
    skip_optimizer = not (has_normal_optimizer or has_rl_optimizer)
    wrap_with_ddp = not skip_optimizer
    if args.use_bridge:
        from megatron.bridge import AutoBridge
        from megatron.training.arguments import core_transformer_config_from_args
        if args.bridge_hf_model is None:
            raise ValueError("When --use-bridge is set, --bridge-hf-model must be provided.")
        bridge = AutoBridge.from_hf_pretrained(args.bridge_hf_model)
        provider = bridge.to_megatron_provider(load_weights=args.load_weights, hf_path=args.bridge_hf_model)
        # Keep the bridge alive past model construction so that checkpoint saves
        # can re-use it to export the model back to HuggingFace format. It owns
        # the HF config / tokenizer / weight mapping needed for the conversion.
        _set_bridge(bridge)

        # For VLM providers, optionally skip the vision tower and only build the
        # language model. get_model → _create_model calls provider.provide(), so
        # rebinding it to provide_language_model routes distributed setup through
        # the LLM-only path (returns MCoreGPTModel instead of the full VL model).
        bridge_language_only = (
            getattr(args, "bridge_language_model_only", False)
            and hasattr(provider, "provide_language_model")
        )
        if bridge_language_only:
            provider.provide = provider.provide_language_model
            # VL providers keep position_embedding_type="mrope" for 3D
            # temporal/height/width position ids. In language-only mode
            # pretrain_gpt.py feeds 2D [B,S] position_ids and the plain
            # MCoreGPTModel wires up upstream MultimodalRotaryEmbedding, whose
            # output layout does not match the downstream attention consumer.
            # For pure text the T/H/W channels collapse to a single sequence
            # axis, so standard partial-RoPE (rotary_percent kept) is
            # mathematically equivalent and works with 2D position_ids.
            if getattr(provider, "position_embedding_type", None) == "mrope":
                provider.position_embedding_type = "rope"
            # VL providers default scatter_embedding_sequence_parallel=False so
            # the vision tower can splice image features into the full-seq
            # embedding before SP scatter. Under language-only, the plain
            # MCoreGPTModel decoder (and MTP) expects SP-scattered
            # [s/tp, b, h] input, so re-enable the scatter.
            if hasattr(provider, "scatter_embedding_sequence_parallel"):
                provider.scatter_embedding_sequence_parallel = True

        if hasattr(provider, "finalize"):
            # Overlay Megatron args-derived runtime knobs onto the HF-derived
            # provider before finalize() locks the config.
            #
            # Coverage — three layers, each with its own selection rule:
            #
            # 1) `_bridge_apply_runtime_overrides` → covers TransformerConfig
            #    fields explicitly requested on the CLI. CLI field names that map
            #    to differently named TransformerConfig fields are handled by
            #    `_BRIDGE_CLI_TO_CONFIG_FIELD_ALIASES`. Fields not specified on
            #    the command line remain HF/provider-owned.
            #    Also overlays RoPE fusion / position-embedding-type (skipped for
            #    mrope/yarn) and use_transformer_engine_op_fuser.
            #
            # 2) `_bridge_apply_vlm_overrides` → covers VLM-only fields that
            #    live on VL providers but not on TransformerConfig:
            #      freeze_language_model, freeze_vision_model,
            #      freeze_vision_projection.
            #    Guarded by hasattr on both sides, so pure-LLM providers skip.
            #
            # 3) The DDP config below (built from args after this block) — the
            #    provider itself never owns DDP settings; they're consumed by
            #    provide_distributed_model(ddp_config=...).
            #
            # NOT overlaid (intentionally) — these stay HF-owned via
            # bridge.to_megatron_provider(). Architecture / model-shape fields
            # whose names don't match any pattern in (1) and aren't in (2):
            #   - Shape: num_layers, hidden_size, ffn_hidden_size,
            #     num_attention_heads, num_query_groups, kv_channels,
            #     vocab_size, seq_length, max_position_embeddings.
            #     Reason: these define the checkpoint's architecture; overriding
            #     from CLI would break weight loading.
            #   - Normalization / activation: normalization, layernorm_epsilon,
            #     activation_func, gated_linear_unit, add_bias_linear,
            #     add_qkv_bias. Reason: HF config.json is authoritative.
            #   - Fusion switches: bias_activation_fusion, bias_dropout_fusion,
            #     masked_softmax_fusion, apply_rotary_pos_emb_in_fp32,
            #     attention_softmax_in_fp32, persist_layer_norm,
            #     deallocate_pipeline_outputs. Reason: providers set these to
            #     model-tested defaults; CLI overriding risks silent perf/
            #     accuracy regressions on models the user didn't intend.
            #   - MoE architecture: num_moe_experts, moe_router_topk,
            #     moe_router_load_balancing_type, moe_aux_loss_coeff,
            #     moe_grouped_gemm, moe_token_dispatcher_type, mlp_only_layers,
            #     decoder_sparse_step. Reason: HF-owned per-model.
            #   - RoPE params: rotary_base, rotary_percent, mrope_section.
            #     Reason: model-specific, part of the checkpoint's contract.
            #   - VL architecture: vision_config, patch_size,
            #     temporal_patch_size, spatial_merge_size, image_token_id,
            #     video_token_id, vision_start/end_token_id, bos/eos_token_id,
            #     deepstack_visual_indexes, language_max_sequence_length.
            #     Reason: HF vision config + tokenizer are the source of truth.
            #   - Qwen3.5-only: use_hf_vision_model, vision_dp_when_cp,
            #     hetereogenous_dist_checkpoint, mtp_num_layers. Reason: not
            #     wired to hcu CLI yet — add to _BRIDGE_VLM_OVERRIDE_FIELDS if
            #     you start using Qwen3.5VL.
            transformer_config = core_transformer_config_from_args(args)
            _bridge_apply_runtime_overrides(provider, transformer_config, args)
            provider.finalize()
            # Stash vision meta so the FLOPs wrapper can include ViT/patch-embed/
            # projector cost in throughput reporting. None for pure LLM providers,
            # and also None when --bridge-language-model-only skips the vision tower.
            args._bridge_vision = None if bridge_language_only else _bridge_extract_vision_meta(provider)
            if torch.distributed.get_rank() in [0]:
                print(f"transformer_config: {transformer_config}")
                print(f"provider: {provider}")

        # DDP config — mirror the kwargs build in get_model()'s wrap_with_ddp
        # branch so the two paths stay in sync. Bridge's provide_distributed_model
        # accepts DistributedDataParallelConfig only (torch_fsdp2 is a separate
        # bool arg it consumes internally), so we don't branch on use_torch_fsdp2
        # like get_model() does. Any field appearing here should also appear in
        # get_model() and vice versa.
        kwargs = {}
        for f in dataclasses.fields(DistributedDataParallelConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
        kwargs['check_for_nan_in_grad'] = args.check_for_nan_in_loss_and_grad
        kwargs['check_for_large_grads'] = args.check_for_large_grads
        if args.ddp_num_buckets is not None:
            # get_model() derives bucket_size from num_parameters at this point
            # in the flow, but under use_bridge the model isn't created until
            # provide_distributed_model() below, so num_parameters isn't known.
            # Splitting build/wrap to expose it isn't supported by Bridge's API.
            raise NotImplementedError(
                "--ddp-num-buckets is not supported with --use-bridge; "
                "use --ddp-bucket-size instead."
            )
        kwargs['bucket_size'] = args.ddp_bucket_size
        kwargs['pad_buckets_for_high_nccl_busbw'] = args.ddp_pad_buckets_for_high_nccl_busbw
        kwargs['reduce_scatter_with_fp32_accumulation'] = args.ddp_reduce_scatter_with_fp32_accumulation
        kwargs['param_name_patterns_for_fp32_local_accumulation'] = \
            tuple(args.ddp_param_name_patterns_for_fp32_local_accumulation)
        kwargs['average_in_collective'] = args.ddp_average_in_collective
        # Megatron-FSDP arguments.
        kwargs['megatron_fsdp_main_params_dtype'] = args.megatron_fsdp_main_params_dtype
        kwargs['megatron_fsdp_main_grads_dtype'] = args.megatron_fsdp_main_grads_dtype
        kwargs['megatron_fsdp_grad_comm_dtype'] = args.megatron_fsdp_grad_comm_dtype
        if args.use_megatron_fsdp and args.use_precision_aware_optimizer:
            kwargs["preserve_fp32_weights"] = False

        # Initialize DDPConfig.
        ddp_config = DistributedDataParallelConfig(**kwargs)

        # bucket_size post-processing (mirror of get_model()): give a sane
        # default when unset, and zero it out when grad-reduce isn't overlapped.
        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(
                40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
            )
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        model = provider.provide_distributed_model(
            wrap_with_ddp=wrap_with_ddp,
            ddp_config=ddp_config,
            use_megatron_fsdp=args.use_megatron_fsdp,
            use_torch_fsdp2=args.use_torch_fsdp2,
            overlap_param_gather_with_optimizer_step=args.overlap_param_gather_with_optimizer_step,
            data_parallel_random_init=args.data_parallel_random_init,
        )
        if torch.distributed.get_rank() in [0]:
            print(f"model rank[{torch.distributed.get_rank()}]: {model}")
    else:
        model = get_model(model_provider_func, model_type, wrap_with_ddp=wrap_with_ddp)
    unwrapped_model = unwrap_model(model)

    one_logger and one_logger.log_metrics({"app_build_optimzer_start_time": one_logger_utils.get_timestamp_in_ms()})
    if skip_optimizer:
        optimizer, opt_param_scheduler = None, None
        # In RL inference-only mode, train_iters must still be set despite having no optimizer.
        if args.perform_rl_step:
            update_train_iters(args)
    else:
        config, config_overrides = get_megatron_optimizer_config(args)
        config.timers = timers
        if getattr(args, "use_mup", False):
            model_config_source = (
                unwrapped_model[0] if isinstance(unwrapped_model, list) else unwrapped_model
            )
            model_config = get_model_config(model_config_source)
            mup_overrides = get_mup_config_overrides(
                config=config,
                mup_width_mult=model_config.mup_width_mult,
                optimizer_type=config.optimizer,
            )
            if mup_overrides:
                config_overrides = {**(config_overrides or {}), **mup_overrides}

        optimizer = get_megatron_optimizer(
            config,
            model,
            config_overrides=config_overrides,
            use_gloo_process_groups=args.use_gloo_process_groups,
            dump_param_to_param_group_map=args.dump_param_to_param_group_map,
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
    ) and not args.moe_use_upcycling and not args.use_bridge:
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

    # Validate that the world size can accommodate the current batch size.
    # This catches the case where GPUs were scaled up mid-training but the
    # current position in the batch size schedule yields a batch size that
    # is too small for the number of data-parallel replicas.
    num_microbatches = get_num_microbatches()
    current_global_batch_size = get_current_global_batch_size()
    data_parallel_size = mpu.get_data_parallel_world_size()
    assert num_microbatches is not None and num_microbatches >= 1, (
        f'current global batch size ({current_global_batch_size}) is too small for '
        f'micro_batch_size ({args.micro_batch_size}) * data_parallel_size ({data_parallel_size}) = '
        f'{args.micro_batch_size * data_parallel_size}. The world size cannot accommodate the '
        f'batch size. This can happen when resuming with more GPUs than the current batch size '
        f'schedule entry supports.'
    )

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


def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, forward_backward_func, iteration=None):
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
    save_params_in_this_iteration = (args.save_params_interval is not None and
                                     (iteration + 1) % args.save_params_interval == 0)
    save_activations_in_this_iteration = (args.save_activations_interval is not None and
                                          (iteration + 1) % args.save_activations_interval == 0)
    save_tpe_in_this_iteration = (args.save_tokens_per_expert_interval is not None and
                                  (iteration + 1) % args.save_tokens_per_expert_interval == 0)
    save_wgrads_in_this_iteration = (args.save_wgrads_interval is not None and
                                     (iteration + 1) % args.save_wgrads_interval == 0)
    save_dgrads_in_this_iteration = (args.save_dgrads_interval is not None and
                                     (iteration + 1) % args.save_dgrads_interval == 0)
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
            # If saving main_grads in this iteration, then all-reduce instead of reduce-scatter.
            model_chunk.force_all_reduce = save_wgrads_in_this_iteration
        optimizer.zero_grad()

        if has_nvidia_modelopt:
            # [ModelOpt]: Pipeline-parallel Distillation stacks student and teacher tensors
            adjust_tensor_shapes_fn = get_tensor_shapes_adjust_fn_for_distillation(
                model,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
            )
        else:
            adjust_tensor_shapes_fn = None

        # For the mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
        # we need to call the _copy_main_params_to_param_buffer() after the grad buffer
        # is zeroed by zero_grad_buffer() because param and grad buffer are shared.
        #
        # However, we should skip this on the first iteration when forward_pre_hook is disabled,
        # because:
        # 1. The first iteration's params are already in param.data (from init or checkpoint).
        # 2. Without forward_pre_hook, finish_param_sync() won't be called to zero the grad buffer,
        #    so the main grads will be polluted by the main params.
        #
        # Exception: when a full-iteration CUDA graph has been captured, the all-gather
        # and subsequent param_data zero are baked into the graph and replay
        # unconditionally. We must populate param_data so the replayed AG gathers
        # correct weights, even when forward pre-hooks are disabled.
        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            # Check if forward_pre_hook is enabled by checking if hooks are registered.
            forward_pre_hook_enabled = len(model[0].remove_forward_pre_hook_handles) > 0
            full_cg_captured = FullCudaGraphWrapper.cuda_graph.get("training") is not None
            if forward_pre_hook_enabled or full_cg_captured:
                for optim_instance in optimizer.chained_optimizers:
                    if isinstance(optim_instance, DistributedOptimizer):
                        optim_instance._copy_main_params_to_param_buffer()

        # Forward pass.
        if save_activations_in_this_iteration:
            enable_activation_logging(model, args.save)
        if save_tpe_in_this_iteration:
            enable_tokens_per_expert_logging(model, args.save)
        if save_dgrads_in_this_iteration:
            enable_dgrad_logging(model, args.save)
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
            force_all_reduce=save_wgrads_in_this_iteration,
        )
        if save_activations_in_this_iteration:
            save_activations(iteration + 1)
            disable_activation_logging()
        if save_tpe_in_this_iteration:
            save_tokens_per_expert(iteration + 1)
            disable_tokens_per_expert_logging()
        if save_dgrads_in_this_iteration:
            save_dgrads(iteration + 1)
            disable_dgrad_logging()

        # Reset force_all_reduce field.
        for model_chunk in model:
            model_chunk.force_all_reduce = False

    def _save_state_dict(attr_name, label):
        # Collect state_dict of the given attribute for each parameter.
        state_dict = defaultdict(dict)
        for model_chunk_id, model_chunk in enumerate(model):
            model_chunk_name = f"model_chunk{model_chunk_id}"
            unwrapped_model_chunk = unwrap_model(model_chunk)
            for param_name, param in unwrapped_model_chunk.named_parameters():
                if getattr(param, attr_name, None) is not None:
                    tensor_on_cpu = getattr(param, attr_name).cpu()
                    state_dict[model_chunk_name][param_name] = tensor_on_cpu

        # iteration is 0-indexed, move to 1-indexed for checkpoint name and logging.
        save_grads(args.save, state_dict, iteration + 1, label)

    # Checkpoint wgrads with parameter names.
    if save_wgrads_in_this_iteration:
        _save_state_dict(attr_name="main_grad", label="wgrads")

    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None, 0

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

    # get max attention logit for logging and run clip_qk()
    # Part of MuonClip Optimizer step
    log_max_attention_logit = 0
    if args.qk_clip or args.log_max_attention_logit:
        log_max_attention_logit = clip_qk(model, log_max_only=not args.qk_clip)

    timers('optimizer').stop()

    # Checkpoint params with parameter names.
    if save_params_in_this_iteration:
        _save_state_dict(attr_name="data", label="params")

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

        results = (
            loss_reduced,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
            log_max_attention_logit,
        )

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

            if args.all_reduce_time:
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
            return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad, log_max_attention_logit, args.params_all_reduce_time

    return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad, log_max_attention_logit


def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate: float | None,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
    max_attention_logit,
    pg_collection=None,
    is_first_iteration=False,
    seqlen_squared_sum_in_batch: float | None = None,
    total_real_tokens_in_batch: float | None = None,
):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()
    energy_monitor = get_energy_monitor()

    # On first iteration, log stats but don't reset accumulators so normal interval stats remain accurate.
    should_reset = not is_first_iteration

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
    timers_to_log = []
    if args.timing_log_level >= 1:
        timers_to_log.extend([
            'forward-backward',
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
        ])
    if args.timing_log_level >= 2:
        timers_to_log.extend([
            'batch-generator',
            'forward-compute',
            'backward-compute',
            'forward-recv',
            'forward-send',
            'backward-recv',
            'backward-send',
            'forward-send-forward-recv',
            'forward-send-backward-recv',
            'backward-send-forward-recv',
            'backward-send-backward-recv',
            'forward-backward-send-forward-backward-recv',
        ])
    # Add timers from RL loop if needed.
    if getattr(args, 'perform_rl_step', False):
        timers_to_log.extend(RL_LOGGABLE_TIMER_NAMES)

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * get_num_microbatches()

    # Track app tag & app tag ID
    one_logger_utils.track_app_tag(batch_size, args.world_size, args.seq_length)

    total_iterations = total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]

    # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
    learning_rate: float | None = reduce_max_stat_across_model_parallel_group(learning_rate)
    # Tensorboard values.
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log({'samples vs steps': args.consumed_train_samples}, iteration)
        if learning_rate is not None:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'learning-rate': learning_rate}, iteration)
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
        if args.perform_rl_step:
            grpo_collection_iteration = iteration // (args.grpo_iterations * ( ( args.grpo_samples_per_iteration )// args.global_batch_size ))
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
        if args.log_max_attention_logit:
            writer.add_scalar('max_attention_logit', max_attention_logit, iteration)
            if wandb_writer:
                wandb_writer.log({'max_attention_logit': max_attention_logit}, iteration)

    # Log MoE metrics.
    moe_log_string = ""
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

        if args.is_hybrid_model:
            from operator import itemgetter

            from megatron.core.ssm.mamba_hybrid_layer_allocation import (
                Symbols,
                get_hybrid_layer_counts,
            )
            layers = itemgetter(Symbols.MOE)(get_hybrid_layer_counts(args.hybrid_layer_pattern))
        else:
            layers = args.num_layers

        moe_log_string = get_moe_metrics_tracker().report(
            loss_scale=moe_loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            per_layer_logging=args.moe_per_layer_logging,
            force_initialize=True,
            track_names=track_names,
            num_layers=layers,
            moe_layer_freq=args.moe_layer_freq,
            mtp_num_layers=args.mtp_num_layers,
            pg_collection=pg_collection,
            total_loss_dict=total_loss_dict,
        )

    # Log MTP metrics.
    if args.mtp_num_layers is not None:
        mtp_loss_scale = 1 / get_num_microbatches()
        MTPLossLoggingHelper.track_mtp_metrics(
            mtp_loss_scale, iteration, writer, wandb_writer, total_loss_dict
        )

    # Track sparse attention indexer loss.
    if args.dsa_indexer_loss_coeff is not None and args.dsa_indexer_loss_coeff > 0:
        indexer_loss_scale = 1 / get_num_microbatches()
        DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=indexer_loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
        )

    # Dump memory snapshot and print metrics to stdout.
    if iteration % args.log_interval == 0 or is_first_iteration:
        if args.record_memory_history and (is_rank0() or torch.distributed.get_backend() == 'fake'):
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            with open(args.memory_snapshot_path, 'wb') as f:
                dump(snapshot, f)

        elapsed_time = timers('interval-time').elapsed(barrier=True, reset=should_reset)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(
            args,
            batch_size,
            seqlen_squared_sum_in_batch=seqlen_squared_sum_in_batch,
            total_real_tokens_in_batch=total_real_tokens_in_batch,
        ) / (
            elapsed_time_per_iteration * 10**12 * args.world_size
        )

        one_logger_utils.track_e2e_metrics(args.log_throughput, throughput)

        # We log to stdout after the first iteration (controlled by `is_first_iteration`)
        # to document initialization overhead. Log statistics to TensorBoard and
        # WandB according to the regular schedule.
        if args.log_timers_to_tensorboard and not is_first_iteration:
            if writer:
                writer.add_scalar('iteration-time', elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration}, iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}]"
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
        if learning_rate is not None:
            log_string += f' learning rate: {learning_rate:.6E} |'
        log_string += f' global batch size: {batch_size:5d} |'
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(
                    max(1, total_loss_dict[advanced_iters_key])
                )
                if avg >= 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                if should_reset:
                    total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        if args.num_experts is not None and moe_log_string:
            log_string += moe_log_string
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

        # RL token throughput metrics.
        if args.perform_rl_step:
            log_string += rl_utils.log_rl_throughput_metrics(
                args, batch_size, elapsed_time_per_iteration, iteration, wandb_writer,
            )

        if should_reset:
            total_loss_dict[advanced_iters_key] = 0
            total_loss_dict[skipped_iters_key] = 0
            total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        reported_memory_in_this_iteration = False
        if report_memory_flag:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory(f'(after {iteration} iterations)')
            reported_memory_in_this_iteration = True
            loaded_iteration = max(get_loaded_iteration() or 0, 0)
            if iteration > (loaded_iteration + 1):
                # Make sure the memory after the second iteration is reported to include optimizer state memory.
                report_memory_flag = False
        if args.log_memory_interval is not None and iteration % args.log_memory_interval == 0 and \
            not reported_memory_in_this_iteration:
            report_memory(f'(after {iteration} iterations)')
        # Write timers to wandb, don't reset the counts.
        if args.log_timers_to_tensorboard:
            timers.write(timers_to_log, writer, iteration, normalizer=args.log_interval, reset=False)
            timers.write(timers_to_log, wandb_writer, iteration, normalizer=args.log_interval, reset=False)
        # Log timers to stdout
        timers.log(timers_to_log, normalizer=args.log_interval, reset=should_reset)

    return report_memory_flag, log_string


def checkpoint_and_decide_exit(
    model,
    optimizer,
    opt_param_scheduler,
    iteration,
    num_floating_point_operations_so_far,
    checkpointing_context,
    train_data_iterator,
    iter_log,
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
    if (
        args.exit_interval
        and iteration % args.exit_interval == 0
    ) or (
        args.phase_transition_iterations
        and iteration in args.phase_transition_iterations
    ):
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
    inference_model=None,
):
    """Training function: run train_step desired number of times, run validation, checkpoint."""
    args = get_args()
    timers = get_timers()

    fault_injector_kwargs = {}
    for f in dataclasses.fields(FaultInjectorConfig):
        if hasattr(args, f.name):
            fault_injector_kwargs[f.name] = getattr(args, f.name)
    fault_injector_config = FaultInjectorConfig(**fault_injector_kwargs)

    _maybe_raise_workload_exception = None
    if (
        fault_injector_config.fault_injector_ranks is not None
        or fault_injector_config.fault_injector_num_ranks is not None
    ):
        from megatron.core.fault_injector import (
            maybe_raise_workload_exception as _maybe_raise_workload_exception,
        )
        from megatron.core.fault_injector import (
            setup_fault_injection,
            should_setup_fault_injection_at_iteration,
            should_setup_fault_injection_at_start,
        )

        if should_setup_fault_injection_at_start(fault_injector_config):
            setup_fault_injection(fault_injector_config)

    if args.perform_rl_step:
        assert has_rl_utils, "RL cannot run without the megatron.rl package"

    # Additional variable initialization for RL training
    if args.perform_rl_step:
        if args.skip_train:
            # In inference-only mode, use current weights as reference.
            print_rank_0("> RL inference-only: using current weights as reference.")
            ref_state_dict = {
                k: (v.cpu() if v is not None else v) for k, v in model[0].state_dict().items()
            }
        else:
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
    if args.perform_rl_step:
        print_rank_0("> Reinitializing microbatch calculator for GRPO training...")
        from megatron.core.num_microbatches_calculator import (
            destroy_num_microbatches_calculator,
            init_num_microbatches_calculator,
        )

        # First destroy the existing calculator
        destroy_num_microbatches_calculator()
        # Then initialize with the correct perform_rl_step=True context
        init_num_microbatches_calculator(
            rank=args.rank,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            data_parallel_size=mpu.get_data_parallel_world_size(),
            decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
            step_batch_size_schedule=args.step_batch_size_schedule,
            seq_length=args.seq_length,
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

    if args.hybrid_context_parallel:
        train_data_iterator = iter(HybridCPDataLoaderWrapper(train_data_iterator, config))

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

    model_pg_collection = get_attr_wrapped_model(model[0], "pg_collection")

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
    config.grad_scale_func = optimizer.scale_loss if optimizer is not None else None
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

    # GPU sniff test at start of training.
    if args.gpu_sniff_test_interval is not None:
        _run_gpu_sniff_test('before training')

    report_memory_flag = True
    pre_hook_enabled = False
    should_exit = False
    exit_code = 0
    is_first_iteration = True

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
    if args.cuda_graph_impl == "full_iteration":
        forward_backward_func = FullCudaGraphWrapper(
            forward_backward_func,
            cuda_graph_warmup_steps=args.cuda_graph_warmup_steps,
            use_single_mempool=config.cuda_graph_use_single_mempool,
        )
    # Wrap forward_backward_func for overflow handling with moe_expert_rank_capacity_factor
    if args.moe_expert_rank_capacity_factor is not None:
        copy_main_params = args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather
        forward_backward_func = PagedStashRunner(
            config,
            copy_main_params,
            model,
            optimizer,
            forward_backward_func,
        )
    if args.optimizer_cuda_graph:
        optimizer.step = OptimizerCudaGraphWrapper(
            optimizer.step,
            cuda_graph_warmup_steps=args.cuda_graph_warmup_steps,
            use_single_mempool=config.cuda_graph_use_single_mempool,
        )

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
    nsys_nvtx_context = None # reference to context for nsys profiling, so it can be cleaned up
    if (
        args.profile
        and (len(args.profile_ranks) == 0 or
             torch.distributed.get_rank() in args.profile_ranks)
        and args.use_pytorch_profiler
    ):
        if args.pytorch_profiler_collect_chakra:
            et_dir = Path(f"{args.tensorboard_dir}/../chakra")
            et_dir.mkdir(parents=True, exist_ok=True)
            et = torch.profiler.ExecutionTraceObserver().register_callback(f"{et_dir}/rank-{torch.distributed.get_rank()}.json.gz")
        else:
            et = None

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
            record_shapes=args.pytorch_profiler_collect_shapes,
            with_stack=args.pytorch_profiler_collect_callstack,
            execution_trace_observer=et,
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
        if (args.profile
            and (len(args.profile_ranks) == 0 or
                 torch.distributed.get_rank() in args.profile_ranks)):
            # Enable NVTX range when profiling starts and nvtx_ranges is set.
            if iteration == args.profile_step_start and args.nvtx_ranges:
                configure_nvtx_profiling(True)
            if args.use_pytorch_profiler:
                prof.step()
            elif args.use_hip_profiler:
                if iteration == args.profile_step_start: roctracer.roctracer_start()
                if iteration == args.profile_step_end: roctracer.roctracer_stop()
            elif iteration == args.profile_step_start:
                torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
                nsys_nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=args.record_shapes)
                nsys_nvtx_context.__enter__()

        ft_integration.on_checkpointing_start()
        maybe_finalize_async_save(blocking=False)
        ft_integration.on_checkpointing_end(is_async_finalization=True)
        # Update the timeout for all process groups after initialization
        # We update the timeout after the first successful iteration,
        # which takes longer than others usually
        if args.distributed_timeout_seconds_after_init is not None and iteration == start_iteration+1:
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
                    f"Number of microbatches should not decrease; "
                    f"going from {num_microbatches} to {get_num_microbatches()}"
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
            and not cuda_graph_helper.capture_finished()
            and iteration - start_iteration == args.cuda_graph_warmup_steps
        ):
            if args.cuda_graph_warmup_steps > 0 and should_disable_forward_pre_hook(args):
                disable_forward_pre_hook(model, param_sync=False)
            cuda_graph_helper.create_cudagraphs()
            if args.cuda_graph_warmup_steps > 0 and should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                cuda_graph_helper.cuda_graph_set_manual_hooks()

        # Completely skip iteration if needed.
        if (iteration + 1) in args.iterations_to_skip:
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

        if args.perform_rl_step:
            if optimizer is None:
                # Release stale CUDA cached memory before inference.
                torch.cuda.empty_cache()
            with torch.no_grad():
                train_data_iterator = rl_utils.get_grpo_data_iterator(
                    model, inference_model, optimizer, iteration, ref_state_dict,
                    grpo_iterations=args.grpo_iterations,
                    grpo_prompts_per_step=args.grpo_prompts_per_step,
                    grpo_group_size=args.grpo_group_size,
                    global_batch_size=args.global_batch_size,
                    sequence_packing=args.rl_use_sequence_packing,
                    buffered_rollouts=buffered_rollouts,
                    is_correction=args.rl_inference_logprobs_is_correction,
                    optimizer_is_on_cpu=args.rl_offload_optimizer_during_inference,
                )
                # Buffered rollouts are used as a state container for setups when
                # we use previously-generated data for an update.
                buffered_rollouts = train_data_iterator

        if args.skip_train:
            # RL inference-only mode: skip gradient updates, just collect rollouts.
            loss_dict = {}
            skipped_iter = 0
            should_checkpoint = False
            should_exit = False
            exit_code = 0
            grad_norm = 0.0
            num_zeros_in_grad = 0
            max_attention_logit = None
        else:
            ft_integration.on_training_step_start()
            if args.enable_dynamic_grad_comp:
                if args.find_rank_upper_limit:
                    loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad, max_attention_logit = \
                        train_step(forward_step_func,
                                   train_data_iterator,
                                   model,
                                   optimizer,
                                   opt_param_scheduler,
                                   config,
                                   forward_backward_func,
                                   iteration=iteration)
                else:
                    args.all_reduce_time = True
                    loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad, max_attention_logit, params_all_reduce_time = \
                        train_step(forward_step_func,
                                   train_data_iterator,
                                   model,
                                   optimizer,
                                   opt_param_scheduler,
                                   config,
                                   forward_backward_func,
                                   iteration=iteration)
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
                    max_attention_logit,
                ) = train_step(
                    forward_step_func, train_data_iterator, model, optimizer, opt_param_scheduler, config, forward_backward_func, iteration=iteration
                )
            ft_integration.on_training_step_end()
            if _maybe_raise_workload_exception is not None and iteration != start_iteration:
                _maybe_raise_workload_exception()
            # Fault delay timing can start at the end of iteration N. Self-firing faults
            # (signals, GIL, GPU) may then manifest in iteration N or N+1 depending on the
            # configured delay; workload-exception faults manifest on a later poll.
            if _maybe_raise_workload_exception is not None and should_setup_fault_injection_at_iteration(
                fault_injector_config, iteration
            ):
                setup_fault_injection(fault_injector_config)
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
                        and args.cuda_graph_warmup_steps == 0
                    ):
                        assert (
                            cuda_graph_helper.capture_finished()
                        ), "CUDA Graph capture should have been finished."
                        cuda_graph_helper.cuda_graph_set_manual_hooks()

        iteration += 1

        # If requested, manually register FSDP communication buffers after a short warmup.
        if (
            getattr(args, "fsdp_manual_registration", False)
            and getattr(args, "nccl_ub", False)
            and getattr(args, "use_megatron_fsdp", False)
            and iteration ==  start_iteration + 1
        ):
            for model_chunk in model:
                if isinstance(model_chunk, megatron_FSDP) and getattr(
                    model_chunk.ddp_config, "fsdp_manual_registration", False
                ):
                    param_and_grad_buffer = getattr(model_chunk, "param_and_grad_buffer", None)
                    if param_and_grad_buffer is not None:
                        param_and_grad_buffer.manual_buffer_registration()

        if args.perform_rl_step and args.rl_use_sequence_packing:
            iteration_sequences = rl_utils.get_iteration_sequence_count(args)
            # Track bins separately for packed mode
            bin_count = (
                mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
            )
            args.consumed_train_bins += bin_count
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
        # Drain the per-iteration packed-sequence stats so the FLOPs computation
        # reflects THD per-chunk causal attention AND excludes padding tokens
        # from token-linear work. Returns ``(None, None)`` for unpacked BSHD
        # runs (no collective issued), letting ``num_floating_point_operations``
        # fall back to its closed-form defaults.
        total_real_tokens_in_batch, seqlen_squared_sum_in_batch = (
            consume_seqlen_stats_in_iteration()
        )
        num_floating_point_operations_in_batch = num_floating_point_operations(
            args,
            batch_size,
            seqlen_squared_sum_in_batch=seqlen_squared_sum_in_batch,
            total_real_tokens_in_batch=total_real_tokens_in_batch,
        )
        num_floating_point_operations_so_far += num_floating_point_operations_in_batch
        num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

        # Logging.
        if optimizer is not None and not optimizer.is_stub_optimizer:
            loss_scale = optimizer.get_loss_scale().item()
        else:
            loss_scale = 1.0
        params_norm = None

        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        if optimizer is not None:
            learning_rate = get_canonical_lr_for_logging(optimizer.param_groups)
        else:
            learning_rate = None
        report_memory_flag, iter_log = training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
            max_attention_logit,
            pg_collection=model_pg_collection,
            is_first_iteration=is_first_iteration,
            seqlen_squared_sum_in_batch=seqlen_squared_sum_in_batch,
            total_real_tokens_in_batch=total_real_tokens_in_batch,
        )
        is_first_iteration = False

        # Evaluation.
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid \
                and (args.start_eval_at_iter is None or iteration >= args.start_eval_at_iter):
            if args.log_energy:
                energy_monitor.pause()
            timers('interval-time').stop()
            if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
                # disable_forward_pre_hook(param_sync=True) below force-syncs params for eval.
                # Copy the main params to param buffer before the forced AllGather.
                for model_chunk in model:
                    model_chunk.zero_grad_buffer()
                for optim_instance in optimizer.chained_optimizers:
                    if isinstance(optim_instance, DistributedOptimizer):
                        optim_instance._copy_main_params_to_param_buffer()
            if should_disable_forward_pre_hook(args):
                disable_forward_pre_hook(model)
                pre_hook_enabled = False
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f'iteration {iteration}'
            timers('eval-time', log_level=0).start(barrier=True)
            if args.perform_rl_step:
                rl_eval_model = model
                rl_training_model = None
                # If separate inference and training models, swap training weights
                # back to the inference model for RL evaluation.
                if inference_model is not None:
                    inf_core = unwrap_model(inference_model[0])
                    rl_utils._maybe_prefetch_separate_inference_model_weights(
                        inf_core, to_cpu=False
                    )
                    swap_model_weights(model, inference_model, args.refit_method)
                    rl_eval_model = inference_model
                    rl_training_model = model
                rl_utils.evaluate_and_print_results_rl(
                    valid_data_iterator,
                    rl_eval_model,
                    optimizer,
                    iteration,
                    write_to_tensorboard=True,
                    training_model=rl_training_model,
                )
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
            if args.num_experts is not None:
                get_moe_metrics_tracker().clear()

        # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
        # Some of these only happen at specific iterations. Capture updated FLOPs accumulator
        # (it is reset inside the callback after logging).
        num_floating_point_operations_since_last_log_event = post_training_step_callbacks(
            model,
            optimizer,
            opt_param_scheduler,
            iteration,
            prof,
            num_floating_point_operations_since_last_log_event,
            nsys_nvtx_context,
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

    # Destroy CUDA Graphs.
    if args.cuda_graph_impl == "transformer_engine" and cuda_graph_helper.graphs_created():
        cuda_graph_helper.delete_cuda_graphs()

    # Call OptimizerCudaGraph destructor to destroy optimizer CUDA graph
    if args.optimizer_cuda_graph:
        del optimizer.step

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

    if args.log_energy:
        energy_monitor.lap()
        total_energy = energy_monitor.get_total()
        print_rank_0(f"Total training energy (GPU): {total_energy / 1e6:.3f} MJ")
        energy_monitor.shutdown()

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        # Deregister NCCL user-buffer memory pools before exit.
        # Without this, ProcessGroupNCCL's destructor calls abort() which uses
        # ncclCommDeregister on handles created by ncclCommWindowRegister,
        # causing "NCCL WARN Deregister: Could not find handle" and a crash.
        torch.distributed.barrier()
        for model_module in model:
            if isinstance(model_module, DDP):
                for buf in model_module.buffers + model_module.expert_parallel_buffers:
                    if getattr(buf, 'nccl_mem_pool', None) is not None:
                        nccl_allocator.deregister_mem_pool(buf.nccl_mem_pool, buf.data_parallel_group)
        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.finish()
        ft_integration.shutdown()
        one_logger_utils.finish()
        if args.perform_rl_step:
            rl_utils.rl_inference_interface_shutdown()
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

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    # Disable result validation during evaluation
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.eval_global_batch_size
    eval_micro_batch_size = args.eval_micro_batch_size
    eval_num_microbatches = eval_batch_size // (eval_micro_batch_size * args.data_parallel_size)
    forward_backward_func = get_forward_backward_func()
    if args.cuda_graph_impl == "full_iteration":
        forward_backward_func = FullCudaGraphWrapper(
            forward_backward_func,
            cuda_graph_warmup_steps=args.cuda_graph_warmup_steps,
            use_single_mempool=config.cuda_graph_use_single_mempool,
        )
    # Wrap forward_backward_func for overflow handling with moe_expert_rank_capacity_factor
    if args.moe_expert_rank_capacity_factor is not None:
        copy_main_params = args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather
        forward_backward_func = PagedStashRunner(
            config,
            copy_main_params,
            model,
            None,
            forward_backward_func,
        )

    if has_nvidia_modelopt:
        # [ModelOpt]: Pipeline-parallel Distillation stacks student and teacher tensors
        adjust_tensor_shapes_fn = get_tensor_shapes_adjust_fn_for_distillation(
            model,
            seq_length=args.seq_length,
            micro_batch_size=eval_micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
        )
    else:
        adjust_tensor_shapes_fn = None

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
                micro_batch_size=eval_micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
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
                        total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float, device='cuda')
                    val = [x[key].view(-1) for x in loss_dicts]

                    if val[0].numel() == 2:
                        if args.sft:
                            # normalize over micro batch instead of global
                            val = torch.vstack(val)
                            val = val[:, 0] / val[:, 1].clamp(min=1)
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
                        else :
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
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=eval_micro_batch_size,
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

        # Export the same iteration in HuggingFace format next to the Megatron
        # checkpoint. Skipped for non-persistent checkpoints, which are local
        # scratch state for fast restart rather than a published checkpoint.
        if getattr(args, "save_hf_weights", False) and not non_persistent_ckpt:
            save_hf_checkpoint(iteration, model)

    return wrapper
