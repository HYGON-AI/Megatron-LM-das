# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Pretrain and SFT GPT."""

# Capture the true program start time BEFORE any heavy imports.
import time
_PROGRAM_START_TIME = time.time()

import json

# Suppress warnings on all ranks but rank 0.
import os
import warnings
rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import List, Optional, Tuple

import torch

from gpt_builders import gpt_builder
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks
from megatron.core.utils import StragglerDetector, get_pretrain_batch_on_this_cp_rank
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.utils import (
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import parse_and_validate_args, core_transformer_config_from_args
from megatron.training.argument_utils import pretrain_cfg_container_from_args

from model_provider import model_provider


from hcu_megatron import megatron_adaptor

from hcu_megatron.core.datasets.vlm_dataset import build_train_valid_test_data_iter
from hcu_megatron.core.datasets.vlm_args import add_vlm_extra_args, parse_dataset_config

stimer = StragglerDetector()

# Model families that use mRoPE and expect Bridge-side position_ids computation.
# Bridge's Qwen VL models (Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL) recompute 3D
# mRoPE position_ids inside forward on every PP stage — the dataset must not
# ship position_ids for them (would be redundant, or in the Qwen3 branch simply
# discarded by qwen3_vl_step which forces position_ids=None).
# GLM 4.1V / 4.5V bridge models follow the same mRoPE pattern.
_QWEN_VL_ARCHS = {"qwen2vl", "qwen2.5vl", "qwen3vl"}
_GLM_VL_ARCHS = {"glm4vl"}
_MROPE_VL_ARCHS = _QWEN_VL_ARCHS | _GLM_VL_ARCHS
# Model families whose Bridge forward performs its own CP split on the
# combined vision+language embeddings. Other Qwen VL variants do NOT split
# inside forward — running them with CP > 1 would be silently incorrect.
_ARCHS_WITH_INTERNAL_CP_SPLIT = {"qwen3vl"}
# Model families whose collator emits per-image CP-split metadata.
_ARCHS_WITH_CP_IMG_META = {"qwen3vl"}
# Model families whose forward needs image_input_mask (positions of <image> tokens).
_ARCHS_WITH_IMAGE_INPUT_MASK = {"qwen3vl"}


def _validate_cp_preconditions(args) -> None:
    """Sanity-check CP-related configuration at startup.

    - Bridge Qwen2-VL / Qwen2.5-VL forward does not split the language
      embeddings along CP, so CP > 1 would leave the LM working on the full
      sequence while ``get_batch`` slices only ``labels`` / ``loss_mask``.
      That mismatch corrupts the loss silently. Reject explicitly.
    - Our zigzag CP partition requires seq_length to be a multiple of
      2 * cp_size. Fail early with a friendly message rather than deep in
      the training loop.
    """
    cp = args.context_parallel_size
    if cp <= 1:
        return

    arch = args.model_arch
    if arch in _MROPE_VL_ARCHS and arch not in _ARCHS_WITH_INTERNAL_CP_SPLIT:
        raise ValueError(
            f"model_arch={arch!r} with context_parallel_size={cp} is not "
            f"supported: Bridge's forward for this family does not split "
            f"the embeddings along CP. Only {sorted(_ARCHS_WITH_INTERNAL_CP_SPLIT)} "
            f"support CP > 1 today. Reduce --context-parallel-size to 1."
        )

    if args.seq_length % (2 * cp) != 0:
        raise ValueError(
            f"--seq-length ({args.seq_length}) must be a multiple of "
            f"2 * context_parallel_size ({2 * cp}) to satisfy the zigzag "
            f"CP partition constraint."
        )


def _split_batch_for_cp(batch: dict, cp_size: int) -> dict:
    """Zigzag partition of ``labels`` and ``loss_mask`` along the seq dim.

    Delegates to megatron's ``get_pretrain_batch_on_this_cp_rank`` so hcu's
    CP behavior stays in lockstep with upstream Megatron / Bridge. We only
    partition labels and loss_mask because Bridge's Qwen3-VL forward expects
    the raw (unsliced) ``input_ids`` and does the embedding CP split internally.
    """
    if cp_size <= 1:
        return batch
    return get_pretrain_batch_on_this_cp_rank(
        batch,
        cp_group=mpu.get_context_parallel_group(),
    )


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a training batch.

    Returns a 10-tuple:
        (tokens, labels, loss_mask, attention_mask, position_ids,
         pixel_values, image_grid_thw, image_input_mask, images_padded, cp_img_num)

    Family-specific outputs:
        qwen2vl / qwen2.5vl : position_ids=None, image_input_mask=None,
                              images_padded=None, cp_img_num=None
        qwen3vl             : position_ids=None, plus full CP-split metadata
        gemma3vl            : position_ids=None, image_grid_thw=None,
                              image_input_mask=None, images_padded=None, cp_img_num=None

    ``position_ids`` is always None for VLM: every supported family computes it
    inside model.forward (see _QWEN_VL_ARCHS docstring; Gemma3-VL LM computes
    standard RoPE internally).

    ``has_image`` is a per-batch flag (True iff any sample in this batch carried
    an image). This matches the collator, which only populates pixel_values /
    image_grid_thw when at least one sample had an image.
    """
    args = get_args()
    arch = args.model_arch

    data = dict(next(data_iterator))
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.is_cpu:
            data[k] = v.cuda(non_blocking=True)

    # ── bool fields ──
    bool_keys = ["has_image"]
    if arch in _ARCHS_WITH_IMAGE_INPUT_MASK:
        bool_keys.append("image_input_mask")
    data_b = tensor_parallel.broadcast_data(bool_keys, data, torch.bool)
    has_image = data_b["has_image"].bool()[0].item()
    image_input_mask = (
        data_b["image_input_mask"].bool().contiguous()
        if arch in _ARCHS_WITH_IMAGE_INPUT_MASK else None
    )

    # ── int64 fields ──
    int_keys = ["input_ids", "labels"]
    if has_image and arch in _MROPE_VL_ARCHS:
        int_keys.append("image_grid_thw")
        if arch in _ARCHS_WITH_CP_IMG_META:
            int_keys.extend(["images_padded", "cp_img_num"])
    data_b = tensor_parallel.broadcast_data(int_keys, data, torch.int64)
    tokens = data_b["input_ids"].long().contiguous()
    labels = data_b["labels"].long().contiguous()
    image_grid_thw = data_b.get("image_grid_thw", None)
    images_padded = (
        data_b["images_padded"].bool().tolist()
        if has_image and arch in _ARCHS_WITH_CP_IMG_META else None
    )
    cp_img_num = (
        data_b["cp_img_num"].long().tolist()
        if has_image and arch in _ARCHS_WITH_CP_IMG_META else None
    )

    # ── float32 fields ──
    float_keys = ["loss_mask"]
    if has_image:
        float_keys.append("pixel_values")
    data_b = tensor_parallel.broadcast_data(float_keys, data, torch.float32)
    loss_mask = data_b["loss_mask"].float().contiguous()

    imgs = None
    if has_image:
        imgs = data_b["pixel_values"].float()
        # QwenVL / GlmVL pack pixel_values as (total_pixels, C) with an extra
        # batch dim from default_collate. Gemma3VL keeps standard (B, C, H, W).
        if arch in _MROPE_VL_ARCHS:
            imgs = imgs.squeeze(0).contiguous()
        imgs = imgs.type(torch.bfloat16)

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"

    attention_mask = None
    position_ids = None  # every supported family computes this inside forward.

    if args.context_parallel_size > 1:
        cp_batch = _split_batch_for_cp(
            {"labels": labels, "loss_mask": loss_mask},
            args.context_parallel_size,
        )
        labels = cp_batch["labels"]
        loss_mask = cp_batch["loss_mask"]
        assert attention_mask is None, "if attention_mask is not None, it should be CP-split too"

    # FLOPs accounting: report per-sample **vision-side** image tokens (pre-merge).
    # hcu_megatron/training/training.py:_vision_module_flops treats
    # image_tokens_per_sample as the number of vision tokens before the spatial
    # merger, then divides by spatial**2 internally for the projector cost. Do
    # not switch to LLM-side tokens without also updating that wrapper.
    # TODO(vision-flops): consolidate the pre-/post-merge convention in one
    # place and document it — right now the meaning lives implicitly in the
    # wrapper's arithmetic.
    if has_image and image_grid_thw is not None:
        total_vision_tokens = int(image_grid_thw.prod(dim=-1).sum().item())
        bs = tokens.size(0)
        args._bridge_image_tokens_per_sample = total_vision_tokens // max(bs, 1)
    else:
        args._bridge_image_tokens_per_sample = 0

    return (
        tokens, labels, loss_mask, attention_mask, position_ids, imgs, image_grid_thw,
        image_input_mask, images_padded, cp_img_num,
    )

def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    real_seqlen = torch.tensor(
        output_tensor.shape[-1] * args.context_parallel_size, dtype=torch.float
    )

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.stack([
        torch.sum(losses * loss_mask).view(1),
        loss_mask.sum().view(1)
    ])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )
        assert not loss.isinf().any(), (
            f"Rank {global_rank}: found Inf in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )
    bwd_loss = loss[0] / loss[1]

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    real_seqlen = torch.tensor(
        output_tensor.shape[-1] * args.context_parallel_size, dtype=torch.float
    )
    report = {'lm loss': averaged_loss, 'real_seqlen': real_seqlen}

    return bwd_loss, num_tokens, report


def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()
    arch = args.model_arch

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        (
            tokens, labels, loss_mask, attention_mask, position_ids, pixel_values, image_grid_thw,
            image_input_mask, images_padded, cp_img_num,
        ) = get_batch(data_iterator)
        timers('batch-generator').stop()

    timers("model-forward-only", log_level=2).start()
    with stimer:
        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask,
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)

        # position_ids is intentionally None for every supported family — Bridge
        # models recompute it inside forward. Passing it here would either be
        # redundant (Qwen2.5-VL) or silently discarded (Qwen3-VL / Qwen3.5-VL
        # via qwen3_vl_step's explicit override; Gemma3-VL LM handles its own
        # RoPE).
        model_kwargs = dict(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            loss_mask=loss_mask,
        )
        if arch == "gemma3vl":
            # Gemma3VL: standard RoPE (LM computes position_ids internally);
            # no image_grid_thw / image_input_mask needed.
            pass
        elif arch == "qwen3vl":
            # Qwen3VL: mRoPE + vision CP split; needs full vision metadata.
            model_kwargs.update(
                image_grid_thw=image_grid_thw,
                image_input_mask=image_input_mask,
                images_padded=images_padded,
                cp_img_num=cp_img_num,
            )
        elif arch in _MROPE_VL_ARCHS:
            # Qwen2VL / Qwen2.5VL / GLM4VL: mRoPE, but no image_input_mask.
            model_kwargs["image_grid_thw"] = image_grid_thw
        else:
            raise ValueError(f"unknown model_arch {arch!r}")

        output_tensor = model(**model_kwargs)
        # Gemma3VLModel.forward() returns (outputs, loss_mask); the returned
        # loss_mask is CP-sliced inside the model, so replace the get_batch one.
        if arch == "gemma3vl":
            output_tensor, loss_mask = output_tensor
    timers("model-forward-only").stop()

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    _validate_cp_preconditions(args)
    parse_dataset_config(args)
    tokenizer = build_tokenizer(args)
    print(f">>> tokenizer: {tokenizer} <<<")

    train_iter, valid_iter, test_iter = build_train_valid_test_data_iter(
        args,
        tokenizer,
        rank=torch.distributed.get_rank(),
        dp_rank=mpu.get_data_parallel_rank(),
        dp_size=mpu.get_data_parallel_world_size(),
    )
    print(
        f"> dp world size {mpu.get_data_parallel_world_size()} rank {mpu.get_data_parallel_rank()} "
        f"finished creating dataloader ..."
    )
    return train_iter, valid_iter, test_iter

def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks

if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=add_vlm_extra_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    full_config = pretrain_cfg_container_from_args(args)

    pretrain(full_config,
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
