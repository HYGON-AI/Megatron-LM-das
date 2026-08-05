# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from megatron.core import parallel_state, mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
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

def split_data_cp_rank(val: torch.Tensor, cp_size: int, seq_dim: int, cp_rank: int = None):
    assert cp_size > 1
    assert 0 == val.shape[seq_dim] % (2 * cp_size), f'{val.shape=} {cp_size=}'
    if cp_rank is None:
        cp_rank = parallel_state.get_context_parallel_rank()
    if val is None:
        return val

    val = val.view(
        *val.shape[0:seq_dim],
        2 * cp_size,
        val.shape[seq_dim] // (2 * cp_size),
        *val.shape[(seq_dim + 1):],
    )

    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
    val = val.index_select(seq_dim, index)
    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])

    return val

def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch.

    返回 9 元组，最后 4 个为视觉相关字段：
      qwen2vl / qwen2.5vl: pixel_values, image_grid_thw, image_input_mask, None, None
      qwen3vl:             pixel_values, image_grid_thw, image_input_mask, images_padded, cp_img_num
    """
    args = get_args()
    imgs, tokens, labels, loss_mask, attention_mask, position_ids = None, None, None, None, None, None
    is_qwen3vl = args.model_arch == "qwen3vl"
    is_gemma3vl = args.model_arch == "gemma3vl"
    is_qwen_vl = not is_gemma3vl  # Qwen 系列含 mRoPE, 需要 collator 计算的 position_ids

    data = next(data_iterator)
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.is_cpu:
            data[k] = v.cuda(non_blocking=True)

    # ── 公共字段 ──
    keys = ["image_input_mask", "has_image"]
    data_b = tensor_parallel.broadcast_data(keys, data, torch.bool)
    image_input_mask = data_b["image_input_mask"].bool().contiguous()
    has_image = data_b["has_image"].bool()[0].item()

    # ── int64 字段 ──
    # GemmaVL: position_ids 由 LM 内部计算, collator 不产出
    keys = ["input_ids", "labels"]
    if is_qwen_vl:
        keys.append("position_ids")
    if has_image and is_qwen_vl:
        keys.append("image_grid_thw")
        if is_qwen3vl:
            # qwen3vl 内部做 CP split，需要 images_padded / cp_img_num
            keys.extend(["images_padded", "cp_img_num"])
    data_b = tensor_parallel.broadcast_data(keys, data, torch.int64)
    tokens = data_b["input_ids"].long().contiguous()
    labels = data_b["labels"].long().contiguous()
    image_grid_thw = data_b.get("image_grid_thw", None)
    images_padded = None
    cp_img_num = None
    if has_image and is_qwen3vl:
        cp_img_num = data_b["cp_img_num"].long().tolist()
        images_padded = data_b["images_padded"].bool().tolist()
    if is_qwen_vl:
        position_ids = data_b["position_ids"].long().contiguous()

    keys = ["loss_mask"]
    if has_image:
        keys.append("pixel_values")
    data_b = tensor_parallel.broadcast_data(keys, data, torch.float32)
    if has_image:
        imgs = data_b["pixel_values"].float()
        if is_gemma3vl:
            # Gemma3VL: pixel_values 是 (B, C, H, W), 不需要 squeeze
            pass
        else:
            # QwenVL: pixel_values 是 (total_pixels, C), 去掉多余的 batch 维
            imgs = imgs.squeeze(0).contiguous()
        imgs = imgs.type(torch.bfloat16)
    loss_mask = data_b["loss_mask"].float().contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"

    if args.context_parallel_size > 1:
        labels = split_data_cp_rank(labels, args.context_parallel_size, 1)
        loss_mask = split_data_cp_rank(loss_mask, args.context_parallel_size, 1)
        assert attention_mask is None, "if attention_mask is not None, it should be split too"
    if has_image and image_grid_thw is not None:
        # image_grid_thw shape: [num_images_in_batch, 3]  (t, h, w in vision patch units)
        # 每张图的 vision tokens 数 = t*h*w；merger 之后语言侧看到的 = t*h*w / (spatial_merge_size^2)
        spatial = getattr(args, "spatial_merge_size", 2) or 1
        total_vision_tokens = int(image_grid_thw.prod(dim=-1).sum().item())
        total_llm_side_tokens = total_vision_tokens // (spatial * spatial)
        bs = tokens.size(0)
        # 我们的 flops wrapper 假设的是 \"per-sample image tokens\"，跟 batch 相乘得总数
        args._bridge_image_tokens_per_sample = total_vision_tokens // max(bs, 1)
    else:
        args._bridge_image_tokens_per_sample = 0
        
    return (
        tokens, labels, loss_mask, attention_mask, position_ids, imgs, image_grid_thw,
        image_input_mask, images_padded, cp_img_num
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
    is_qwen3vl = args.model_arch == "qwen3vl"
    is_gemma3vl = args.model_arch == "gemma3vl"

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        # tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage, microbatch_id=microbatch_id)
        (
            tokens, labels, loss_mask, attention_mask, position_ids, pixel_values, image_grid_thw,
            image_input_mask, images_padded, cp_img_num
        ) = get_batch(data_iterator)
        timers('batch-generator').stop()

    timers("model-forward-only", log_level=2).start()
    with stimer:
        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)

        model_kwargs = dict(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            loss_mask=loss_mask,
        )
        if is_gemma3vl:
            # Gemma3VL: 标准 RoPE (LM 内部计算 position_ids)，
            # 不需要 image_grid_thw / image_input_mask
            pass
        elif is_qwen3vl:
            # Qwen3VL: mRoPE + vision CP split, 需要全部视觉参数
            model_kwargs.update(
                image_grid_thw=image_grid_thw,
                image_input_mask=image_input_mask,
                images_padded=images_padded,
                cp_img_num=cp_img_num,
            )
        else:
            # Qwen2VL / Qwen2.5VL: mRoPE, 但不需要 image_input_mask
            model_kwargs["image_grid_thw"] = image_grid_thw
        output_tensor = model(**model_kwargs)
        # Gemma3VLModel.forward() 返回 (outputs, loss_mask) 元组，
        # loss_mask 已在模型内部经过 CP slice，需要替换 get_batch 的版本
        if is_gemma3vl:
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
