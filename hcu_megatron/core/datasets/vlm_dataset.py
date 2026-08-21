# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""VLM dataset factory (public entry points).

External callers only need three functions from this module:

    - ``get_processor(args)``               — build an HF ``AutoProcessor``
    - ``build_train_valid_test_datasets``   — build the per-family Datasets
    - ``build_train_valid_test_data_iter``  — build DataLoaders + iterators

Family-specific behavior (dataset class, collator, tokenizer setup, HF-config
verification) lives on ``VLMFamily`` entries in ``VLM_REGISTRY``. Adding a new
VLM family means declaring a new entry, not touching this factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Type

import torch
from transformers import AutoProcessor


# HF logs a per-call warning when apply_chat_template sees kwargs the chat
# template's Jinja source doesn't mention. GLM 4.1V's template only references
# `add_generation_prompt`, so every dataset call trips the warning for
# tokenize/tools/etc. The rest of the codebase is written against Qwen-style
# templates that DO reference those vars, so the warning is noise here.
class _DropChatTemplateKwargsWarning(logging.Filter):
    _NEEDLE = "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict"

    def filter(self, record: logging.LogRecord) -> bool:
        return self._NEEDLE not in record.getMessage()


logging.getLogger("transformers.processing_utils").addFilter(_DropChatTemplateKwargsWarning())

from hcu_megatron.core.datasets.utils import get_iterator
from hcu_megatron.core.datasets.mm_dataset import (
    MultiModalDataset,
    convert_conversations,
    remove_bos,
)
from hcu_megatron.core.datasets.vlm_collator import (
    DataCollatorForGemmaVL,
    DataCollatorForQwenVL,
)
from hcu_megatron.core.datasets.vlm_gemma import GemmaVLDataset
from hcu_megatron.core.datasets.vlm_glm import GlmVLDataset
from hcu_megatron.core.datasets.vlm_images import (
    IMAGE_FACTOR,
    MAX_PIXELS,
    MAX_RATIO,
    MIN_PIXELS,
    fetch_images,
    pad_and_split_pixel_values,
    resize_image,
    smart_resize,
)
from hcu_megatron.core.datasets.vlm_qwen import QwenVLDataset
from hcu_megatron.core.datasets.vlm_tokenizer import (
    setup_gemma_tokenizer,
    setup_glm_tokenizer,
    setup_qwen_tokenizer,
)


# Preserve historical import surface — a few call sites import these symbols
# from ``vlm_dataset`` directly.
__all__ = [
    "IMAGE_FACTOR",
    "MIN_PIXELS",
    "MAX_PIXELS",
    "MAX_RATIO",
    "smart_resize",
    "resize_image",
    "pad_and_split_pixel_values",
    "fetch_images",
    "convert_conversations",
    "remove_bos",
    "MultiModalDataset",
    "QwenVLDataset",
    "GemmaVLDataset",
    "GlmVLDataset",
    "DataCollatorForQwenVL",
    "DataCollatorForGemmaVL",
    "get_processor",
    "build_train_valid_test_datasets",
    "build_train_valid_test_data_iter",
]


# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VLMFamily:
    """Everything that varies between VLM model families lives here.

    - ``dataset_cls``     — the ``MultiModalDataset`` subclass to instantiate
    - ``tokenizer_setup`` — injects family-specific ids/attributes on the tokenizer
    - ``build_collator``  — builds the batch collator given ``args`` + tokenizer
    - ``build_processor`` — optional overrides for AutoProcessor kwargs; falls
      back to a Qwen-style default
    - ``verify_hf_config`` — optional HF-config cross-check (Qwen3VL)
    """

    dataset_cls: Type[MultiModalDataset]
    tokenizer_setup: Callable
    build_collator: Callable
    build_processor: Callable
    verify_hf_config: Callable = None


def _build_qwen_processor(args):
    """Qwen family AutoProcessor kwargs."""
    if args.model_arch == "qwen3vl":
        min_pixels = args.min_pixels_num
        max_pixels = args.max_pixels_num
    else:
        min_pixels = args.min_pixels_num if args.min_pixels_num else MIN_PIXELS
        max_pixels = args.max_pixels_num if args.max_pixels_num else MAX_PIXELS
    init_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "token": None,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "use_fast": True,
    }
    processor = AutoProcessor.from_pretrained(args.processor_path, **init_kwargs)
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor


def _build_gemma_processor(args):
    """Gemma family AutoProcessor kwargs — no min/max pixels."""
    init_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "token": None,
        "use_fast": True,
    }
    return AutoProcessor.from_pretrained(args.processor_path, **init_kwargs)


def _build_glm_processor(args):
    """GLM 4.1V/4.5V AutoProcessor kwargs — Glm4vProcessor resizes internally
    per its own ``size`` config, so we don't pipe min/max pixels through here.
    """
    init_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "token": None,
        "use_fast": True,
    }
    processor = AutoProcessor.from_pretrained(args.processor_path, **init_kwargs)
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor


def _build_qwen_collator(args, tokenizer):
    hw_factor = args.context_parallel_size
    if args.sequence_parallel:
        hw_factor *= args.tensor_model_parallel_size
    # qwen3vl uses a different pixel_values alignment strategy; skip extra padding.
    if args.model_arch == "qwen3vl":
        hw_factor = 1

    if args.model_arch == "qwen3vl":
        # Sanity-check the tokenizer against the HF config so a mismatched
        # processor path or a stale vocab surfaces here, not deep in the model.
        from transformers import Qwen3VLConfig
        hf_config = Qwen3VLConfig.from_pretrained(args.processor_path)
        _verify_qwen3vl_config(tokenizer, hf_config)

    return DataCollatorForQwenVL(
        hw_factor=hw_factor,
        model_arch=args.model_arch,
        tokenizer=tokenizer,
        cp_size=args.context_parallel_size,
    )


def _build_gemma_collator(args, tokenizer):
    return DataCollatorForGemmaVL(tokenizer=tokenizer)


def _build_glm_collator(args, tokenizer):
    """GLM 4.1V/4.5V uses the Qwen collator (same pixel_values layout) but
    without Qwen3VL's CP image-split path (see ``_ARCHS_WITH_CP_IMG_SPLIT`` in
    vlm_collator.py). Config verification uses ``Glm4vConfig``.
    """
    hw_factor = args.context_parallel_size
    if args.sequence_parallel:
        hw_factor *= args.tensor_model_parallel_size

    from transformers import Glm4vConfig
    hf_config = Glm4vConfig.from_pretrained(args.processor_path)
    _verify_glm4v_config(tokenizer, hf_config)

    return DataCollatorForQwenVL(
        hw_factor=hw_factor,
        model_arch=args.model_arch,
        tokenizer=tokenizer,
        cp_size=args.context_parallel_size,
    )


def _verify_qwen3vl_config(tokenizer, hf_config):
    assert hf_config.image_token_id == tokenizer.image_token_id
    assert hf_config.video_token_id == tokenizer.video_token_id
    assert hf_config.vision_start_token_id == tokenizer.vision_start_token_id
    assert hf_config.vision_end_token_id == tokenizer.vision_end_token_id


def _verify_glm4v_config(tokenizer, hf_config):
    """GLM 4.1V/4.5V exposes image_token_id / video_token_id at the top level
    and uses image_start/end_token_id (not vision_*) for the wrappers.
    """
    assert hf_config.image_token_id == tokenizer.image_token_id, (
        f"image_token_id mismatch: hf={hf_config.image_token_id}, "
        f"tok={tokenizer.image_token_id}"
    )
    assert hf_config.video_token_id == tokenizer.video_token_id, (
        f"video_token_id mismatch: hf={hf_config.video_token_id}, "
        f"tok={tokenizer.video_token_id}"
    )
    assert hf_config.image_start_token_id == tokenizer.vision_start_token_id, (
        f"image_start_token_id mismatch: hf={hf_config.image_start_token_id}, "
        f"tok={tokenizer.vision_start_token_id}"
    )
    assert hf_config.image_end_token_id == tokenizer.vision_end_token_id, (
        f"image_end_token_id mismatch: hf={hf_config.image_end_token_id}, "
        f"tok={tokenizer.vision_end_token_id}"
    )


VLM_REGISTRY: dict[str, VLMFamily] = {
    "qwen2vl": VLMFamily(
        dataset_cls=QwenVLDataset,
        tokenizer_setup=setup_qwen_tokenizer,
        build_collator=_build_qwen_collator,
        build_processor=_build_qwen_processor,
    ),
    "qwen2.5vl": VLMFamily(
        dataset_cls=QwenVLDataset,
        tokenizer_setup=setup_qwen_tokenizer,
        build_collator=_build_qwen_collator,
        build_processor=_build_qwen_processor,
    ),
    "qwen3vl": VLMFamily(
        dataset_cls=QwenVLDataset,
        tokenizer_setup=setup_qwen_tokenizer,
        build_collator=_build_qwen_collator,
        build_processor=_build_qwen_processor,
    ),
    "gemma3vl": VLMFamily(
        dataset_cls=GemmaVLDataset,
        tokenizer_setup=setup_gemma_tokenizer,
        build_collator=_build_gemma_collator,
        build_processor=_build_gemma_processor,
    ),
    "glm4vl": VLMFamily(
        dataset_cls=GlmVLDataset,
        tokenizer_setup=setup_glm_tokenizer,
        build_collator=_build_glm_collator,
        build_processor=_build_glm_processor,
    ),
}


def _get_family(args) -> VLMFamily:
    model_arch = getattr(args, "model_arch", "qwen2vl")
    if model_arch not in VLM_REGISTRY:
        raise ValueError(
            f"Unknown model_arch {model_arch!r}. "
            f"Known families: {sorted(VLM_REGISTRY)}"
        )
    return VLM_REGISTRY[model_arch]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_processor(args):
    """Build the HF ``AutoProcessor`` for the current model family."""
    return _get_family(args).build_processor(args)


def build_train_valid_test_datasets(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    use_for_hf=False,
):
    """Build train / valid / test Datasets for the current model family."""
    family = _get_family(args)
    processor = get_processor(args)

    common_kwargs = dict(
        min_pixels_num=args.min_pixels_num,
        max_pixels_num=args.max_pixels_num,
        use_for_hf=use_for_hf,
        mask_history=args.mask_history,
        tokenizer=tokenizer,
        max_seq_len=args.seq_length,
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        num_workers=args.num_workers,
        seed=args.seed,
        top_domains_to_cut=args.vlm_top_domains_to_cut,
        processor=processor,
        tar_dir=args.tarfile_path,
        image_token_id=getattr(args, "image_token_id", None),
        moe_pad_with_random_token=getattr(args, "moe_pad_with_random_token", False),
    )

    train_ds = family.dataset_cls(
        path_likes=args.data_path,
        domain_probabilities=args.vlm_domain_probabilities,
        domain_names=args.vlm_train_data_domain_names,
        global_batch_size=args.global_batch_size,
        train=True,
        **common_kwargs,
    )

    eval_ds = None
    if args.vlm_eval_data_path is not None:
        eval_ds = family.dataset_cls(
            path_likes=args.vlm_eval_data_path,
            domain_probabilities=[1.0] * len(args.vlm_eval_data_path),
            domain_names=args.vlm_eval_data_domain_names,
            global_batch_size=args.global_batch_size,
            train=False,
            **common_kwargs,
        )

    return train_ds, eval_ds, None


def build_train_valid_test_data_iter(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    use_for_hf=False,
):
    """Build the DataLoader / iterator triple.

    Layout:
        1. run family-specific tokenizer setup
        2. build Datasets + Collator
        3. wrap in ``torch.utils.data.DataLoader``
        4. wrap in Megatron's rerun iterator (unless ``use_for_hf``)
    """
    family = _get_family(args)
    family.tokenizer_setup(tokenizer, args)

    train_ds, eval_ds, test_ds = build_train_valid_test_datasets(
        args, tokenizer, rank, dp_rank, dp_size, use_for_hf=use_for_hf,
    )
    collate_func = family.build_collator(args, tokenizer)

    batch_size = args.micro_batch_size
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_func,
        prefetch_factor=args.vlm_dataloader_prefetch_factor,
    )
    train_dl = torch.utils.data.DataLoader(train_ds, **loader_kwargs)
    eval_dl = torch.utils.data.DataLoader(eval_ds, **loader_kwargs) if eval_ds is not None else None
    test_dl = torch.utils.data.DataLoader(test_ds, **loader_kwargs) if test_ds is not None else None

    if use_for_hf:
        return train_dl, eval_dl, test_dl
    return get_iterator(train_dl), get_iterator(eval_dl), get_iterator(test_dl)
