# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Base ``MultiModalDataset`` shared by all VLM families.

Family datasets (``QwenVLDataset``, ``GemmaVLDataset``) override:

    * ``process_vision`` — turn PIL images into model-ready tensors
    * ``convert_example`` — full single-sample pipeline (text → tensors)
    * ``gen_label_mask``  — decide which spans are labels

The base class owns:

    * dataset iteration and image loading (``__iter__`` + validation)
    * a ``_pre_convert_hook`` for family-specific per-sample steps
      (e.g., Qwen3VL needs an extra resize before ``process_vision``)
    * ``_pad_truncate_shift`` — common ``max_seq_len + 1`` pad / truncate / shift
      logic

Input JSONL sample shape (SFT):

    {
        "conversations": [
            {"role": "user", "content": "text<image>"},
            {"role": "assistant", "content": "reply"}
        ],
        "images": [{"image_path": "img.png"}]     # optional
    }
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from types import SimpleNamespace
from typing import Optional

import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from hcu_megatron.core.datasets.indexed_jsonl_dataset import MultimodalIndexedJsonlDataset
from hcu_megatron.core.datasets.utils import random_pad_list
from hcu_megatron.core.datasets.vlm_images import (
    IMAGE_FACTOR,
    MAX_RATIO,
    fetch_images,
)


# ---------------------------------------------------------------------------
# Conversation preprocessing helpers
# ---------------------------------------------------------------------------
def convert_pattern(
    user_input: str,
    image_pattern: str = "<image>",
    video_pattern: str = "<video>",
):
    """Split ``user_input`` into text / image / video segments.

    Example::

        "look<image>reply"
        -> [
            {"type": "text",  "text": "look"},
            {"type": "image", "image": "0"},
            {"type": "text",  "text": "reply"},
        ]
    """
    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({"type": "text", "text": user_input[cur:start]})

        token_name = matched.string[start:end][1:-1]   # strip < >
        contents.append({"type": token_name, token_name: str(mm_idx[token_name])})

        cur = end
        mm_idx[token_name] += 1

    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur:]})

    return contents


def convert_conversations(conversations):
    """Apply ``convert_pattern`` to each turn's ``content``."""
    res = []
    for conversation in conversations:
        new_conversation = copy.deepcopy(conversation)
        new_conversation["content"] = convert_pattern(conversation["content"])
        res.append(new_conversation)
    return res


def remove_bos(text: str) -> str:
    """Strip a duplicate ``<bos>`` produced by nested apply_chat_template calls."""
    bos = "<bos>"
    assert text.startswith(bos)
    return text[len(bos):]


# ---------------------------------------------------------------------------
# Base VLM dataset
# ---------------------------------------------------------------------------
class MultiModalDataset(TorchIterableDataset):
    """Iterable base class for VLM datasets.

    Subclasses must implement ``process_vision`` and ``convert_example``.
    Optionally override ``_pre_convert_hook`` to inject family-specific
    per-sample preprocessing (e.g. Qwen3VL resize).
    """

    # ── image validation thresholds — subclasses may override ──
    image_min_size: int = IMAGE_FACTOR
    image_max_ratio: float = MAX_RATIO

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        path_likes,
        domain_probabilities,
        domain_names,
        global_batch_size,
        rank=0,
        dp_rank=0,
        dp_size=1,
        num_workers=1,
        seed=0,
        train=False,
        top_domains_to_cut=1,
        processor=None,
        tar_dir="/",
        image_token_id=None,
        moe_pad_with_random_token=False,
    ):
        self.underlying = MultimodalIndexedJsonlDataset(
            path_likes=path_likes,
            domain_probabilities=domain_probabilities,
            domain_names=domain_names,
            global_batch_size=global_batch_size,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            num_workers=num_workers,
            seed=seed,
            train=train,
            top_domains_to_cut=top_domains_to_cut,
        )
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = processor.image_processor if processor is not None else None
        self.max_seq_len = max_seq_len
        self.tar_dir = tar_dir
        self.image_token_id = image_token_id
        self.moe_pad_with_random_token = moe_pad_with_random_token

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def process_vision(self, images, videos=None):
        raise NotImplementedError

    def convert_example(self, example, conversations, imgs, domain_states, tools=None):
        raise NotImplementedError

    def gen_label_mask(self, conversations, imgs=None, tools=None, image_grid_thw=None,
                       label_role=("assistant",), rm_bos=True):
        """Default implementation for Qwen-style processors.

        Uses ``processor.apply_chat_template`` iteratively and measures token
        lengths to mark non-``label_role`` spans. Overridden by ``GemmaVLDataset``
        which does not have a chat_template.
        """
        pre_len = 0
        mask_indexs = []
        for i in range(len(conversations)):
            if conversations[i]["role"] in ("system",):
                continue
            add_generation_prompt = conversations[i]["role"] in ("user",)
            text = self.processor.apply_chat_template(
                conversations[:i + 1],
                tools=tools,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            if image_grid_thw is not None:
                n_imgs = text.count(self.tokenizer.image_token)
                thw_prefix = image_grid_thw[:n_imgs] if n_imgs > 0 else None
                text = self.padding_vision_token(text, thw_prefix)
            if rm_bos:
                text = remove_bos(text)
            cur_len = len(self.tokenizer._tokenizer(text, padding=False).input_ids)
            if conversations[i]["role"] not in label_role:
                mask_indexs.append([pre_len, cur_len])
            pre_len = cur_len
        if getattr(self, "mask_history", False):
            mask_indexs = [[mask_indexs[0][0], mask_indexs[-1][-1]]]
        return mask_indexs

    def _pre_convert_hook(self, imgs, json_data):
        """Family-specific per-sample step run after image validation.

        Default no-op. ``QwenVLDataset`` overrides this to apply an extra
        Qwen3VL resize when the processor is ``Qwen3VLProcessor``.
        """
        return imgs

    # ------------------------------------------------------------------
    # Shared pad / truncate / shift for ``convert_example`` implementations
    # ------------------------------------------------------------------
    def _pad_truncate_shift(
        self,
        input_ids: list,
        labels: list,
        attention_mask: list,
        *,
        use_for_hf: bool,
        pad_token_id: int,
        random_pad: bool = False,
        random_pad_ban_ids: Optional[list] = None,
        truncate_before_shift: bool = False,
    ):
        """Bring ``input_ids/labels/attention_mask`` to ``max_seq_len`` after shift.

        Autoregressive training uses ``max_seq_len + 1`` tokens: after the AR
        shift we end up with ``max_seq_len`` (input, label) pairs.

        Args:
            truncate_before_shift: If True (Gemma-style), head-truncate to
                ``max_seq_len + 1`` before shifting so the sample is aligned to
                the target length before shift. If False (Qwen-style), let
                overlong samples fall through and rely on the tail truncate
                after shift to keep the trailing tokens.
        """
        target_len = self.max_seq_len + 1
        if len(input_ids) < target_len:
            pad_len = target_len - len(input_ids)
            if random_pad:
                input_ids = random_pad_list(input_ids, pad_len, random_pad_ban_ids)
            else:
                input_ids = input_ids + [pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        elif truncate_before_shift and len(input_ids) > target_len:
            input_ids = input_ids[:target_len]
            labels = labels[:target_len]
            attention_mask = attention_mask[:target_len]

        # AR shift
        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]
        labels = labels[:-1] if use_for_hf else labels[1:]

        # Overlong: keep the tail
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len:]
            labels = labels[-self.max_seq_len:]
            attention_mask = attention_mask[-self.max_seq_len:]

        return input_ids, labels, attention_mask

    # ------------------------------------------------------------------
    # Iteration — same skeleton across families
    # ------------------------------------------------------------------
    def __iter__(self):
        """Common per-sample loop.

        1. pull a raw json line
        2. load + validate images (skip on failure)
        3. convert_conversations
        4. ``_pre_convert_hook`` (family-specific)
        5. ``convert_example`` -> training tensors
        6. yield / skip
        """
        domain_states = SimpleNamespace(domain_lines=0)
        for example in self.underlying:
            json_data = example["json_data"]

            imgs = None
            if "images" in json_data and len(json_data["images"]) > 0:
                try:
                    imgs = fetch_images(json_data["images"], self.tar_dir)
                except Exception as e:
                    domain_states.domain_lines += example["domain_line"]
                    print(
                        f"Abort Sample at dp-rank:{self.underlying.dp_rank}"
                        f"[image load failed: {e}]"
                    )
                    continue

                if not self._images_are_valid(imgs):
                    domain_states.domain_lines += example["domain_line"]
                    print(f"Abort Sample at dp-rank:{self.underlying.dp_rank}[invalid image]")
                    continue

            conversations = convert_conversations(json_data["conversations"])
            tools = json_data.get("tools", None)
            assert len(conversations) > 1, "SFT 至少需要一轮对话 (user + assistant)"

            imgs = self._pre_convert_hook(imgs, json_data)

            del example["json_data"]
            example_copy = copy.deepcopy(example)
            example_copy = self.convert_example(
                example_copy, conversations, imgs, domain_states, tools,
            )
            if example_copy is None:
                continue
            yield example_copy

    def _images_are_valid(self, imgs) -> bool:
        for img in imgs:
            if img is None:
                return False
            width, height = img.size
            if width < self.image_min_size or height < self.image_min_size:
                return False
            if max(height, width) / min(height, width) > self.image_max_ratio:
                return False
        return True
