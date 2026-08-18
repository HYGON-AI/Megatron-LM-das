# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Gemma3-VL dataset.

Much simpler than the Qwen family:
    - no chat_template — turns are stitched manually
    - no image_grid_thw — pixel_values are standard (B, C, H, W)
    - no mRoPE — collator does not compute position_ids
    - no vision-token placeholder expansion — ``<image>`` is replaced with
      ``GEMMA_TOKENS_PER_IMAGE`` copies of ``<image_soft_token>``

Gemma3 conversation format::

    <bos><start_of_turn>user
    {text}<end_of_turn>
    <start_of_turn>model
    {text}<end_of_turn><eos>
"""

from __future__ import annotations

import torch

from hcu_megatron.core.datasets.mm_dataset import MultiModalDataset, remove_bos


GEMMA_IMAGE_TOKEN = "<image_soft_token>"
GEMMA_TOKENS_PER_IMAGE = 256


class GemmaVLDataset(MultiModalDataset):
    """Gemma3-VL SFT dataset."""

    def __init__(
        self,
        min_pixels_num,
        max_pixels_num,
        use_for_hf,
        mask_history,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_pixels_num = min_pixels_num
        self.max_pixels_num = max_pixels_num
        self.use_for_hf = use_for_hf
        self.mask_history = mask_history

    # ------------------------------------------------------------------
    # Text formatting (no chat_template)
    # ------------------------------------------------------------------
    @staticmethod
    def _format_content(content) -> str:
        """Turn segmented content (list of dict) into plain text.

        Maps ``<image>`` / ``<video>`` back to a run of Gemma soft tokens.
        """
        if isinstance(content, str):
            return content.replace("<image>", GEMMA_IMAGE_TOKEN * GEMMA_TOKENS_PER_IMAGE)
        parts = []
        for seg in content:
            seg_type = seg.get("type")
            if seg_type == "text":
                parts.append(seg.get("text", ""))
            elif seg_type in ("image", "video"):
                parts.append(GEMMA_IMAGE_TOKEN * GEMMA_TOKENS_PER_IMAGE)
        return "".join(parts)

    def _format_conversation(self, conversations) -> str:
        """Stitch a Gemma3 conversation string (bos + turns + eos)."""
        parts = [self.tokenizer._tokenizer.bos_token]
        for turn in conversations:
            role = turn.get("role", "")
            content = self._format_content(turn.get("content", ""))
            if role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")
            elif role == "system":
                parts.append(f"{content}\n")
        parts.append(self.tokenizer._tokenizer.eos_token)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Vision preprocessing — standard SigLIP-style processor
    # ------------------------------------------------------------------
    def process_vision(self, images, videos=None):
        if images is not None:
            return self.image_processor(images=images, return_tensors="pt")
        return {}

    # ------------------------------------------------------------------
    # Label mask — Gemma has no apply_chat_template, so we override the base
    # ------------------------------------------------------------------
    def gen_label_mask(self, conversations, imgs=None, tools=None, image_grid_thw=None,
                       label_role=("assistant",), rm_bos=True):
        parts = [self.tokenizer._tokenizer.bos_token]
        pre_len = 0
        mask_indexs = []
        for turn in conversations:
            role = turn.get("role")
            if role in ("system",):
                continue
            content = self._format_content(turn.get("content", ""))
            if role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")
            text = "".join(parts)
            if rm_bos:
                text = remove_bos(text)
            cur_len = len(self.tokenizer._tokenizer(text, padding=False).input_ids)
            if role not in label_role:
                mask_indexs.append([pre_len, cur_len])
            pre_len = cur_len

        if self.mask_history:
            mask_indexs = [[mask_indexs[0][0], mask_indexs[-1][-1]]]
        return mask_indexs

    # ------------------------------------------------------------------
    # Full single-sample pipeline (simpler than Qwen: 6 steps)
    # ------------------------------------------------------------------
    def convert_example(self, example, conversations, imgs, domain_states, tools=None):
        # 1. vision
        media_info = self.process_vision(imgs)
        pixel_values = media_info.get("pixel_values", None)

        # 2. text (bos + turns + eos, with soft-token expansion)
        all_text = self._format_conversation(conversations)

        # 3. tokenize
        tokenized = self.tokenizer._tokenizer(all_text, padding=False)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # 4. label mask
        labels = torch.tensor(input_ids, dtype=torch.int64)
        label_mask = self.gen_label_mask(conversations, imgs, tools, rm_bos=False)
        for mask in label_mask:
            labels[mask[0]:mask[1]] = -100
        prompt_len = label_mask[-1][-1]
        labels = labels.tolist()

        # 5. pad / truncate / shift (shared)
        input_ids, labels, attention_mask = self._pad_truncate_shift(
            input_ids, labels, attention_mask,
            use_for_hf=self.use_for_hf,
            pad_token_id=self.tokenizer._tokenizer.pad_token_id,
            random_pad=False,
            random_pad_ban_ids=None,
            truncate_before_shift=True,
        )

        # 6. package
        example["input_ids"] = torch.tensor(input_ids, dtype=torch.int64)
        example["labels"] = torch.tensor(labels, dtype=torch.int64)
        example["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
        example["pixel_values"] = pixel_values
        example["image_grid_thw"] = None  # Gemma has no per-image grid
        example["image_input_mask"] = example["input_ids"] == self.tokenizer.image_token_id

        domain_states.domain_lines += example["domain_line"]

        if torch.all(example["labels"] == -100).item():
            print(f"Abort Sample at dp-rank:{self.underlying.dp_rank}[all ignore]")
            return None

        example["domain_line"] = torch.tensor(domain_states.domain_lines, dtype=torch.int64)
        example["prompt_len"] = torch.tensor(prompt_len, dtype=torch.int64)
        domain_states.domain_lines = 0
        return example
