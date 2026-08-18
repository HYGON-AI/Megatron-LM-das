# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Qwen VL family dataset (qwen2vl / qwen2.5vl / qwen3vl).

Only the Qwen-specific single-sample pipeline lives here. The dataset loop,
image loading/validation, and pad/shift live on ``MultiModalDataset`` (see
``mm_dataset.py``). Qwen3VL adds a ``_pre_convert_hook`` step because the
``Qwen3VLProcessor`` skips the internal image resize.
"""

from __future__ import annotations

import torch
from transformers.feature_extraction_utils import BatchFeature

try:
    from transformers import Qwen3VLProcessor
except ImportError:
    Qwen3VLProcessor = None

from hcu_megatron.core.datasets.mm_dataset import MultiModalDataset
from hcu_megatron.core.datasets.vlm_images import resize_image


class QwenVLDataset(MultiModalDataset):
    """Dataset for Qwen2-VL / Qwen2.5-VL / Qwen3-VL SFT.

    Family differences are minimal:
      - Qwen2VL / Qwen2.5VL: ``Qwen2VLProcessor`` resizes internally.
      - Qwen3VL: ``Qwen3VLProcessor`` (fast) does NOT resize internally, so we
        apply ``resize_image`` before ``process_vision``.
    """

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
    # Qwen3VL: manual resize before ``process_vision``
    # ------------------------------------------------------------------
    def _pre_convert_hook(self, imgs, json_data):
        if imgs is None:
            return imgs
        if Qwen3VLProcessor is not None and isinstance(self.processor, Qwen3VLProcessor):
            imgs = [
                resize_image(ele, img, self.min_pixels_num, self.max_pixels_num)
                for ele, img in zip(json_data["images"], imgs)
            ]
        return imgs

    # ------------------------------------------------------------------
    # Vision preprocessing
    # ------------------------------------------------------------------
    def process_vision(self, images, videos=None):
        """PIL images → pixel_values + image_grid_thw.

        Qwen2/Qwen2.5/Qwen3 image_processor outputs a flat pixel sequence and a
        per-image (t, h, w) grid, unlike a standard ViT. Example: a 560x560 image
        with merge_size=2, patch_size=14 produces image_grid_thw == [[1, 20, 20]]
        and 400 vision tokens.
        """
        image_inputs = self.image_processor(images=images, return_tensors="pt") if images is not None else {}
        videos_inputs = (
            self.image_processor(images=None, videos=videos, return_tensors="pt")
            if videos is not None else {}
        )
        return BatchFeature(data={**image_inputs, **videos_inputs})

    # ------------------------------------------------------------------
    # <|image_pad|> placeholder expansion
    # ------------------------------------------------------------------
    def padding_vision_token(self, text: str, image_grid_thw, video_grid_thw=None):
        """Expand each ``<|image_pad|>`` in the chat_template output to N tokens.

        N = t * h * w / merge_size^2 per image. The chat template emits a single
        placeholder per image; the model's decoder expects the exact vision
        token count.
        """
        merge_length = self.image_processor.merge_size ** 2
        if image_grid_thw is not None:
            index = 0
            while self.tokenizer.image_token in text:
                text = text.replace(
                    self.tokenizer.image_token,
                    "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1,
                )
                index += 1
            text = text.replace("<|placeholder|>", self.tokenizer.image_token)

        if video_grid_thw is not None:
            index = 0
            while self.tokenizer.video_token in text:
                text = text.replace(
                    self.tokenizer.video_token,
                    "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1,
                )
                index += 1
            text = text.replace("<|placeholder|>", self.tokenizer.video_token)

        return text

    # ------------------------------------------------------------------
    def get_image_token_cnt(self, image_grid_thw, video_grid_thw=None):
        """Vision token count implied by ``image_grid_thw`` / ``video_grid_thw``.

        Used to check that ``input_ids`` contain exactly this many
        ``<|image_pad|>`` after tokenization.
        """
        merge_length = self.image_processor.merge_size ** 2
        total_cnt = torch.tensor(0, dtype=torch.long)
        if image_grid_thw is not None:
            for i in range(image_grid_thw.shape[0]):
                total_cnt += image_grid_thw[i].prod() // merge_length
        if video_grid_thw is not None:
            for i in range(video_grid_thw.shape[0]):
                total_cnt += video_grid_thw.prod() // merge_length
        return total_cnt.item()

    # ------------------------------------------------------------------
    # Full single-sample pipeline
    # ------------------------------------------------------------------
    def convert_example(
        self,
        example,
        conversations,
        imgs,
        domain_states,
        tools=None,
    ):
        """Convert one conversation sample into training tensors.

        Steps:
            1. vision → pixel_values + image_grid_thw
            2. chat_template → text with a single ``<|image_pad|>`` per image
            3. expand ``<|image_pad|>`` to the exact vision token count
            4. tokenize → input_ids / attention_mask
            5. build label mask (prompt/history spans → -100)
            6. pad / truncate / shift (shared with GemmaVL via base class)
            7. package tensors, verify image token count matches
        """
        # 1. vision
        media_info = self.process_vision(imgs)
        image_grid_thw = media_info.get("image_grid_thw", None)

        # 2. chat_template → text
        all_text = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=False,
        )
        # 3. expand image placeholder
        all_text = self.padding_vision_token(all_text, image_grid_thw)
        # 4. tokenize
        tokenized = self.tokenizer._tokenizer(all_text, padding=False)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # 5. label mask
        labels = torch.tensor(input_ids, dtype=torch.int64)
        label_mask = self.gen_label_mask(conversations, imgs, tools, image_grid_thw, rm_bos=False)
        for mask in label_mask:
            labels[mask[0]:mask[1]] = -100
        prompt_len = label_mask[-1][-1]
        labels = labels.tolist()

        # 6. pad / truncate / shift (shared)
        ban_token_ids = [
            self.tokenizer.image_token_id, self.tokenizer.video_token_id,
            self.tokenizer.vision_start_token_id, self.tokenizer.vision_end_token_id,
        ]
        input_ids, labels, attention_mask = self._pad_truncate_shift(
            input_ids, labels, attention_mask,
            use_for_hf=self.use_for_hf,
            pad_token_id=self.tokenizer._tokenizer.pad_token_id,
            random_pad=self.moe_pad_with_random_token,
            random_pad_ban_ids=ban_token_ids,
            truncate_before_shift=False,
        )

        # 7. package + verify
        example["input_ids"] = torch.tensor(input_ids, dtype=torch.int64)
        example["labels"] = torch.tensor(labels, dtype=torch.int64)
        example["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
        example["pixel_values"] = media_info.get("pixel_values", None)
        example["image_grid_thw"] = image_grid_thw
        if self.image_token_id is not None:
            assert self.tokenizer.image_token_id == self.image_token_id
        example["image_input_mask"] = example["input_ids"] == self.tokenizer.image_token_id

        domain_states.domain_lines += example["domain_line"]

        # image token count must line up with image_grid_thw
        sum_image_token = example["image_input_mask"].sum().cpu().item()
        total_image_token = self.get_image_token_cnt(image_grid_thw)
        all_ignore = torch.all(example["labels"] == -100).item()
        assert total_image_token >= sum_image_token, (
            f"image token mismatch: in_ids={sum_image_token}, grid={total_image_token}"
        )
        if total_image_token > sum_image_token or all_ignore:
            print(f"Abort Sample at dp-rank:{self.underlying.dp_rank}")
            return None

        example["domain_line"] = torch.tensor(domain_states.domain_lines, dtype=torch.int64)
        example["prompt_len"] = torch.tensor(prompt_len, dtype=torch.int64)
        domain_states.domain_lines = 0
        return example
