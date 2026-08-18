# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Batch collators for VLM datasets.

Both collators share pixel_values / has_image / loss_mask assembly. Neither
collator computes ``position_ids`` — under ``--use-bridge`` the Qwen VL model
(Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL) recomputes 3D mRoPE position ids inside
its own forward on every PP stage, and Gemma3-VL leaves position ids to the
underlying LM. Dataset-side computation would be redundant and can drift from
the Bridge-side implementation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data.dataloader import default_collate

from hcu_megatron.core.datasets.vlm_images import pad_and_split_pixel_values


# Only Qwen3-VL's Bridge forward consumes ``cp_img_num`` / ``images_padded`` and
# distributes vision tokens across CP ranks; other Qwen VL variants expect the
# full pixel_values tensor. Keep pad_and_split_pixel_values gated on this set.
_ARCHS_WITH_CP_IMG_SPLIT = {"qwen3vl"}


def _build_loss_mask(labels: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    """Standard SFT loss mask: 1 for supervised tokens, 0 for -100/pad."""
    loss_mask = torch.ones(labels.size(), dtype=torch.float, device=labels.device)
    loss_mask[labels == -100] = 0.0
    if pad_token_id is not None:
        loss_mask[labels == pad_token_id] = 0.0
    return loss_mask


def _split_pixel_values_from_instances(
    instances: Sequence[Dict],
    require_image_grid_thw: bool,
) -> tuple[List[Dict], List[torch.Tensor], List[Optional[torch.Tensor]]]:
    """Peel per-sample vision tensors out of ``instances`` for later concat.

    Returns the reduced (image-free) instance list, plus parallel lists of
    pixel_values and image_grid_thw (grid may be None for families that don't
    use it, but every collator drops the key regardless).

    Rejects mixed batches (some samples with images, some without) — hcu's
    downstream forward assumes per-batch consistency, and mixing would leave
    ``image_grid_thw`` and ``<image>`` token counts out of sync.
    """
    new_instances: list[Dict] = []
    pixel_values: list[torch.Tensor] = []
    image_grid_thws: list[Optional[torch.Tensor]] = []
    for instance in instances:
        pv = instance.get("pixel_values")
        if pv is not None:
            pixel_values.append(pv)
            image_grid_thws.append(instance.get("image_grid_thw"))
        # Drop vision keys before collate — they're variable-shape.
        instance.pop("pixel_values", None)
        instance.pop("image_grid_thw", None)
        new_instances.append(instance)

    if pixel_values and len(pixel_values) != len(instances):
        raise ValueError(
            f"Heterogeneous VLM batch: {len(pixel_values)} of {len(instances)} "
            f"samples have images. Batches must be all-image or all-text so that "
            f"image_grid_thw and <image> token counts stay aligned in forward."
        )

    if require_image_grid_thw and pixel_values:
        for i, g in enumerate(image_grid_thws):
            if g is None:
                raise ValueError(
                    f"Sample {i} has pixel_values but no image_grid_thw — "
                    f"Qwen VL collator requires the grid alongside every image."
                )

    return new_instances, pixel_values, image_grid_thws


class DataCollatorForQwenVL:
    """Batch collator for Qwen VL family.

    Responsibilities:
        - build loss_mask
        - Qwen3-VL only: split / redistribute pixel_values across CP ranks
          (with padding to ``hw_factor`` alignment when needed) and emit
          ``cp_img_num`` / ``images_padded`` metadata; Qwen2-VL / Qwen2.5-VL
          expect the full pixel_values tensor untouched.

    Not computed here:
        - position_ids: Bridge-side Qwen VL models compute 3D mRoPE inside
          ``forward`` on every PP stage. See:
            * bridge qwen25_vl model: get_rope_index invoked in forward
            * bridge qwen3_vl_step: forcefully sets ``forward_args["position_ids"] = None``
          so anything the collator produces would be ignored.
    """

    def __init__(
        self,
        hw_factor: int = 1,
        model_arch: str = "qwen2vl",
        tokenizer=None,
        cp_size: int = 1,
    ):
        # hw_factor = context_parallel_size * (tp_size if sequence_parallel else 1) * 4
        # qwen2vl/qwen2.5vl/qwen3vl merge_size == 2, so raw pixel_values are 4-aligned.
        self.hw_factor = hw_factor * 4
        self.model_arch = model_arch
        self.tokenizer = tokenizer
        self.cp_size = cp_size

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        new_instances, pixel_values, image_grid_thws = _split_pixel_values_from_instances(
            instances, require_image_grid_thw=True,
        )
        res = default_collate(new_instances)

        if pixel_values:
            if self.model_arch in _ARCHS_WITH_CP_IMG_SPLIT:
                self._pack_qwen3_vision(res, pixel_values, image_grid_thws)
            else:
                # qwen2vl / qwen2.5vl: forward expects the full pixel_values
                # tensor on every rank; no CP split, no padding entries.
                res["pixel_values"] = torch.cat(pixel_values, dim=0)
                res["image_grid_thw"] = torch.cat(image_grid_thws, dim=0)
                res["has_image"] = torch.tensor([True], dtype=torch.bool)
        else:
            res["has_image"] = torch.tensor([False], dtype=torch.bool)

        res["loss_mask"] = _build_loss_mask(res["labels"], self.tokenizer.pad_token_id)
        return res

    def _pack_qwen3_vision(
        self,
        res: Dict[str, torch.Tensor],
        pixel_values: List[torch.Tensor],
        image_grid_thws: List[torch.Tensor],
    ) -> None:
        """Qwen3-VL: CP-split pixel_values + emit cp_img_num / images_padded."""
        pixel_values, image_grid_thws, cp_img_num, images_padded = pad_and_split_pixel_values(
            self.cp_size,
            self.hw_factor,
            pixel_values,
            image_grid_thws,
        )
        for image_padded in images_padded:
            assert not image_padded, "not support image padded now"
        res["pixel_values"] = torch.cat(pixel_values, dim=0)
        res["image_grid_thw"] = torch.cat(image_grid_thws, dim=0)
        res["has_image"] = torch.tensor([True], dtype=torch.bool)
        res["images_padded"] = torch.tensor(images_padded, dtype=torch.int64)
        res["cp_img_num"] = torch.tensor(cp_img_num, dtype=torch.int64)


class DataCollatorForGemmaVL:
    """Batch collator for Gemma3-VL.

    Simpler than QwenVL: no CP-rank pixel_values reshuffle. Just the standard
    collate + loss_mask. position_ids are computed inside the language model.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        new_instances, pixel_values, _ = _split_pixel_values_from_instances(
            instances, require_image_grid_thw=False,
        )
        res = default_collate(new_instances)

        if pixel_values:
            res["pixel_values"] = torch.cat(pixel_values, dim=0)
            res["has_image"] = torch.tensor([True], dtype=torch.bool)
        else:
            res["has_image"] = torch.tensor([False], dtype=torch.bool)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else None
        res["loss_mask"] = _build_loss_mask(res["labels"], pad_id)
        return res
