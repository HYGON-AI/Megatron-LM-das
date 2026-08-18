# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Image loading and preprocessing utilities shared by all VLM datasets.

Split from the original ``mm_dataset.py`` / ``vlm_dataset.py`` so the individual
family dataset modules stay lean.

Contents:
    - tar-backed image reader with a small LRU-ish cache
    - ``fetch_image`` / ``fetch_images`` — single-sample image loaders
    - Qwen-style ``smart_resize`` + ``resize_image``
    - ``pad_and_split_pixel_values`` — CP-rank pixel_values redistribution
"""

from __future__ import annotations

import io
import math
import os
from typing import List, Tuple

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Qwen VL preprocessing constants
# ---------------------------------------------------------------------------
# copy from: https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


# ---------------------------------------------------------------------------
# Tar-backed image reader
# ---------------------------------------------------------------------------
# Simple approximate-LRU cache keyed by tar file path. Access counts monotonically
# increase; the smallest count is evicted when the cache is full. Keeping this
# module-global (as before) lets multiple call sites in the same worker share
# in-memory tar blobs without extra wiring.
_tar_file_cache: dict[str, list] = {}
_tar_file_cache_size = 25
_tar_access_cnt = 0


def get_image_from_tar(tar_filepath: str, offset: int, size: int) -> Image.Image:
    global _tar_access_cnt
    _tar_access_cnt += 1
    if tar_filepath in _tar_file_cache:
        tar_context = _tar_file_cache[tar_filepath][0]
        _tar_file_cache[tar_filepath][1] = _tar_access_cnt
    else:
        with open(tar_filepath, "rb") as tar_fd:
            tar_context = tar_fd.read()
        if len(_tar_file_cache) > _tar_file_cache_size:
            min_key = ""
            min_cnt = _tar_access_cnt * 10
            for k, v in _tar_file_cache.items():
                if v[1] < min_cnt:
                    min_key = k
                    min_cnt = v[1]
            _tar_file_cache.pop(min_key)
        _tar_file_cache[tar_filepath] = [tar_context, _tar_access_cnt]

    return Image.open(io.BytesIO(tar_context[offset:offset + size]))


def fetch_image(ele: dict, tar_dir: str) -> Image.Image:
    """Load one image entry.

    Supported entry shapes:
      - ``{"image_path": "/abs/path.png"}``          — read from disk
      - ``{"image_path": ..., "tar_name": ..., "offset": ..., "size": ...}`` — read from tar

    Returned image is converted to RGB.
    """
    image_path = ele["image_path"]
    if "tar_name" in ele:
        image_obj = get_image_from_tar(
            os.path.join(tar_dir, ele["tar_name"]), ele["offset"], ele["size"],
        )
    else:
        image_obj = Image.open(image_path)
    return image_obj.convert("RGB")


def fetch_images(images: List[dict], tar_dir: str) -> List[Image.Image]:
    return [fetch_image(ele, tar_dir) for ele in images]


# ---------------------------------------------------------------------------
# Qwen-style smart resize
# ---------------------------------------------------------------------------
def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Rescale so that both dims are divisible by ``factor``, total pixels stay
    within ``[min_pixels, max_pixels]``, and aspect ratio is preserved.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_image(
    ele: dict,
    image: Image.Image,
    default_min_pixels: int | None = None,
    default_max_pixels: int | None = None,
    size_factor: int = IMAGE_FACTOR,
) -> Image.Image:
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        default_min_pixels = default_min_pixels or MIN_PIXELS
        default_max_pixels = default_max_pixels or MAX_PIXELS
        width, height = image.size
        min_pixels = ele.get("min_pixels", default_min_pixels)
        max_pixels = ele.get("max_pixels", default_max_pixels)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    return image.resize((resized_width, resized_height))


# ---------------------------------------------------------------------------
# CP-rank pixel_values redistribution
# ---------------------------------------------------------------------------
def pad_and_split_pixel_values(
    cp_size: int,
    hw_factor: int,
    pixel_values: List[torch.Tensor],
    image_grid_thws: List[torch.Tensor],
):
    """Split ``pixel_values`` per-image, redistribute across CP ranks, pad to ``hw_factor``.

    ``hw_factor`` = ``context_parallel_size * (tp_size if sequence_parallel else 1) * 4``.
    qwen2vl/qwen2.5vl/qwen3vl all use merge_size=2, so raw pixel_values are already
    4-aligned; the extra ``*4`` in the caller keeps this function agnostic.

    Returns:
        new_pixel_values:    re-ordered and possibly zero-padded pixel_values list
        new_image_grid_thws: matching grid_thw list (padding entries inserted where needed)
        cp_img_num:          image count per CP rank
        images_padded:       0/1 flag per CP rank for whether padding was inserted
    """
    assert len(pixel_values) == len(image_grid_thws)

    # 1) split multi-image pixel_values into single-image pieces
    split_pixel_values: list[torch.Tensor] = []
    split_image_grid_thws: list[torch.Tensor] = []
    for pixel_value, image_grid_thw in zip(pixel_values, image_grid_thws):
        split_image_grid_thw = list(torch.split(image_grid_thw, 1, dim=0))
        split_image_grid_thws.extend(split_image_grid_thw)
        slice_begin = 0
        for ele in split_image_grid_thw:
            slice_end = slice_begin + ele.prod().item()
            split_pixel_values.append(pixel_value[slice_begin:slice_end].clone())
            slice_begin = slice_end

    pixel_values = split_pixel_values
    image_grid_thws = split_image_grid_thws
    img_num = len(image_grid_thws)

    # 2) evenly distribute images across CP ranks
    img_num_per_rank = img_num // cp_size
    img_num_remain = img_num % cp_size
    cp_img_num = [img_num_per_rank + (1 if i < img_num_remain else 0) for i in range(cp_size)]

    # 3) per rank: gather pixel_values, pad-align if needed
    img_idx = 0
    new_pixel_values: list[torch.Tensor] = []
    new_image_grid_thws: list[torch.Tensor] = []
    images_padded: list[int] = []
    for i in range(cp_size):
        seq_len = 0
        img_begin_idx = img_idx
        img_end_idx = img_begin_idx + cp_img_num[i]
        img_idx += cp_img_num[i]

        for j in range(img_begin_idx, img_end_idx):
            seq_len += pixel_values[j].size(0)
            new_pixel_values.append(pixel_values[j])
            new_image_grid_thws.append(image_grid_thws[j])

        image_padded = 0 != seq_len % hw_factor
        if image_padded:
            padded_seqlen = (seq_len + hw_factor - 1) // hw_factor * hw_factor - seq_len
            # hw_factor already includes *4, so padded_seqlen is a multiple of 4.
            assert padded_seqlen > 0 and padded_seqlen % 4 == 0
            # Insert a zero-padded pixel_values tensor and a placeholder grid.
            # Downstream uses images_padded to skip the placeholder.
            new_pixel_values.append(
                torch.zeros(
                    [padded_seqlen, pixel_values[0].size(-1)],
                    dtype=pixel_values[0].dtype,
                    device=pixel_values[0].device,
                )
            )
            new_image_grid_thws.append(
                torch.tensor(
                    [[1, 2, padded_seqlen // 2]],
                    dtype=image_grid_thws[0].dtype,
                    device=image_grid_thws[0].device,
                )
            )
            cp_img_num[i] += 1
        images_padded.append(int(image_padded))

    return new_pixel_values, new_image_grid_thws, cp_img_num, images_padded
