# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer property injection for VLM training.

The Megatron ``DefaultTokenizerText`` wrapper doesn't expose Qwen/Gemma vision
tokens as attributes. We monkey-patch a small set of attributes onto the
wrapper so the datasets/collators can access things like
``tokenizer.image_token_id`` uniformly.

This module also swaps ``tokenizer._tokenizer`` from the Megatron
``HuggingFaceTokenizer`` wrapper to the underlying HF ``AutoTokenizer`` so the
datasets can call it directly.
"""

from __future__ import annotations


# Qwen family — all three variants share the same vision token strings
_QWEN_VISION_TOKENS = (
    ("<|image_pad|>",     "image_token"),
    ("<|video_pad|>",     "video_token"),
    ("<|vision_start|>",  "vision_start_token"),
    ("<|vision_end|>",    "vision_end_token"),
)


# GLM 4.1V / 4.5V — vision placeholder is <|image|> (id 151343), with separate
# begin/end wrappers that the collator/dataset use for the equivalent of Qwen's
# vision_start / vision_end. See config.json image_token_id / image_start_token_id.
_GLM_VISION_TOKENS = (
    ("<|image|>",           "image_token"),
    ("<|video|>",           "video_token"),
    ("<|begin_of_image|>",  "vision_start_token"),
    ("<|end_of_image|>",    "vision_end_token"),
)


def _unwrap_hf_tokenizer(tokenizer):
    """Return the HF tokenizer and replace tokenizer._tokenizer with it.

    tokenizer type: DefaultTokenizerText -> MegatronTokenizerText
        tokenizer._tokenizer            = Megatron-LM HuggingFaceTokenizer
        tokenizer._tokenizer.tokenizer  = HF AutoTokenizer
    """
    hf = tokenizer._tokenizer.tokenizer
    tokenizer._tokenizer = hf
    return hf


def _inject_common_ids(tokenizer, hf):
    """Attributes present on any HF tokenizer that DefaultTokenizerText misses."""
    tokenizer.pad_token_id = hf.pad_token_id
    tokenizer.eos_token_id = hf.eos_token_id
    tokenizer.bos_token_id = hf.bos_token_id


def setup_qwen_tokenizer(tokenizer, args):
    """Inject Qwen vision token ids + shared pad/eos/bos ids."""
    hf = _unwrap_hf_tokenizer(tokenizer)
    for token, attr in _QWEN_VISION_TOKENS:
        setattr(tokenizer, attr, token)
        setattr(tokenizer, f"{attr}_id", hf.convert_tokens_to_ids(token))
    _inject_common_ids(tokenizer, hf)


def setup_gemma_tokenizer(tokenizer, args):
    """Inject Gemma soft-image token + shared pad/eos/bos ids."""
    hf = _unwrap_hf_tokenizer(tokenizer)
    tokenizer.image_token = "<image_soft_token>"
    tokenizer.image_token_id = hf.convert_tokens_to_ids("<image_soft_token>")
    _inject_common_ids(tokenizer, hf)


def setup_glm_tokenizer(tokenizer, args):
    """Inject GLM 4.1V/4.5V vision token ids + shared pad/eos/bos ids."""
    hf = _unwrap_hf_tokenizer(tokenizer)
    for token, attr in _GLM_VISION_TOKENS:
        setattr(tokenizer, attr, token)
        setattr(tokenizer, f"{attr}_id", hf.convert_tokens_to_ids(token))
    _inject_common_ids(tokenizer, hf)
