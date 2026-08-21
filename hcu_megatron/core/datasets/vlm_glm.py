# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GLM 4.1V / 4.5V dataset.

Reuses the Qwen VL pipeline (Glm4v processor also emits a flat pixel sequence
plus per-image ``image_grid_thw`` with ``merge_size=2``). The only
family-specific bit is the vision placeholder token — ``<|image|>`` on GLM vs
``<|image_pad|>`` on Qwen — which is picked up transparently from the injected
``tokenizer.image_token`` set by :func:`setup_glm_tokenizer`.
"""

from __future__ import annotations

from hcu_megatron.core.datasets.vlm_qwen import QwenVLDataset


class GlmVLDataset(QwenVLDataset):
    """GLM 4.1V / 4.5V SFT dataset — Qwen-style pipeline with GLM tokens.

    ``_pre_convert_hook`` intentionally inherits the base class no-op behavior:
    ``Glm4vImageProcessor`` (or its fast counterpart) already resizes internally
    per its ``size`` config, unlike Qwen3VL's processor which required an
    external resize step.
    """
    pass
