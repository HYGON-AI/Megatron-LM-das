# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CLI arguments and dataset-config JSON parsing for VLM training.

Public API:
    * ``add_vlm_extra_args(parser)`` — register VLM CLI arguments
    * ``parse_dataset_config(args)`` — load the ``--vlm-data-config-path`` JSON
      and populate ``args.data_path`` / ``args.vlm_*`` fields

Expected JSON layout at ``--vlm-data-config-path``::

    {
        "train_data_infos": {
            "math":  {"path": "/data/math_jsonl_dir",  "probability": 0.5},
            "chat":  {"path": "/data/chat_jsonl_dir",  "probability": 0.5}
        },
        "eval_data_infos": {                                # optional
            "math_eval": {"path": "/data/math_eval_dir"}
        }
    }

Fields:
    path         — directory (containing ``*.jsonl``) or a single ``.jsonl`` file
    probability  — sampling weight for this domain in the global batch
                   (required for train_data_infos; ignored for eval_data_infos)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from hcu_megatron.core.datasets.utils import print_rank_0


# =============================================================================
# CLI registration
# =============================================================================
def add_vlm_extra_args(parser):
    """Register VLM training CLI arguments (model + data)."""

    # ── model architecture + preprocessing ──
    model_group = parser.add_argument_group(title="vlm model arguments")
    model_group.add_argument(
        "--model-arch",
        type=str,
        default="qwen2vl",
        choices=["qwen2vl", "qwen2.5vl", "qwen3vl", "gemma3vl"],
        help="model architecture; drives processor / tokenizer defaults",
    )
    model_group.add_argument("--processor-path", type=str, default=None)
    model_group.add_argument("--tarfile-path", type=str, default="/")
    model_group.add_argument(
        "--min-pixels-num", type=int, default=None,
        help="min image width*height for the image processor",
    )
    model_group.add_argument(
        "--max-pixels-num", type=int, default=None,
        help="max image width*height for the image processor",
    )
    model_group.add_argument("--spatial-merge-size", type=int, default=2)
    model_group.add_argument(
        "--mask-history", action="store_true",
        help="in multi-turn chats, keep only the last turn as loss target",
    )

    # ── VLM freeze switches (mirrored onto the bridge provider) ──
    model_group.add_argument(
        "--freeze-language-model", action="store_true", default=False,
        help="freeze language model weights during fine-tuning",
    )
    model_group.add_argument(
        "--freeze-vision-model", action="store_true", default=False,
        help="freeze vision encoder weights during fine-tuning",
    )
    model_group.add_argument(
        "--freeze-vision-projection", action="store_true", default=False,
        help="freeze vision-to-language projection weights during fine-tuning",
    )

    # ── data config entry ──
    parser.add_argument(
        "--vlm-data-config-path",
        type=str,
        default=None,
        help="path to the VLM SFT dataset JSON config",
    )

    # ── data loading knobs ──
    data_group = parser.add_argument_group(title="vlm dataset arguments")
    data_group.add_argument(
        "--vlm-dataloader-prefetch-factor",
        type=int,
        default=4,
        help="DataLoader prefetch_factor",
    )
    data_group.add_argument(
        "--vlm-top-domains-to-cut",
        type=int,
        default=1,
        help=(
            "number of largest domains to redistribute when the raw "
            "int(prob*gbs) allocation doesn't sum to gbs"
        ),
    )

    return parser


# =============================================================================
# JSON dataset-config parsing
# =============================================================================
@dataclass(frozen=True)
class _SplitConfig:
    """Parsed per-split (train/eval) config, sorted by path for stable domain ids."""

    paths: List[str]
    domain_names: List[str]
    probabilities: Optional[List[float]]     # None for eval; required for train


def _parse_split(
    split_dict: Optional[dict],
    *,
    require_probability: bool,
    split_name: str,
) -> Optional[_SplitConfig]:
    """Turn a ``{domain_name: {path, probability?}, ...}`` dict into a ``_SplitConfig``.

    Sorted by ``path`` so ``domain_id`` is stable across runs regardless of dict
    iteration order. This stability matters if you later re-introduce anything
    that indexes by domain_id (metrics, per-domain counters, etc.).
    """
    if not split_dict:
        return None

    domain_names: List[str] = []
    paths: List[str] = []
    probabilities: List[float] = []
    for name, values in split_dict.items():
        if "path" not in values:
            raise ValueError(f"{split_name}[{name!r}] missing required key 'path'")
        domain_names.append(name)
        paths.append(values["path"])
        if require_probability:
            if "probability" not in values:
                raise ValueError(f"{split_name}[{name!r}] missing required key 'probability'")
            probabilities.append(float(values["probability"]))

    # Sort by path so domain_id is deterministic.
    order = sorted(range(len(paths)), key=paths.__getitem__)
    paths = [paths[i] for i in order]
    domain_names = [domain_names[i] for i in order]
    probabilities = [probabilities[i] for i in order] if require_probability else None

    if require_probability and probabilities is not None:
        total = sum(probabilities)
        if total <= 0:
            raise ValueError(f"{split_name} probabilities must sum to > 0, got {total}")

    return _SplitConfig(paths=paths, domain_names=domain_names, probabilities=probabilities)


def parse_dataset_config(args) -> None:
    """Parse ``args.vlm_data_config_path`` and populate ``args`` in place."""
    if not args.vlm_data_config_path:
        raise ValueError("--vlm-data-config-path must be specified for VLM SFT training")

    with open(args.vlm_data_config_path, "r") as f:
        data_config = json.load(f)

    if "train_data_infos" not in data_config:
        raise ValueError(
            f"'train_data_infos' not found in {args.vlm_data_config_path}"
        )

    train = _parse_split(
        data_config["train_data_infos"],
        require_probability=True,
        split_name="train_data_infos",
    )
    assert train is not None  # train_data_infos is required
    args.data_path = train.paths
    args.vlm_domain_probabilities = train.probabilities
    args.vlm_train_data_domain_names = train.domain_names

    # Eval split is optional; also disabled when --eval-iters == 0.
    eval_split = None
    if args.eval_iters > 0:
        eval_split = _parse_split(
            data_config.get("eval_data_infos"),
            require_probability=False,
            split_name="eval_data_infos",
        )
    if eval_split is not None:
        args.vlm_eval_data_path = eval_split.paths
        args.vlm_eval_data_domain_names = eval_split.domain_names
    else:
        args.vlm_eval_data_path = None
        args.vlm_eval_data_domain_names = None

    print_rank_0(
        f"parse_dataset_config: "
        f"data_path={args.data_path} "
        f"vlm_domain_probabilities={args.vlm_domain_probabilities} "
        f"train_data_domain_names={args.vlm_train_data_domain_names} "
        f"vlm_eval_data_path={args.vlm_eval_data_path}"
    )
