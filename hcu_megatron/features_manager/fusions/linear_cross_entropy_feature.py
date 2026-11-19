# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Feature registration for the optional HCU Linear CE training path."""

from argparse import ArgumentParser, Namespace

from ..feature import AbstractFeature


class LinearCrossEntropyFeature(AbstractFeature):
    """Validate the optional fused output projection and CE training path."""

    def __init__(self) -> None:
        # Upstream flags are absent from the early adaptor-only parser. Always
        # run validation, then select the backend using the fully parsed args.
        super().__init__("cross-entropy-loss-fusion", optimization_level=0)

    def register_args(self, parser: ArgumentParser) -> None:
        """Extend the upstream backend choices without adding a new argument."""
        # The early adaptor-only parser does not contain the upstream option.
        for action in parser._actions:
            if "--cross-entropy-fusion-impl" in action.option_strings:
                if action.choices is not None and "linear" not in action.choices:
                    action.choices = [*action.choices, "linear"]
                break

    def validate_args(self, args: Namespace) -> Namespace:
        if not (
            getattr(args, "cross_entropy_loss_fusion", False)
            and getattr(args, "cross_entropy_fusion_impl", "native") == "linear"
        ):
            return args
        unsupported = []
        for name in ("tensor_model_parallel_size", "context_parallel_size"):
            if getattr(args, name, 1) != 1:
                unsupported.append(f"{name} must be 1")
        for name in (
            "sequence_parallel",
            "mtp_num_layers",
            "defer_embedding_wgrad_compute",
            "use_mup",
            "enable_vocab_parallel",
        ):
            if getattr(args, name, False):
                unsupported.append(f"{name} must be disabled")
        if unsupported:
            raise ValueError("HCU Linear CE does not support: " + "; ".join(unsupported))
        if not getattr(args, "bf16", False):
            raise ValueError("HCU Linear CE requires --bf16")
        return args
