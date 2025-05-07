from typing import Optional
from functools import wraps
from dataclasses import dataclass

from megatron.training import get_args
from megatron.core.transformer.transformer_config import TransformerConfig, MLATransformerConfig


def transformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        fn(self)
        args = get_args()

        """Number of Multi-Token Prediction (MTP) Layers."""
        self.mtp_num_layers = args.mtp_num_layers

        """Weighting factor of Multi-Token Prediction (MTP) loss."""
        self.mtp_loss_scaling_factor = args.mtp_loss_scaling_factor

        ##################
        # flux
        ##################
        self.flux_transpose_weight = args.flux_transpose_weight


    return wrapper


@dataclass
class ExtraTransformerConfig:
    ##################
    # flux
    ##################
    flux_transpose_weight: bool = False

    combined_1f1b: bool = False
    """If true, use combined 1F1B for communication hiding."""

    combined_1f1b_recipe: str = 'ep_a2a'
    """Recipe to use for combined 1F1B. Currently only 'ep_a2a' and 'golden' are supported."""


@dataclass
class TransformerConfigPatch(TransformerConfig, ExtraTransformerConfig):
    pass


@dataclass
class MLATransformerConfigPatch(MLATransformerConfig, ExtraTransformerConfig):
    pass
