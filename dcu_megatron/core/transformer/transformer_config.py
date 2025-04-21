from dataclasses import dataclass

from megatron.core.transformer.transformer_config import TransformerConfig, MLATransformerConfig


@dataclass
class ExtraTransformerConfig:
    ##################
    # multi-token prediction
    ##################
    num_nextn_predict_layers: int = 0
    """The number of multi-token prediction layers"""

    mtp_loss_scale: float = 0.3
    """Multi-token prediction loss scale"""

    recompute_mtp_norm: bool = False
    """Whether to recompute mtp normalization"""

    recompute_mtp_layer: bool = False
    """Whether to recompute mtp layer"""

    share_mtp_embedding_and_output_weight: bool = False
    """share embedding and output weight with mtp layer."""

    ##################
    # flux
    ##################
    flux_transpose_weight: bool = False


@dataclass
class TransformerConfigPatch(TransformerConfig, ExtraTransformerConfig):
    pass


@dataclass
class MLATransformerConfigPatch(MLATransformerConfig, ExtraTransformerConfig):
    pass
