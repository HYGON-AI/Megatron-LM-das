import warnings

from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer import ModuleSpec
from .multi_token_predictor import (
    MultiTokenPredicationSubmodules,
    MultiTokenPredictor
)


try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TENorm
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    LNImpl = FusedLayerNorm
except ImportError:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


def get_mtp_spec(transformer_layer, use_te=False, use_flux=False):
    """
    Multi Token Predication Layer Specification.
    """
    use_te = use_te & HAVE_TE
    mtp_spec = ModuleSpec(
        module=MultiTokenPredictor,
        submodules=MultiTokenPredicationSubmodules(
            embedding=None,
            enorm=TENorm if use_te or use_flux else LNImpl,
            hnorm=TENorm if use_te or use_flux else LNImpl,
            eh_proj=TEColumnParallelLinear if use_te else ColumnParallelLinear,
            transformer_layer=transformer_layer,
            final_layernorm=TENorm if use_te or use_flux else LNImpl,
            output_layer=None,
        )
    )

    return mtp_spec
