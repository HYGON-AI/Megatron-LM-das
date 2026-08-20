# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import warnings
from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.models.gpt.gpt_layer_specs import TESpecProvider
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    FusedMLASelfAttention,
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_inference_spec,
    get_mlp_module_spec_for_backend,
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.typed_torch import copy_signature
from megatron.core.utils import is_te_min_version

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TENorm,
        TELinear,
    )
except ImportError:
    warnings.warn('transformer_engine is not installed.')

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
except ImportError:
    warnings.warn('Apex is not installed.')

try:
    from megatron.core.extensions.kitchen import HAVE_KITCHEN, KitchenSpecProvider

except ImportError:
    HAVE_KITCHEN = False

from hcu_megatron.core.tensor_parallel.layers import (
    FluxColumnParallelLinear,
    FluxRowParallelLinear
)
from hcu_megatron.core.transformer.hyper_connection import HyperConnectionModule
from hcu_megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer, TransformerLayerSubmodules
from hcu_megatron.training.arguments import get_adaptor_args


def get_gpt_layer_with_flux_submodules(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    qk_l2_norm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,  # pylint: disable=unused-argument
    use_kitchen: bool = False,  # pylint: disable=unused-argument
    use_te_activation_func: bool = False,  # pylint: disable=unused-argument
    use_kitchen_attention: bool = False,  # pylint: disable=unused-argument
    kitchen_attention_backend: str = "sdpa",  # pylint: disable=unused-argument
    mla_down_proj_fusion: bool = False,
    use_grouped_gemm_for_dense_mlp: bool = False, # pylint: disable=unused-argument
    enable_hyper_connection: bool = False,
) -> ModuleSpec:
    """Use this spec to use flux modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MLA. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        enable_hyper_connection (bool): Use HyperConnectionTransformerLayer with
            HyperConnectionModule instead of plain TransformerLayer. Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with flux modules
    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    mlp = get_mlp_module_flux_spec(
        use_te=False,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    hc_module = HyperConnectionModule if enable_hyper_connection else IdentityOp

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        return TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MLASelfAttentionSubmodules(
                    linear_q_proj=FluxColumnParallelLinear,
                    linear_q_down_proj=TELinear,
                    linear_q_up_proj=FluxColumnParallelLinear,
                    linear_kv_down_proj=TELinear,
                    linear_kv_up_proj=FluxColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=FluxRowParallelLinear,
                    q_layernorm=TENorm if qk_layernorm else IdentityOp,
                    kv_layernorm=TENorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=hc_module,
            pre_mlp_layernorm=TENorm,
            mlp=mlp,
            mlp_hyper_connection=hc_module,
            mlp_bda=get_bias_dropout_add,
        )
    else:

        # TENorm significantly harms convergence when used
        # for QKLayerNorm if TE Version < 1.9;
        # we instead use the Apex implementation.
        qk_norm = TENorm if is_te_min_version("1.9.0") else FusedLayerNorm

        return TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=FluxColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=FluxRowParallelLinear,
                    q_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                    k_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=hc_module,
            pre_mlp_layernorm=TENorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            mlp_hyper_connection=hc_module,
        )


@copy_signature(get_gpt_layer_with_flux_submodules)
def get_gpt_layer_with_flux_spec(*args, **kwargs) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training)."""
    enable_hyper_connection = kwargs.get('enable_hyper_connection', False)
    layer_module = HyperConnectionTransformerLayer if enable_hyper_connection else TransformerLayer
    return ModuleSpec(
        module=layer_module,
        submodules=get_gpt_layer_with_flux_submodules(*args, **kwargs),
    )


def get_gpt_layer_with_transformer_engine_submodules(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    qk_l2_norm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_kitchen: bool = False,
    use_te_activation_func: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
    mla_down_proj_fusion: bool = False,
    use_grouped_gemm_for_dense_mlp: bool = False,
    enable_hyper_connection: bool = False,
) -> TransformerLayerSubmodules:
    """Use these submodules to use lower-level Transformer Engine modules (required for fp8
    training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MLA. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_te_op_fuser (bool, optional): Use Transformer Engine's operation-based API, which may
                                          enable certain operation fusions. Defaults to False.
        mla_down_proj_fusion (bool, optional): Enable fused q/kv down-projection and fused input
                                               layernorm when backend supports. Otherwise fall back
                                               to the unfused MLA.

    Returns:
        TransformerLayerSubmodules: TE modules to construct a TransformerLayer

    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    if use_kitchen:
        assert HAVE_KITCHEN
        backend: BackendSpecProvider = KitchenSpecProvider(
            fallback=TESpecProvider(),
            use_kitchen_attention=use_kitchen_attention,
            kitchen_attention_backend=kitchen_attention_backend,
        )
        if use_te_op_fuser:
            raise AssertionError("use_te_op_fuser not compatible with using kitchen in mlp.")
        if use_te_activation_func:
            raise AssertionError("use_te_activation_func not compatible with using kitchen.")
    else:
        backend = TESpecProvider()

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te_op_fuser=use_te_op_fuser,
        use_te_activation_func=use_te_activation_func,
        use_grouped_gemm_for_dense_mlp=use_grouped_gemm_for_dense_mlp,
    )

    hc_module = HyperConnectionModule if enable_hyper_connection else IdentityOp

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        linear_q_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )
        linear_kv_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )

        if mla_down_proj_fusion:
            fuse_input_layernorm = backend.column_parallel_layer_norm_linear() is not None
            input_layernorm = IdentityOp if fuse_input_layernorm else backend.layer_norm()
            down_proj_linear = (
                backend.column_parallel_layer_norm_linear()
                if fuse_input_layernorm
                else backend.linear()
            )
            return TransformerLayerSubmodules(
                input_layernorm=input_layernorm,
                self_attention=ModuleSpec(
                    module=FusedMLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=backend.column_parallel_linear(),
                        linear_qkv_down_proj=down_proj_linear,
                        linear_q_up_proj=linear_q_up_proj,
                        linear_kv_up_proj=linear_kv_up_proj,
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                self_attention_hyper_connection=hc_module,
                pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
                mlp=mlp,
                mlp_hyper_connection=hc_module,
                mlp_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map=(
                    {
                        "self_attention.linear_q_down_proj.layer_norm_": "input_layernorm.",
                        "self_attention.linear_kv_down_proj.layer_norm_": "input_layernorm.",
                        "self_attention.linear_qkv_down_proj.layer_norm_": "input_layernorm.",
                    }
                    if fuse_input_layernorm
                    else {}
                ),
            )
        return TransformerLayerSubmodules(
            input_layernorm=backend.layer_norm(has_residual=True),
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MLASelfAttentionSubmodules(
                    linear_q_proj=backend.column_parallel_linear(),
                    linear_q_down_proj=backend.linear(),
                    linear_q_up_proj=linear_q_up_proj,
                    linear_kv_down_proj=backend.linear(),
                    linear_kv_up_proj=linear_kv_up_proj,
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=IdentityOp,
                    kv_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=hc_module,
            pre_mlp_layernorm=backend.layer_norm(has_residual=True) if num_experts else IdentityOp,
            mlp=mlp,
            mlp_hyper_connection=hc_module,
            mlp_bda=get_bias_dropout_add,
        )
    else:
        qk_norm = backend.layer_norm(for_qk=True)
        return TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_layer_norm_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                    k_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=hc_module,
            pre_mlp_layernorm=backend.layer_norm(has_residual=True) if num_experts else IdentityOp,
            mlp=mlp,
            mlp_hyper_connection=hc_module,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "mlp.0.weight": "mlp.linear_fc1.layer_norm_weight",
                "mlp.0.bias": "mlp.linear_fc1.layer_norm_bias",
                "mlp.1.basic_ops.0.weight": "mlp.linear_fc1.weight",
                "mlp.1.basic_ops.1.bias": "mlp.linear_fc1.bias",
                "mlp.3.basic_ops.0.weight": "mlp.linear_fc2.weight",
                "mlp.3.basic_ops.1.bias": "mlp.linear_fc2.bias",
            },
        )


@copy_signature(get_gpt_layer_with_transformer_engine_submodules)
def get_gpt_layer_with_transformer_engine_spec(*args, **kwargs) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training)."""
    enable_hyper_connection = kwargs.get('enable_hyper_connection', False)
    layer_module = HyperConnectionTransformerLayer if enable_hyper_connection else TransformerLayer
    return ModuleSpec(
        module=layer_module,
        submodules=get_gpt_layer_with_transformer_engine_submodules(*args, **kwargs),
    )


def get_mlp_module_flux_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=FluxColumnParallelLinear,
                linear_fc2=FluxRowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec(
            use_te=True,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
        )


def get_gpt_layer_local_submodules(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    use_kitchen: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
    enable_hyper_connection: bool = False,
) -> TransformerLayerSubmodules:
    """Use these submodules for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MLA. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        enable_hyper_connection (bool): Use HyperConnectionTransformerLayer with
            HyperConnectionModule instead of plain TransformerLayer. Defaults to False.

    Returns:
        TransformerLayerSubmodules: Megatron-Core modules to construct a TransformerLayer
    """

    if use_kitchen:
        assert HAVE_KITCHEN
        backend = KitchenSpecProvider(
            fallback=LocalSpecProvider(),
            use_kitchen_attention=use_kitchen_attention,
            kitchen_attention_backend=kitchen_attention_backend,
        )
    else:
        backend = LocalSpecProvider()
    # Adjust for RMS norm.
    if normalization == "RMSNorm":
        layer_norm = backend.layer_norm(rms_norm=True, for_qk=False, has_residual=True)
        qk_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    else:
        layer_norm = backend.layer_norm(rms_norm=False, for_qk=False, has_residual=True)
        qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)

    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_local_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    mlp = get_mlp_module_spec_for_backend(
        backend=backend, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )

    hc_module = HyperConnectionModule if enable_hyper_connection else IdentityOp

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        return TransformerLayerSubmodules(
            input_layernorm=layer_norm,
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MLASelfAttentionSubmodules(
                    linear_q_proj=backend.column_parallel_linear(),
                    linear_q_down_proj=backend.column_parallel_linear(),
                    linear_q_up_proj=backend.column_parallel_linear(),
                    linear_kv_down_proj=backend.column_parallel_linear(),
                    linear_kv_up_proj=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    kv_layernorm=qk_norm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=hc_module,
            pre_mlp_layernorm=layer_norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            mlp_hyper_connection=hc_module,
        )
    else:
        return TransformerLayerSubmodules(
            input_layernorm=layer_norm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                    k_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=hc_module,
            pre_mlp_layernorm=layer_norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            mlp_hyper_connection=hc_module,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        )


@copy_signature(get_gpt_layer_local_submodules)
def get_gpt_layer_local_spec(*args, **kwargs) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core."""
    enable_hc = kwargs.get('enable_hyper_connection', False)
    layer_module = HyperConnectionTransformerLayer if enable_hc else TransformerLayer
    return ModuleSpec(
        module=layer_module, submodules=get_gpt_layer_local_submodules(*args, **kwargs)
    )


def get_gpt_decoder_layer_specs(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """GPT block spec."""

    if use_transformer_engine:
        if get_adaptor_args().parallel_linear_impl == 'flux':
            gpt_layer_spec_clz = get_gpt_layer_with_flux_spec
        else:
            gpt_layer_spec_clz = get_gpt_layer_with_transformer_engine_spec

        dense_layer_spec = gpt_layer_spec_clz(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_te_activation_func=config.use_te_activation_func,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
            mla_down_proj_fusion=getattr(config, "mla_down_proj_fusion", False),
            enable_hyper_connection=config.enable_hyper_connections,
        )
        moe_layer_spec = gpt_layer_spec_clz(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_te_activation_func=config.use_te_activation_func,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
            mla_down_proj_fusion=getattr(config, "mla_down_proj_fusion", False),
            enable_hyper_connection=config.enable_hyper_connections,
        )
    elif config.transformer_impl == "inference_optimized":
        layer_norm_impl = TENorm
        dense_layer_spec = get_gpt_layer_with_inference_spec(
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            qk_l2_norm=qk_l2_norm,
        )
        moe_layer_spec = get_gpt_layer_with_inference_spec(
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            qk_l2_norm=qk_l2_norm,
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        )
    else:
        dense_layer_spec = get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            normalization=normalization,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
            enable_hyper_connection=config.enable_hyper_connections,
        )
        moe_layer_spec = get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            normalization=normalization,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
            enable_hyper_connection=config.enable_hyper_connections,
        )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    return layer_specs
