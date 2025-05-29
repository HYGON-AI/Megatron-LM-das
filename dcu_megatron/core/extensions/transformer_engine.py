import os
import copy
import torch
import dataclasses
import transformer_engine as te

from functools import wraps
from typing import Any, Optional, Callable
from packaging.version import Version as PkgVersion

from megatron.training import get_args
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.extensions.transformer_engine import TELinear as MegatronCoreTELinear
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear as MegatronCoreTELayerNormColumnParallelLinear

from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
)


def _get_extra_te_kwargs_wrapper(_get_extra_te_kwargs_func):
    @wraps(_get_extra_te_kwargs_func)
    def wrapper(config: TransformerConfig):
        extra_transformer_engine_kwargs = _get_extra_te_kwargs_func(config)
        if hasattr(config, "split_bw"):
            extra_transformer_engine_kwargs["delay_wgrad_compute"] = config.split_bw
        return extra_transformer_engine_kwargs

    if is_te_min_version("2.3.0.dev0"):
        return wrapper
    return _get_extra_te_kwargs_func


class TELinear(MegatronCoreTELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().

    parallel_mode currently supports 3 different values:
        - "column": Split the weight matrix along output dimension (used in TEColumnParallelLinear)
        - "row": Split the weight matrix along input dimension (used in TERowParallelLinear)
        - "duplicated": No tensor parallelism and weight is duplicated across TP ranks
        - Note: For expert linear layers, we will disable communication logic here
                as TP communication is handled in token_dispatcher.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: Optional[str] = None,
        is_expert: bool = False,
    ):
        args = get_args()
        self.split_bw = args.split_bw if hasattr(args, "split_bw") else False
        if not is_te_min_version("2.3.0.dev0"):
            assert not self.split_bw, "split_bw is currently not supported"

        if self.split_bw:
            config = copy.copy(config)
            config.split_bw = True

        super().__init__(
            input_size,
            output_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=is_expert,
        )

    def backward_dw(self):
        if not self.split_bw:
            return

        return super(MegatronCoreTELinear, self).backward_dw()


class TELayerNormColumnParallelLinear(MegatronCoreTELayerNormColumnParallelLinear):
    """
    Wrapper for the Transformer-Engine's `LayerNormLinear` layer that combines
    layernorm and linear layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
    ):
        args = get_args()
        self.split_bw = args.split_bw if hasattr(args, "split_bw") else False
        if not is_te_min_version("2.3.0.dev0"):
            assert not self.split_bw, "split_bw is currently not supported"

        if self.split_bw:
            config = copy.copy(config)
            config.split_bw = True

        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

    def backward_dw(self):
        if not self.split_bw:
            return

        return super(MegatronCoreTELayerNormColumnParallelLinear, self).backward_dw()


class TEDotProductAttentionPatch(te.pytorch.DotProductAttention):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
    ):
        self.config = config
        self.te_forward_mask_type = False
        self.qkv_format: str = 'sbhd'

        if self.config.apply_query_key_layer_scaling != bool(
            int(os.getenv('NVTE_APPLY_QK_LAYER_SCALING', '0'))
        ):
            raise ValueError(
                f"apply_query_key_layer_scaling is {self.config.apply_query_key_layer_scaling} "
                f"but environment variable NVTE_APPLY_QK_LAYER_SCALING is "
                f"{os.getenv('NVTE_APPLY_QK_LAYER_SCALING')}. Transformer Engine does not support "
                f"setting query key layer scaling via argument, so these two must match."
            )

        extra_kwargs: dict[str, Any] = {}
        if is_te_min_version("0.11.0"):
            extra_kwargs["num_gqa_groups"] = self.config.num_query_groups
        elif self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Transformer Engine v{get_te_version()} does not support Grouped Query Attention, "
                f"use a newer version of Transformer Engine. "
                f"(num_query_groups ({self.config.num_query_groups}) != "
                f"num_attention_heads ({self.config.num_attention_heads}))"
            )

        if is_te_min_version("0.10.0"):
            extra_kwargs["attention_type"] = attention_type
            # older version don't need attention_type

        if is_te_min_version("0.12.0", check_equality=False):
            self.te_forward_mask_type = True

        # This check is important as CP config can be disabled while having a valid CP group
        # Example - Disabling CP for encoder while a valid CP group exists for decoder
        if self.config.context_parallel_size > 1:
            assert is_te_min_version(
                "1.0.0"
            ), "Only Transformer-Engine version >= 1.0.0 supports context parallelism!"
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()
            extra_kwargs["cp_group"] = get_context_parallel_group(check_initialized=False)
            extra_kwargs["cp_global_ranks"] = get_context_parallel_global_ranks(
                check_initialized=False
            )
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
            if is_te_min_version("1.10.0"):
                if cp_comm_type is None:
                    extra_kwargs["cp_comm_type"] = "p2p"
                elif cp_comm_type == "a2a+p2p":
                    assert is_te_min_version("1.12.0"), (
                        f"Transformer-Engine v{get_te_version()} must be >= 1.12.0 to support"
                        "hierarchical cp commucation."
                    )
                    extra_kwargs["cp_comm_type"] = "a2a+p2p"
                    extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups(
                        check_initialized=False
                    )
                else:
                    extra_kwargs["cp_comm_type"] = cp_comm_type

        if self.config.deterministic_mode:
            if int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")) != 0:
                raise RuntimeError(
                    "deterministic_mode is on and we are using DotProductAttention from "
                    "Transformer Engine, but NVTE_ALLOW_NONDETERMINISTIC_ALGO is not 0. "
                    f"Currently set to: {os.getenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', 'not set')}."
                )

        if config.window_size is not None:
            # Check version
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "sliding window attention."
            )
            extra_kwargs['window_size'] = config.window_size

        if is_te_min_version("1.9.0"):
            # TE 1.10.0 introduces the ability to set the different k and v channels
            kv_channels = (
                (k_channels, v_channels)
                if k_channels is not None and v_channels is not None
                else self.config.kv_channels
            )
            extra_kwargs['softmax_scale'] = softmax_scale
        else:
            kv_channels = self.config.kv_channels

        self.kept_packed_seq_params = set(
            field.name for field in dataclasses.fields(PackedSeqParams)
        )
        if get_te_version() < PkgVersion("1.3.0"):
            # TE 1.3.0 introduces precomputing max_seqlen to remove unnecessary kernels and D2H
            # copies (#555)
            # These two arguments did not exist prior to 1.3.0
            self.kept_packed_seq_params.discard("max_seqlen_q")
            self.kept_packed_seq_params.discard("max_seqlen_kv")

        if get_te_version() < PkgVersion("1.10.0"):
            # TE 1.8.0 introduces cu_seqlens_padded which is the cu_seqlens with paddings counted
            # in each individual sequence in THD format dataset
            # These two arguments did not exist prior to 1.8.0. Full support added in 1.10.0 (#1012)
            self.kept_packed_seq_params.discard("cu_seqlens_q_padded")
            self.kept_packed_seq_params.discard("cu_seqlens_kv_padded")

        super(TEDotProductAttention, self).__init__(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=kv_channels,
            attention_dropout=(
                self.config.attention_dropout if attention_dropout is None else attention_dropout
            ),
            attn_mask_type=attn_mask_type.name,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
            layer_number=layer_number,
            **extra_kwargs,
        )


if is_te_min_version("1.9.0.dev0"):
    from megatron.core.extensions.transformer_engine import TEGroupedLinear as MegatronCoreTEGroupedLinear

    class TEGroupedLinear(MegatronCoreTEGroupedLinear):
        """
        Wrapper for the Transformer-Engine's `GroupedLinear` layer.

        Note that if Megatron's parallel_state has not been initialized
        yet, the tp_group passed to TE will be None and must be set later
        via set_tensor_parallel_group().
        """

        def __init__(
            self,
            num_gemms: int,
            input_size: int,
            output_size: int,
            *,
            parallel_mode: Optional[str],
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool = False,
            tp_comm_buffer_name: Optional[str] = None,
        ):
            args = get_args()
            self.split_bw = args.split_bw if hasattr(args, "split_bw") else False
            if not is_te_min_version("2.3.0.dev0"):
                assert not self.split_bw, "split_bw is currently not supported"

            if self.split_bw:
                config = copy.copy(config)
                config.split_bw = True

            super().__init__(
                num_gemms,
                input_size,
                output_size,
                parallel_mode=parallel_mode,
                config=config,
                init_method=init_method,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )

        def backward_dw(self):
            if not self.split_bw:
                return

            return super(MegatronCoreTEGroupedLinear, self).backward_dw()
