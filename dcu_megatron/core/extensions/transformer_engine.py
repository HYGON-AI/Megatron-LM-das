import dataclasses
import os
from typing import Any, Optional
from functools import wraps

import torch
from packaging.version import Version as PkgVersion

from megatron.training import get_args
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_te_version,
    is_te_min_version,
)

try:
    import transformer_engine as te

    HAVE_TE = True
except ImportError:
    from unittest.mock import MagicMock

    te = MagicMock()
    HAVE_TE = False

from megatron.core.extensions.transformer_engine import TEDotProductAttention


def te_module_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        megatron_args = get_args()
        kwargs["fine_grained_activation_offloading"] = megatron_args.fine_grained_activation_offloading
        fn(self, *args, **kwargs)

    return wrapper


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
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        self.config = config
        self.te_forward_mask_type = False
        self.qkv_format: str = "sbhd"

        if self.config.apply_query_key_layer_scaling != bool(
            int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))
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

        if pg_collection is None:
            pg_collection = ProcessGroupCollection(
                tp=get_tensor_model_parallel_group(check_initialized=False),
                cp=get_context_parallel_group(check_initialized=False),
                hcp=get_hierarchical_context_parallel_groups(check_initialized=False),
            )
        else:
            assert hasattr(
                pg_collection, "tp"
            ), "TEDotProductAttention pg_collection must have tp pg"
            assert hasattr(
                pg_collection, "cp"
            ), "TEDotProductAttention pg_collection must have cp pg"
            if cp_comm_type == "a2a+p2p":
                assert hasattr(
                    pg_collection, "hcp"
                ), "TEDotProductAttention pg_collection must have hierarchical cp pg"

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
            extra_kwargs["cp_group"] = pg_collection.cp
            extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(
                pg_collection.cp
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

        if is_layer_window_attention(
            config.window_size, config.window_attn_skip_freq, layer_number
        ):
            # Check version
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "sliding window attention."
            )
            extra_kwargs["window_size"] = config.window_size

        if is_te_min_version("1.9.0"):
            # TE 1.10.0 introduces the ability to set the different k and v channels
            kv_channels = (
                (k_channels, v_channels)
                if k_channels is not None and v_channels is not None
                else self.config.kv_channels
            )
            extra_kwargs["softmax_scale"] = softmax_scale
        else:
            kv_channels = self.config.kv_channels

        if self.config.softmax_type != "vanilla":
            assert is_te_min_version("2.8.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 2.8.0 to support"
                "`softmax_type`."
            )
            extra_kwargs["softmax_type"] = self.config.softmax_type

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
            tp_group=pg_collection.tp,
            layer_number=layer_number,
            **extra_kwargs,
        )


try:
    # pylint: disable=unused-import
    from transformer_engine.pytorch import cpu_offload_v1 as cpu_offload
except ImportError:
    try:
        from transformer_engine.pytorch import cpu_offload
    except ImportError:
        cpu_offload = None
try:
    # pylint: disable=unused-import
    from transformer_engine.pytorch.float8_tensor import Float8Tensor
except ImportError:
    Float8Tensor = None
