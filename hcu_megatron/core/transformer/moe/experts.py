# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Some of this code was adopted from https://github.com/AMD-AGI/Primus
import torch
import torch.nn.functional as F

from typing import Optional, Tuple, Union, List
from megatron.core import tensor_parallel
from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP as MegatronCoreTEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.training import print_rank_0

from hcu_megatron.training import get_args

_TURBO_FUSED_GROUPED_GEMM_FALLBACK_WARNED = False


def is_turbo_fused_grouped_mlp_supported() -> Tuple[bool, Optional[str]]:
    from hcu_megatron.core.extensions.primus_turbo import (
        PrimusTurboLowPrecisionGlobalStateManager,
    )

    if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
        quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
        if not (quant_config.current_scaling() or quant_config.mxfp8_scaling()):
            return False, "turbo_fused_grouped_gemm only supports current scaling or mxfp8 scaling currently"
    elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
        quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
        if not quant_config.mxfp4_scaling():
            return False, "turbo_fused_grouped_gemm only supports mxfp4 scaling currently"
    else:
        return False, "`turbo_fused_grouped_gemm` requires Turbo FP8 (current or mxfp8 scaling) or Turbo FP4."

    return True, None


def te_grouped_mlp_sharded_state_dict_wrapper(original_func):
    """Restore logical local expert count while constructing a dist checkpoint."""
    from functools import wraps
    import re

    @wraps(original_func)
    def wrapper(self, prefix='', sharded_offsets=(), metadata=None):
        num_local_master = getattr(self, 'num_local_master_experts', None)
        if num_local_master is None or self.num_local_experts <= num_local_master:
            return original_func(
                self, prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
            )

        real_num_local_experts = self.num_local_experts
        self.num_local_experts = num_local_master
        try:
            state_dict = original_func(
                self, prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
            )
        finally:
            self.num_local_experts = real_num_local_experts

        filtered = {}
        for key, value in state_dict.items():
            match = re.search(r'(?:weight|bias)(\d+)', key)
            if match and int(match.group(1)) >= num_local_master:
                continue
            filtered[key] = value
        return filtered

    return wrapper


class TEGroupedMLP():
    def forward(
        self,
        permuted_local_hidden_states,
        tokens_per_expert,
        permuted_probs,
    ):
        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            # Whether activation function is interleaved GLU
            with_glu_interleaving = (
                self.config.gated_linear_unit
                and self.config.moe_mlp_glu_interleave_size is not None
            )
            if self.config.use_te_activation_func:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                if with_glu_interleaving:
                    intermediate_parallel = self._remove_glu_interleaving(
                        intermediate_parallel, self.config.moe_mlp_glu_interleave_size
                    )
                intermediate_parallel = self.activation_func(intermediate_parallel)
                if permuted_probs is not None:
                    original_dtype = intermediate_parallel.dtype
                    intermediate_parallel = intermediate_parallel * permuted_probs
                    intermediate_parallel = intermediate_parallel.to(original_dtype)
            elif self.config.bias_activation_fusion and not with_glu_interleaving:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu in TEGroupedMLP."
                    )
            elif (
                self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu
            ):
                assert (
                    bias_parallel is None
                ), "Bias is not supported with fused weighted squared relu."
                intermediate_parallel = weighted_squared_relu_impl(
                    intermediate_parallel, permuted_probs
                )
            else:
                from hcu_megatron.core.fusions.fused_bias_gelu import fused_bias_gelu

                intermediate_parallel = fused_bias_gelu(self, intermediate_parallel, permuted_probs)

            return intermediate_parallel


class PrimusGroupedMLP(MegatronCoreTEGroupedMLP):
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: GroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ):
        args = get_args()

        super().__init__(
            num_local_experts,
            config,
            submodules,
            pg_collection,
            name=name,
        )

        self.use_turbo_fused_act_with_probs = args.use_turbo_fused_act_with_probs
        self.moe_router_padding_for_quantization = args.moe_router_padding_for_quantization

        self.turbo_fused_grouped_gemm = getattr(args, "use_turbo_op_fuser", False)
        if self.turbo_fused_grouped_gemm:
            self._check_turbo_fused_grouped_gemm_contract()

    def _check_turbo_fused_grouped_gemm_contract(self) -> None:
        """Fail early on configs the fused grouped-MLP op does not implement."""
        from hcu_megatron.core.extensions.primus_turbo import (
            PrimusTurboGroupedLinear,
        )

        assert isinstance(self.linear_fc1, PrimusTurboGroupedLinear) and isinstance(
            self.linear_fc2, PrimusTurboGroupedLinear
        ), "`turbo_fused_grouped_gemm` requires PrimusTurboGroupedLinear (enable `use_turbo_grouped_gemm`)."
        assert (
            not self.config.add_bias_linear
        ), "`turbo_fused_grouped_gemm` does not support `add_bias_linear`."
        assert (
            self.config.glu_linear_offset == 0.0
        ), "`turbo_fused_grouped_gemm` does not support a non-zero glu_linear_offset."
        assert (
            not self.config.activation_func_fp8_input_store
        ), "`turbo_fused_grouped_gemm` does not support activation_func_fp8_input_store."
        assert not self.config.moe_apply_probs_on_input, (
            "`turbo_fused_grouped_gemm` applies probs inside the fused GLU; "
            "`moe_apply_probs_on_input` (scale before fc1) is a different contract."
        )
        assert (
            not self.offload_expert_fc1 and not self.offload_moe_act
        ), "`turbo_fused_grouped_gemm` does not support expert fc1 / moe-act offload."

    def _use_explicit_quantization_padding(self) -> bool:
        """Whether this forward must pad expert groups before quantization."""
        if not (self.config.fp8 or self.config.fp4):
            return False
        if self.moe_router_padding_for_quantization:
            return False
        return True

    def bias_act_func_with_mask(
        self,
        intermediate_parallel: torch.Tensor,
        bias_parallel: torch.Tensor,
        permuted_probs: torch.Tensor,
        tokens_per_experts: Union[torch.Tensor, None] = None,
    ):

        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            # Whether activation function is interleaved GLU
            with_glu_interleaving = (
                self.config.gated_linear_unit
                and self.config.moe_mlp_glu_interleave_size is not None
            )
            if self.config.use_te_activation_func:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                if with_glu_interleaving:
                    intermediate_parallel = self._remove_glu_interleaving(
                        intermediate_parallel, self.config.moe_mlp_glu_interleave_size
                    )
                intermediate_parallel = self.activation_func(intermediate_parallel)
                if permuted_probs is not None:
                    original_dtype = intermediate_parallel.dtype
                    intermediate_parallel = intermediate_parallel * permuted_probs
                    intermediate_parallel = intermediate_parallel.to(original_dtype)
            elif self.config.bias_activation_fusion and not with_glu_interleaving:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu in TEGroupedMLP."
                    )
            elif (
                self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu
            ):
                assert (
                    bias_parallel is None
                ), "Bias is not supported with fused weighted squared relu."
                intermediate_parallel = weighted_squared_relu_impl(
                    intermediate_parallel, permuted_probs
                )
            else:
                from hcu_megatron.core.fusions.fused_bias_gelu import fused_bias_gelu

                intermediate_parallel = fused_bias_gelu(self, intermediate_parallel, permuted_probs)

            return intermediate_parallel

        if self.use_turbo_fused_act_with_probs:
            from hcu_megatron.core.extensions.primus_turbo import (
                fused_bias_act_with_probs,
            )

            assert (
                tokens_per_experts is not None
            ), "tokens_per_experts is required when `use_turbo_fused_act_with_probs` is True."

            if self.activation_func == F.silu and self.config.gated_linear_unit:
                activation = "silu"
            elif self.activation_func == F.gelu and self.config.gated_linear_unit:
                activation = "gelu"
            else:
                raise ValueError(
                    "Only support fusion of swiglu and gelu in PrimusGroupedMLP when `use_turbo_fused_act_with_probs` is True."
                )

            # `forward()` unsqueeze(-1)'s `permuted_probs` to [tokens, 1] so the non-fused
            # asserts ndim == 1. Squeeze back to 1D for the fused kernel only.
            probs_1d = permuted_probs.squeeze(-1) if permuted_probs.dim() == 2 else permuted_probs
            # dtype is handled inside the fused kernel
            return fused_bias_act_with_probs(
                intermediate_parallel,
                bias_parallel,
                probs_1d,
                tokens_per_experts,
                activation,
                self.config.activation_func_clamp_value,
            )
        else:
            # use the original bias_act_func from TEGroupedMLP, ignore the tokens_per_experts
            return bias_act_func(intermediate_parallel, bias_parallel, permuted_probs)

    def _forward_with_turbo_fused_grouped_mlp(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        from hcu_megatron.core.extensions.primus_turbo import (
            PrimusTurboLowPrecisionGlobalStateManager,
        )

        assert (
            tokens_per_expert.device == permuted_local_hidden_states.device
        ), "tokens_per_expert and permuted_local_hidden_states must be on the same device"

        if self.activation_func == F.silu and self.config.gated_linear_unit:
            activation = "silu"
        elif self.activation_func == F.gelu and self.config.gated_linear_unit:
            activation = "gelu"
        else:
            raise ValueError(
                "Only support fusion of silu and gelu in PrimusGroupedMLP when `turbo_fused_grouped_gemm` is True."
            )

        x, w1, fuse_w1_grad_accum = self.linear_fc1.prepare_weights(permuted_local_hidden_states)
        x, w2, fuse_w2_grad_accum = self.linear_fc2.prepare_weights(x)
        assert fuse_w1_grad_accum == fuse_w2_grad_accum, (
            "fc1 and fc2 resolved different fuse_wgrad_accum_pattern values: "
            f"fc1={fuse_w1_grad_accum!r}, fc2={fuse_w2_grad_accum!r}"
        )
        fuse_wgrad_accum_pattern = fuse_w1_grad_accum

        if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
            from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8

            mlp_op = grouped_mlp_fp8
            quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
        elif PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp4_enabled():
            from primus_turbo.pytorch.ops.grouped_mlp_fp4 import grouped_mlp_fp4

            mlp_op = grouped_mlp_fp4
            quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
        else:
            raise ValueError(
                "`turbo_fused_grouped_gemm` only available when Turbo FP8 "
                "(current or mxfp8 scaling) or Turbo FP4 is enabled."
            )

        def _turbo_fused_grouped_mlp(hidden, group_lens, probs):
            # w1/w2 stay in the closure: they may be a QuantizedTensorPair (not a
            # Tensor checkpoint arg). Weight prep stays outside the checkpoint so
            # the first-microbatch cache / de-osc marker / grad bridge run once.
            return mlp_op(
                hidden,
                w1,
                w2,
                group_lens,
                probs=probs,
                trans_w1=True,
                trans_w2=True,
                config=quant_config.data(),
                activation=activation,
                fuse_wgrad_accum_pattern=fuse_wgrad_accum_pattern,
                clamp_limit=self.config.activation_func_clamp_value,
            )

        if self.activation_recompute:
            output = tensor_parallel.checkpoint(
                _turbo_fused_grouped_mlp, False, x, tokens_per_expert, permuted_probs
            )
        else:
            output = _turbo_fused_grouped_mlp(x, tokens_per_expert, permuted_probs)

        # NOTE: None means no bias
        return output, None

    @staticmethod
    def _apply_bias(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
        if bias_parallel is None:
            return intermediate_parallel

        # NOTE: tokens_per_expert is on GPU, so we need to convert it to a list of ints.
        tokens_per_expert_cpu = tokens_per_expert.tolist()

        return MegatronCoreTEGroupedMLP._apply_bias(
            intermediate_parallel, bias_parallel, tokens_per_expert_cpu, permuted_probs
        )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        output_buffer: Optional[torch.Tensor] = None,  # unused
        grad_input_buffer: Optional[torch.Tensor] = None,  # unused
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of PrimusGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.
            output_buffer (torch.Tensor, optional): Preallocated buffer to write the fc2 output into
            (NCCL-EP zero-copy fwd combine); only the fused op-fuser path supports it.
            grad_input_buffer (torch.Tensor, optional): Preallocated buffer to write the fc1 dgrad
            into (NCCL-EP zero-copy bwd dispatch); only the fused op-fuser path supports it.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        if self.turbo_fused_grouped_gemm:
            supported, reason = is_turbo_fused_grouped_mlp_supported()
            if supported:
                return self._forward_with_turbo_fused_grouped_mlp(
                    permuted_local_hidden_states, tokens_per_expert, permuted_probs
                )
            global _TURBO_FUSED_GROUPED_GEMM_FALLBACK_WARNED
            if not _TURBO_FUSED_GROUPED_GEMM_FALLBACK_WARNED:
                print_rank_0(
                    f"Warning: `turbo_fused_grouped_gemm` is not used because {reason}. "
                    "Falling back to unfused PrimusGroupedMLP."
                )
                _TURBO_FUSED_GROUPED_GEMM_FALLBACK_WARNED = True

        use_explicit_quantization_padding = self._use_explicit_quantization_padding()
        if use_explicit_quantization_padding:
            # NOTE: When moe_router_padding_for_quantization is true the token is padded. So we can skip the padding here to reduce cpu sync.
            tokens_per_expert_cpu: list[int] = tokens_per_expert.tolist()
            actual_tokens_per_expert_cpu: list[int] = tokens_per_expert_cpu
            permuted_local_hidden_states, tokens_per_expert_cpu = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert_cpu
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert_cpu
            )
            tokens_per_expert = torch.tensor(
                tokens_per_expert_cpu, device=permuted_local_hidden_states.device
            )
        else:
            permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        expert_fc1_manager = off_interface(
            self.offload_expert_fc1, permuted_local_hidden_states, "expert_fc1"
        )
        with expert_fc1_manager as permuted_local_hidden_states:
            fc1_output, bias_parallel = apply_module(self.linear_fc1)(
                permuted_local_hidden_states, tokens_per_expert
            )
        fc1_output = expert_fc1_manager.group_offload(
            fc1_output,
            forced_released_tensors=[permuted_local_hidden_states],
            delay_offload=self.config.delay_offload_until_cuda_graph,
        )

        moe_act_manager = off_interface(self.offload_moe_act, fc1_output, "moe_act")
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with moe_act_manager as fc1_output:
                # NOTE: use the bias_act_func_with_mask instead of the bias_act_func to reduce the extra compute when stage of `sync_free_moe` is 3.
                bias_act_output = self.activation_checkpoint.checkpoint(
                    self.bias_act_func_with_mask, fc1_output, bias_parallel, permuted_probs, tokens_per_expert
                )
        else:
            with moe_act_manager as fc1_output:
                bias_act_output = self.bias_act_func_with_mask(fc1_output, bias_parallel, permuted_probs, tokens_per_expert)
        output, output_bias = apply_module(self.linear_fc2)(bias_act_output, tokens_per_expert)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        # Delay the offload of the moe act until after the linear_fc2 has been computed
        # to make sure the fc1_output is reloaded to GPU before recomputing moe_act.
        output = moe_act_manager.group_offload(
            output,
            forced_released_tensors=[fc1_output],
            delay_offload=self.config.delay_offload_until_cuda_graph,
        )
        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)

        # upad and concat the output
        if use_explicit_quantization_padding:
            output = self.quantization_unpadding(output, actual_tokens_per_expert_cpu)

        output_bias = None

        return output, output_bias


# Bridge TE fused wgrad buffers and differentiable echo weight transport.
#
# Like Loong-Megatron's HybridEPExpertDispatch.backward, replica gradients are
# explicitly added to home main_grad: MCore's DDP hook skips ordinary gradients
# once the home fused GEMM has set grad_added_to_main_grad.
#
# Synchronous AlltoAll/DeepEP and overlap weight backward run on the compute
# stream. Overlap registers its autograd exchange on that stream too, so home
# and replica additions are ordered without a global pending-gradient queue.
class _EchoHomeWeightGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        if getattr(weight, 'main_grad', None) is None:
            raise RuntimeError('Fused echo dispatch requires a home weight main_grad buffer')
        ctx.weight = weight
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.weight
        weight.main_grad.add_(grad_output)
        weight.grad_added_to_main_grad = True
        # Trigger the leaf/DDP hook, which must not accumulate this branch again.
        return torch.zeros_like(weight)


class _EchoReplicaWeightGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, main_grad):
        # Each in-flight forward owns its buffer. Do not reuse a module slot's
        # buffer: pipeline microbatches can replace slots before backward.
        ctx.main_grad = main_grad
        ctx.weight_dtype = weight.dtype
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx, dummy_grad):
        # TE has already computed wgrad into this buffer. Its autograd output
        # is only a scheduling dummy, and can contain uninitialized values.
        return ctx.main_grad.to(dtype=ctx.weight_dtype), None


def echo_dispatch_home_weights(weights, fused):
    """Wrap only the weight-dispatch branch; TE keeps original home Parameters."""
    if not fused:
        return weights
    return [_EchoHomeWeightGrad.apply(weight) for weight in weights]


def echo_replica_weight(weight, grad_dtype):
    """Expose a uniform-dtype wgrad output to TE and return it to transport."""
    main_grad = torch.zeros_like(weight, dtype=grad_dtype)
    replica = _EchoReplicaWeightGrad.apply(weight, main_grad)
    replica.main_grad = main_grad
    replica.grad_added_to_main_grad = False
    return replica


class DummyFunction(torch.autograd.Function):
    """Autograd placeholder for empty echo expert weight tensors.

    When a spare-expert slot receives no tokens on this rank, its weight
    tensor is empty (numel==0). We still need to place *some* tensor at
    `weight{i}` so TE's GroupedLinear doesn't crash. DummyFunction wraps an
    empty tensor into a graph node whose backward returns zero gradient.
    """

    @staticmethod
    def forward(ctx, x, weight_shape):
        ctx.input = x
        dummy_weight = torch.empty(weight_shape, dtype=x.dtype, device=x.device)
        return dummy_weight

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(ctx.input), None


def _grouped_mlp_free_expert_parameters(self, expert_indices: List[int]):
    """Free the nn.Parameter slots for the given expert indices on this rank.

    Called once at init (or before first echo forward) so the runtime can
    later reinstall dispatched weights via `set_expert_weights` without
    conflicting with autograd registration.
    """
    to_free_weight_names = [f'weight{i}' for i in expert_indices]
    for module in [self.linear_fc1, self.linear_fc2]:
        for name, _ in list(module.named_parameters()):
            if name in to_free_weight_names:
                delattr(module, name)
                module._parameters.pop(name, None)


def _grouped_mlp_get_expert_weights(
    self, module: str, expert_indices: List[int]
) -> List[torch.Tensor]:
    """Return per-expert weight tensors for the given indices."""
    assert module in ("fc1", "fc2"), f"Invalid module: {module}"
    expert_layer = self.linear_fc1 if module == "fc1" else self.linear_fc2
    return [getattr(expert_layer, f"weight{i}") for i in expert_indices]


def _grouped_mlp_set_expert_weights(
    self,
    module: str,
    expert_weights: List[torch.Tensor],
    expert_indices: List[int],
):
    """Install (in-place) dispatched weights onto per-expert slots.

    Uses setattr rather than nn.Parameter registration so echo can rewrite
    weights each forward without triggering optimizer state confusion.

    """
    assert module in ("fc1", "fc2"), f"Invalid module: {module}"
    expert_layer = self.linear_fc1 if module == "fc1" else self.linear_fc2
    fused = self.config.gradient_accumulation_fusion
    if fused:

        grad_dtype = getattr(expert_layer, "weight0").main_grad.dtype
    shape_attr = f"{module}_weight_shape"
    weight_shape = getattr(self, shape_attr, None)
    if weight_shape is None:
        weight_shape = getattr(expert_layer, "weight0").shape
        setattr(self, shape_attr, weight_shape)
    # Persistent per-slot main_grad sinks, allocated once and reused every
    # forward instead of a fresh torch.zeros per slot per call. Contents are
    # never read (echo slots are not optimizer Parameters).
    sinks = getattr(expert_layer, "_echo_main_grad_sinks", None)
    if sinks is None:
        sinks = {}
        setattr(expert_layer, "_echo_main_grad_sinks", sinks)
    for i, expert_index in enumerate(expert_indices):
        w = expert_weights[i]
        if w.numel() == 0:
            w = DummyFunction.apply(w, weight_shape)
        if isinstance(w, torch.nn.Parameter):
            # Restoring a home slot must never mark its DDP buffer as written.
            setattr(expert_layer, f"weight{expert_index}", w)
            continue
        if fused:
            w = echo_replica_weight(w, grad_dtype)
            setattr(expert_layer, f"weight{expert_index}", w)
            continue
        # Attach TE-compatible attrs so backward doesn't crash on missing
        # main_grad. Echo slots don't accumulate gradients across steps.
        if not hasattr(w, "main_grad"):
            sink = sinks.get(expert_index)
            if sink is None or sink.dtype != w.dtype or sink.device != w.device:
                sink = torch.zeros(weight_shape, dtype=w.dtype, device=w.device)
                sinks[expert_index] = sink
            w.main_grad = sink
        w.grad_added_to_main_grad = True
        setattr(expert_layer, f"weight{expert_index}", w)


class TEGroupedMLP():
    def forward(
        self,
        permuted_local_hidden_states,
        tokens_per_expert,
        permuted_probs,
    ):
        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            # Whether activation function is interleaved GLU
            with_glu_interleaving = (
                self.config.gated_linear_unit
                and self.config.moe_mlp_glu_interleave_size is not None
            )
            if self.config.use_te_activation_func:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                if with_glu_interleaving:
                    intermediate_parallel = self._remove_glu_interleaving(
                        intermediate_parallel, self.config.moe_mlp_glu_interleave_size
                    )
                intermediate_parallel = self.activation_func(intermediate_parallel)
                if permuted_probs is not None:
                    original_dtype = intermediate_parallel.dtype
                    intermediate_parallel = intermediate_parallel * permuted_probs
                    intermediate_parallel = intermediate_parallel.to(original_dtype)
            elif self.config.bias_activation_fusion and not with_glu_interleaving:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu in TEGroupedMLP."
                    )
            elif (
                self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu
            ):
                assert (
                    bias_parallel is None
                ), "Bias is not supported with fused weighted squared relu."
                intermediate_parallel = weighted_squared_relu_impl(
                    intermediate_parallel, permuted_probs
                )
            else:
                from hcu_megatron.core.fusions.fused_bias_gelu import fused_bias_gelu

                intermediate_parallel = fused_bias_gelu(self, intermediate_parallel, permuted_probs)

            return intermediate_parallel


class PrimusTurboGroupedMLP(MegatronCoreTEGroupedMLP):
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: GroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        args = get_args()

        super().__init__(
            num_local_experts,
            config,
            submodules,
            pg_collection,
        )

        self.use_primus_fused_act_with_probs = args.use_primus_fused_act_with_probs

    def bias_act_func_with_mask(
        self,
        intermediate_parallel: torch.Tensor,
        bias_parallel: torch.Tensor,
        permuted_probs: torch.Tensor,
        tokens_per_experts: Union[torch.Tensor, None] = None,
    ):
        if self.use_primus_fused_act_with_probs:
            from hcu_megatron.core.extensions.primus_turbo import (
                fused_bias_act_with_probs,
            )

            assert (
                tokens_per_experts is not None
            ), "tokens_per_experts is required when `use_primus_fused_act_with_probs` is True."

            if self.activation_func == F.silu and self.config.gated_linear_unit:
                activation = "silu"
            elif self.activation_func == F.gelu and self.config.gated_linear_unit:
                activation = "gelu"
            else:
                raise ValueError(
                    "Only support fusion of swiglu and gelu in PrimusGroupedMLP when `use_turbo_fused_act_with_probs` is True."
                )

            # `forward()` unsqueeze(-1)'s `permuted_probs` to [tokens, 1] so the non-fused
            # asserts ndim == 1. Squeeze back to 1D for the fused kernel only.
            probs_1d = permuted_probs.squeeze(-1) if permuted_probs.dim() == 2 else permuted_probs
            # dtype is handled inside the fused kernel
            return fused_bias_act_with_probs(
                intermediate_parallel, bias_parallel, probs_1d, tokens_per_experts, activation
            )
        else:
            # use the original bias_act_func from TEGroupedMLP, ignore the tokens_per_experts
            return self.bias_act_func(intermediate_parallel, bias_parallel, permuted_probs)

    @staticmethod
    def _apply_bias(intermediate_parallel, bias_parallel, tokens_per_expert, permuted_probs):
        if bias_parallel is None:
            return intermediate_parallel

        # NOTE: tokens_per_expert is on GPU, so we need to convert it to a list of ints.
        tokens_per_expert_cpu = tokens_per_expert.tolist()

        return MegatronCoreTEGroupedMLP._apply_bias(
            intermediate_parallel, bias_parallel, tokens_per_expert_cpu, permuted_probs
        )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of PrimusGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        if self.config.fp8 or self.config.fp4:
            # NOTE: When moe_router_padding_for_quantization is true the token is padded. So we can skip the padding here to reduce cpu sync.
            tokens_per_expert_cpu: list[int] = tokens_per_expert.tolist()
            actual_tokens_per_expert_cpu: list[int] = tokens_per_expert_cpu
            permuted_local_hidden_states, tokens_per_expert_cpu = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert_cpu
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert_cpu
            )
            tokens_per_expert = torch.tensor(
                tokens_per_expert_cpu, device=permuted_local_hidden_states.device
            )
        else:
            permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        with off_interface(
            self.offload_expert_fc1, permuted_local_hidden_states, "expert_fc1"
        ) as permuted_local_hidden_states:
            fc1_output, bias_parallel = apply_module(self.linear_fc1)(
                permuted_local_hidden_states, tokens_per_expert
            )
        if self.offload_expert_fc1:
            fc1_output = off_interface.group_commit(
                fc1_output,
                name="expert_fc1",
                forced_released_tensors=[permuted_local_hidden_states],
            )

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                # NOTE: use the bias_act_func_with_mask instead of the bias_act_func to reduce the extra compute when stage of `sync_free_moe` is 3.
                bias_act_output = self.activation_checkpoint.checkpoint(
                    self.bias_act_func_with_mask, fc1_output, bias_parallel, permuted_probs, tokens_per_expert
                )
        else:
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                bias_act_output = self.bias_act_func(fc1_output, bias_parallel, permuted_probs)
        output, output_bias = apply_module(self.linear_fc2)(bias_act_output, tokens_per_expert)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        # Delay the offload of the moe act until after the linear_fc2 has been computed
        # to make sure the fc1_output is reloaded to GPU before recomputing moe_act.
        if self.offload_moe_act:
            output = off_interface.group_commit(output, name="moe_act", forced_released_tensors=[fc1_output])
        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)

        # upad and concat the output
        if self.config.fp8 or self.config.fp4:
            output = self.quantization_unpadding(output, actual_tokens_per_expert_cpu)

        output_bias = None

        return output, output_bias


from .experts_overlap import _grouped_mlp_forward_with_dispatch_overlap

__all__ = [
    'DummyFunction',
    '_grouped_mlp_free_expert_parameters',
    '_grouped_mlp_get_expert_weights',
    '_grouped_mlp_set_expert_weights',
    '_grouped_mlp_forward_with_dispatch_overlap',
]
