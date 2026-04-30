import torch
import torch.nn.functional as F

from typing import Optional, Tuple
from megatron.core import tensor_parallel
from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import get_args
from megatron.core.typed_torch import apply_module


class TEGroupedMLP():
    def bias_act_func(self, intermediate_parallel, bias_parallel, permuted_probs):
        """
        Applies bias and activation function to the output of linear_fc1.
        """
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if permuted_probs is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * permuted_probs
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
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
                raise ValueError("Only support fusion of swiglu and quick_gelu in TEGroupedMLP.")
        elif (
            self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu
        ):
            assert bias_parallel is None, "Bias is not supported with fused weighted squared relu."
            intermediate_parallel = weighted_squared_relu_impl(
                intermediate_parallel, permuted_probs
            )
        else:
            from dcu_megatron.core.fusions.fused_bias_gelu import fused_bias_gelu

            intermediate_parallel = fused_bias_gelu(self, intermediate_parallel, permuted_probs)

        return intermediate_parallel

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        tokens_per_expert: list[int] = tokens_per_expert.tolist()
        if self.config.fp8 or self.config.fp4:
            actual_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.quantization_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.quantization_padding(
                permuted_probs.unsqueeze(-1), actual_tokens_per_expert
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
                bias_act_output = self.activation_checkpoint.checkpoint(
                    self.bias_act_func, fc1_output, bias_parallel, permuted_probs
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
            output = off_interface.group_commit(
                output, name="moe_act", forced_released_tensors=[fc1_output]
            )
        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)

        # upad and concat the output
        if self.config.fp8 or self.config.fp4:
            output = self.quantization_unpadding(output, actual_tokens_per_expert)

        output_bias = None

        return output, output_bias


class PrimusTurboGroupedMLP():
    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        import primus_turbo.pytorch as pt

        args = get_args()

        super().__init__(
            num_local_experts,
            config,
            pg_collection,
        )
        self.use_primus_fused_act_with_probs = args.use_primus_fused_act_with_probs

        self.grouped_gemm = pt.ops.grouped_gemm

        if self.use_primus_fused_act_with_probs:
            assert self.config.gated_linear_unit, "turbo_fused_act_with_probs only support with GLU."

            if self.config.activation_func == F.silu:
                turbo_fused_act_with_probs = pt.ops.swiglu_with_probs
            elif self.config.activation_func == F.gelu:
                turbo_fused_act_with_probs = pt.ops.geglu_with_probs
            else:
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            def _activation_func_with_probs(x, probs, tokens_per_experts):
                assert x.ndim == 2
                assert probs.ndim == 1
                num_tokens = x.shape[0]
                row_mask = pt.ops.tokens_per_expert_to_mask(tokens_per_experts, num_tokens)
                return turbo_fused_act_with_probs(x, probs, row_mask)

            self.activation_func_with_probs = _activation_func_with_probs

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the GroupedMLP."""
        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        gemm_kargs = [dict(), dict()]

        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            tokens_per_expert = tokens_per_expert.to(w1.device)
            assert w1.is_contiguous(), "w1 must be contiguous"
            assert w2.is_contiguous(), "w2 must be contiguous"

            fc1_output = self.grouped_gemm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False, **(gemm_kargs[0])
            )
            if self.activation_recompute:
                if self.use_primus_fused_act_with_probs:
                    intermediate_parallel = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs,
                        fc1_output,
                        permuted_probs,
                        tokens_per_expert,
                    )
                else:
                    intermediate_parallel = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs, fc1_output, permuted_probs.unsqueeze(-1)
                    )

                fc2_output = self.grouped_gemm(
                    intermediate_parallel, w2, tokens_per_expert, trans_b=False, **(gemm_kargs[1])
                )
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                if self.use_primus_fused_act_with_probs:
                    intermediate_parallel = self.activation_func_with_probs(
                        fc1_output, permuted_probs, tokens_per_expert
                    )
                else:
                    intermediate_parallel = self.activation_func_with_probs(
                        fc1_output, permuted_probs.unsqueeze(-1)
                    )
                fc2_output = self.grouped_gemm(
                    intermediate_parallel, w2, tokens_per_expert, trans_b=False, **(gemm_kargs[1])
                )
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0
            # Make sure params of experts still have gradients even given zero tokens.
            assert (
                not self.patch_zero_bubble and not self.patch_primus_pipeline
            ), "Zero bubble or primus pipeline not support torch.matmul backend yet"
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                if self.use_primus_fused_act_with_probs:
                    h = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs, h, permuted_probs, tokens_per_expert
                    )
                else:
                    h = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs, h, permuted_probs.unsqueeze(-1)
                    )
                fc2_output = torch.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                if self.use_primus_fused_act_with_probs:
                    h = self.activation_func_with_probs(h, permuted_probs, tokens_per_expert)
                else:
                    h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
                fc2_output = torch.matmul(h, w2)

        return fc2_output, None
