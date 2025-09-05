import torch
import torch.nn.functional as F

from typing import Optional, Tuple
from megatron.core import tensor_parallel
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.transformer.moe import grouped_gemm_util as gg


class GroupedMLP():
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
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            from dcu_megatron.core.pipeline_parallel.cpu_offload import (
                get_offload_context,
                offload_checker_ctx,
            )

            if self.activation_recompute:
                with get_offload_context(self.config), offload_checker_ctx(
                    self.config, lambda x: True
                ):
                    fc1_output = gg.ops.gmm(
                        permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
                    )
            else:
                fc1_output = gg.ops.gmm(
                    permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
                )

            if self.activation_recompute:
                with get_offload_context(self.config), offload_checker_ctx(
                    self.config, lambda x: True
                ):
                    intermediate_parallel = self.activation_checkpoint.checkpoint(
                        self.activation_func_with_probs, fc1_output, permuted_probs.unsqueeze(-1)
                    )
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                h = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, h, permuted_probs.unsqueeze(-1)
                )
                fc2_output = torch.matmul(h, w2)
                self.activation_checkpoint.discard_output_and_register_recompute(fc2_output)
            else:
                h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
                fc2_output = torch.matmul(h, w2)

        return fc2_output, None

    def backward_dw(self):
        """Performs backward pass for weight gradients in Experts.
        Empty implementation for compatibility with SequentialMLP and TEGroupedMLP.
        """
        pass

    
class TEGroupedMLP():
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
        tokens_per_expert = tokens_per_expert.tolist()
        if self.config.fp8:
            actual_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.fp8_padding(
                permuted_local_hidden_states, tokens_per_expert
            )
            permuted_probs, _ = self.fp8_padding(
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

        from dcu_megatron.core.pipeline_parallel.cpu_offload import (
            get_offload_context,
            offload_checker_ctx,
            TECpuOffloadContextManager,
        )

        if self.activation_recompute:
            with get_offload_context(self.config), offload_checker_ctx(self.config, lambda x: True), \
                 TECpuOffloadContextManager(self.config.offload_moe_mlp_input):
                intermediate_parallel, bias_parallel = self.linear_fc1(
                    permuted_local_hidden_states, tokens_per_expert
                )
        else:
            intermediate_parallel, bias_parallel = self.linear_fc1(
                permuted_local_hidden_states, tokens_per_expert
            )

        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            if self.config.bias_activation_fusion:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        permuted_probs,
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of swiglu in TEGroupedMLP.")
            else:
                from dcu_megatron.core.fusions.fused_bias_gelu import fused_bias_gelu
                intermediate_parallel = fused_bias_gelu(
                    bias_parallel,
                    intermediate_parallel,
                    tokens_per_expert,
                    permuted_probs,
                    self.config.gated_linear_unit,
                    self.activation_func,
                )
            return intermediate_parallel

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()

            with get_offload_context(self.config), offload_checker_ctx(self.config, lambda x: True):
                intermediate_parallel = self.activation_checkpoint.checkpoint(
                    bias_act_func, intermediate_parallel, bias_parallel, permuted_probs
                )

            output, output_bias = self.linear_fc2(intermediate_parallel, tokens_per_expert)
            self.activation_checkpoint.discard_output_and_register_recompute(output)
        else:
            intermediate_parallel = bias_act_func(
                intermediate_parallel, bias_parallel, permuted_probs
            )
            output, output_bias = self.linear_fc2(intermediate_parallel, tokens_per_expert)

        # upad and concat the output
        if self.config.fp8:
            output = self.fp8_unpadding(output, actual_tokens_per_expert)

        return output, output_bias

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


class SequentialMLP():
    def backward_dw(self):
        """Backward pass for weight gradients in SequentialMLP."""
        try:
            for expert in self.local_experts:
                expert.backward_dw()
        except Exception as e:
            raise Exception(
                f"Unknown error occurred during SequentialMLP backward_dw() execution: {str(e)}"
            )
