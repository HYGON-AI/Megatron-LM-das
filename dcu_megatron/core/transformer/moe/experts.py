import torch
import contextlib
import torch.nn.functional as F

from typing import Optional, Tuple
from functools import wraps
from megatron.core import tensor_parallel
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.transformer.moe import grouped_gemm_util as gg

from dcu_megatron.core.pipeline_parallel import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)


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

            from dcu_megatron.core.transformer.cpu_offload import (
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


def te_grouped_mlp_init_wrapper(te_grouped_mlp_init_func):
    @wraps(te_grouped_mlp_init_func)
    def wrapper(
        self,
        num_local_experts,
        config,
        submodules,
        pg_collection,
    ):
        te_grouped_mlp_init_func(
            self,
            num_local_experts=num_local_experts,
            config=config,
            submodules=submodules,
            pg_collection=pg_collection,
        )

        self.offload_expert_fc1 = (
            config.fine_grained_activation_offloading
            and "expert_fc1" in config.offload_modules
        )

        self.offload_moe_act = (
            config.fine_grained_activation_offloading
            and "moe_act" in config.offload_modules
        )

        self.offload_expert_fc2 = (
            config.fine_grained_activation_offloading
            and "expert_fc2" in config.offload_modules
        )

        # This is to avoid the CPU overhead of multiple d2h copies
        if self.offload_expert_fc1 and not self.config.fp8:
            from dcu_megatron.core.extensions.transformer_engine import set_save_original_input
            # set_save_original_input(self.linear_fc1)

    return wrapper


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

        if self.offload_expert_fc1:
            permuted_local_hidden_states = fine_grained_offloading_group_start(
                permuted_local_hidden_states, name="expert_fc1"
            )
        with get_fine_grained_offloading_context(self.offload_expert_fc1):
            fc1_output, bias_parallel = self.linear_fc1(
                permuted_local_hidden_states, tokens_per_expert
            )
        if self.offload_expert_fc1:
            fc1_output, bias_parallel = fine_grained_offloading_group_commit(
                fc1_output,
                bias_parallel,
                name="expert_fc1",
                forced_released_tensors=[permuted_local_hidden_states]
            )

        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
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
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu in TEGroupedMLP."
                    )
            elif (
                self.activation_func == squared_relu and self.config.use_fused_weighted_squared_relu
            ):
                assert bias_parallel is None
                intermediate_parallel = weighted_squared_relu_impl(
                    intermediate_parallel, permuted_probs
                )
            else:
                from dcu_megatron.core.fusions.fused_bias_gelu import fused_bias_gelu
                intermediate_parallel = fused_bias_gelu(
                    intermediate_parallel,
                    permuted_probs,
                    self.config.gated_linear_unit,
                    self.activation_func,
                )

            return intermediate_parallel

        if self.offload_moe_act:
            fc1_output = fine_grained_offloading_group_start(fc1_output, name="moe_act")

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with get_fine_grained_offloading_context(self.offload_moe_act):
                bias_act_output = self.activation_checkpoint.checkpoint(
                    bias_act_func, fc1_output, bias_parallel, permuted_probs
                )
        else:
            with get_fine_grained_offloading_context(self.offload_moe_act):
                bias_act_output = bias_act_func(fc1_output, bias_parallel, permuted_probs)

        if self.offload_moe_act:
            (bias_act_output,) = fine_grained_offloading_group_commit(
                bias_act_output, name="moe_act", forced_released_tensors=[fc1_output]
            )

        if self.offload_expert_fc2:
            bias_act_output = fine_grained_offloading_group_start(
                bias_act_output, name="expert_fc2"
            )
        with get_fine_grained_offloading_context(self.offload_expert_fc2):
            output, output_bias = self.linear_fc2(bias_act_output, tokens_per_expert)
        if self.offload_expert_fc2:
            output, output_bias = fine_grained_offloading_group_commit(
                output, output_bias,
                name="expert_fc2",
                forced_released_tensors=[bias_act_output]
            )

        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        # upad and concat the output
        if self.config.fp8:
            output = self.fp8_unpadding(output, actual_tokens_per_expert)

        output = self._apply_bias(output, output_bias, tokens_per_expert, permuted_probs)
        output_bias = None

        return output, output_bias
