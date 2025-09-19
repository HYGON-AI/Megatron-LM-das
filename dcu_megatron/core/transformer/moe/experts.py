import torch
import torch.nn.functional as F

from typing import Optional, Tuple
from functools import wraps
from megatron.core import tensor_parallel
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
from megatron.core.transformer.moe import grouped_gemm_util as gg

from dcu_megatron.core.pipeline_parallel.cpu_offload import (
    PipelineOffloadManager,
    group_prefetch_offload_start,
    group_prefetch_offload_commit,
)


class GroupedMLP():
    def backward_dw(self):
        """Performs backward pass for weight gradients in Experts.
        Empty implementation for compatibility with SequentialMLP and TEGroupedMLP.
        """
        pass


def te_grouped_mlp_init_wrapper(te_grouped_mlp_init_func):
    @wraps(te_grouped_mlp_init_func)
    def wrapper(
        self,
        num_local_experts,
        config,
        submodules,
        model_comm_pgs,
    ):
        te_grouped_mlp_init_func(
            self,
            num_local_experts=num_local_experts,
            config=config,
            submodules=submodules,
            model_comm_pgs=model_comm_pgs,
        )

        self.offload_router_fc1 = (
            config.offload_activation
            and "router_fc1" in config.offload_modules
        )

        self.offload_router_fc2 = (
            config.offload_activation
            and "router_fc2" in config.offload_modules
        )

    return wrapper


class TEGroupedMLP():
    def _offload_router_fc1_forward(
        self,
        permuted_local_hidden_states,
        tokens_per_expert,
    ):
        """Forward method with router fc1 activation offloading."""
        if not permuted_local_hidden_states.is_contiguous():
            permuted_local_hidden_states = permuted_local_hidden_states.contiguous()

        permuted_local_hidden_states = group_prefetch_offload_start(permuted_local_hidden_states)
        permuted_local_hidden_states.offloading_activation = True

        with PipelineOffloadManager.get_instance():
            intermediate_parallel, bias_parallel = self.linear_fc1(
                permuted_local_hidden_states, tokens_per_expert
            )

        intermediate_parallel, bias_parallel = group_prefetch_offload_commit(
            intermediate_parallel,
            bias_parallel,
            release_tensors=[permuted_local_hidden_states]
        )

        return intermediate_parallel, bias_parallel

    def _offload_router_fc2_forward(
        self,
        intermediate_parallel,
        tokens_per_expert,
    ):
        """Forward method with router fc2 activation offloading."""
        if not intermediate_parallel.is_contiguous():
            intermediate_parallel = intermediate_parallel.contiguous()

        intermediate_parallel = group_prefetch_offload_start(intermediate_parallel)

        intermediate_parallel.offloading_activation = True

        with PipelineOffloadManager.get_instance():
            output, output_bias = self.linear_fc2(
                intermediate_parallel, tokens_per_expert
            )

        def call_back():
            cur_stream = torch.cuda.current_stream()
            intermediate_parallel.record_stream(cur_stream)
            intermediate_parallel.untyped_storage().resize_(0)

        output, output_bias = group_prefetch_offload_commit(
            output,
            output_bias,
            offloaded_call_back=call_back
        )
        return output, output_bias

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

        if self.offload_router_fc1:
            intermediate_parallel, bias_parallel = self._offload_router_fc1_forward(
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

            intermediate_parallel = self.activation_checkpoint.checkpoint(
                bias_act_func, intermediate_parallel, bias_parallel, permuted_probs
            )
            if self.offload_router_fc2:
                output, output_bias = self._offload_router_fc2_forward(intermediate_parallel, tokens_per_expert)
            else:
                output, output_bias = self.linear_fc2(intermediate_parallel, tokens_per_expert)

            self.activation_checkpoint.discard_output_and_register_recompute(output)
        else:
            intermediate_parallel = bias_act_func(
                intermediate_parallel, bias_parallel, permuted_probs
            )
            if self.offload_router_fc2:
                output, output_bias = self._offload_router_fc2_forward(intermediate_parallel, tokens_per_expert)
            else:
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
