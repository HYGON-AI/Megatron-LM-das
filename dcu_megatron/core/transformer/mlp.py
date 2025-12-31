import torch
import torch.nn.functional as F

from functools import wraps
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.utils import (
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from dcu_megatron.core.pipeline_parallel import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)


def mlp_init_wrapper(mlp_init_func):
    @wraps(mlp_init_func)
    def wrapper(
        self,
        config,
        submodules,
        is_expert=False,
        input_size=None,
        ffn_hidden_size=None,
        tp_group=None,
    ):
        mlp_init_func(
            self,
            config,
            submodules,
            is_expert=is_expert,
            input_size=input_size,
            ffn_hidden_size=ffn_hidden_size,
            tp_group=tp_group,
        )

        self.offload_shared_fc1 = (
            config.fine_grained_activation_offloading
            and "shared_fc1" in config.offload_modules
        )

        self.offload_shared_fc2 = (
            config.fine_grained_activation_offloading
            and "shared_fc2" in config.offload_modules
        )

    return wrapper


class MLP():

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        if self.offload_shared_fc1 and self.training:
            hidden_states = fine_grained_offloading_group_start(hidden_states, name="linear_fc1")
        with get_fine_grained_offloading_context(self.offload_shared_fc1):
            intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        if self.offload_shared_fc1 and self.training:
            intermediate_parallel, bias_parallel = fine_grained_offloading_group_commit(
                intermediate_parallel, bias_parallel, name="linear_fc1", forced_released_tensors=[hidden_states]
            )
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")
        if self.offload_shared_fc2 and self.training:
            intermediate_parallel = fine_grained_offloading_group_start(intermediate_parallel, name="linear_fc2")
        with get_fine_grained_offloading_context(self.offload_shared_fc2):
            output, output_bias = self.linear_fc2(intermediate_parallel)
        if self.offload_shared_fc2 and self.training:
            output, output_bias = fine_grained_offloading_group_commit(
                output, output_bias, name="linear_fc2", forced_released_tensors=[intermediate_parallel]
            )
        nvtx_range_pop(suffix="linear_fc2")

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, output_bias
