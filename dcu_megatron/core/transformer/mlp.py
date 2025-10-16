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

from dcu_megatron.core.transformer.cpu_offload import (
    PipelineOffloadManager,
    group_prefetch_offload_start,
    group_prefetch_offload_commit,
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
    def _offload_shared_fc1_forward(
        self,
        hidden_states,
    ):
        """Forward method with router fc1 activation offloading."""
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        hidden_states = group_prefetch_offload_start(hidden_states)

        hidden_states.offloading_activation = True

        with PipelineOffloadManager.get_instance():
            intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        def call_back():
            cur_stream = torch.cuda.current_stream()
            hidden_states.record_stream(cur_stream)
            hidden_states.untyped_storage().resize_(0)

        intermediate_parallel, bias_parallel = group_prefetch_offload_commit(
            intermediate_parallel,
            bias_parallel,
            offloaded_call_back=call_back
        )
        return intermediate_parallel, bias_parallel

    def _offload_shared_fc2_forward(
        self,
        intermediate_parallel,
    ):
        """Forward method with router fc2 activation offloading."""
        if not intermediate_parallel.is_contiguous():
            intermediate_parallel = intermediate_parallel.contiguous()

        intermediate_parallel = group_prefetch_offload_start(intermediate_parallel)

        intermediate_parallel.offloading_activation = True

        with PipelineOffloadManager.get_instance():
            output, output_bias = self.linear_fc2(intermediate_parallel)

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

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        if self.offload_shared_fc1:
            intermediate_parallel, bias_parallel = self._offload_shared_fc1_forward(hidden_states)
        else:
            intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        if self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of swiglu with per_token_scale in MLP.")
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
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

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
        if self.offload_shared_fc2:
            output, output_bias = self._offload_shared_fc2_forward(intermediate_parallel)
        else:
            output, output_bias = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")

        if per_token_scale is not None:
            assert output_bias is None, "Bias is not supported with per_token_scale"

        return output, output_bias

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()
	