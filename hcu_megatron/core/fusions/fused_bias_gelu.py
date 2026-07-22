import torch


@torch.compile(fullgraph=True)
def fused_bias_gelu(
    self,
    intermediate_parallel, 
    permuted_probs,
):

    if self.config.gated_linear_unit:
        with_glu_interleaving = (
            self.config.gated_linear_unit
            and self.config.moe_mlp_glu_interleave_size is not None
        )

        def glu(x):
            if with_glu_interleaving:
                x = self._remove_glu_interleaving(
                    x, self.config.moe_mlp_glu_interleave_size
                )
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
    original_dtype = intermediate_parallel.dtype
    intermediate_parallel = intermediate_parallel * permuted_probs
    intermediate_parallel = intermediate_parallel.to(original_dtype)

    return intermediate_parallel
