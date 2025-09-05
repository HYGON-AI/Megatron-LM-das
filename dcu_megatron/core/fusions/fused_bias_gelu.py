import torch

@torch.compile(fullgraph=True)
def fused_bias_gelu(
    bias_parallel,
    intermediate_parallel, 
    tokens_per_expert,
    permuted_probs,
    gated_linear_unit,
    activation_func,
):
    if bias_parallel is not None:
        shape = intermediate_parallel.shape
        intermediate_parallel = torch.cat(
            [
                t + b
                for t, b in zip(
                    torch.split(
                        intermediate_parallel.view(-1, shape[-1]), tokens_per_expert
                    ),
                    bias_parallel,
                )
            ]
        ).view(shape)
    if gated_linear_unit:

        def glu(x):
            x = torch.chunk(x, 2, dim=-1)
            return activation_func(x[0]) * x[1]

        intermediate_parallel = glu(intermediate_parallel)
    else:
        intermediate_parallel = activation_func(intermediate_parallel)
    original_dtype = intermediate_parallel.dtype
    intermediate_parallel = intermediate_parallel * permuted_probs
    intermediate_parallel = intermediate_parallel.to(original_dtype)

    return intermediate_parallel