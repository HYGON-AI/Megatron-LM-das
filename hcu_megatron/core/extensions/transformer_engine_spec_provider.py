from functools import partial, wraps

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.training import get_args


def te_spec_provider_grouped_mlp_modules_wrapper(fn):
    @wraps(fn)
    def wrapper(self, moe_use_grouped_gemm: bool):
        """Which module and submodules to use for grouped mlp"""

        args = get_args()

        if moe_use_grouped_gemm and TEColumnParallelGroupedLinear is not None:
            return partial(
                TEGroupedMLP,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=TEColumnParallelGroupedLinear,
                    linear_fc2=TERowParallelGroupedLinear,
                    activation_func=self.activation_func(),
                ),
            )

        if (
            args.use_primus_grouped_gemm
            and moe_use_grouped_gemm
            and TEColumnParallelGroupedLinear is not None
        ):
            from hcu_megatron.core.transformer.moe.experts import PrimusTurboGroupedMLP

            return partial(
                PrimusTurboGroupedMLP,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=TEColumnParallelGroupedLinear,
                    linear_fc2=TERowParallelGroupedLinear,
                    activation_func=self.activation_func(),
                ),
            )

        return fn(self, moe_use_grouped_gemm)

    return wrapper
