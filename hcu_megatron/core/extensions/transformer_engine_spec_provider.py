# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from functools import partial, wraps

from megatron.core.transformer.moe.experts import GroupedMLPSubmodules
from megatron.training import get_args


def te_spec_provider_grouped_mlp_modules_wrapper(fn):
    @wraps(fn)
    def wrapper(self, moe_use_grouped_gemm: bool):
        """Which module and submodules to use for grouped mlp"""

        args = get_args()

        if (
            args.use_primus_grouped_gemm
            and moe_use_grouped_gemm
        ):
            from hcu_megatron.core.transformer.moe.experts import PrimusTurboGroupedMLP

            from hcu_megatron.core.extensions.primus_turbo import (
                PrimusTurboColumnParallelGroupedLinear,
                PrimusTurboRowParallelGroupedLinear,
            )

            return partial(
                PrimusTurboGroupedMLP,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=PrimusTurboColumnParallelGroupedLinear,
                    linear_fc2=PrimusTurboRowParallelGroupedLinear,
                    activation_func=self.activation_func(),
                ),
            )

        return fn(self, moe_use_grouped_gemm)

    return wrapper
