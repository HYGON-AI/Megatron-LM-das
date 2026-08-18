# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from functools import partial, wraps

from megatron.core.transformer.moe.experts import GroupedMLPSubmodules

from hcu_megatron.training.arguments import get_adaptor_args


def te_spec_provider_grouped_mlp_modules_wrapper(fn):
    @wraps(fn)
    def wrapper(self, moe_use_grouped_gemm: bool):
        """Which module and submodules to use for grouped mlp"""

        if (
            get_adaptor_args().use_primus_grouped_gemm
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
