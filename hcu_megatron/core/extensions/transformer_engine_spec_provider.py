from functools import wraps

from megatron.training import get_args


def te_spec_provider_grouped_mlp_modules_wrapper(fn):
    @wraps(fn)
    def wrapper(self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool):
        """Which module and submodules to use for grouped mlp"""

        args = get_args()

        if (
            moe_use_grouped_gemm
            and args.use_primus_grouped_mlp
        ):
            from hcu_megatron.core.transformer.moe.experts import PrimusTurboGroupedMLP
            return PrimusTurboGroupedMLP, None

        return fn(self, moe_use_grouped_gemm, moe_use_legacy_grouped_gemm)

    return wrapper
