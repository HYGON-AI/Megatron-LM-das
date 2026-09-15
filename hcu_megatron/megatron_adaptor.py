# This code was adopted from https://gitcode.com/Ascend/MindSpeed
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.training.global_vars import destroy_global_vars, set_global_variables
from megatron.training.arguments import parse_args, validate_args

from .features_manager import ADAPTOR_FEATURES
from .patch_utils import MegatronPatchesManager
from hcu_megatron.training.arguments import (
    destroy_adaptor_args,
    get_adaptor_args,
    parse_adaptor_args,
    set_adaptor_args,
)


def patch_features(adaptor_args):
    set_adaptor_args(adaptor_args)
    for feature in ADAPTOR_FEATURES:
        if (
            (getattr(adaptor_args, feature.feature_name, None) and feature.optimization_level == 2)
            or feature.default_patches
        ):
            feature.register_patches(MegatronPatchesManager, get_adaptor_args())

    MegatronPatchesManager.apply_patches()


def repatch(patch_adaptor_args=None, patch_megatron_args=None, skip_validate=True):
    MegatronPatchesManager.remove_patches()

    destroy_global_vars()
    destroy_num_microbatches_calculator()
    megatron_args = parse_args()

    destroy_adaptor_args()
    adaptor_args = parse_adaptor_args()
    if patch_adaptor_args is not None:
        for k, v in patch_adaptor_args.items():
            setattr(adaptor_args, k, v)
            setattr(megatron_args, k, v)
    set_adaptor_args(adaptor_args)

    if patch_megatron_args is not None:
        for k, v in patch_megatron_args.items():
            setattr(megatron_args, k, v)

    if not skip_validate:
        validate_args(megatron_args)
    set_global_variables(megatron_args, False)
    patch_features(get_adaptor_args())


patch_features(parse_adaptor_args())
