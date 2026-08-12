# This code was adopted from https://gitcode.com/Ascend/MindSpeed
import argparse

from megatron.training import get_args

from .features_manager import ADAPTOR_FEATURES
from .patch_utils import MegatronPatchesManager
from hcu_megatron.training.arguments import destroy_adaptor_args, get_adaptor_args


def patch_features():
    adaptor_args = get_adaptor_args()

    for feature in ADAPTOR_FEATURES:
        if (
            (getattr(adaptor_args, feature.feature_name, None) and feature.optimization_level == 2)
            or feature.default_patches
        ):
            feature.register_patches(MegatronPatchesManager, adaptor_args)

    MegatronPatchesManager.apply_patches()


def repatch(patch_adaptor_args, patch_megatron_args):
    destroy_adaptor_args()
    MegatronPatchesManager.remove_patches()

    adaptor_args = get_adaptor_args()
    for k, v in patch_adaptor_args.items():
        setattr(adaptor_args, k, v)

    megatron_args = get_args()
    for k, v in patch_megatron_args.items():
        setattr(megatron_args, k, v)

    patch_features()


patch_features()
