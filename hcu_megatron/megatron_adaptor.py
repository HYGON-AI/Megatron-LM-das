# This code was adopted from https://gitcode.com/Ascend/MindSpeed
import argparse

from megatron.training import get_args

from .features_manager import ADAPTOR_FEATURES
from .patch_utils import MegatronPatchesManager
from hcu_megatron.training.arguments import process_adaptor_args


_ADAPTOR_ARGS = None


def add_args(args, key, value):
    if key is not None:
        key = key[2:].replace('-', '_')
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        setattr(args, key, value)


def parser_unknown_args(args, unknown):
    i = 0
    key = value = None
    while i < len(unknown):
        if unknown[i].startswith("--"):
            add_args(args, key, value)
            key = unknown[i]
            value = None
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)


def get_adaptor_args():
    global _ADAPTOR_ARGS
    if _ADAPTOR_ARGS is None:
        parser = argparse.ArgumentParser(description='Adaptor Arguments', allow_abbrev=False)
        _ADAPTOR_ARGS, unknown = process_adaptor_args(parser).parse_known_args()
        parser_unknown_args(_ADAPTOR_ARGS, unknown)
    return _ADAPTOR_ARGS


def destroy_adaptor_args():
    global _ADAPTOR_ARGS
    _ADAPTOR_ARGS = None


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
