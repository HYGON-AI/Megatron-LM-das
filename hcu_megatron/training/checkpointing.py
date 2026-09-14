# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""UltraEP checkpoint helpers."""

from functools import wraps
import re


_ULTRAEP_EXPERT_KEY = re.compile(r'(?:linear_fc1|linear_fc2)\.(?:weight|bias)(\d+)$')


def _filter_ultraep_native_model_state_dict(model_state_dict, model_chunks):
    """Drop physical replica slots from native model checkpoint state dicts."""
    if not model_chunks:
        return model_state_dict

    module = model_chunks[0]
    config = getattr(module, 'config', None)
    while config is None and hasattr(module, 'module'):
        module = module.module
        config = getattr(module, 'config', None)
    if config is None:
        return model_state_dict
    ep_size = getattr(config, 'expert_model_parallel_size', None)
    if not ep_size:
        return model_state_dict
    num_local_master = config.num_moe_experts // ep_size

    filtered = {}
    for key, value in model_state_dict.items():
        match = _ULTRAEP_EXPERT_KEY.search(key)
        if match and int(match.group(1)) >= num_local_master:
            continue
        filtered[key] = value
    return filtered


def generate_state_dict_ultraep_wrapper(original_func):
    """Filter UltraEP replica parameters from native torch checkpoints."""

    @wraps(original_func)
    def wrapper(args, model, optimizer, opt_param_scheduler, rng_state, *extra_args, **kwargs):
        state_dict = original_func(
            args, model, optimizer, opt_param_scheduler, rng_state, *extra_args, **kwargs
        )
        if getattr(args, 'ckpt_format', None) != 'torch_dist':
            for index in range(len(model)):
                key = 'model' if len(model) == 1 else f'model{index}'
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict[key] = _filter_ultraep_native_model_state_dict(
                        state_dict[key], [model[index]]
                    )
        return state_dict

    return wrapper
