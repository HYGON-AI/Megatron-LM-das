# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from functools import wraps


def linear_forward_main_grad_wrapper(forward_func):
    @wraps(forward_func)
    def linear_forward_main_grad(ctx,
                                 inputs,
                                 weight,
                                 bias,
                                 gradient_accumulation_fusion,
                                 allreduce_dgrad,
                                 sequence_parallel,
                                 grad_output_buffer,
                                 wgrad_deferral_limit,
                                 tp_group,
                                 ):
        output = forward_func(ctx,
                              inputs,
                              weight,
                              bias,
                              gradient_accumulation_fusion,
                              allreduce_dgrad,
                              sequence_parallel,
                              grad_output_buffer,
                              wgrad_deferral_limit,
                              tp_group,
                              )
        ctx.weight = weight
        return output

    return linear_forward_main_grad


def linear_backward_main_grad_wrapper(backward_func):
    @wraps(backward_func)
    def linear_backward_main_grad(ctx, grad_output):
        class NewCtx:
            pass
        new_ctx = NewCtx()
        inputs, _ = ctx.saved_tensors
        for key in dir(ctx):
            if key == 'saved_tensors':
                setattr(new_ctx, 'saved_tensors', (inputs, ctx.weight))
            elif key.startswith('__') or key == 'saved_variables':
                continue
            else:
                try:
                    getattr(ctx, key)
                except AttributeError:
                    continue
                setattr(new_ctx, key, getattr(ctx, key))
        return backward_func(new_ctx, grad_output)

    return linear_backward_main_grad
