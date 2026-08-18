# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
import torch

from megatron.training import get_args

from hcu_megatron.core.quantization_utils import (
    float_to_int8s,
    int8s_to_float,
    destindex_copy_quantize_int8,
    destindex_dequantize_int8,
    destindex_copy_quantize_int4,
    destindex_dequantize_int4,
)
from hcu_megatron.features_manager.communication.quantize_comm_feature import (
    QUANT_BIT_DEFAULT_GROUP_SIZE_MAP,
    QUANT_BIT_GROUP_SIZE_CHOICES_MAP,
)

DTYPE_NUM_BYTES_MAP = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
}

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def q_alltoall_int8(input, quant_group_size, quant_scale_dtype, output_split_sizes, input_split_sizes, group, use_nccl_stream=False):
    t, s = input.shape[0], input.shape[1]

    assert s % quant_group_size == 0, f"size {s} should be divided by quant_group_size {quant_group_size}."
    num_quant_groups = s // quant_group_size
    num_scale_bytes = 2 * DTYPE_NUM_BYTES_MAP[quant_scale_dtype]
    input_all = torch.empty((t, num_quant_groups, quant_group_size + num_scale_bytes), dtype=torch.int8, device="cuda")
    buffer_scales = torch.empty((t, num_quant_groups, 2), dtype=DTYPE_MAP[quant_scale_dtype], device="cuda")
    input_q = input.view(-1, num_quant_groups, quant_group_size)

    destindex_copy_quantize_int8(
        input_q,
        input_all[:, :, :quant_group_size],
        buffer_scales,
    )
    input_all[:, :, -num_scale_bytes:] = float_to_int8s(buffer_scales)  # size: [t, num_quant_groups, num_scale_bytes]


    if output_split_sizes is None:
        output = input.new_empty(
            size=[t, s + num_quant_groups * num_scale_bytes],      # allocate memory for scale
            dtype=torch.int8,
        )
    else:
        output = input.new_empty(
            size=[sum(output_split_sizes)] + list(input.size()[1:-1]) + [s + num_quant_groups * num_scale_bytes],
            dtype=torch.int8,
        )

    if use_nccl_stream:
        handle = torch.distributed.all_to_all_single(
            output,
            input_all.view(t, -1),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        handle.wait()
    else:
        torch.distributed.all_to_all_single(
            output,
            input_all.view(t, -1),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

    output = output.view(-1, num_quant_groups, quant_group_size + num_scale_bytes)
    scales = int8s_to_float(output[:, :, -num_scale_bytes:], output_dtype=DTYPE_MAP[quant_scale_dtype])
    dequant_out = torch.empty((output.shape[0], num_quant_groups, quant_group_size), dtype=torch.bfloat16, device="cuda")
    destindex_dequantize_int8(output[:, :, :-num_scale_bytes], scales, dequant_out)

    return dequant_out.view(-1, s)


def q_alltoall_int4(input, quant_group_size, quant_scale_dtype, output_split_sizes, input_split_sizes, group, use_nccl_stream=False):
    t, s = input.shape[0], input.shape[1]
    assert s % 2 == 0, f"size {s} should be an even number."
    assert s % quant_group_size == 0, f"size {s} should be divided by quant_group_size {quant_group_size}."

    num_quant_groups = s // quant_group_size
    num_scale_bytes = 2 * DTYPE_NUM_BYTES_MAP[quant_scale_dtype]
    input_all = torch.empty((t, 1, s // 2 + num_scale_bytes * num_quant_groups), dtype=torch.int8, device="cuda")
    buffer_scales = torch.empty((t, 1, num_quant_groups * 2), dtype=DTYPE_MAP[quant_scale_dtype], device="cuda")
    input_q = input.unsqueeze(1)

    destindex_copy_quantize_int4(
        input_q,
        input_all[:, :, :(s // 2)],
        buffer_scales,
        quant_group_size,
    )

    input_all = input_all.squeeze(1)
    input_all[:, (s // 2):] = float_to_int8s(buffer_scales).squeeze(1)  # size: [t, num_quant_groups * k], (scale_high, scale_low, shift_high, shift_low)

    if output_split_sizes is None:
        output = input.new_empty(
            size=[t, s // 2 + num_quant_groups * num_scale_bytes],
            dtype=torch.int8,
        )
    else:
        output = input.new_empty(
            size=[sum(output_split_sizes)] + list(input.size()[1:-1]) + [s // 2 + num_quant_groups * num_scale_bytes],
            dtype=torch.int8,
        )

    if use_nccl_stream:
        handle = torch.distributed.all_to_all_single(
            output,
            input_all,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        handle.wait()
    else:
        torch.distributed.all_to_all_single(
            output,
            input_all,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

    scales = int8s_to_float(output[:, (s // 2):], output_dtype=DTYPE_MAP[quant_scale_dtype]).unsqueeze(1)
    dequant_out = torch.empty((output.shape[0], 1, s), dtype=torch.bfloat16, device="cuda")
    destindex_dequantize_int4(output[:, :(s // 2)].unsqueeze(1), scales, dequant_out, quant_group_size)

    return dequant_out.squeeze(1)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            group,
            input,
            output_split_sizes,
            input_split_sizes,
            use_nccl_stream=False,
            use_quantize_comm=False,
            quant_comm_bits: int = None,
            quant_group_size: int = None,
            quant_scale_dtype: str = "bf16",
        ):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.use_nccl_stream = use_nccl_stream
        ctx.use_quantize_comm = use_quantize_comm
        ctx.quant_comm_bits = quant_comm_bits
        ctx.quant_group_size = quant_group_size
        ctx.quant_scale_dtype = quant_scale_dtype

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        if use_quantize_comm:
            assert input.dtype == torch.bfloat16, "Only bfloat16 is supported"

        input = input.contiguous()
        input_dim = input.dim()
        assert input_dim <= 2, "Only supports 1-D or 2-D tensors."

        if use_quantize_comm and input_dim > 1:
            if quant_comm_bits == 8:
                output = q_alltoall_int8(
                    input,
                    quant_group_size,
                    quant_scale_dtype,
                    output_split_sizes,
                    input_split_sizes,
                    group,
                    use_nccl_stream=use_nccl_stream,
                )
            else:
                output = q_alltoall_int4(
                    input,
                    quant_group_size,
                    quant_scale_dtype,
                    output_split_sizes,
                    input_split_sizes,
                    group,
                    use_nccl_stream=use_nccl_stream,
                )
        else:
            if output_split_sizes is None:
                output = torch.empty_like(input)
            else:
                # Unequal split (all2all-v)
                output = input.new_empty(
                    size=[sum(output_split_sizes)] + list(input.size()[1:]),
                    dtype=input.dtype,
                    device=torch.cuda.current_device(),
                )

            if use_nccl_stream:
                handle = torch.distributed.all_to_all_single(
                    output,
                    input,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                    async_op=True,
                )
                handle.wait()
            else:
                torch.distributed.all_to_all_single(
                    output,
                    input,
                    output_split_sizes=output_split_sizes,
                    input_split_sizes=input_split_sizes,
                    group=group,
                )

        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        input_grad = _AllToAll.apply(
            ctx.group,
            *grad_output,
            ctx.input_split_sizes,
            ctx.output_split_sizes,
            ctx.use_nccl_stream,
            ctx.use_quantize_comm,
            ctx.quant_comm_bits,
            ctx.quant_group_size,
            ctx.quant_scale_dtype,
        )

        return (
            None,          # group
            input_grad,    # input
            None,          # output_split_sizes
            None,          # input_split_sizes
            None,          # use_nccl_stream
            None,          # use_quantize_comm
            None,          # quant_comm_bits
            None,          # quant_group_size
            None,          # quant_scale_dtype
        )


def all_to_all(
        group,
        input_,
        output_split_sizes_=None,
        input_split_sizes=None,
        use_nccl_stream=False,
        use_quantize_comm=None,
        quant_comm_bits=None,
        quant_group_size=None,
        quant_scale_dtype="bf16",
    ):
    """Wrapper for autograd function"""
    args = get_args()
    if use_quantize_comm is None:
        use_quantize_comm = args.use_quantize_comm if hasattr(args, "use_quantize_comm") else False
        quant_comm_bits = args.quant_comm_bits
        quant_group_size = args.quant_group_size
        quant_scale_dtype = args.quant_scale_dtype

    if input_.dtype != torch.bfloat16:
        use_quantize_comm = False

    if use_quantize_comm:
        if quant_comm_bits is None:
            quant_comm_bits = 8
        assert quant_comm_bits in {4, 8}, f"quant_comm_bits [{quant_comm_bits}] only accepts values from [4, 8]"

        if quant_group_size is None:
            quant_group_size = QUANT_BIT_DEFAULT_GROUP_SIZE_MAP[args.quant_comm_bits]
        assert quant_group_size in QUANT_BIT_GROUP_SIZE_CHOICES_MAP[args.quant_comm_bits]

    return _AllToAll.apply(
        group,
        input_,
        output_split_sizes_,
        input_split_sizes,
        use_nccl_stream,
        use_quantize_comm,
        quant_comm_bits,
        quant_group_size,
        quant_scale_dtype,
    )
