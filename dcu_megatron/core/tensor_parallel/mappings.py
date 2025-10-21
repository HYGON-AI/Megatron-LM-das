import torch

from megatron.training import get_args

from dcu_megatron.core.quantization_utils import (
    fp16_to_int8s,
    int8s_to_fp16,
    destindex_copy_quantize_int8,
    destindex_dequantize_int8,
    destindex_copy_quantize_int4,
    destindex_dequantize_int4,
)
from dcu_megatron.adaptor.features_manager.communication.quantize_comm_feature import (
    QUANT_BIT_DEFAULT_GROUP_SIZE_MAP,
    QUANT_BIT_GROUP_SIZE_CHOICES_MAP,
)


def q_alltoall_int8(output, input, quant_group_size, output_split_sizes, input_split_sizes, group):
    t, s = input.shape[0], input.shape[1]

    assert s % quant_group_size == 0, f"size {s} should be divided by quant_group_size {quant_group_size}."
    num_quant_groups = s // quant_group_size

    input_all = torch.empty((t, num_quant_groups, quant_group_size + 4), dtype=torch.int8, device="cuda")
    buffer_scales = torch.empty((t, num_quant_groups, 2), dtype=torch.bfloat16, device="cuda")
    input_q = input.view(-1, num_quant_groups, quant_group_size)

    destindex_copy_quantize_int8(
        input_q,
        input_all[:, :, :quant_group_size],
        buffer_scales,
    )
    input_all[:, :, -4:] = fp16_to_int8s(buffer_scales) # size: [t, num_quant_groups, 4]. (scale_high, scale_low, shift_high, shift_low)

    torch.distributed.all_to_all_single(
        output,
        input_all.view(t, -1),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )

    output = output.view(-1, num_quant_groups, quant_group_size + 4)
    scales = int8s_to_fp16(output[:, :, -4:])
    dequant_out = torch.empty((output.shape[0], num_quant_groups, quant_group_size), dtype=torch.bfloat16, device="cuda")
    destindex_dequantize_int8(output[:, :, :-4], scales, dequant_out)

    return dequant_out.view(-1, s)


def q_alltoall_int4(output, input, quant_group_size, output_split_sizes, input_split_sizes, group):
    t, s = input.shape[0], input.shape[1]
    assert s % 2 == 0, f"size {s} should be an even number."
    assert s % quant_group_size == 0, f"size {s} should be divided by quant_group_size {quant_group_size}."

    num_quant_groups = s // quant_group_size

    input_all = torch.empty((t, 1, s // 2 + 4 * num_quant_groups), dtype=torch.int8, device="cuda")
    buffer_scales = torch.empty((t, 1, num_quant_groups * 2), dtype=torch.bfloat16, device="cuda")
    input_q = input.unsqueeze(1)

    destindex_copy_quantize_int4(
        input_q,
        input_all[:, :, :(s // 2)],
        buffer_scales,
        quant_group_size,
    )

    input_all = input_all.squeeze()
    input_all[:, (s // 2):] = fp16_to_int8s(buffer_scales.squeeze())  # size: [t, num_quant_groups * 4], (scale_high, scale_low, shift_high, shift_low)

    torch.distributed.all_to_all_single(
        output,      # output: [, s // 2 + num_quant_groups * 4]
        input_all,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )

    scales = int8s_to_fp16(output[:, (- 4 * num_quant_groups):]).view(-1, 2 * num_quant_groups).unsqueeze(1)

    dequant_out = torch.empty((output.shape[0], 1, s), dtype=torch.bfloat16, device="cuda")
    destindex_dequantize_int4(output[:,:s // 2].unsqueeze(1), scales, dequant_out, quant_group_size)

    return dequant_out.squeeze()


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            group,
            input,
            output_split_sizes,
            input_split_sizes,
            use_quantize_comm=False,
            quant_comm_bits: int = None,
            quant_group_size: int = None
        ):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.use_quantize_comm = use_quantize_comm
        ctx.quant_comm_bits = quant_comm_bits
        ctx.quant_group_size = quant_group_size

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        if use_quantize_comm:
            assert input.dtype == torch.bfloat16, "Only bfloat16 is supported"

        input = input.contiguous()
        input_dim = input.dim()
        assert input_dim <= 2, "Only supports 1-D or 2-D tensors."

        if output_split_sizes is None:
            # Equal split (all2all)
            if use_quantize_comm and input_dim > 1:
                if quant_comm_bits == 8:
                    output = input.new_empty(
                        size=[input.shape[0], input.shape[1] + input.shape[1] // ctx.quant_group_size * 4],
                        dtype=torch.int8,
                        device=torch.cuda.current_device(),
                    )
                else:
                    output = input.new_empty(
                        size=[input.shape[0], input.shape[1] // 2 + input.shape[1] // ctx.quant_group_size * 4],
                        dtype=torch.int8,
                        device=torch.cuda.current_device(),
                    )
            else:
                output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            if use_quantize_comm and input_dim > 1:
                if quant_comm_bits == 8:
                    output = input.new_empty(
                        size=[sum(output_split_sizes)] + list(input.size()[1:-1]) + [input.shape[-1]+ input.shape[-1] // ctx.quant_group_size * 4],
                        dtype=torch.int8,
                        device=torch.cuda.current_device(),
                    )
                else:
                    output = input.new_empty(
                        size=[sum(output_split_sizes)] + list(input.size()[1:-1]) + [input.shape[1] // 2 + input.shape[1] // ctx.quant_group_size * 4],
                        dtype=torch.int8,
                        device=torch.cuda.current_device(),
                    )

            else:
                output = input.new_empty(
                    size=[sum(output_split_sizes)] + list(input.size()[1:]),
                    dtype=input.dtype,
                    device=torch.cuda.current_device(),
            )

        if use_quantize_comm and input_dim > 1:
            if quant_comm_bits == 8:
                output = q_alltoall_int8(output, input, ctx.quant_group_size, output_split_sizes, input_split_sizes, group)
            else:
                output = q_alltoall_int4(output, input, ctx.quant_group_size, output_split_sizes, input_split_sizes, group)
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
        return (
            None,    # group
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes, ctx.use_quantize_comm, ctx.quant_comm_bits, ctx.quant_group_size),
            None,    # output_split_sizes
            None,    # input_split_sizes
            None,    # use_quantize_comm
            None,    # quant_comm_bits
            None,    # quant_group_size
        )


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None, use_quantize_comm=None, quant_comm_bits=None, quant_group_size=None):
    """Wrapper for autograd function"""
    if use_quantize_comm is None:
        args = get_args()
        use_quantize_comm = args.use_quantize_comm if hasattr(args, "use_quantize_comm") else False
        quant_comm_bits = args.quant_comm_bits
        quant_group_size = args.quant_group_size

    if input_.dtype != torch.bfloat16:
        use_quantize_comm = False

    if use_quantize_comm:
        if quant_comm_bits is None:
            quant_comm_bits = 8
        assert quant_comm_bits in {4, 8}, f"quant_comm_bits [{quant_comm_bits}] only accepts values from [4, 8]"

        if quant_group_size is None:
            quant_group_size = QUANT_BIT_DEFAULT_GROUP_SIZE_MAP[args.quant_comm_bits]
        assert quant_group_size in QUANT_BIT_GROUP_SIZE_CHOICES_MAP[args.quant_comm_bits]

    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes, use_quantize_comm, quant_comm_bits, quant_group_size)
