import torch

from .qcomm import q_alltoall


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes, use_quantize_comm=False):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.use_quantize_comm = use_quantize_comm

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            if use_quantize_comm:
                output = input.new_empty(
                    size=[input.shape[0], input.shape[1]+4],
                    dtype=torch.int8,
                    device=torch.cuda.current_device(),
                )
            else:
                output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            if use_quantize_comm:
                output = input.new_empty(
                    size=[sum(output_split_sizes)] + list(input.size()[1:]),
                    dtype=torch.int8,
                    device=torch.cuda.current_device(),
                )
            else:
                output = input.new_empty(
                    size=[sum(output_split_sizes)] + list(input.size()[1:]),
                    dtype=input.dtype,
                    device=torch.cuda.current_device(),
            )

        if use_quantize_comm:
            output = q_alltoall(output, input, output_split_sizes, input_split_sizes,group)
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
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes, ctx.use_quantize_comm),
            None,
            None,
            None,
        )


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None, use_quantize_comm=False):
    """Wrapper for autograd function"""
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes, use_quantize_comm)
