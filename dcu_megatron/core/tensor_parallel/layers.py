import os
import warnings
from functools import wraps
from typing import Callable, List, Optional

try:
    import flux
    HAS_FLUX = True
except ImportError:
    HAS_FLUX = False

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.training import print_rank_0
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import (
    is_torch_min_version,
    prepare_input_tensors_for_wgrad_compute
)
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    VocabParallelEmbedding,
)
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    _reduce_scatter_along_first_dim,
    _gather_along_first_dim,
)
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.tensor_parallel.mappings import _reduce
from megatron.core.tensor_parallel.layers import (
    custom_fwd,
    custom_bwd,
    dist_all_gather_func,
    linear_with_frozen_weight,
    linear_with_grad_accumulation_and_async_allreduce
)

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False


def vocab_parallel_embedding_init(
    self,
    num_embeddings: int,
    embedding_dim: int,
    *,
    init_method: Callable,
    reduce_scatter_embeddings: bool = False,
    config: ModelParallelConfig,
    skip_weight_param_allocation: bool = False
):
    super(VocabParallelEmbedding, self).__init__()
    # Keep the input dimensions.
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.reduce_scatter_embeddings = reduce_scatter_embeddings
    self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
    # Divide the weight matrix along the vocaburaly dimension.
    (self.vocab_start_index, self.vocab_end_index) = (
        VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings,
            get_tensor_model_parallel_rank(),
            self.tensor_model_parallel_size,
        )
    )
    self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
    self.deterministic_mode = config.deterministic_mode

    # Allocate weights and initialize.
    if not skip_weight_param_allocation:
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)
    else:
        self.weight = None


@torch.compile(mode='max-autotune-no-cudagraphs')
def vocab_parallel_embedding_forward(self, input_, weight=None):
    """Forward.

    Args:
        input_ (torch.Tensor): Input tensor.
    """
    if weight is None:
        if self.weight is None:
            raise RuntimeError(
                "weight was not supplied to VocabParallelEmbedding forward pass "
                "and skip_weight_param_allocation is True."
            )
        weight = self.weight

    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
    else:
        masked_input = input_
    # Get the embeddings.
    if self.deterministic_mode:
        output_parallel = weight[masked_input]
    else:
        # F.embedding currently has a non-deterministic backward function
        output_parallel = F.embedding(masked_input, weight)
    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel[input_mask, :] = 0.0

    if self.reduce_scatter_embeddings:
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        output_parallel = output_parallel.transpose(0, 1).contiguous()
        output = reduce_scatter_to_sequence_parallel_region(output_parallel)
    else:
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output


class AGLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        transpose_weight=False,
        fw_ag_gemm_op=None,
        bw_gemm_rs_op=None,
    ):
        """Forward."""
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer
        ctx.transpose_weight = transpose_weight
        ctx.bw_gemm_rs_op = bw_gemm_rs_op

        if sequence_parallel:
            sequence_len, batch_size, input_hidden_size = input.size()
            output_hidden_size = weight.size(0)
            world_size = get_tensor_model_parallel_world_size()

            if fw_ag_gemm_op is None:
                fw_ag_gemm_op = flux.AGKernel(
                    get_tensor_model_parallel_group(),
                    1, # torch.distributed.get_world_size() // torch.cuda.device_count(),
                    sequence_len * batch_size * world_size,
                    output_hidden_size,
                    input_hidden_size,
                    input.dtype,
                    output_dtype=input.dtype,
                    transpose_weight=transpose_weight,
                    local_copy=False,
                    ring_mode=flux.AgRingMode.Auto,
                )

            output = fw_ag_gemm_op.forward(
                input.view(sequence_len * batch_size, -1),
                weight.t().contiguous() if transpose_weight else weight,
                bias=bias,
                input_scale=None,
                weight_scale=None,
                output_scale=None,
                fast_accum=False
            )
            output = output.view(sequence_len * world_size, batch_size, -1)
        else:
            output = torch.matmul(input, weight.t())

        torch.cuda.current_stream().synchronize()

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward."""
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit
        transpose_weight = ctx.transpose_weight
        bw_gemm_rs_op = ctx.bw_gemm_rs_op

        wgrad_compute = True
        if grad_output_buffer is not None:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        world_size = get_tensor_model_parallel_world_size()
        if wgrad_compute:
            if ctx.sequence_parallel:
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(
                    dim_size, input.dtype, "mpu"
                )
                handle = dist_all_gather_func(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input

        if ctx.sequence_parallel:
            sequence_len, batch_size, _ = grad_output.size()

            if bw_gemm_rs_op is None:
                input_hidden_size = weight.size(-1)
                bw_gemm_rs_op = flux.GemmRS(
                    get_tensor_model_parallel_group(),
                    1, # world_size // torch.cuda.device_count(),
                    sequence_len * batch_size,
                    input_hidden_size,
                    input.dtype,
                    input.dtype,
                    transpose_weight=transpose_weight,
                    fuse_reduction=False
                )

            grad_input = bw_gemm_rs_op.forward(
                grad_output.view(sequence_len * batch_size, -1),
                weight if transpose_weight else weight.t().contiguous(),
                bias=None,
                input_scale=None,
                weight_scale=None,
                output_scale=None,
                fast_accum=False
            )

            torch.cuda.current_stream().synchronize()
            grad_input = grad_input.view(sequence_len // get_tensor_model_parallel_world_size(), batch_size, -1)
        else:
            grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel and wgrad_compute:
            handle.wait()

        if wgrad_compute:
            grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
                grad_output, total_input
            )

        if not ctx.sequence_parallel and ctx.allreduce_dgrad:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if not ctx.sequence_parallel and ctx.allreduce_dgrad:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None


def ag_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = 0,
    transpose_weight: Optional[bool] = False,
    fw_ag_gemm_op=None,
    bw_gemm_rs_op=None
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Args:
        input (torch.Tensor required): input like torch.nn.functional.linear

        weight (torch.Tensor required): weight like torch.nn.functional.linear

        bias (torch.Tensor optional): bias like torch.nn.functional.linear

        gradient_accumulation_fusion (bool required): Perform the gradient
            accumulation fusion, requires the custom CUDA extension
            fused_weight_gradient_mlp_cuda module. To use
            gradient_accumulation_fusion you must install APEX with
            --cpp_ext and --cuda_ext. For example: "pip install
            --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
            " Note that the extension requires CUDA>=11. Otherwise, you
            must turn off gradient accumulation fusion."

        allreduce_dgrad (bool required): Do the allreduce of input gradients.
            The allreduce is done asynchronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.

        sequence_parallel (bool required): Indicates that sequence
            parallelism is used and thus in the forward pass the input is
            all gathered, and the backward pass the input gradients are
            reduce scattered.

        grad_output_buffer (List[torch.Tensor] optional): Buffer used to save
            output gradients when embedding table wgrad compute is deferred.
            Defaults to None.

        wgrad_deferral_limit (int optional): Limit on the number of
            micro-batches for which embedding weight gradient GEMM should be
            deferred. Disable by setting this to 0. Defaults to 0.

        transpose_weight: transpose weight.

        fw_ag_gemm_op: flux AGKernel for forward.

        bw_gemm_rs_op: flux GemmRS for backward.

    """

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        transpose_weight,
        fw_ag_gemm_op,
        bw_gemm_rs_op,
    ]

    if not ag_linear.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                ag_linear.warned = True

            if allreduce_dgrad:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                ag_linear.warned = True

    return AGLinear.apply(*args)


ag_linear.warned = False


class LinearRS(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        transpose_weight=False,
    ):
        """Forward."""
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.grad_output_buffer = grad_output_buffer
        ctx.transpose_weight = transpose_weight

        world_size = get_tensor_model_parallel_world_size()

        sequence_len, batch_size, _ = input.size()
        # input: 3D tensor whose order of dimension is [sequence, batch, hidden]
        input = input.view(sequence_len * batch_size, -1)
        output_hidden_size = weight.size(0)

        if sequence_parallel:
            gemm_rs_op = flux.GemmRS(
                get_tensor_model_parallel_group(),
                1, #world_size // torch.cuda.device_count(),
                sequence_len * batch_size,
                output_hidden_size,
                input.dtype,
                input.dtype,
                transpose_weight=transpose_weight,
                fuse_reduction=False,
            )
            output = gemm_rs_op.forward(
                input,
                weight.t().contiguous() if transpose_weight else weight,
                bias=bias,
                input_scale=None,
                weight_scale=None,
                output_scale=None,
                fast_accum=False,
            )
            output = output.view(sequence_len // world_size, batch_size, -1)
        else:
            output_buf = torch.empty(
                [sequence_len * batch_size, output_hidden_size],
                dtype=input.dtype,
                device=input.device,
                requires_grad=False
            )
            gemm_only_op = flux.GemmOnly(
                input_dtype=input.dtype,
                output_dtype=input.dtype,
                transpose_weight=transpose_weight,
                use_fp8_gemm=False,
            )
            output = gemm_only_op.forward(
                input,
                weight.t().contiguous() if transpose_weight else weight,
                bias=bias,
                output_buf=output_buf,
                input_scale=None,
                weight_scale=None,
                output_scale=None,
                fast_accum=False,
            )
            output = output.view(sequence_len, batch_size, -1)
            output = _reduce(output)

        # torch.cuda.current_stream().synchronize()
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Backward."""
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer
        wgrad_deferral_limit = ctx.wgrad_deferral_limit
        transpose_weight = ctx.transpose_weight

        wgrad_compute = True
        if grad_output_buffer is not None:
            if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
                grad_output_buffer.append(grad_output)
                wgrad_compute = False

        if wgrad:
            if ctx.sequence_parallel:
                dim_size = list(grad_output.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(
                    dim_size, grad_output.dtype, "mpu"
                )
                handle = dist_all_gather_func(
                    all_gather_buffer, grad_output, group=get_tensor_model_parallel_group(), async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_grad_output = all_gather_buffer
            else:
                total_grad_output = grad_output

        if ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()

            sequence_len, batch_size, output_hidden_size = grad_output.size()
            input_hidden_size = weight.size(-1)

            ag_kernel = flux.AGKernel(
                get_tensor_model_parallel_group(),
                1, #world_size // torch.cuda.device_count(),
                sequence_len * batch_size * world_size,
                input_hidden_size,
                output_hidden_size,
                grad_output.dtype,
                output_dtype=input.dtype,
                transpose_weight=transpose_weight,
                local_copy=False,
                ring_mode=flux.AgRingMode.Auto,
            )
            grad_input = ag_kernel.forward(
                grad_output.view(sequence_len * batch_size, -1),
                weight if transpose_weight else weight.t().contiguous(),
                bias=None,
                input_scale=None,
                weight_scale=None,
                output_scale=None,
                fast_accum=False,
            )

            torch.distributed.barrier()
            torch.cuda.current_stream().synchronize()
            grad_input = grad_input.contiguous().view(sequence_len * world_size, batch_size, -1)
        else:
            grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel and wgrad_compute:
            handle.wait()

        if wgrad_compute:
            total_grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
                total_grad_output, input
            )

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, total_grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, total_grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = total_grad_output.t().matmul(total_input)
        grad_bias = total_grad_output.sum(dim=0) if use_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


def linear_rs(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = 0,
    transpose_weight: Optional[bool] = False,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Args:
        input (torch.Tensor required): input like torch.nn.functional.linear

        weight (torch.Tensor required): weight like torch.nn.functional.linear

        bias (torch.Tensor optional): bias like torch.nn.functional.linear

        gradient_accumulation_fusion (bool required): Perform the gradient
            accumulation fusion, requires the custom CUDA extension
            fused_weight_gradient_mlp_cuda module. To use
            gradient_accumulation_fusion you must install APEX with
            --cpp_ext and --cuda_ext. For example: "pip install
            --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
            " Note that the extension requires CUDA>=11. Otherwise, you
            must turn off gradient accumulation fusion."

        allreduce_dgrad (bool required): Do the allreduce of input gradients.
            The allreduce is done asynchronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.

        sequence_parallel (bool required): Indicates that sequence
            parallelism is used and thus in the forward pass the input is
            all gathered, and the backward pass the input gradients are
            reduce scattered.

        grad_output_buffer (List[torch.Tensor] optional): Buffer used to save
            output gradients when embedding table wgrad compute is deferred.
            Defaults to None.

        wgrad_deferral_limit (int optional): Limit on the number of
            micro-batches for which embedding weight gradient GEMM should be
            deferred. Disable by setting this to 0. Defaults to 0.

        transpose_weight: transpose weight.
    """

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        transpose_weight,
    ]

    if not linear_rs.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_rs.warned = True

            if allreduce_dgrad:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_rs.warned = True

    return LinearRS.apply(*args)


linear_rs.warned = False


def column_parallel_linear_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        # flux params
        self.use_flux = False
        if "use_flux" in kwargs:
            self.use_flux = kwargs["use_flux"]
        elif hasattr(self.config, "use_flux"):
            self.use_flux = self.config.use_flux

        self.flux_transpose_weight = False
        if "flux_transpose_weight" in kwargs:
            self.flux_transpose_weight = kwargs["flux_transpose_weight"]
        elif hasattr(self.config, "flux_transpose_weight"):
            self.flux_transpose_weight = self.config.flux_transpose_weight

        if self.sequence_parallel:
            self.previous_flux_params = (None,) * 5
            self.fw_ag_gemm_op = None
            self.bw_gemm_rs_op = None

    return wrapper


class ColumnParallelLinearPatch(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    """

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        """Forward of ColumnParallelLinear

        Args:
            input_:
                3D tensor whose order of dimension is [sequence, batch, hidden]
            weight (optional):
                weight tensor to use, compulsory when skip_weight_param_allocation is True.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `gather_output` arg in the constructor will be used.

        Returns:
            - output
            - bias

        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                assert (
                    self.config.cpu_offloading is False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias = self.bias if not self.skip_bias_add else None

        if (
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        if self.config.defer_embedding_wgrad_compute:
            if (
                self.config.wgrad_deferral_limit == 0
                or len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit
            ):
                self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        if self.use_flux:
            assert HAS_FLUX, "flux is NOT installed"

            sequence_len, batch_size, input_hidden_size = input_parallel.size()
            output_hidden_size = weight.size(0)
            world_size = get_tensor_model_parallel_world_size()
            if self.sequence_parallel:
                current_flux_params = (
                    sequence_len,
                    batch_size,
                    input_hidden_size,
                    output_hidden_size,
                    input_parallel.dtype
                )

                if (
                    self.fw_ag_gemm_op is None
                    or current_flux_params != self.previous_flux_params
                ):
                    self.fw_ag_gemm_op = flux.AGKernel(
                        get_tensor_model_parallel_group(),
                        1, # torch.distributed.get_world_size() // torch.cuda.device_count(),
                        sequence_len * batch_size * world_size,
                        output_hidden_size,
                        input_hidden_size,
                        input_parallel.dtype,
                        output_dtype=input_parallel.dtype,
                        transpose_weight=self.flux_transpose_weight,
                        local_copy=False,
                        ring_mode=flux.AgRingMode.Auto,
                    )

                    self.bw_gemm_rs_op = flux.GemmRS(
                        get_tensor_model_parallel_group(),
                        1, # world_size // torch.cuda.device_count(),
                        sequence_len * batch_size * world_size,
                        input_hidden_size,
                        input_parallel.dtype,
                        input_parallel.dtype,
                        transpose_weight=self.flux_transpose_weight,
                        fuse_reduction=False
                    )

                self.previous_flux_params = current_flux_params

            self._forward_impl = ag_linear
        elif not weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad

        forward_params = {
            "input": input_parallel,
            "weight": weight,
            "bias": bias,
            "gradient_accumulation_fusion": self.gradient_accumulation_fusion,
            "allreduce_dgrad": allreduce_dgrad,
            "sequence_parallel": False if self.explicit_expert_comm else self.sequence_parallel,
            "grad_output_buffer": self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None,
            "wgrad_deferral_limit": self.config.wgrad_deferral_limit if self.config.defer_embedding_wgrad_compute else None,
        }
        if self.use_flux:
            forward_params.update({
                "transpose_weight": self.flux_transpose_weight,
                "fw_ag_gemm_op": self.fw_ag_gemm_op,
                "bw_gemm_rs_op": self.bw_gemm_rs_op,
            })

        output_parallel = self._forward_impl(**forward_params)

        gather_output = self.gather_output
        # Use the runtime gather output if it's set explicitly.
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output

        if gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


def row_parallel_linear_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        # flux params
        self.use_flux = False
        if "use_flux" in kwargs:
            self.use_flux = kwargs["use_flux"]
        elif hasattr(self.config, "use_flux"):
            self.use_flux = self.config.use_flux

        self.flux_transpose_weight = False
        if "flux_transpose_weight" in kwargs:
            self.flux_transpose_weight = kwargs["flux_transpose_weight"]
        elif hasattr(self.config, "flux_transpose_weight"):
            self.flux_transpose_weight = self.config.flux_transpose_weight

    return wrapper


class RowParallelLinearPatch(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X
    along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    """

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                assert (
                    self.config.cpu_offloading is False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if self.use_flux:
            self._forward_impl = linear_rs
        elif not self.weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        allreduce_dgrad = False

        forward_params = {
            "input": input_parallel,
            "weight": self.weight,
            "bias": self.bias if self.use_flux or not self.skip_bias_add else None,
            "gradient_accumulation_fusion": self.gradient_accumulation_fusion,
            "allreduce_dgrad": allreduce_dgrad,
            "sequence_parallel": False if not self.use_flux else self.sequence_parallel,
            "grad_output_buffer": None,
        }

        if self.use_flux:
            forward_params.update({"transpose_weight": self.flux_transpose_weight})

        output_parallel = self._forward_impl(**forward_params)

        if self.use_flux:
            return output_parallel, None if not self.skip_bias_add else self.bias

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
