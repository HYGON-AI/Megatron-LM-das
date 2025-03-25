from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import is_torch_min_version
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    VocabParallelEmbedding,
)

from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import VocabUtility


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
