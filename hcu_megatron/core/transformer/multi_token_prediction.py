

import contextlib
import warnings
from typing import Optional
from functools import wraps

import torch
from torch import Tensor

from megatron.training import get_args
from megatron.core.transformer.enums import LayerType
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.multi_token_prediction import get_mtp_layer_offset


def tie_word_embeddings_state_dict_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if get_args().schedule_method == "dualpipev":
            return

        fn(*args, **kwargs)

    return wrapper


def mtp_on_this_rank(
    config: TransformerConfig, ignore_virtual: Optional[bool] = True, vp_stage: Optional[int] = None
) -> bool:
    """
    Check if there is MTP on the current rank.

    Behavior:
        - If a custom pipeline model parallel layout is provided in the config:
            - If virtual pipeline parallelism is enabled (and `ignore_virtual` is False), checks
              whether any MTP layers are present on this (pp_rank, vp_stage) pair.
            - Otherwise, checks all virtual pipeline ranks of the current pipeline rank. Returns
              True if any virtual sub-rank includes at least one MTP layer.
        - If no custom layout is provided, assumes all MTP layers (if any) are placed on the last
          pipeline stage. The function returns True only on the last pipeline stage.
    """
    mtp_on_this_rank = False
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if config.pipeline_model_parallel_layout is not None:
        # with custom PP layout, we support put MTP layers on any pipeline stage
        layout = config.pipeline_model_parallel_layout.layout
        if (
            not ignore_virtual
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
            num_layers_to_build = layout[pp_rank][vp_stage].count(LayerType.mtp)
            mtp_on_this_rank = num_layers_to_build > 0
        else:
            for vpp_rank in range(len(layout[pp_rank])):
                num_layers_to_build = layout[pp_rank][vpp_rank].count(LayerType.mtp)
                if num_layers_to_build > 0:
                    mtp_on_this_rank = True
                    break
    else:
        # without custom PP layout, we only support put all of MTP layers on the last pipeline stage
        if config.mtp_num_layers is not None:
            if get_args().schedule_method == 'dualpipev':
                mtp_on_this_rank = parallel_state.is_pipeline_first_stage(
                    ignore_virtual=True
                )
            else:
                mtp_on_this_rank = parallel_state.is_pipeline_last_stage(
                    ignore_virtual=ignore_virtual, vp_stage=vp_stage
                )
        else:
            mtp_on_this_rank = False
    return mtp_on_this_rank


def get_mtp_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None, model=None,
) -> int:
    """Get the number of MTP layers to build."""

    args = get_args()
    dualpipev_first_chunk = getattr(model, "dualpipev_first_chunk", False) if model is not None else getattr(args, "dualpipev_first_chunk", False)
    if args.schedule_method == "dualpipev":
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True) and not dualpipev_first_chunk:
            return config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            return 0

    if config.pipeline_model_parallel_layout is not None:
        # If we have a custom PP layout, get the number of mtp layers in the layout array.
        num_layers_to_build = config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.mtp, vp_stage=vp_stage
        )
        assert num_layers_to_build == config.mtp_num_layers or num_layers_to_build == 0, (
            f"Currently, we only support put all of MTP layers on the last pipeline stage, "
            f"so the number of MTP layers to build ({num_layers_to_build}) must match "
            f"mtp_num_layers ({config.mtp_num_layers}) or be 0."
        )
    else:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
            num_layers_to_build = config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            num_layers_to_build = 0
    return num_layers_to_build


RecomputeMTPLayerFlag = False

def get_recompute_mtp_layer_flag():
    global RecomputeMTPLayerFlag
    return RecomputeMTPLayerFlag


def set_recompute_mtp_layer_flag(recompute_mtp_layer_flag):
    global RecomputeMTPLayerFlag
    RecomputeMTPLayerFlag = recompute_mtp_layer_flag


@contextlib.contextmanager
def _fork_recompute_mtp_layer_flag():
    # Store the current states.
    current_recompute_mtp_layer_flag = get_recompute_mtp_layer_flag()
    try:
        yield
    finally:
        # Set the states back to what it was at the start of this function.
        set_recompute_mtp_layer_flag(current_recompute_mtp_layer_flag)


class MultiTokenPredictionLayer:
    def _checkpointed_forward(self, forward_func, *args, **kwargs):
        def checkpoint_handler():
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8:
                from megatron.core.extensions.transformer_engine import te_checkpoint

                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    *args,
                    **kwargs,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func, self.config.distribute_saved_activations, *args, *kwargs.values()
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert (
                self.config.recompute_num_layers == 1
            ), "recompute_num_layers must be 1 for MTP recompute"
            outputs = checkpoint_handler()
        elif (
            get_args().recompute_layer_ids is not None
            or get_args().recompute_mtp_layer_ids is not None
        ):
            # layer id is in recompute_mtp_layer_ids
            if get_recompute_mtp_layer_flag():
                outputs = checkpoint_handler()
            else:
                outputs = forward_func(*args, **kwargs)
        elif self.config.recompute_method == 'block':
            # TODO: implement block-based recompute for MTP
            warnings.warn(
                "recompute_method == 'block' is not supported for MTP yet." " Skipping recompute."
            )
            outputs = forward_func(*args, **kwargs)
        else:
            raise ValueError("Invalid activation recompute method.")

        return outputs

    def backward_dw(self):
        self.eh_proj.backward_dw()
        if (
            hasattr(self, "mtp_model_layer")
            and hasattr(self.mtp_model_layer, "backward_dw")
            and callable(self.mtp_model_layer.backward_dw)
        ):
            self.mtp_model_layer.backward_dw()


class MultiTokenPredictionBlock:
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        extra_block_kwargs: Optional[dict] = None,
        embedding=None,
    ) -> Tensor:
        """
        Perform the forward pass through all of the MTP modules.

        Args:
            hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.

        Returns:
            (Tensor): The mtp loss tensor of shape [b, s].
        """

        if (
            get_args().schedule_method == "dualpipev"
            and embedding.word_embeddings.weight is None
        ):
            from hcu_megatron.core.models.common.language_module.language_module import get_shared_embedding_from_dual_chunk
            embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()

        # get hidden states from previous mtp stages
        offset = get_mtp_layer_offset(self.config, self.vp_stage)
        hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
        hidden_states = hidden_states_list[offset]
        for iteration in range(self.config.mtp_num_layers):
            layer_idx = 0 if self.mtp_use_repeated_layer else iteration
            global_iteration = iteration + offset
            with _fork_recompute_mtp_layer_flag():
                if get_args().recompute_mtp_layer_ids is not None:
                    set_recompute_mtp_layer_flag(global_iteration in get_args().recompute_mtp_layer_ids)
                (hidden_states, input_ids, position_ids) = self.layers[layer_idx](
                    input_ids=input_ids,
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    embedding=embedding,
                    **(extra_block_kwargs or {}),
                )

            # append the output hidden states of the current mtp layer
            # to the hidden_states_list
            hidden_states_list.append(hidden_states)

        # concat the hidden states of all mtp layers
        hidden_states = torch.cat(hidden_states_list, dim=0)
        return hidden_states

    def backward_dw(self):
        for layer in self.layers:
            layer.backward_dw()
