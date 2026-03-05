from typing import Optional
from functools import wraps

import torch
from torch import Tensor

from megatron.training import get_args
from megatron.core import InferenceParams, parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.pipeline_parallel.utils import is_vp_last_stage
from megatron.core.transformer.multi_token_prediction import get_mtp_layer_offset

from dcu_megatron.core.pipeline_parallel import (
    fine_grained_offloading_set_last_layer,
)


def tie_word_embeddings_state_dict_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if get_args().schedule_method == "dualpipev":
            return

        fn(*args, **kwargs)

    return wrapper


def get_mtp_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None, model=None,
) -> int:
    """Get the number of MTP layers to build."""

    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    args = get_args()
    dualpipev_first_chunk = getattr(model, "dualpipev_first_chunk", False) if model is not None else getattr(args, "dualpipev_first_chunk", False)
    if args.schedule_method == "dualpipev":
        if is_first_pp_stage and not dualpipev_first_chunk:
            return config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            return 0

    vp_size = config.virtual_pipeline_model_parallel_size
    if is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_last_pp_stage:
        return config.mtp_num_layers if config.mtp_num_layers else 0
    else:
        return 0


class MultiTokenPredictionLayer:
    def backward_dw(self):
        self.eh_proj.backward_dw()
        self.transformer_layer.backward_dw()


class MultiTokenPredictionBlock:
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
        extra_block_kwargs: dict = None,
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
            from dcu_megatron.core.models.common.language_module.language_module import get_shared_embedding_from_dual_chunk
            embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()

        # get hidden states from previous mtp stages
        offset = get_mtp_layer_offset(self.config)
        hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
        hidden_states = hidden_states_list[offset]
        for layer_number in range(len(self.layers)):
            if self.config.fine_grained_activation_offloading:
                fine_grained_offloading_set_last_layer(layer_number == len(self.layers) - 1)

            (hidden_states, input_ids, position_ids) = self.layers[layer_number](
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
